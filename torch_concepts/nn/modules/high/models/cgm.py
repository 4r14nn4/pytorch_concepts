"""Causal Concept Graph Model with joint structure and parameter learning."""

from __future__ import annotations

import copy
import random
from functools import cached_property, partial
from typing import Optional, Sequence, Union

import networkx as nx
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal, OneHotCategorical

from .....annotations import Annotations
from .....concept_graph import ConceptGraph
from .....construct_graph import GraphGeneratorLearnable
from .....distributions import Delta
from .....utils import ensure_list
from ...low.dense_layers import MLP
from ...low.graph_aggregator import GraphAggregator
from ...low.intervention.intervention import InterventionModule
from ...low.base.intervention import BaseInterventionPolicy
from ...low.intervention.strategy.do import DoIntervention
from ...low.sequential import Sequential
from ...low.priors import LearnablePrior
from ...low.predictors.mix import (
    MixConceptEmbeddingToConceptEmbedding,
)
from ...utils import state_embedding_counts
from ...low.predictors.neural_structural_equations import (
    NeuralStructuralEquations,
)
from ...mid.distributions import DEFAULT_DIST_KWARGS
from ...mid.factors.cpd import ParametricCPD
from ...mid.graph.bayesian_network import BayesianNetwork
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.deterministic import DeterministicInference
from ...mid.inference.torch.independent import IndependentInference
from ...mid.variable import ConceptVariable, EmbeddingVariable
from ...outputs import ModelOutput
from ..base.graph import DirectedGraphModel


def _project_to_dag(adjacency: torch.Tensor) -> torch.Tensor:
    """Project an adjacency to a DAG as in the original CausalCGM."""
    adjacency = adjacency.detach().clone()
    while True:
        graph = nx.from_numpy_array(
            adjacency.cpu().numpy(), create_using=nx.DiGraph
        )
        try:
            list(nx.topological_sort(graph))
            return adjacency
        except nx.NetworkXUnfeasible:
            cyclic_edges = {
                edge
                for component in nx.strongly_connected_components(graph)
                if len(component) > 1
                for edge in graph.subgraph(component).edges
            }
            candidates = adjacency.clone()
            candidates[candidates == 0] = 100
            mask = torch.ones_like(candidates, dtype=torch.bool)
            for edge in cyclic_edges:
                mask[edge] = False
            candidates[mask] = 100
            weakest = torch.unravel_index(
                candidates.argmin(), candidates.shape
            )
            adjacency[weakest] = 0


class _CGMInterventionPolicy(BaseInterventionPolicy):
    """Select explicit flattened concept columns per batch row."""

    def __init__(self, indices: torch.Tensor):
        super().__init__()
        self.register_buffer("indices", indices.long())

    def forward(self, concepts, *args, **kwargs):
        indices = self.indices.to(concepts.device)
        if indices.dim() == 1:
            indices = indices.unsqueeze(-1)
        return torch.ones_like(concepts).scatter(1, indices, 0.0)


def _select_params(output, names, prefix=""):
    """Select named variable views and optionally prefix their quantities."""
    selected = {}
    for quantity, tensor in output.params.items():
        labels = {
            *tensor.annotation.label_to_index,
            *tensor.annotation.label_groups,
        }
        present = [name for name in names if name in labels]
        if present:
            selected[f"{prefix}{quantity}"] = tensor[present]
    return selected


class CausalCGM(DirectedGraphModel):
    """CGM trained as local conditionals and unfolded to an evaluation BN.

    Training keeps the original exogenous contexts ``U`` and independent
    copies ``V_prime``. Endogenous variables are grouped into the minimum
    number of homogeneous plates, so each plate mixes its inputs, aggregates
    the graph, and evaluates its structural equations once. The learned or
    fixed adjacency is produced by a neural graph layer, not by an extra PGM variable.
    Structural equations use a shared transform followed by child-specific
    MLPs. Because the relaxed adjacency may be cyclic, these plate conditionals
    form the training objective. During evaluation the adjacency
    is projected to a DAG and unfolded into a :class:`BayesianNetwork`, queried
    directly by the configured inference engine.
    For more details, see the `paper <https://arxiv.org/abs/2405.16507>`_.

    Parameters
    ----------
    input_size : int
        Dimensionality of input features (after the backbone, if any).
    annotations : Annotations
        Concept annotations (labels, cardinalities, types).
    task_names : Union[Sequence[str], str], optional
        Names of task variables. They are excluded from random training
        interventions and, by default, cannot have outgoing edges.
    embedding_size : int, default 8
        Width of each per-concept state embedding.
    shared_n_layers : int, default 0
        Hidden layers in the shared structural equations.
    shared_hidden_size : Optional[int], default None
        Shared-equation hidden width; defaults to ``embedding_size``.
    concept_n_layers : int, default 1
        Hidden layers in each concept-specific structural equation.
    concept_hidden_size : Optional[int], default None
        Concept-specific hidden width; defaults to ``embedding_size``.
    shared_activation : str, default "leaky_relu"
        Activation inside the shared structural MLP and automatically applied
        to its output.
    concept_activation : str, default "leaky_relu"
        Activation inside each concept-specific structural MLP.
    graph : Optional[ConceptGraph], default None
        A pre-learned or otherwise fixed DAG. Pass either ``graph`` or
        ``graph_generator``; when both are omitted, DAGMA-CGM is created.
    graph_generator : Optional[GraphGeneratorLearnable], default None
        Learnable graph generator; ``None`` creates the paper's DAGMA-CGM
        generator with its default configuration. Graph-specific options such
        as ``no_out_task``, ``edges_to_check``, ``initialization``, and
        ``initialization_data`` belong to the generator. If provided,
        it must be a ``GraphGeneratorLearnable``.
    inference : Optional[type[BaseInference]], default DeterministicInference
        Inference engine class for evaluation (see :class:`BaseInference`).
    inference_kwargs : Optional[dict], default None
        Keyword arguments for the evaluation inference engine.
    train_inference : Optional[type[BaseInference]], default IndependentInference
        Inference engine for the training Bayesian network.
    train_inference_kwargs : Optional[dict], default None
        Keyword arguments for the training inference engine; teacher forcing
        Keyword arguments forwarded unchanged to the training engine.
    run_interventions : bool, default True
        If True, each training forward also evaluates the random intervention
        scenarios used by the CACE regularizer. Disable this to compute only
        the observational prior and posterior predictions.
    lightning : bool, default False
        If True, adds Lightning training capabilities.
    **kwargs
        Forwarded to :class:`BaseModel` (e.g. ``backbone``, ``latent_size``).
    """

    supported_concept_types = frozenset({"binary", "categorical", "continuous"})

    # Per-type distribution policy: how this model models each concept type.
    variable_distributions = {
        "binary": Bernoulli,
        "categorical": OneHotCategorical,
        "continuous": Normal,
    }
    variable_dist_kwargs = dict(DEFAULT_DIST_KWARGS)

    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        task_names: Optional[Union[Sequence[str], str]] = None,
        embedding_size: int = 8,
        shared_n_layers: int = 0,
        shared_hidden_size: Optional[int] = None,
        concept_n_layers: int = 1,
        concept_hidden_size: Optional[int] = None,
        shared_activation: str = "leaky_relu",
        concept_activation: str = "leaky_relu",
        graph: Optional[ConceptGraph] = None,
        graph_generator: Optional[GraphGeneratorLearnable] = None,
        inference: Optional[type[BaseInference]] = DeterministicInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[type[BaseInference]] = IndependentInference,
        train_inference_kwargs: Optional[dict] = None,
        run_interventions: bool = True,
        lightning: bool = False,
        **kwargs,
    ) -> None:
        task_names = ensure_list(task_names) if task_names is not None else []
        if graph is not None and graph_generator is not None:
            raise ValueError("Pass either `graph` or `graph_generator`, not both.")
        if graph is None and graph_generator is None:
            graph_generator = GraphGeneratorLearnable(
                name="dagma_cgm",
                source="DAGMA_CGM",
                concept_names=list(annotations.labels),
                task_names=task_names,
            )
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            lightning=lightning,
            plate=True,
            graph=graph,
            **kwargs,
        )
        self.task_names = task_names
        self.copy_names = [f"{name}__copy" for name in self.concept_names]
        if not set(self.task_names).issubset(self.concept_names):
            raise ValueError("task_names must be present in annotations.")
        self.n_concepts = len(self.concept_names) - len(self.task_names)
        if self.n_concepts == 0:
            raise ValueError("CausalCGM requires at least one intervenable concept.")
        self.run_interventions = bool(run_interventions)
        self.intervention_indices = [
            index for index, name in enumerate(self.concept_names)
            if name not in self.task_names
        ]
        self.embedding_size = embedding_size
        self.structural_equation_kwargs = {
            "shared_n_layers": shared_n_layers,
            "shared_hidden_size": shared_hidden_size,
            "concept_n_layers": concept_n_layers,
            "concept_hidden_size": concept_hidden_size,
            "shared_activation": shared_activation,
            "concept_activation": concept_activation,
        }

        self.endogenous_types = [
            self.concept_annotations.concept(name).type
            for name in self.concept_names
        ]
        self.cardinalities = [
            self.concept_annotations.concept(name).cardinality
            for name in self.concept_names
        ]
        self.n_state_embeddings = state_embedding_counts(
            self.endogenous_types, self.cardinalities
        )
        if graph_generator is not None and not isinstance(graph_generator, GraphGeneratorLearnable):
            raise TypeError("`graph_generator` must be a GraphGeneratorLearnable.")
        if graph_generator is not None and list(graph_generator.concept_names) != list(self.concept_names):
            raise ValueError("Graph generator concept names must match annotations.")
        self.graph_layer = GraphAggregator(
            generator=graph_generator,
            adjacency=None if graph_generator is not None else self.graph.data,
        )
        self._graph_generator_validated = False
        configure_loss_terms = getattr(
            getattr(self, "loss", None), "configure_terms", None
        )
        if configure_loss_terms is not None:
            configure_loss_terms(self.graph_generator)
            if hasattr(self.loss, "task_names"):
                self.loss.task_names = list(self.task_names)
        if self.graph_generator is not None:
            self._validate_graph_generator_compatibility()
        self._build_model()
        self.setup_inference(
            inference or DeterministicInference,
            inference_kwargs,
            train_inference or IndependentInference,
            train_inference_kwargs,
        )

    def _resolve_graph(self):
        """Use an empty DAG until the learned structure is materialized."""
        if self._given_graph is not None:
            return self._given_graph
        adjacency = torch.zeros(len(self.concept_names), len(self.concept_names))
        return ConceptGraph(adjacency, node_names=list(self.concept_names))

    @property
    def graph_generator(self):
        return self.graph_layer.generator

    @property
    def fixed_adjacency(self):
        return self.graph_layer.fixed_adjacency

    def _adjacency(self) -> torch.Tensor:
        """Return the trainable or fixed graph adjacency."""
        adjacency = self.graph_layer.graph()
        if self.graph_generator is not None and not self._graph_generator_validated:
            self._validate_graph_generator_compatibility(adjacency)
        return adjacency

    def _validate_graph_generator_compatibility(
        self, adjacency: Optional[torch.Tensor] = None,
    ) -> None:
        """Validate the adjacency contract required by CausalCGM."""
        if adjacency is None:
            with torch.no_grad():
                adjacency = self.graph_generator()
        expected = (len(self.concept_names), len(self.concept_names))
        if not isinstance(adjacency, torch.Tensor) or adjacency.shape != expected:
            raise ValueError(
                "CausalCGM graph generators must return a tensor with shape "
                f"{expected}."
            )
        if not torch.isfinite(adjacency).all():
            raise ValueError("CausalCGM graph adjacencies must be finite.")
        if (adjacency < 0).any():
            raise ValueError("CausalCGM graph adjacencies must be non-negative.")
        if not torch.allclose(
            adjacency.diagonal(), torch.zeros_like(adjacency.diagonal())
        ):
            raise ValueError("CausalCGM graph adjacencies must have a zero diagonal.")
        if (
            getattr(self.graph_generator, "no_out_task", False)
            and self.task_names
            and adjacency[
                [self.concept_names.index(name) for name in self.task_names]
            ].any()
        ):
            raise ValueError(
                "With no_out_task=True, task rows in the adjacency must be zero."
            )
        self._graph_generator_validated = True

    def _input_latent_block(self):
        """Build the input and shared latent variables and their CPDs.

        The extra MLP after the backbone matches the original CausalCGM encoder.
        """
        input_var = EmbeddingVariable(
            "input", distribution=Delta, shape=self.input_size
        )
        shared_var = EmbeddingVariable(
            "shared_embedding", distribution=Delta, size=self.embedding_size
        )
        shared_encoder = Sequential(
            self.backbone,
            MLP(
                self.latent_size,
                self.embedding_size,
                output_size=self.embedding_size,
                n_layers=2,
                activation="leaky_relu",
            ),
        )
        input_cpd = ParametricCPD(input_var, LearnablePrior(input_var.shape))
        shared_cpd = ParametricCPD(
            shared_var, shared_encoder, parents=[input_var]
        )
        return input_var, shared_var, input_cpd, shared_cpd

    def build_concept_embedding_variables(
        self, names, embedding_size, plate_name, name_fmt="{}_embedding",
    ):
        """Build one state-conditioned embedding bank per concept."""
        return [
            EmbeddingVariable(
                name_fmt.format(name), distribution=Delta,
                size=n_states * embedding_size,
            )
            for name, n_states in zip(names, self.n_state_embeddings)
        ]

    @property
    def mixer(self):
        return self._mixer

    @property
    def structural_equations(self):
        return self._structural_equations

    def _parametrization(self, variable, head):
        """Build one head, plus an independent scale head for a Normal."""
        second = copy.deepcopy(head) if "loc" in variable.param_sizes else None
        return self._flexible_parametrization(variable, head, second=second)

    def _aggregate_inputs(
        self, inputs, *, exogenous, endogenous, sources, root=None,
    ):
        """Prepare the inputs consumed by a structural equation."""
        if not sources:
            return inputs[exogenous[root]]
        aggregated = {
            "concept_embeddings": [
                inputs[exogenous[source]].unflatten(
                    -1, (
                        self.n_state_embeddings[source],
                        self.embedding_size,
                    ),
                )
                for source in sources
            ],
            "concept_values": [
                inputs[endogenous[source]] for source in sources
            ],
        }
        if root is not None:
            aggregated["source_concepts"] = sources
        return aggregated

    def _build_model(self) -> None:
        input_var, shared_var, self.input_cpd, self.shared_cpd = self._input_latent_block()
        exogenous = self.build_concept_embedding_variables(
            self.concept_names, self.embedding_size, "exogenous",
            name_fmt="{}__u",
        )
        endogenous = self.build_concept_variables(
            self.concept_names, "endogenous"
        )
        endogenous_copies = ConceptVariable(
            self.copy_names,
            distribution=[
                self.distribution_of(name) for name in self.concept_names
            ],
            size=self.cardinalities,
            dist_kwargs=[
                self.dist_kwargs_of(name) for name in self.concept_names
            ],
        )

        exogenous_encoders = [
            Sequential(
                MLP(
                    self.embedding_size, variable.size,
                    output_size=variable.size, n_layers=1,
                    activation="leaky_relu",
                ),
                nn.LeakyReLU(),
            )
            for variable in exogenous
        ]
        structural_equations = NeuralStructuralEquations(
            in_embeddings=self.embedding_size,
            out_concept_types=self.endogenous_types,
            cardinalities_out_concepts=self.cardinalities,
            **self.structural_equation_kwargs,
        )
        object.__setattr__(self, "_structural_equations", structural_equations)
        self._mixer = MixConceptEmbeddingToConceptEmbedding(
            in_embeddings=self.embedding_size,
            concept_types=self.endogenous_types,
            cardinalities=self.cardinalities,
            expand_binary_embeddings=False,
            complete_output=True,
        )
        self.exogenous_cpds = nn.ModuleList([
            ParametricCPD(variable, encoder, parents=[shared_var])
            for variable, encoder in zip(exogenous, exogenous_encoders)
        ])
        self.endogenous_copy_cpds = nn.ModuleList([
            ParametricCPD(
                copy_var,
                self._parametrization(
                    copy_var,
                    structural_equations.concept_structural_equations[node],
                ),
                parents=[exogenous_var],
            )
            for node, (copy_var, exogenous_var) in enumerate(zip(
                endogenous_copies, exogenous
            ))
        ])

        aggregate = partial(
            self._aggregate_inputs,
            exogenous=exogenous,
            endogenous=endogenous_copies,
            sources=tuple(range(len(self.concept_names))),
        )

        indices = {name: index for index, name in enumerate(self.concept_names)}
        self.endogenous_cpds = nn.ModuleList()
        for variable in endogenous:
            targets = [indices[name] for name in variable.members]

            equations = (
                structural_equations
                if targets == list(range(len(self.concept_names)))
                else structural_equations.for_targets(targets)
            )
            cpd = ParametricCPD(
                variable,
                self._parametrization(variable, equations),
                parents=[*exogenous, *endogenous_copies],
                aggregate=aggregate,
                trunk=Sequential(self.mixer, self.graph_layer),
            )
            self.endogenous_cpds.append(cpd)

        self.pgm = BayesianNetwork(
            variables=[
                input_var, shared_var, *exogenous,
                *endogenous_copies, *endogenous,
            ],
            factors=[
                self.input_cpd, self.shared_cpd,
                *self.exogenous_cpds, *self.endogenous_copy_cpds,
                *self.endogenous_cpds,
            ],
        )

    def setup_inference(
        self,
        inference=None,
        inference_kwargs=None,
        train_inference=None,
        train_inference_kwargs=None,
    ):
        """Configure ordinary inference engines for the two CGM graphs.

        Training-specific graph inputs and intervention scenarios are prepared
        by :meth:`_training_run` and :meth:`_training_interventions`; they are
        deliberately not responsibilities of the inference engine.
        """
        if train_inference is not None:
            super().setup_inference(
                train_inference=train_inference,
                train_inference_kwargs=train_inference_kwargs,
            )
        if inference is not None:
            self._eval_inference_cls = inference
            self._eval_inference_kwargs = dict(inference_kwargs or {})

    def materialize_graph(self):
        """Materialize the learned graph through its generator pipeline."""
        if self.graph_generator is None:
            graph = ConceptGraph(
                _project_to_dag(self._adjacency()),
                node_names=self.concept_names,
            )
        else:
            graph = self.graph_generator.construct_graph(
                transform=_project_to_dag,
            )
        self._set_graph(graph)
        return graph

    def materialize_bayesian_network(self):
        """Materialize and install the evaluation Bayesian network."""
        graph = self.materialize_graph()
        exogenous = [cpd.variable for cpd in self.exogenous_cpds]
        endogenous = [
            self._make_concept_variable(name) for name in self.concept_names
        ]
        indices = {name: index for index, name in enumerate(self.concept_names)}
        factors = []
        for node, name in enumerate(self.concept_names):
            parents = [
                indices[parent] for parent in graph.get_predecessors(name)
            ]

            aggregate = partial(
                self._aggregate_inputs,
                exogenous=exogenous,
                endogenous=endogenous,
                sources=parents,
                root=node,
            )

            parametrization = (
                dict(self.endogenous_copy_cpds[node].parametrization)
                if not parents else
                self._parametrization(
                    endogenous[node],
                    self.structural_equations.for_targets([node]),
                )
            )
            factors.append(ParametricCPD(
                endogenous[node], parametrization,
                parents=(
                    [exogenous[node]] if not parents else
                    [*(exogenous[i] for i in parents),
                     *(endogenous[i] for i in parents)]
                ),
                aggregate=aggregate,
                trunk=None if not parents else Sequential(
                    self.mixer,
                    GraphAggregator(adjacency=graph.data),
                ),
            ))

        prefix_cpds = [self.input_cpd, self.shared_cpd, *self.exogenous_cpds]
        eval_pgm = BayesianNetwork(
            variables=[*(cpd.variable for cpd in prefix_cpds), *endogenous],
            factors=[*prefix_cpds, *factors],
        )
        eval_inference = self._eval_inference_cls(
            eval_pgm, **self._eval_inference_kwargs
        )
        self._modules.pop("eval_pgm", None)
        self._modules.pop("eval_inference", None)
        object.__setattr__(self, "eval_pgm", eval_pgm)
        object.__setattr__(self, "eval_inference", eval_inference)
        self._eval_pgm_stale = False
        return eval_pgm

    def train(self, mode: bool = True):
        """Refresh the evaluation model only when leaving training mode."""
        was_training = self.training
        result = super().train(mode)
        if was_training and not mode and hasattr(self, "_eval_inference_cls"):
            self.materialize_bayesian_network()
        return result

    def _apply(self, fn):
        result = super()._apply(fn)
        if hasattr(self, "eval_pgm"):
            self._eval_pgm_stale = True
        return result

    @cached_property
    def _query_plan(self):
        """Map concept ground truth to the teacher-forced copy variables, to query inference during training."""
        axis = self.concept_annotations
        return [
            (
                copy_name,
                [(axis.get_index(name), axis.concept(name).cardinality)],
            )
            for copy_name, name in zip(self.copy_names, self.concept_names)
        ]

    def _training_run(self, source_values, evidence, **kwargs):
        """Execute one training query through the endogenous plate CPDs."""
        query = {
            **dict(zip(self.copy_names, source_values)),
            **dict.fromkeys(self.concept_names),
        }
        return self.train_inference.query(
            query=query,
            evidence=evidence,
            **kwargs,
        )

    def _training_interventions(self, observed):
        """Build the random intervention scenarios from the original CGM.

        One non-task concept is selected independently in every batch row.
        Binary concepts generate low/high scenarios; homogeneous categorical
        concepts generate one scenario per state. Other variable mixtures do
        not currently define a common intervention grid.
        """
        offsets = [0]
        for cardinality in self.cardinalities:
            offsets.append(offsets[-1] + cardinality)
        eligible_columns = [
            column
            for node in self.intervention_indices
            for column in range(offsets[node], offsets[node + 1])
        ]

        def intervene(constants, selected_columns):
            values = torch.cat(observed, dim=-1)
            module = InterventionModule(
                nn.Identity(), DoIntervention(constants),
                _CGMInterventionPolicy(selected_columns),
                out_concepts_to_intervene_on=eligible_columns,
                quantile=0.0,
            )
            return list(module(values).split(self.cardinalities, dim=-1))

        if all(type_name == "binary" for type_name in self.endogenous_types):
            leading_shape = observed[0].shape[:-1]
            selected = torch.tensor(
                [
                    random.sample(range(len(self.intervention_indices)), 1)[0]
                    for _ in range(leading_shape.numel())
                ],
                device=observed[0].device,
            ).reshape(leading_shape)
            selected_columns = torch.tensor(
                [offsets[self.intervention_indices[choice]] for choice in selected],
                device=observed[0].device,
            ).reshape(leading_shape)
            return {
                "low": intervene(0.0, selected_columns),
                "high": intervene(1.0, selected_columns),
            }
        if (
            len(set(self.endogenous_types)) == 1
            and len(set(self.cardinalities)) == 1
            and self.endogenous_types[0] == "categorical"
        ):
            leading_shape = observed[0].shape[:-1]
            selected = torch.tensor(
                [
                    random.sample(range(len(self.intervention_indices)), 1)[0]
                    for _ in range(leading_shape.numel())
                ],
                device=observed[0].device,
            ).reshape(leading_shape)
            selected_columns = torch.stack([
                torch.arange(
                    offsets[self.intervention_indices[choice]],
                    offsets[self.intervention_indices[choice] + 1],
                    device=observed[0].device,
                )
                for choice in selected.flatten()
            ]).reshape(*leading_shape, self.cardinalities[0])
            interventions = {}
            for category in range(self.cardinalities[0]):
                constants = torch.zeros_like(torch.cat(observed, dim=-1))
                constants.scatter_(
                    1, selected_columns[:, category:category + 1], 1.0
                )
                interventions[f"category_{category}"] = intervene(
                    constants, selected_columns
                )
            return interventions
        return {}

    def _training_forward(self, query, evidence, **kwargs):
        """Assemble prior, posterior, graph, and optional intervention output."""
        source_names = self.copy_names
        observed = [query[name] for name in source_names]
        self.graph_layer.clear()
        if self.graph_generator is not None and not self._graph_generator_validated:
            self._adjacency()
        output = self._training_run(observed, evidence, **kwargs)
        adjacency = self.graph_layer.adjacency
        params = _select_params(output, self.concept_names)
        params.update(_select_params(output, source_names, "prior_"))
        if self.run_interventions:
            for label, values in self._training_interventions(observed).items():
                intervened = self._training_run(values, evidence, **kwargs)
                params.update(_select_params(
                    intervened, self.concept_names, f"{label}_",
                ))
        params["adjacency"] = adjacency
        return ModelOutput(
            params=params,
            extra={"task_names": tuple(self.task_names)},
        )

    def forward(
        self, query=None, evidence=None, input=None, target=None, **inference_kwargs
    ):
        """Run CGM inference without exposing its internal query layout.

        Standard PyTorch training passes ``target``; evaluation needs only the
        input. Lightning may continue to pass the query prepared by its shared
        step. An explicit ``query`` remains available for advanced inference.
        """
        if not self.training and getattr(self, "_eval_pgm_stale", False):
            self.materialize_bayesian_network()
        if query is not None and target is not None:
            raise ValueError("Pass either `query` or `target`, not both.")
        if query is None:
            if self.training and target is None:
                raise ValueError("CausalCGM training requires `target`.")
            query = self.default_query(target)
        if self.training:
            evidence = dict(evidence or {})
            if input is not None:
                evidence["input"] = input
            return self._training_forward(
                query, evidence, **inference_kwargs
            )
        return super().forward(
            query=query, evidence=evidence, input=input, **inference_kwargs
        )

    def default_query(self, ground_truth):
        if self.training:
            return super().fully_observed_query(ground_truth)
        return dict.fromkeys(self.concept_names)


__all__ = ["CausalCGM"]
