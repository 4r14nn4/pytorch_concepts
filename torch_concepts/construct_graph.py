"""
Concept graph generation utilities.

- :class:`GraphGenerator` — abstract base class defining source registration,
  generator state, refinement, validation, and graph materialization.
- :class:`GraphGeneratorFixed` — fixed generation. Its source provides
  ``compute``. Built-in sources:
  ``'GroundTruth'``, ``'Causallearn'``, and ``'LLM'``.
- :class:`GraphGeneratorLearnable` — differentiable generation as a
  :class:`torch.nn.Module`. Its source provides ``forward`` and its
  initialization strategies. ``construct_graph`` materializes the trained graph.
  Built-in sources: ``'WANDA'`` and
  ``'DAGMA_CGM'``.

Both concrete APIs return a :class:`ConceptGraph`, which owns inspection and
plotting. Optional refinement accepts any ``ConceptGraph -> ConceptGraph``
callable, including the standalone :func:`refine_llm` function.

Caching
-------
Only fixed graphs are cacheable. Calling ``precompute_graph`` with a
:class:`GraphGeneratorFixed` may load a compatible graph from disk or compute,
validate, and save a new one. Fixed generators also reuse their materialized
graph in memory unless ``force=True``. Learnable generators are registered for
end-to-end training instead: their graphs are never persisted or reused from a
cache. After training, ``construct_graph`` materializes the latest valid
``forward`` result and then applies refinement and validation.

Concept descriptions
--------------------
Direct LLM generation resolves descriptions from ``concept_descriptions`` and
then ``dataset.label_descriptions``. The standalone ``refine_llm`` function
uses the ``concept_descriptions`` mapping passed to it.

Extensibility:

- **refinement**: pass a callable through ``refinement``::

      from functools import partial

      generator = GraphGeneratorFixed(
          name="ges",
          refinement=partial(refine_llm, llm_backend=backend, domain="weather"),
      )

- **per-name** (``'ges'`` vs ``'pc'``, or one LLM model vs another): pass the
  new ``name`` to an existing source; no new callbacks are needed::

      generator = GraphGeneratorFixed(name="ges", source="Causallearn")

- **per-fixed-source**: define ``compute`` and return
  :class:`GraphGeneratorFixedSpec`::

      @GraphGeneratorFixed.register_source("mylab")
      def _load_mylab(generator, name, **kwargs):
          return GraphGeneratorFixedSpec(compute=_compute_mylab)

- **per-learnable-source**: create the trainable parameters and return
  :class:`GraphGeneratorLearnableSpec` with ``forward`` and the supported
  initializations::

      @GraphGeneratorLearnable.register_source("mylab")
      def _load_mylab_learnable(generator, name, **kwargs):
          generator.weight = nn.Parameter(torch.randn(1))
          return GraphGeneratorLearnableSpec(
              forward=_mylab_forward,
              initializations={
                  "random": GraphGeneratorInitializationSpec(
                      _initialize_mylab_random,
                  ),
              },
          )

Use a new ``name`` for another model or method within an existing source; use
``register_source`` only for a new implementation family.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from torch_concepts.concept_graph import ConceptGraph
#from torch_concepts.data.concept_generator import llm_backends


DEFAULT_REFINEMENT_MODEL = "groq/openai/gpt-oss-20b"


if TYPE_CHECKING:
    from torch_concepts.data.base.dataset import ConceptDataset


#class _LiteLLMBackendWithMillisecondRetry(llm_backends.LiteLLMBackend):
#    @staticmethod
#    def _rate_limit_wait_seconds(error: Exception) -> float | None:
#        wait_seconds = llm_backends.LiteLLMBackend._rate_limit_wait_seconds(error)
#        if wait_seconds is not None:
#            return wait_seconds

#        match = re.search(
#            r"Please try again in (?P<delay>\d+(?:\.\d+)?)(?P<unit>ms|s)",
#            str(error),
#            flags=re.IGNORECASE,
#        )
#        if match:
#            delay = float(match.group("delay"))
#            if match.group("unit").lower() == "ms":
#                delay /= 1000.0
#            return delay + 1.0
#        return None


class GraphGenerator:
    """Abstract base class shared by both graph-generator APIs.

    The base class owns the source registry and common generator state.
    Subclasses keep separate source registries and implement their fixed or learnable contract
    according to their fixed or learnable semantics.

    Dataset identity is assumed to be determined by the ordered
    ``dataset.concept_names`` and ``dataset.n_samples``. Datasets with the
    same concept names and sample count share a cache identity, even when
    their values differ; use ``force=True`` to recompute in that case.
    Learnable generators also include parameter versions in their own
    in-memory cache key to detect weight updates. Without a dataset, the
    dataset identity is ``None``.

    Parameters
    ----------
    name : str
        Method or model name understood by ``source``.
    source : str, optional
        Registered implementation family. It is inferred when ``name`` maps
        to exactly one registered source; otherwise it must be provided.
    refinement : callable, optional
        A ``ConceptGraph -> ConceptGraph`` callable applied after generation
        and before validation. Use ``partial(refine_llm, llm_backend=backend)`` for LLM edge orientation.
        ``None`` disables refinement.
    require_dag : bool, default True
        Validate the final graph after generation and refinement, raising an
        error unless it is a directed acyclic graph.
    concept_descriptions : dict, optional
        Explicit descriptions keyed by concept name. Missing entries are
        filled from ``dataset.label_descriptions`` during ``construct_graph``.

    Attributes
    ----------
    name : str
        Configured method or model name.
    source : str
        Configured source family.
    graph : ConceptGraph or None
        Most recently materialized graph, or ``None`` before generation.
    fitted : bool
        Whether ``construct_graph`` has materialized and stored at least one
        graph snapshot. This is state information, especially useful for
        learnable generators; dataset ground-truth assignment does not rely
        on this flag.
    trainable : bool
        Class-level flag distinguishing fixed and learnable generators.
    """

    trainable: bool
    _sources: dict[str, Callable] = {}
    _name_sources: dict[str, set[str]] = {}

    def __init__(
        self,
        name: str,
        source: Optional[str] = None,
        refinement: Optional[Callable[[ConceptGraph], ConceptGraph]] = None,
        require_dag: bool = True,
        concept_descriptions: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ):
        if type(self) is GraphGenerator:
            raise TypeError(
                "GraphGenerator is abstract; instantiate "
                "GraphGeneratorFixed or GraphGeneratorLearnable."
            )
        super().__init__()
        self.name = name
        self.source = self.resolve_source(name, source)
        self.require_dag = bool(require_dag)
        self.graph: Optional[ConceptGraph] = None
        self.fitted = False
        self._graph_cache_key: Any = None
        self._concept_descriptions = dict(concept_descriptions or {})
        if self.source not in self._sources:
            raise ValueError(
                f"Unknown source {self.source!r} for {type(self).__name__}; "
                f"registered sources: {sorted(self._sources)}. Register new "
                f"ones with @{type(self).__name__}.register_source(...)."
            )
        spec = self._sources[self.source](self, self.name, **kwargs)
        if refinement is not None and not callable(refinement):
            raise TypeError("`refinement` must be callable or None.")
        self._spec = replace(spec, refinement=refinement)

    @classmethod
    def register_source(
        cls, source: str, names: Optional[Sequence[str]] = None,
    ) -> Callable:
        """Register a source initializer and the method names it provides."""
        def decorator(fn: Callable) -> Callable:
            cls._sources[source] = fn
            for name in names or ():
                cls._name_sources.setdefault(name, set()).add(source)
            return fn
        return decorator

    @classmethod
    def resolve_source(cls, name: str, source: Optional[str] = None) -> str:
        """Infer a unique source from a registered method name."""
        if source is not None:
            return source
        matches = sorted(cls._name_sources.get(name, set()))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(
                f"Cannot infer a source for method {name!r}; specify `source`."
            )
        raise ValueError(
            f"Method {name!r} is provided by multiple sources {matches}; "
            "specify `source`."
        )

    def _resolve_context(self, dataset: Optional[ConceptDataset]) -> None:
        """Fill missing concept descriptions from the current dataset."""
        if dataset is None:
            return
        dataset_descriptions = getattr(dataset, "label_descriptions", None) or {}
        self._concept_descriptions = {
            name: str(
                self._concept_descriptions.get(
                    name, dataset_descriptions.get(name, "")
                )
            )
            for name in dataset.concept_names
        }

    def _validate_graph(self, graph: ConceptGraph) -> None:
        if self.require_dag and not graph.is_directed_acyclic():
            raise ValueError(
                f"Graph method {self.name!r} produced a graph that is not a "
                "directed acyclic graph (DAG). DAG validation is enabled by "
                "default. Choose another method, for example "
                "`GraphGeneratorFixed(name='ges')`, or orient ambiguous edges "
                "with an LLM refinement, for example "
                "`GraphGeneratorFixed(name='pc', "
                "refinement=partial(refine_llm, llm_backend=backend))`. Pass "
                "`require_dag=False` only when a non-DAG is intentional."
            )



    def _cache_key(self, dataset) -> Any:
        """Identify a graph by configuration, dataset metadata and weight versions."""
        refinement = self._spec.refinement
        versions = self._parameter_versions() if self.trainable else None
        dataset_key = (
            (tuple(dataset.concept_names), dataset.n_samples)
            if dataset is not None else None
        )
        return (
            type(self).__name__, self.source, self.name, refinement,
            dataset_key, versions,
        )

    def _cached_graph(self, cache_key: Any, force: bool) -> Optional[ConceptGraph]:
        if not force and self.fitted and self._graph_cache_key == cache_key:
            warnings.warn(
                "Using a graph already materialized in memory; pass "
                "`force=True` to rebuild it.",
                UserWarning, stacklevel=3,
            )
            return self.graph
        return None

    def invalidate_cache(self) -> None:
        """Discard the materialized graph without changing generator parameters."""
        self.graph = None
        self.fitted = False
        self._graph_cache_key = None

    def construct_graph(
        self,
        dataset: Optional[ConceptDataset] = None,
        *,
        force: bool = False,
    ) -> ConceptGraph:
        """Construct, refine, validate, and retain the resulting graph.

        Complete in-memory graphs are reused only when method, dataset,
        refinement and (for learnable generators) parameter
        versions still match. ``force=True`` bypasses that cache and also
        refreshes a learnable adjacency. Disk persistence is handled only by
        fixed-graph ``precompute_graph``.
        """
        self._resolve_context(dataset)
        cache_key = self._cache_key(dataset)
        # if source, name or dataset changes, we may need to rebuild the graph. Not only materialize it. Warning message.
        context = (self.source, self.name, cache_key[4])
        previous_context = getattr(self, "_materialization_context", None)
        if self.trainable and previous_context not in (None, context):
            warnings.warn(
                "The dataset, source, or method of a learnable graph generator "
                "changed. Rebuilding the graph does not retrain its parameters; "
                "rerun the complete training pipeline for the new context.",
                UserWarning, stacklevel=2,
            )

        cached = self._cached_graph(cache_key, force)
        if cached is not None:
            return cached

        if self.trainable:
            with torch.no_grad():
                generated = self()
            cache_key = self._cache_key(dataset)
        else:
            if dataset is None:
                raise ValueError("Fixed graph construction requires a dataset.")
            generated = self._spec.compute(self, dataset)
        if isinstance(generated, ConceptGraph):
            graph = generated
        elif isinstance(generated, torch.Tensor):
            concept_names = list(
                dataset.concept_names
                if dataset is not None
                else self.concept_names
            )
            expected = list(getattr(self, "concept_names", concept_names))
            if concept_names != expected:
                raise ValueError(
                    "Dataset concept names must match the generator concepts."
                )
            graph = ConceptGraph(generated.detach(), node_names=concept_names)
        else:
            raise TypeError(
                "Graph generator callbacks must return ConceptGraph or Tensor."
            )

        if self._spec.refinement is not None:
            graph = self._spec.refinement(graph)
            if not isinstance(graph, ConceptGraph):
                raise TypeError("Refinement must return a ConceptGraph.")
        self._validate_graph(graph)
        self.graph = graph
        self.fitted = True
        self._graph_cache_key = cache_key
        self._materialization_context = context
        return graph

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"source={self.source!r}, trainable={self.trainable})"
        )



@dataclass(frozen=True, kw_only=True)
class GraphGeneratorSpec:
    """Options shared by fixed and learnable graph sources."""

    refinement: Optional[Callable[[ConceptGraph], ConceptGraph]] = None


@dataclass(frozen=True, kw_only=True)
class GraphGeneratorFixedSpec(GraphGeneratorSpec):
    """Implementation contract for a fixed graph source."""
    compute: Callable

class GraphGeneratorInitializationSpec(NamedTuple):
    """Initialization callback and its data requirement."""

    initialize: Callable
    requires_data: bool = False


@dataclass(frozen=True, kw_only=True)
class GraphGeneratorLearnableSpec(GraphGeneratorSpec):
    """Implementation contract for a learnable graph source."""

    forward: Callable
    initializations: dict[str, GraphGeneratorInitializationSpec] = field(
        default_factory=dict,
    )


class GraphGeneratorLearnable(GraphGenerator, nn.Module):
    """Differentiable graph generator.

    A registered learnable source supplies ``forward`` and initialization
    strategies. Every source must provide ``random``, which is selected by
    default. Data are accepted only by strategies declaring that they require
    them. :meth:`construct_graph` materializes a detached
    :class:`ConceptGraph` snapshot and records it in the shared generator state. Optional refinement accepts a graph callable.
    """

    trainable = True
    _sources: dict[str, Callable] = {}
    _name_sources: dict[str, set[str]] = {}
    def __init__(
        self,
        name: str,
        source: Optional[str] = None,
        refinement: Optional[Callable[[ConceptGraph], ConceptGraph]] = None,
        require_dag: bool = True,
        initialization: str = "random",
        initialization_data: Any = None,
        concept_descriptions: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            source=source,
            refinement=refinement,
            require_dag=require_dag,
            concept_descriptions=concept_descriptions,
            **kwargs,
        )
        self.initialization = initialization
        self._apply_initialization(initialization, initialization_data)

    def _initialization_spec(
        self, strategy: str,
    ) -> GraphGeneratorInitializationSpec:
        """Return and validate one initialization strategy."""
        initializers = self._spec.initializations
        if strategy not in initializers:
            supported = sorted(initializers)
            raise ValueError(
                f"Unknown initialization {strategy!r} for source {self.source!r}; "
                f"supported initializations: {supported}."
            )
        return initializers[strategy]

    def _apply_initialization(
        self,
        strategy: str,
        data: Any,
    ) -> None:
        """Apply the selected source initializer."""
        spec = self._initialization_spec(strategy)
        if spec.requires_data and data is None:
            raise ValueError(
                f"Initialization {strategy!r} for source {self.source!r} "
                "requires `initialization_data`."
            )
        if not spec.requires_data and data is not None:
            raise ValueError(
                f"Initialization {strategy!r} for source {self.source!r} "
                "does not accept data."
            )
        spec.initialize(self, data)

    def forward(self) -> torch.Tensor:
        """Return the current differentiable adjacency matrix."""
        adjacency = self._spec.forward(self)
        self.invalidate_cache()
        return adjacency

    def _parameter_versions(self) -> tuple[int, ...]:
        return tuple(parameter._version for parameter in self.parameters())

# ------------------------------------------------------------------
# DAGMA-CGM: CausalCGM's modified DAGMA adjacency
# ------------------------------------------------------------------
def _dagma_cgm_forward(
    generator: GraphGeneratorLearnable, _dataset=None,
) -> torch.Tensor:
    """Return the thresholded straight-through adjacency used by CausalCGM."""
    weights = (generator.fc1.weight * generator.edge_mask).abs()
    for source, target in generator.edges_to_check:
        weights[source, target] += torch.sigmoid(
            generator.edge_matrix[source, target]
        )
    for source, target in generator.edges_to_check:
        weights[target, source] += 1 - weights[source, target]
    soft = torch.sigmoid(5 * (weights - generator.threshold))
    hard = (weights > generator.threshold).to(weights.dtype)
    return weights * (soft + (hard - soft).detach())


@torch.no_grad()
def _initialize_dagma_cgm_random(
    generator: GraphGeneratorLearnable, _data: Any,
) -> None:
    generator.fc1.reset_parameters()
    generator.edge_matrix.zero_()


def _entropy(values: torch.Tensor) -> torch.Tensor:
    _, counts = torch.unique(values, dim=0, return_counts=True)
    probabilities = counts / counts.sum()
    return -(probabilities * probabilities.log2()).sum()


@torch.no_grad()
def _initialize_dagma_cgm_entropy(
    generator: GraphGeneratorLearnable, data: Any,
) -> torch.Tensor:
    """Initialize DAGMA-CGM from complete training-set conditional entropy."""
    values = data.concepts if hasattr(data, "concepts") else data
    values = values.tensor if hasattr(values, "tensor") else values
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise ValueError(
            "Entropy initialization data must be a 2D tensor or expose "
            "a 2D `concepts` tensor."
        )
    if values.shape[1] != generator.n_concepts:
        raise ValueError(
            "Entropy initialization data must have one column per graph node."
        )
    values = values.to(generator.fc1.weight.device)
    adjacency = torch.zeros_like(generator.fc1.weight)
    entropies = [
        _entropy(values[:, index:index + 1])
        for index in range(generator.n_concepts)
    ]
    for source in range(generator.n_concepts):
        for target in range(generator.n_concepts):
            if source != target:
                joint = _entropy(values[:, [source, target]])
                adjacency[source, target] = 1 - (joint - entropies[target])
    if generator.no_out_task and generator.task_indices:
        adjacency[generator.task_indices, :] = 0
    mean = adjacency.mean()
    if mean != 0:
        adjacency = adjacency / mean
    adjacency.clamp_(0, 0.99)
    generator.fc1.weight.copy_(adjacency)
    return adjacency


@GraphGeneratorLearnable.register_source("DAGMA_CGM", names=["dagma_cgm"])
def _load_dagma_cgm_source(
    generator: GraphGeneratorLearnable,
    name: str,
    concept_names: List[str],
    n_tasks: int = 0,
    task_names: Optional[List[str]] = None,
    threshold: float = 0.02,
    no_out_task: bool = True,
    edges_to_check=None,
    **_,
) -> GraphGeneratorLearnableSpec:
    """Initialize the DAGMA variant defined by the CausalCGM paper."""
    if name != "dagma_cgm":
        raise ValueError("The DAGMA_CGM source supports only name='dagma_cgm'.")
    if not 0 <= n_tasks <= len(concept_names):
        raise ValueError("n_tasks must be between zero and the number of nodes.")
    generator.concept_names = list(concept_names)
    generator.n_concepts = len(concept_names)
    if task_names is None:
        task_names = concept_names[-n_tasks:] if n_tasks else []
    missing = set(task_names) - set(concept_names)
    if missing:
        raise ValueError(f"task_names must be graph nodes; missing: {sorted(missing)}.")
    generator.task_names = list(task_names)
    generator.task_indices = [concept_names.index(task) for task in task_names]
    generator.n_tasks = len(task_names)
    generator.fc1 = nn.Linear(
        generator.n_concepts, generator.n_concepts, bias=False
    )
    generator.edge_matrix = nn.Parameter(torch.zeros(
        generator.n_concepts, generator.n_concepts
    ))
    generator.no_out_task = bool(no_out_task)
    generator.edges_to_check = list(edges_to_check or [])
    edge_mask = torch.ones(generator.n_concepts, generator.n_concepts)
    edge_mask.fill_diagonal_(0)
    if no_out_task and generator.task_indices:
        edge_mask[generator.task_indices, :] = 0
    generator.register_buffer("edge_mask", edge_mask)
    generator.threshold = float(threshold)
    return GraphGeneratorLearnableSpec(
        forward=_dagma_cgm_forward,
        initializations={
            "random": GraphGeneratorInitializationSpec(
                _initialize_dagma_cgm_random,
            ),
            "entropy": GraphGeneratorInitializationSpec(
                _initialize_dagma_cgm_entropy,
                requires_data=True,
            ),
        },
    )

class GraphGeneratorFixed(GraphGenerator):
    """Fixed graph generator.

    A registered fixed source supplies the generation callback.
    Calling :meth:`construct_graph` records the
    resulting graph in the common generator state.
    """

    trainable = False
    _sources: dict[str, Callable] = {}
    _name_sources: dict[str, set[str]] = {}

    def __init__(
        self,
        name: str,
        source: Optional[str] = None,
        refinement: Optional[Callable[[ConceptGraph], ConceptGraph]] = None,
        require_dag: bool = True,
        concept_descriptions: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            source=source,
            refinement=refinement,
            require_dag=require_dag,
            concept_descriptions=concept_descriptions,
            **kwargs,
        )


#
# ------------------------------------------------------------------
# Ground-truth graph generator
# ------------------------------------------------------------------
def _compute_ground_truth(
    self: GraphGeneratorFixed,
    dataset: ConceptDataset,
) -> ConceptGraph:
    if dataset.graph_native is None:
        raise ValueError("The GroundTruth source requires `dataset.graph_native`.")
    return dataset.graph_native


@GraphGeneratorFixed.register_source("GroundTruth", names=["ground_truth"])
def _load_ground_truth_source(
    generator: GraphGeneratorFixed,
    name: str,
) -> GraphGeneratorFixedSpec:
    if name != "ground_truth":
        raise ValueError(
            "The GroundTruth source supports only name='ground_truth'."
        )
    return GraphGeneratorFixedSpec(compute=_compute_ground_truth)


# ------------------------------------------------------------------
# CausalLearn graph generator
# ------------------------------------------------------------------
_CONSTRAINT_BASED = {"pc"}
_SCORE_BASED = {"ges"}


def _import_causallearn(method: str):
    """Lazily import and return the requested CausalLearn algorithm.

    Args:
        method: One of ``'pc'``,``'ges'``.

    Raises:
        ValueError: If ``method`` is not supported.
        ImportError: If ``causallearn`` is not installed.
    """
    try:
        if method == "pc":
            from causallearn.search.ConstraintBased.PC import pc
            return pc
        elif method == "ges":
            from causallearn.search.ScoreBased.GES import ges
            return ges
        else:
            raise ValueError(
                f"Unknown causallearn method '{method}'. "
                f"Supported: {sorted(_CONSTRAINT_BASED | _SCORE_BASED)}."
            )
    except ImportError as exc:
        raise ImportError(
            "CausalLearn-based graph generator requires the `causallearn` package. "
            "Install it with: pip install causal-learn"
        ) from exc


def _cl_graph_to_adj(cl_graph: Any) -> torch.Tensor:
    """Convert CausalLearn endpoints without dropping ambiguous edges."""
    adj_np = np.array(cl_graph.graph, dtype=np.float32, copy=True)
    diff = adj_np - adj_np.T
    adj_np[diff == -2] = 1.0
    adj_np[diff == 2] = 0.0
    return torch.from_numpy(adj_np)


def _compute_causallearn(
    self: GraphGeneratorFixed,
    dataset: ConceptDataset,
) -> ConceptGraph:
    algorithm = _import_causallearn(self.name)
    data = dataset.concepts.detach().cpu().numpy()

    if self.name in _CONSTRAINT_BASED:
        result = algorithm(data, self.alpha, self.indep_test)
        cl_graph = result[0] if isinstance(result, tuple) else result.G
    else:
        cl_graph = algorithm(data, score_func=self.score_func)["G"]

    concept_names = list(dataset.concept_names)
    return ConceptGraph(
        _cl_graph_to_adj(cl_graph),
        node_names=concept_names,
    )


@GraphGeneratorFixed.register_source("Causallearn", names=["pc", "ges"])
def _load_causallearn_source(
    generator: GraphGeneratorFixed,
    name: str,
    alpha: float = 0.05,
    indep_test: str = "chisq",
    score_func: str = "local_score_BDeu",
) -> GraphGeneratorFixedSpec:
    supported = _CONSTRAINT_BASED | _SCORE_BASED
    if name not in supported:
        raise ValueError(
            f"Unknown CausalLearn name {name!r}. "
            f"Supported names: {sorted(supported)}."
        )
    if name in _CONSTRAINT_BASED and not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")
    generator.alpha = alpha
    generator.indep_test = indep_test
    generator.score_func = score_func
    return GraphGeneratorFixedSpec(compute=_compute_causallearn)

# ------------------------------------------------------------------
# LLM graph generator
# ------------------------------------------------------------------

# Allowed response tokens
_EDGE_TOKENS = ("A->B", "B->A", "none")

# Prompt template
_PROMPT_TEMPLATE = (
    "You are a causal-inference expert {domain_clause}.\n"
    "Assess the direct causal relationship between:\n"
    "A: {concept_1_details}\n"
    "B: {concept_2_details}\n"
    "{context_section}\n"
    "Choose one answer, accounting for confounding, indirect effects, and "
    "mere association:\n"
    "A->B: A directly causes B\n"
    "B->A: B directly causes A\n"
    "none: no direct causal relationship\n\n"
    "Reason internally. Output exactly A->B, B->A, or none. "
    "No other response is allowed."
)

def _compute_llm(
    self: GraphGeneratorFixed,
    dataset: ConceptDataset,
) -> ConceptGraph:
    """Build a graph by querying the LLM for every concept pair."""
    concept_names = list(dataset.concept_names)
    adjacency = torch.zeros(len(concept_names), len(concept_names))
    for i in range(len(concept_names)):
        for j in range(i + 1, len(concept_names)):
            concept_a, concept_b = concept_names[i], concept_names[j]
            response = _query_pair(
                self.llm_backend,
                concept_a,
                self._concept_descriptions.get(concept_a, ""),
                concept_b,
                self._concept_descriptions.get(concept_b, ""),
                domain=self.domain, repeats=self.repeats,
            )
            if response == "A->B":
                adjacency[i, j] = 1.0
            elif response == "B->A":
                adjacency[j, i] = 1.0
    return ConceptGraph(adjacency, node_names=concept_names)


def refine_llm(
    graph: ConceptGraph,
    *,
    llm_backend: Callable[..., str],
    domain: str = "",
    concept_descriptions: Optional[dict[str, str]] = None,
    repeats: int = 1,
) -> ConceptGraph:
    """Orient reciprocal edges, leaving absent and directed edges unchanged.

    ``llm_backend`` is a configured callable accepting a prompt and ``repeats``.
    Missing concept descriptions default to empty strings. Returns a new graph;
    the input is unchanged. Bind keyword arguments with ``functools.partial``
    to use this function as a generator's ``refinement`` callback.
    """
    if not callable(llm_backend):
        raise TypeError("`llm_backend` must be callable.")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer.")
    descriptions = concept_descriptions or {}
    concept_names = list(graph.node_names)
    adjacency = graph.data.clone()
    for i in range(len(concept_names)):
        for j in range(i + 1, len(concept_names)):
            if adjacency[i, j] == 0 or adjacency[j, i] == 0:
                continue
            concept_a, concept_b = concept_names[i], concept_names[j]
            response = _query_pair(
                llm_backend,
                concept_a,
                descriptions.get(concept_a, ""),
                concept_b,
                descriptions.get(concept_b, ""),
                domain=domain, repeats=repeats,
            )
            if response == "A->B":
                adjacency[i, j] = 1.0
                adjacency[j, i] = 0.0
            elif response == "B->A":
                adjacency[i, j] = 0.0
                adjacency[j, i] = 1.0
    return ConceptGraph(adjacency, node_names=concept_names)


def _query_pair(
    llm_backend: Callable[..., str],
    concept_a: str,
    concept_a_description: str,
    concept_b: str,
    concept_b_description: str,
    *,
    domain: str = "",
    repeats: int = 1,
) -> str:
    domain_clause = f"in the domain of {domain}" if domain else ""
    concept_a_details = _concept_details(concept_a, concept_a_description)
    concept_b_details = _concept_details(concept_b, concept_b_description)

    context = ""

    prompt = _PROMPT_TEMPLATE.format(
        domain_clause=domain_clause,
        concept_1_details=concept_a_details,
        concept_2_details=concept_b_details,
        context_section=(
            f"\nRelevant context:\n{context}\n" if context else ""
        ),
    )
    response = llm_backend(prompt, repeats=repeats)
    return _most_frequent_token(response)


def _concept_details(name: str, description: str) -> str:
    return f"{name} — {description}" if description else name


def _most_frequent_token(response: Any) -> str:
    """Return the majority token, using ``none`` for invalid votes or ties."""
    raw_tokens = [
        line.strip()
        for line in str(response).splitlines()
        if line.strip()
    ]
    counts = {token: 0 for token in _EDGE_TOKENS}
    for token in raw_tokens:
        vote = token if token in _EDGE_TOKENS else "none"
        counts[vote] += 1

    highest_count = max(counts.values())
    winners = [token for token, count in counts.items() if count == highest_count]
    return winners[0] if len(winners) == 1 else "none"

@GraphGeneratorFixed.register_source(
    "LLM", names=[DEFAULT_REFINEMENT_MODEL],
)
def _load_llm_source(
    self: GraphGeneratorFixed,
    name: str,
    api_key: Optional[str] = None,
    llm_backend: Optional[Callable[..., str]] = None,
    completion_kwargs: Optional[dict[str, Any]] = None,
    repeats: int = 1,
    domain: str = "",
    use_rag: Optional[bool] = None,
    rag: Optional[Any] = None,
    documents: Optional[List[str]] = None,
    n_retrieved: int = 3,
    embedding_model: str = "openai/text-embedding-3-small",
    embedding_backend: Optional[Callable[..., Any]] = None,
    embedding_kwargs: Optional[dict[str, Any]] = None,
) -> GraphGeneratorFixedSpec:
    rag_enabled = (
        bool(rag is not None or documents)
        if use_rag is None
        else use_rag
    )
    if rag_enabled:
        raise NotImplementedError("RAG support is not implemented yet.")

    if n_retrieved < 1:
        raise ValueError("n_retrieved must be at least 1.")
    if (
        not isinstance(repeats, int)
        or isinstance(repeats, bool)
        or repeats < 1
    ):
        raise ValueError("repeats must be a positive integer.")
    self.model = name
    self.api_key = api_key
    self.domain = domain
    self.repeats = repeats

    llm_options = {
        "temperature": 0,
        "max_tokens": 200,
        "retry_on_rate_limit": True,
        "max_rate_limit_wait": 120.0,
        **(completion_kwargs or {}),
    }
    if api_key is not None:
        llm_options["api_key"] = api_key
    if llm_backend is None:
        from torch_concepts.data.concept_generator.llm_backends import LiteLLMBackend

        llm_backend = LiteLLMBackend(model=name, **llm_options)
    self.llm_backend = llm_backend
    if not callable(self.llm_backend):
        raise TypeError("`llm_backend` must be callable.")
    self.rag = rag
    self.documents: List[str] = list(documents or [])
    self.use_rag = rag_enabled
    self.n_retrieved = n_retrieved
    self.embedding_model = embedding_model
    embedding_options = dict(embedding_kwargs or {})
    if api_key is not None:
        embedding_options["api_key"] = api_key
    self.embedding_backend = embedding_backend
    if (
        self.use_rag
        and self.rag is None
        and self.documents
        and self.embedding_backend is None
    ):
        backend_type = getattr(
            llm_backends,
            "LiteLLMEmbeddingBackend",
            None,
        )
        if backend_type is None:
            raise ImportError(
                "Document RAG requires either an `embedding_backend` or "
                "`llm_backends.LiteLLMEmbeddingBackend`."
            )
        self.embedding_backend = backend_type(
            model=embedding_model,
            **embedding_options,
        )
    if self.embedding_backend is not None and not callable(
        self.embedding_backend
    ):
        raise TypeError("`embedding_backend` must be callable.")
    self._doc_embeddings: Optional[np.ndarray] = None

    if self.use_rag and self.rag is None and not self.documents:
        raise ValueError(
            "RAG is enabled, but neither `rag` nor `documents` was provided."
        )
    if self.rag is not None and not (
        callable(self.rag) or callable(getattr(self.rag, "retrieve", None))
    ):
        raise TypeError(
            "`rag` must be callable or expose a callable `retrieve(query, k)`."
        )
    return GraphGeneratorFixedSpec(
        compute=_compute_llm,
    )

__all__ = [
    "GraphGenerator",
    "refine_llm",
    "GraphGeneratorSpec",
    "GraphGeneratorFixedSpec",
    "GraphGeneratorLearnableSpec",
    "GraphGeneratorInitializationSpec",
    "GraphGeneratorLearnable",
    "GraphGeneratorFixed",
]
