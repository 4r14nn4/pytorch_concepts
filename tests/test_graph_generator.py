"""Tests for the current graph_generator API.

Run from the repository root with either:

    pytest -q tests/test_graph_generator.py
    python tests/test_graph_generator.py
"""

from types import SimpleNamespace

import matplotlib.style as mpl_style
import numpy as np
import pytest
import torch

if not hasattr(mpl_style, "core"):
    mpl_style.core = mpl_style

import torch_concepts.graph_generator as graph_module
from torch_concepts.concept_graph import ConceptGraph
from torch_concepts.graph_generator import (
    GraphGenerator,
    GraphGeneratorFixed,
    GraphGeneratorLearnable,
    compose_refinements,
    entropy_initialization,
    fixed_dagma_initialization,
    random_initialization,
    refine_llm,
    remove_weakest_cycles,
)


@pytest.fixture
def dataset():
    names = ["rain", "wet", "traffic"]
    return SimpleNamespace(
        name="toy",
        concept_names=names,
        concepts=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
        ),
        graph_native=ConceptGraph(
            torch.tensor(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
            node_names=names,
        ),
        label_descriptions={"rain": "whether it rains", "wet": "wet grass"},
        n_samples=3,
        seed=7,
    )


def test_base_generator_is_abstract_and_registries_are_separate():
    with pytest.raises(TypeError):
        GraphGenerator(name="x", source="x")
    assert "GroundTruth" in GraphGeneratorFixed._sources
    assert "DAGMA_CGM" in GraphGeneratorLearnable._sources
    assert "GroundTruth" not in GraphGeneratorLearnable._sources
    assert "DAGMA_CGM" not in GraphGeneratorFixed._sources


@pytest.mark.parametrize("cls", [GraphGeneratorFixed, GraphGeneratorLearnable])
def test_unknown_source_reports_registered_sources(cls):
    with pytest.raises(ValueError, match="Unknown source.*registered sources"):
        cls(name="missing", source="Missing")


def test_ground_truth_construct_graph_returns_native_and_caches(dataset):
    generator = GraphGeneratorFixed(name="ground_truth")
    graph = generator.construct_graph(dataset)
    assert graph is dataset.graph_native
    assert generator.graph is graph
    assert generator.fitted
    second = generator.construct_graph(dataset)
    assert second is graph


def test_ground_truth_requires_native_graph(dataset):
    dataset.graph_native = None
    generator = GraphGeneratorFixed(name="ground_truth")
    with pytest.raises(ValueError, match="graph_native"):
        generator.construct_graph(dataset)


def test_cache_key_uses_stable_dataset_metadata(dataset):
    generator = GraphGeneratorFixed(name="ground_truth")
    clone = SimpleNamespace(**dataset.__dict__)
    assert generator._cache_key(dataset) == generator._cache_key(clone)
    clone.seed = 8
    assert generator._cache_key(dataset) != generator._cache_key(clone)
    clone.seed = dataset.seed
    clone.concept_names = ["wet", "rain", "traffic"]
    assert generator._cache_key(dataset) != generator._cache_key(clone)


def test_callable_refinement_runs_on_materialized_copy(dataset):
    calls = []

    def refine(graph):
        calls.append(True)
        data = graph.data.clone()
        data[1, 2] = 1
        return ConceptGraph(data, node_names=list(graph.node_names))

    generator = GraphGeneratorFixed(name="ground_truth", refinement=refine)
    graph = generator.construct_graph(dataset)
    assert calls == [True]
    assert graph.data[1, 2] == 1
    assert dataset.graph_native.data[1, 2] == 0


def test_compose_refinements_and_validation(dataset):
    def a(graph):
        data = graph.data.clone()
        data[0, 2] = 1
        return ConceptGraph(data, node_names=list(graph.node_names))

    def b(graph):
        data = graph.data.clone()
        data[2, 1] = 1
        return ConceptGraph(data, node_names=list(graph.node_names))

    refined = compose_refinements(a, b)
    graph = GraphGeneratorFixed(
        name="ground_truth", refinement=refined, require_dag=False
    ).construct_graph(dataset)
    assert graph.data[0, 2] == 1
    assert graph.data[2, 1] == 1
    with pytest.raises(ValueError):
        compose_refinements()
    with pytest.raises(TypeError):
        compose_refinements(a, object())


def test_invalid_refinement_rejected(dataset):
    with pytest.raises(TypeError, match="must be callable"):
        GraphGeneratorFixed(name="ground_truth", refinement={"bad": True})
    generator = GraphGeneratorFixed(
        name="ground_truth", refinement=lambda graph: graph.data
    )
    with pytest.raises(TypeError, match="must return a ConceptGraph"):
        generator.construct_graph(dataset)


class _CausalLearnGraph:
    def __init__(self, adjacency):
        self.graph = np.asarray(adjacency)


def test_causallearn_pc_adapter(monkeypatch, dataset):
    calls = []

    def pc(data, alpha, indep_test):
        calls.append((data, alpha, indep_test))
        return SimpleNamespace(
            G=_CausalLearnGraph([[0, -1, 0], [1, 0, -1], [0, -1, 0]])
        )

    monkeypatch.setattr(graph_module, "_import_causallearn", lambda name: pc)
    graph = GraphGeneratorFixed(
        name="pc",
        source="Causallearn",
        alpha=0.2,
        indep_test="fisherz",
        require_dag=False,
    ).construct_graph(dataset)
    np.testing.assert_array_equal(calls[0][0], dataset.concepts.numpy())
    assert calls[0][1:] == (0.2, "fisherz")
    torch.testing.assert_close(
        graph.data,
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]]),
    )


def test_causallearn_ges_adapter(monkeypatch, dataset):
    calls = []

    def ges(data, score_func):
        calls.append((data, score_func))
        return {"G": _CausalLearnGraph([[0, -1, 0], [1, 0, 0], [0, 0, 0]])}

    monkeypatch.setattr(graph_module, "_import_causallearn", lambda name: ges)
    graph = GraphGeneratorFixed(
        name="ges", source="Causallearn", score_func="custom"
    ).construct_graph(dataset)
    assert calls[0][1] == "custom"
    torch.testing.assert_close(
        graph.data,
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1])
def test_pc_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        GraphGeneratorFixed(name="pc", source="Causallearn", alpha=alpha)


def test_llm_generation_and_refinement(dataset):
    prompts = []
    responses = iter(["A->B\nA->B\nnone", "B->A", "none"])

    def backend(prompt, repeats=1, **kwargs):
        prompts.append((prompt, repeats))
        return next(responses)

    graph = GraphGeneratorFixed(
        name="fake",
        source="LLM",
        llm_backend=backend,
        repeats=3,
        domain="weather",
    ).construct_graph(dataset)
    torch.testing.assert_close(
        graph.data,
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    assert len(prompts) == 3
    assert "weather" in prompts[0][0]

    calls = []

    def orient(prompt, **kwargs):
        calls.append(prompt)
        return "B->A"

    reciprocal = ConceptGraph(
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]), node_names=["a", "b"]
    )
    refined = refine_llm(llm_backend=orient)(reciprocal)
    torch.testing.assert_close(refined.data, torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    assert len(calls) == 1


@pytest.mark.parametrize("repeats", [0, -1, 1.2, True])
def test_llm_repeats_validation(repeats):
    with pytest.raises(ValueError, match="positive integer"):
        GraphGeneratorFixed(
            name="fake",
            source="LLM",
            llm_backend=lambda *_a, **_k: "none",
            repeats=repeats,
        )


def test_learnable_dagma_initialization_forward_and_materialization():
    data = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]
    )
    generator = GraphGeneratorLearnable(
        name="dagma_cgm",
        concept_names=["a", "b", "task"],
        task_names=["task"],
        require_dag=False,
        initialization=entropy_initialization(data),
    )
    assert generator.fc1.weight.abs().sum() > 0
    assert torch.all(generator.fc1.weight[2] == 0)
    adjacency = generator()
    assert adjacency.requires_grad
    assert torch.all(adjacency.diagonal() == 0)
    assert torch.all(adjacency[2] == 0)
    graph = generator.construct_graph()
    assert graph.node_names == ["a", "b", "task"]
    assert generator.fitted


def test_learnable_construct_graph_tracks_parameter_versions():
    generator = GraphGeneratorLearnable(
        name="dagma_cgm",
        concept_names=["a", "b"],
        require_dag=False,
        initialization=random_initialization,
    )
    first = generator.construct_graph()
    second = generator.construct_graph()
    assert second is not first
    torch.testing.assert_close(second.data, first.data)
    with torch.no_grad():
        generator.fc1.weight.add_(1.0)
    third = generator.construct_graph()
    assert third is not second
    assert not torch.equal(third.data, second.data)
    assert generator.graph is third
    assert generator.fitted


def test_fixed_dagma_initialization_freezes_weights():
    adjacency = torch.tensor([[0.0, 0.5], [0.0, 0.0]])
    generator = GraphGeneratorLearnable(
        name="dagma_cgm",
        concept_names=["a", "b"],
        initialization=fixed_dagma_initialization(adjacency),
    )
    torch.testing.assert_close(generator.fc1.weight, adjacency)
    assert not generator.fc1.weight.requires_grad
    assert torch.all(generator.edge_matrix == 0)


def test_remove_weakest_cycles_breaks_cycle():
    graph = ConceptGraph(
        torch.tensor(
            [[0.0, 0.2, 0.0], [0.0, 0.0, 0.3], [0.1, 0.0, 0.0]]
        ),
        node_names=["a", "b", "c"],
    )
    refined = remove_weakest_cycles(graph)
    assert refined.data[2, 0] == 0
    assert refined.data[0, 1] == 0.2
    assert refined.data[1, 2] == 0.3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
