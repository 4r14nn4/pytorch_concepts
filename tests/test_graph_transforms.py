"""Uniform callable graph refinement."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from torch_concepts.concept_graph import ConceptGraph
from torch_concepts.construct_graph import GraphGeneratorFixed, GraphGeneratorLearnable, refine_llm


def test_callable_refinement_and_cache():
    native = ConceptGraph(torch.zeros(2, 2), node_names=["a", "b"])
    calls = []

    def refine(graph, weight):
        calls.append(True)
        adjacency = graph.data.clone()
        adjacency[0, 1] = weight
        return ConceptGraph(adjacency, node_names=list(graph.node_names))

    generator = GraphGeneratorFixed(
        name="ground_truth", refinement=partial(refine, weight=0.5),
    )
    dataset = SimpleNamespace(graph_native=native, concept_names=["a", "b"], n_samples=2)
    graph = generator.construct_graph(dataset)
    assert graph.data[0, 1] == 0.5
    assert native.data.sum() == 0
    with pytest.warns(UserWarning, match="already materialized"):
        assert generator.construct_graph(dataset) is graph
    assert calls == [True]


def test_llm_refinement_is_standalone_callable():
    prompts = []

    def backend(prompt, **kwargs):
        prompts.append(prompt)
        return "A->B"

    refine = partial(
        refine_llm, llm_backend=backend, domain="weather",
        concept_descriptions={"a": "rain"},
    )
    native = ConceptGraph(torch.tensor([[0., 1.], [1., 0.]]), node_names=["a", "b"])
    generator = GraphGeneratorFixed(name="ground_truth", refinement=refine)
    dataset = SimpleNamespace(graph_native=native, concept_names=["a", "b"], n_samples=2)
    graph = generator.construct_graph(dataset)
    assert torch.equal(graph.data, torch.tensor([[0., 1.], [0., 0.]]))
    assert native.data[1, 0] == 1
    assert "rain" in prompts[0] and "weather" in prompts[0]
    refine(graph)
    assert len(prompts) == 1


def test_learnable_refinement_only_runs_at_materialization():
    calls = []

    def refine(graph):
        calls.append(True)
        return ConceptGraph(torch.zeros_like(graph.data), node_names=list(graph.node_names))

    generator = GraphGeneratorLearnable(
        name="dagma_cgm", concept_names=["a", "b"], refinement=refine,
    )
    assert generator().requires_grad
    assert calls == []
    assert generator.construct_graph().data.sum() == 0
    assert calls == [True]


def test_none_disables_refinement():
    native = ConceptGraph(torch.zeros(2, 2), node_names=["a", "b"])
    dataset = SimpleNamespace(graph_native=native, concept_names=["a", "b"], n_samples=2)
    generator = GraphGeneratorFixed(name="ground_truth")
    assert generator.construct_graph(dataset) is native


def test_invalid_refinement_rejected():
    with pytest.raises(TypeError, match="must be callable"):
        GraphGeneratorFixed(name="ground_truth", refinement={"name": "llm"})
    generator = GraphGeneratorLearnable(
        name="dagma_cgm", concept_names=["a", "b"], refinement=lambda graph: graph.data,
    )
    with pytest.raises(TypeError, match="must return a ConceptGraph"):
        generator.construct_graph()


def test_learnable_materialization_uses_parameter_cache():
    generator = GraphGeneratorLearnable(
        name="dagma_cgm", concept_names=["a", "b"], require_dag=False,
    )
    calls = []
    handle = generator.register_forward_hook(
        lambda module, args, output: calls.append(torch.is_grad_enabled()),
    )
    try:
        first = generator.construct_graph()
        assert calls == [False]
        with pytest.warns(UserWarning, match="already materialized"):
            assert generator.construct_graph() is first
        assert calls == [False]

        with torch.no_grad():
            generator.fc1.weight.add_(1)
        second = generator.construct_graph()
        assert second is not first
        assert not torch.equal(second.data, first.data)
        assert calls == [False, False]

        generator.construct_graph(force=True)
        assert calls == [False, False, False]
        assert generator().requires_grad
        assert not generator.fitted
        generator.construct_graph()
        assert calls == [False, False, False, True, False]
    finally:
        handle.remove()


def test_dataset_cache_identity_uses_names_and_sample_count():
    generator = GraphGeneratorFixed(name="ground_truth")
    first = SimpleNamespace(concept_names=["a", "b"], n_samples=10)
    second = SimpleNamespace(concept_names=["a", "b"], n_samples=10)
    assert generator._cache_key(first) == generator._cache_key(second)
    second.n_samples = 11
    assert generator._cache_key(first) != generator._cache_key(second)
    second.n_samples = 10
    second.concept_names = ["b", "a"]
    assert generator._cache_key(first) != generator._cache_key(second)


@pytest.mark.parametrize("empty_graph", [False, True])
def test_llm_distinguishes_construction_from_refinement(empty_graph):
    calls = []
    responses = iter(["A->B", "B->A", "none"])

    def backend(prompt, **kwargs):
        calls.append(prompt)
        return next(responses)

    llm = GraphGeneratorFixed(name="test", source="LLM", llm_backend=backend)
    names = ["a", "b", "c"]
    if empty_graph:
        graph = ConceptGraph(torch.zeros(3, 3), node_names=names)
        result = refine_llm(graph, llm_backend=backend)
        assert graph.data.sum() == 0
        assert result.data.sum() == 0
        assert calls == []
        return
    else:
        dataset = SimpleNamespace(concept_names=names, n_samples=2)
        result = llm.construct_graph(dataset)
    assert len(calls) == 3
    torch.testing.assert_close(
        result.data, torch.tensor([[0., 1., 0.], [0., 0., 0.], [1., 0., 0.]]),
    )


def test_llm_only_queries_reciprocal_edges_in_nonempty_graph():
    calls = []

    def backend(prompt, **kwargs):
        calls.append(prompt)
        return "B->A"

    llm = GraphGeneratorFixed(name="test", source="LLM", llm_backend=backend)
    adjacency = torch.tensor([[0., 1., 0.7], [1., 0., 0.], [0., 0., 0.]])
    graph = ConceptGraph(adjacency, node_names=["a", "b", "c"])
    result = refine_llm(graph, llm_backend=backend)
    assert len(calls) == 1
    torch.testing.assert_close(
        result.data, torch.tensor([[0., 0., 0.7], [1., 0., 0.], [0., 0., 0.]]),
    )
    torch.testing.assert_close(graph.data, adjacency)

