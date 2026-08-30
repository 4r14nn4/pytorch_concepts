"""Graph generation and embedding aggregation."""

import torch
import torch.nn as nn


class GraphAggregator(nn.Module):
    """Learn or store a graph and aggregate source embeddings through it."""

    def __init__(self, generator=None, adjacency=None):
        super().__init__()
        if (generator is None) == (adjacency is None):
            raise ValueError("Pass exactly one of generator or adjacency.")
        self.generator = generator
        self.register_buffer(
            "fixed_adjacency",
            None if adjacency is None else adjacency.detach().clone(),
        )

    def graph(self):
        """Return the learned or fixed adjacency."""
        adjacency = (
            self.generator()
            if self.generator is not None
            else self.fixed_adjacency
        )
        self._last_adjacency = adjacency
        return adjacency

    def clear(self):
        """Make the next forward generate a fresh graph."""
        self._last_adjacency = None

    @property
    def adjacency(self):
        if getattr(self, "_last_adjacency", None) is None:
            raise RuntimeError("The graph layer has not generated a graph yet.")
        return self._last_adjacency

    def forward(
        self, source_embeddings, *, adjacency=None,
        source_concepts=None, target_concept=None,
    ):
        if adjacency is None:
            adjacency = (
                self.graph()
                if getattr(self, "_last_adjacency", None) is None
                else self._last_adjacency
            )
        # Evaluation graphs can be materialized after the parent model has
        # already been moved to an accelerator. Modules created at that point
        # retain the graph original (usually CPU) device, so align the graph
        # with the runtime embeddings before the batched matrix product.
        adjacency = adjacency.to(source_embeddings)
        if adjacency.shape[-2] != adjacency.shape[-1]:
            raise ValueError("adjacency must be square.")
        if source_concepts is not None:
            adjacency = adjacency[..., source_concepts, :]
        if source_embeddings.shape[-2] != adjacency.shape[-2]:
            raise ValueError("source embeddings and adjacency rows must match.")
        if target_concept is not None:
            n_targets = adjacency.shape[-1]
            if not 0 <= target_concept < n_targets:
                raise IndexError(
                    f"target_concept must be in [0, {n_targets}), got {target_concept}."
                )
            return torch.einsum(
                "...se,...s->...e", source_embeddings,
                adjacency[..., :, target_concept],
            )
        return torch.einsum("...se,...st->...te", source_embeddings, adjacency)


__all__ = ["GraphAggregator"]
