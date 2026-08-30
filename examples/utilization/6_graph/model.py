"""
Example: Graph Generator Inside CausalCGM

The same components prepared on the data side in ``caching.py`` can instead
run inside the high-level model:

- ``GraphGeneratorLearnable('dagma_cgm')`` is a trainable model component,
  so its adjacency is optimized jointly with the CGM mechanisms.
- ``DAGMALoss`` penalizes cyclic learned structures.

This example does not precompute embeddings or a fixed graph. The graph is
randomly initialized and is then learned by gradient descent.

Flow:
1. Generate samples from the Asia Bayesian network and create the data splits.
2. Initialize a DAGMA-CGM graph generator without preparing data splits.
3. Build ``CausalCGM`` with the learnable graph generator.
4. Train all non-frozen components with PyTorch Lightning.
5. Switch to evaluation mode and inspect the graph materialized by CausalCGM.
"""
import torch
from pytorch_lightning import Trainer

from torch_concepts import seed_everything
from torch_concepts.construct_graph import GraphGeneratorLearnable
from torch_concepts.data import BnLearnDataModule
from torch_concepts.nn import CGMTrainingLoss, CausalCGM, MLP


def main():
    seed_everything(42)
    dm = BnLearnDataModule(
        name="asia", n_gen=500, batch_size=64, seed=42,
    )
    # Random initialization needs no data split; Trainer.fit calls setup().
    graph_generator = GraphGeneratorLearnable(
        name="dagma_cgm",
        concept_names=list(dm.annotations.labels),
        n_tasks=1,
    )
    model = CausalCGM(
        input_size=dm.n_features[-1],
        annotations=dm.annotations,
        task_names="dysp",
        backbone=MLP(dm.n_features[-1], 128),
        latent_size=128,
        embedding_size=8,
        graph_generator=graph_generator,
        lightning=True,
        loss=CGMTrainingLoss(),
        optim_class=torch.optim.AdamW,
        optim_kwargs={"lr": 0.01},
    )
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=3,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=dm)

    # Take one final snapshot, then construct and refine the learned graph.
    with torch.no_grad():
        graph_generator()
    learned_graph = graph_generator.construct_graph(dm.dataset)
    print("Learned Asia graph:\n", learned_graph.to_pandas())


if __name__ == "__main__":
    main()
