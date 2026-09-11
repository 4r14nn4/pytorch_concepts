"""Compare fixed-graph and learned-graph CGMs with mixed concept types.

Run from the repository root:
    python examples/utilization/2_model/20_cgm_mixed_variables.py
Quick end-to-end check:
    python examples/utilization/2_model/20_cgm_mixed_variables.py --epochs 1 --samples 256

Targets contain four columns: binary, category INDEX (0/1/2), continuous,
and continuous task. Output parameters instead have four logits (1 + 3)
and two locations/scales. MSE supervises locations, not Normal scales.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from pytorch_lightning import Trainer
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
from torchmetrics.regression import MeanSquaredError

from torch_concepts import Annotations, seed_everything
from torch_concepts.construct_graph import GraphGeneratorLearnable
from torch_concepts.data.base import ConceptDataModule, ConceptDataset
from torch_concepts.nn import (
    CausalCGM, CGMTrainingLoss, ConceptLoss, ConceptMetrics, MLP,
)


SEED = 42
LABELS = ["binary", "category", "continuous", "task"]
TASK = "task"
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "20_cgm_mixed_variables"
# Rows are sources, columns are targets: binary -> continuous,
# category -> continuous, category -> task, continuous -> task.
ADJACENCY = torch.tensor([
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
], dtype=torch.float32)


def make_datamodule(n_samples, batch_size):
    """Generate an SCM and noisy input measurements without task leakage."""
    rng = torch.Generator().manual_seed(SEED)
    binary = torch.randint(2, (n_samples,), generator=rng).float()
    category = torch.randint(3, (n_samples,), generator=rng)
    category_effect = torch.tensor([-1.0, 0.0, 1.0])[category]
    continuous = (
        0.8 * (2 * binary - 1) + 0.7 * category_effect
        + 0.25 * torch.randn(n_samples, generator=rng)
    )
    task = (
        0.8 * continuous + 0.3 * torch.sin(continuous)
        + 0.4 * category_effect + 0.15 * torch.randn(n_samples, generator=rng)
    )
    targets = torch.stack([binary, category.float(), continuous, task], dim=1)

    # Measurements of the three concepts, mixed into 12 observed features.
    # The task and its independent noise are never used to construct x.
    signals = torch.cat([
        (2 * binary - 1)[:, None],
        torch.nn.functional.one_hot(category, num_classes=3).float(),
        continuous[:, None],
    ], dim=1)
    projection = torch.randn(5, 12, generator=rng) / (5 ** 0.5)
    inputs = signals @ projection + 0.15 * torch.randn(n_samples, 12, generator=rng)
    annotations = Annotations(
        labels=LABELS, cardinalities=[1, 3, 1, 1],
        types=["binary", "categorical", "continuous", "continuous"],
    )
    dataset = ConceptDataset(
        input_data=inputs, concepts=targets, annotations=annotations,
        graph=pd.DataFrame(ADJACENCY.numpy(), index=LABELS, columns=LABELS),
        name="mixed_cgm_toy",
    )
    dm = ConceptDataModule(
        dataset=dataset, batch_size=batch_size, val_size=0.1, test_size=0.2,
        seed=SEED, workers=0, drop_last=False,
    )
    dm.setup("fit")
    return dm


def make_model(dm, learn_graph):
    graph_kwargs = (
        {"graph_generator": GraphGeneratorLearnable(
            name="dagma_cgm", source="DAGMA_CGM", concept_names=LABELS,
            task_names=[TASK], no_out_task=True, initialization="random",
        )}
        if learn_graph else {"graph": dm.graph}
    )
    return CausalCGM(
        input_size=12, annotations=dm.annotations, task_names=[TASK],
        embedding_size=16,
        backbone=MLP(input_size=12, hidden_size=64, n_layers=1),
        latent_size=64, lightning=True,
        # Mixed variables currently have no common random intervention grid.
        run_interventions=False,
        loss=CGMTrainingLoss(
            prediction_loss=ConceptLoss(
                binary=torch.nn.BCEWithLogitsLoss(),
                categorical=torch.nn.CrossEntropyLoss(),
                continuous=torch.nn.MSELoss(),
            ),
            lambda_dag=3.0, lambda_cace=0.0,
        ),
        metrics=ConceptMetrics(
            annotations=dm.annotations, summary=True, per_concept=True,
            binary={"accuracy": BinaryAccuracy()},
            categorical={"accuracy": MulticlassAccuracy(num_classes=3)},
            continuous={"mse": MeanSquaredError()},
        ),
        optim_class=torch.optim.AdamW, optim_kwargs={"lr": 0.001},
        **graph_kwargs,
    )


def main(epochs=100, n_samples=5000, batch_size=256):
    seed_everything(SEED, workers=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Step 1: Generate mixed synthetic data")
    dm = make_datamodule(n_samples, batch_size)
    print(f"Inputs: {tuple(dm.dataset.input_data.shape)}")
    print(f"Targets: {tuple(dm.dataset.concepts.shape)}; columns: {LABELS}")
    print("Category targets are indices 0/1/2, not one-hot vectors.")
    results = []
    graphs = {"Ground truth": ADJACENCY}

    for name, learn_graph in [("CGM_given", False), ("CGM_learned", True)]:
        seed_everything(SEED, workers=True)
        print(f"\nStep 2: Initialize {name}")
        model = make_model(dm, learn_graph)

        print("Step 3: Check training forward pass")
        inputs = dm.dataset.input_data[:batch_size]
        targets = dm.dataset.concepts[:batch_size]
        with torch.no_grad():
            output = model(input=inputs, target=targets)
        for quantity in ("logits", "loc", "scale", "prior_logits", "prior_loc"):
            print(f"  {quantity}: {tuple(output.params[quantity].shape)}")

        print("Step 4: Train with Lightning")
        trainer = Trainer(
            max_epochs=epochs, accelerator="auto", devices=1,
            logger=False, enable_checkpointing=False,
            default_root_dir=str(OUTPUT_DIR / name), log_every_n_steps=1,
        )
        trainer.fit(model, datamodule=dm)

        print("Step 5: Evaluate on the shared held-out test split")
        metrics = trainer.test(model, datamodule=dm)[0]
        results.append({"model": name, **metrics})
        model.eval()
        graphs[name] = model.graph.data.detach().cpu()
        pd.DataFrame(graphs[name].numpy(), index=LABELS, columns=LABELS).to_csv(
            OUTPUT_DIR / f"{name}_adjacency.csv"
        )

    print("\nStep 6: Save metrics and compare materialized DAGs")
    table = pd.DataFrame(results).set_index("model")
    table.to_csv(OUTPUT_DIR / "metrics.csv")
    print(table.to_string())
    figure, axes = plt.subplots(1, len(graphs), figsize=(13, 4))
    vmax = max(1.0, max(float(matrix.max()) for matrix in graphs.values()))
    for axis, (name, matrix) in zip(axes, graphs.items()):
        axis.imshow(matrix.numpy(), vmin=0, vmax=vmax, cmap="Blues")
        axis.set_xticks(range(4), LABELS, rotation=45, ha="right")
        axis.set_yticks(range(4), LABELS)
        axis.set(title=name, xlabel="Target", ylabel="Source")
        for source in range(4):
            for target in range(4):
                axis.text(target, source, f"{matrix[source, target]:.2f}",
                          ha="center", va="center")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "graphs.png", dpi=160)
    plt.close(figure)
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 32 or args.batch_size < 1:
        parser.error("Require epochs >= 1, samples >= 32 and batch-size >= 1.")
    main(args.epochs, args.samples, args.batch_size)
