"""Reproduce the CausalCGM paper runner on dSprites with torch-concepts.

This is the dSprites-only counterpart of the authors' ``run.py``:
https://github.com/gabriele-dominici/CausalCGM/blob/main/run.py

It compares the paper's two CGM variants over five seeds:

* ``CausalCGM`` learns a DAG, initialized from conditional entropy;
* ``CausalCGM_given`` uses the known dSprites DAG.

The arrays and graph are the same as in ``15_causal_cgm_demo_reproduction.py``.
It reports observational accuracy and the intervention metrics used by the paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.classification import BinaryAccuracy

from torch_concepts import Annotations, seed_everything
from torch_concepts.construct_graph import GraphGeneratorLearnable
from torch_concepts.data.base import ConceptDataModule, ConceptDataset
from torch_concepts.data.splitters import FixedIndicesSplitter
from torch_concepts.nn import (
    CGMTrainingLoss,
    CausalCGM,
    ConceptMetrics,
    WeightedConceptLoss,
)
from torch_concepts.nn.functional import cace_score


LABELS = ["Shape", "Size", "PosY", "PosX", "Color", "Label"]
TASK = "Label"
PERTURB = "PosX"
BLOCK = "Size"
N_SEEDS = 5
EPOCHS = 200
BATCH_SIZE = 128
OUTPUT_DIR = Path("outputs/19_cgm_paper")

# adjacency[source, target] = 1 means source -> target
ADJACENCY = torch.tensor(
    [
        [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ],
    dtype=torch.float32,
)


def load_data(data_dir: Path):
    """Load exactly the preprocessed arrays consumed by experiment 15."""
    def load(name):
        path = data_dir / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing dSprites array: {path}")
        value = torch.from_numpy(np.load(path)).float()
        return value[:, None] if value.ndim == 1 else value

    train_x = load("train_features")
    train_target = torch.cat(
        [load("train_concepts"), load("train_tasks")], dim=1,
    )
    test_x = load("test_features")
    test_target = torch.cat(
        [load("test_concepts"), load("test_tasks")], dim=1,
    )
    return train_x, train_target, test_x, test_target


def make_datamodule(train_x, train_target, test_x, test_target, seed, batch_size):
    """Shuffle and split the original training set as in the paper runner."""
    permutation = torch.randperm(
        len(train_x), generator=torch.Generator().manual_seed(seed),
    )
    train_x = train_x[permutation]
    train_target = train_target[permutation]
    annotations = Annotations(
        labels=LABELS,
        cardinalities=[1] * len(LABELS),
        types=["binary"] * len(LABELS),
    )
    dataset = ConceptDataset(
        input_data=torch.cat([train_x, test_x]),
        concepts=torch.cat([train_target, test_target]),
        annotations=annotations,
        graph=pd.DataFrame(ADJACENCY.numpy(), index=LABELS, columns=LABELS),
        name="dSprites",
    )
    split = int(0.8 * len(train_x))
    datamodule = ConceptDataModule(
        dataset=dataset,
        splitter=FixedIndicesSplitter(
            train_idxs=range(split),
            val_idxs=range(split, len(train_x)),
            test_idxs=range(len(train_x), len(dataset)),
        ),
        batch_size=batch_size
    )
    datamodule.setup("fit")
    return datamodule, train_target[:split]


def make_model(datamodule, **kwargs):
    """Create one Lightning CausalCGM with the shared configuration."""
    prediction_loss = WeightedConceptLoss(
        concept_weight=1.0, task_weight=1.0, task_names=[TASK],
        binary=torch.nn.BCEWithLogitsLoss(),
    )
    return CausalCGM(
        input_size=datamodule.dataset.input_data.shape[1],
        annotations=datamodule.annotations,
        task_names=TASK,
        embedding_size=8,
        lightning=True,
        loss=CGMTrainingLoss(
            prediction_loss=prediction_loss, lambda_dag=3.0, lambda_cace=0.0,
        ),
        metrics=ConceptMetrics(
            datamodule.annotations, summary=True, per_concept=[TASK],
            binary={"accuracy": BinaryAccuracy()},
        ),
        optim_class=torch.optim.AdamW,
        optim_kwargs={"lr": 0.01},
        **kwargs,
    )


def probabilities(output):
    """Return binary node probabilities in LABELS order."""
    return torch.cat(
        [output.params[name]["logits"].sigmoid() for name in LABELS], dim=1,
    )


def intervene(model, inputs, values: dict[str, torch.Tensor]):
    query = dict.fromkeys(LABELS)
    query.update(values)
    return probabilities(model(input=inputs, query=query))


def evaluate(model, inputs, target):
    """Compute the observational and PosX/Size intervention metrics."""
    perturb_index = LABELS.index(PERTURB)
    block_index = LABELS.index(BLOCK)
    low = torch.zeros(len(inputs), 1, device=inputs.device)
    high = torch.ones_like(low)
    observed_block = target[:, block_index:block_index + 1]
    flipped = 1 - target[:, perturb_index:perturb_index + 1]

    with torch.no_grad():
        observed = probabilities(model(input=inputs))
        do_low = intervene(model, inputs, {PERTURB: low})
        do_high = intervene(model, inputs, {PERTURB: high})
        blocked_low = intervene(
            model, inputs, {PERTURB: low, BLOCK: observed_block},
        )
        blocked_high = intervene(
            model, inputs, {PERTURB: high, BLOCK: observed_block},
        )
        perturbed = intervene(model, inputs, {PERTURB: flipped})
        perturbed_blocked = intervene(
            model, inputs, {PERTURB: flipped, BLOCK: observed_block},
        )

    correct = ((observed > 0.5) == target.bool()).float()
    return {
        "accuracy": correct.mean().item(),
        "concept_accuracy": correct[:, :-1].mean().item(),
        "task_accuracy": correct[:, -1].mean().item(),
        "task_accuracy_perturb": (
            (perturbed[:, -1] > 0.5) == target[:, -1].bool()
        ).float().mean().item(),
        "task_accuracy_perturb_block": (
            (perturbed_blocked[:, -1] > 0.5) == target[:, -1].bool()
        ).float().mean().item(),
        "cace": cace_score(do_low[:, -1], do_high[:, -1]).item(),
        "cace_block": cace_score(
            blocked_low[:, -1], blocked_high[:, -1],
        ).item(),
    }


def main():
    """Train and compare the learned-graph and given-graph CGMs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    train_x, train_target, test_x, test_target = load_data(
        root / "data" / "dsprites_demo"
    )
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    rows = []

    for seed in range(N_SEEDS):
        seed_everything(seed, workers=True)
        datamodule, fit_target = make_datamodule(
            train_x, train_target, test_x, test_target, seed, BATCH_SIZE
        )
        graph_generator = GraphGeneratorLearnable(
            name="dagma_cgm", concept_names=LABELS, task_names=[TASK],
            initialization="entropy", initialization_data=fit_target,
        )
        models = {
            "CausalCGM": make_model(
                datamodule, graph_generator=graph_generator,
            ),
            "CausalCGM_given": make_model(
                datamodule, graph=datamodule.graph,
            ),
        }

        for name, model in models.items():
            print(f"Training {name}, seed {seed + 1}/{N_SEEDS}")
            checkpoint = ModelCheckpoint(
                dirpath=OUTPUT_DIR / "checkpoints" / name / str(seed),
                monitor="val/SUMMARY-binary_accuracy", mode="max",
                save_top_k=1, save_weights_only=True,
            )
            trainer = Trainer(
                max_epochs=EPOCHS, accelerator=accelerator, devices=1,
                logger=False, callbacks=[checkpoint],
            )
            trainer.fit(model, datamodule=datamodule)
            state = torch.load(
                checkpoint.best_model_path, map_location="cpu",
                weights_only=False,
            )
            model.load_state_dict(state["state_dict"])
            model.to("cpu").eval()
            rows.append({
                "model": name, "seed": seed,
                **evaluate(model, test_x, test_target),
            })

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "results_raw.csv", index=False)
    summary = results.groupby("model").agg(["mean", "std"])
    summary.to_csv(OUTPUT_DIR / "results_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
