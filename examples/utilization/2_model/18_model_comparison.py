"""Compare CBM, CEM, two CGM variants and C2BM on the Asia network.

GES + LLM supplies the fixed graph used by C2BM and CGM-GraphFixed.
CGM-Learnable instead learns its own graph.
All models are compared on observational accuracy and causal interventions.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.classification import BinaryAccuracy

from torch_concepts import seed_everything
from torch_concepts.construct_graph import GraphGeneratorFixed, GraphGeneratorLearnable
from torch_concepts.data import BnLearnDataModule
from torch_concepts.nn import (
    CGMTrainingLoss,
    CausalCGM,
    CausallyReliableConceptBottleneckModel,
    ConceptBottleneckModel,
    ConceptEmbeddingModel,
    ConceptMetrics,
    MLP,
    WeightedConceptLoss,
)


SEED = 42
TASK = "dysp"
OUTPUT_DIR = Path("outputs/18_comparison")
INTERVENTION_PROBABILITY = 0.8
TEST_INTERVENTION_NOISE = 0.8
LLM_MODEL = "groq/openai/gpt-oss-20b"
LLM_API_KEY = ""  # Paste your Groq API key here.


def make_model(
    model_class, dm, loss=None, hidden_size=64, learning_rate=0.00075,
    **kwargs,
):
    """Create one Lightning model with the shared configuration."""
    input_size = dm.n_features[-1]
    if model_class is not CausalCGM:
        kwargs["train_inference_kwargs"] = {
            "p_int": INTERVENTION_PROBABILITY,
        }
    return model_class(
        input_size=input_size,
        annotations=dm.annotations,
        backbone=MLP(input_size, hidden_size, activation="leaky_relu"),
        latent_size=hidden_size,
        lightning=True,
        loss=loss or WeightedConceptLoss(
            concept_weight=0.8 * (len(dm.annotations.labels) - 1),
            task_weight=0.2, task_names=[TASK],
            binary=torch.nn.BCEWithLogitsLoss(),
        ),
        metrics=ConceptMetrics(
            dm.annotations, summary=True, per_concept=[TASK],
            binary={"accuracy": BinaryAccuracy()},
        ),
        optim_class=torch.optim.Adam,
        optim_kwargs={"lr": learning_rate},
        **kwargs,
    )


def intervention_label_accuracy(
    model, batches, names, evaluated_names, interventions,
):
    """Score cumulative ground-truth interventions as in C2BM Fig. 4."""
    interventions = set(interventions)
    indices = {name: index for index, name in enumerate(names)}
    query_names = [name for name in evaluated_names if name not in interventions]
    correct = {name: 0 for name in evaluated_names}
    n_samples = 0
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for inputs, target in batches:
            inputs = inputs.to(device)
            target = target.to(device)
            evidence = {
                name: target[:, indices[name]:indices[name] + 1].float()
                for name in interventions
            }
            interventional = model(
                input=inputs, query=query_names, evidence=evidence,
            )
            for name in query_names:
                predicted = (
                    interventional.logits[name].tensor.squeeze(-1).sigmoid() > 0.5
                )
                correct[name] += (
                    predicted == target[:, indices[name]].bool()
                ).sum().item()
            for name in interventions:
                correct[name] += len(target)
            n_samples += len(target)

    return {name: correct[name] / n_samples for name in evaluated_names}


def plot_interventions(rows):
    """Save Figure-4-style intervention curves."""
    table = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(6, 4))
    for name, values in table.groupby("model"):
        axis.plot(
            values["n_interventions"],
            values["label_accuracy"],
            marker="o", label=name,
        )
    axis.set(
        xlabel="Number of ground-truth interventions",
        ylabel="Label accuracy (%)",
        title="Asia intervention comparison",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "interventions.png", dpi=200)
    plt.close(figure)
    table.to_csv(OUTPUT_DIR / "interventions.csv", index=False)


def main():
    """Train, evaluate and compare all models."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)

    print("Step 1: generate Asia data and learn a GES + LLM graph")
    dm = BnLearnDataModule(
        name="asia", root="data/asia_paper_protocol",
        n_gen=10_000, batch_size=512, seed=SEED, generation_seed=SEED,
        autoencoder_exclude=[TASK],
        autoencoder_kwargs={
            "noise": 0.5, "latent_dim": 32, "lr": 0.0005,
            "epochs": 2_000, "batch_size": 512, "patience": 50,
        },
    )
    if not LLM_API_KEY:
        raise RuntimeError("Set LLM_API_KEY in this file before running the example.")
    dm.precompute_graph(
        GraphGeneratorFixed(
            name="ges",
            refinement={
                "name": LLM_MODEL,
                "api_key": LLM_API_KEY,
                "domain": "medical diagnosis",
                "use_rag": False,
            },
        ),
        cache=True,
    )
    dm.graph.plot(OUTPUT_DIR / "ges_llm", title="Asia: GES + LLM")

    print("Step 2: create CBM, CEM, two CGM variants and C2BM")
    dm.setup("fit")
    names = list(dm.annotations.labels)
    train_targets = dm.dataset.concepts[dm.trainset.indices]
    graph_generator = GraphGeneratorLearnable(
        name="dagma_cgm", concept_names=names, task_names=[TASK],
        initialization="entropy",
        initialization_data=train_targets,
    )
    models = {
        "CBM": make_model(ConceptBottleneckModel, dm, task_names=TASK),
        "CEM": make_model(
            ConceptEmbeddingModel, dm, task_names=TASK, embedding_size=4,
            hidden_size=32,
        ),
        "CGM-Learnable": make_model(
            CausalCGM,
            dm,
            task_names=TASK,
            embedding_size=8,
            graph_generator=graph_generator,
            loss=CGMTrainingLoss(),
        ),
        "CGM-GraphFixed": make_model(
            CausalCGM,
            dm,
            task_names=TASK,
            embedding_size=8,
            graph=dm.graph,
            loss=CGMTrainingLoss(),
        ),
        "C2BM": make_model(
            CausallyReliableConceptBottleneckModel,
            dm,
            graph=dm.graph,
            embedding_size=4,
            hypernet_hidden_size=4,
            hidden_size=64,
            learning_rate=0.0004,
        ),
    }

    print("Step 3: train the models")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    accuracy_rows = []
    for name, model in models.items():
        print(f"Training {name}")
        trainer = Trainer(
            max_epochs=500,
            accelerator=accelerator,
            devices=1,
            logger=False,
            enable_progress_bar=False,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=30, mode="min"),
                ModelCheckpoint(
                    dirpath=OUTPUT_DIR / "checkpoints" / name,
                    monitor="val_loss", mode="min", save_top_k=1,
                ),
            ],
        )
        trainer.fit(model, datamodule=dm)
        scores = trainer.test(
            model, datamodule=dm, ckpt_path="best", verbose=False,
        )[0]
        accuracy_rows.append({
            "model": name,
            "label_accuracy": scores["test/SUMMARY-binary_accuracy"],
            "task_accuracy": scores[f"test/{TASK}_accuracy"],
        })

    print("Step 4: materialize and plot the graph learned by CGM")
    cgm_learnable = models["CGM-Learnable"]
    cgm_graph = cgm_learnable.graph_generator.graph
    cgm_graph.plot(
        OUTPUT_DIR / "cgm_learned_graph",
        title="Asia: CGM-Learnable graph",
    )

    print("Step 5: compare observational accuracy and interventions")
    print(pd.DataFrame(accuracy_rows).to_string(index=False))
    order = [
        name
        for level in dm.graph_native.get_levels()
        for name in level
        if name != TASK
    ]
    evaluated_names = order + [TASK]
    intervention_rows = []
    for model_name, model in models.items():
        noisy_batches = [
            (
                batch["inputs"]["x"]
                + TEST_INTERVENTION_NOISE * torch.randn_like(batch["inputs"]["x"]),
                batch["concepts"]["c"],
            )
            for batch in dm.test_dataloader()
        ]
        model_rows = []
        for n_interventions in range(len(order) + 1):
            cumulative_interventions = order[:n_interventions]
            node_accuracy = intervention_label_accuracy(
                model, noisy_batches, names, evaluated_names, cumulative_interventions,
            )
            concept_accuracy = [
                node_accuracy[name] for name in order
            ]
            model_rows.append({
                "model": model_name,
                "n_interventions": n_interventions,
                "intervened_labels": ",".join(cumulative_interventions),
                "label_accuracy": 100 * sum(node_accuracy.values()) / len(node_accuracy),
                "concept_accuracy": 100 * sum(concept_accuracy) / len(concept_accuracy),
                "task_accuracy": 100 * node_accuracy[TASK],
                "node_accuracy": node_accuracy,
            })
        for row in model_rows:
            row.pop("node_accuracy")
        intervention_rows.extend(model_rows)

    accuracy_table = pd.DataFrame(accuracy_rows)
    accuracy_table.to_csv(OUTPUT_DIR / "accuracy.csv", index=False)
    plot_interventions(intervention_rows)
    print(f"Results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
