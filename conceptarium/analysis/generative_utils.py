"""Shared machinery for ``analysis/run_generative_analysis.py``.

Four independent pieces, in the order the script uses them:

1. **Run discovery and rebuild** — find finished Hydra jobs on disk and
   reconstruct their datamodule + model from the config saved beside them.
2. **Modalities** — everything that knows what a *sample* is. An image today; a
   token sequence tomorrow. Metrics never touch raw model output, so adding a
   generative model over discrete data is one :class:`Modality` subclass and no
   change to any metric.
3. **Metrics** — each one a small class in :data:`METRICS`, taking an
   :class:`EvalContext` and returning ``{column_name: value}``. Adding a metric
   is one class plus one registry entry; nothing else moves.
4. **Results** — the per-run and seed-averaged CSVs.

The split that makes this extensible is between the *shape* of a metric and the
*modality* it is measured in. FID, for instance, is a Fréchet distance between
feature vectors — computed by ``torchmetrics`` — and only
:meth:`Modality.features` knows those come from an Inception network for images
and would come from a language model for text.
"""

from __future__ import annotations

import csv
import logging
import math
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf

from conceptarium.utils import (
    attach_latent_encoder,
    resolve_graph,
    setup_run_env,
    update_config_from_data,
)
from torch_concepts.nn import ReconstructionLoss
from torch_concepts.nn.modules.mid.distributions import spec_for

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Run discovery and rebuild
# ===========================================================================
def is_job_dir(path: Path) -> bool:
    """Is ``path`` a finished Hydra *job* directory?

    Needs both halves of a rebuildable run: the config Hydra wrote, and at least
    one checkpoint. The checkpoint also stands in for "the run succeeded" — one
    that crashed before saving cannot be analysed — and rules out a multirun
    *sweep* directory, which has a ``.hydra`` of its own but no checkpoints.
    """
    return (path / ".hydra" / "config.yaml").is_file() and any(
        (path / "checkpoints").glob("*.ckpt")
    )


def matches_filters(job_dir: Path, filters: Optional[dict]) -> bool:
    """Does the run's own saved config satisfy every ``dotted.key: value``?"""
    if not filters:
        return True
    try:
        job_cfg = OmegaConf.load(job_dir / ".hydra" / "config.yaml")
    except Exception:  # unreadable config -- treat as non-matching, not fatal
        return False
    return all(
        (found := OmegaConf.select(job_cfg, key)) is not None and str(found) == str(want)
        for key, want in filters.items()
    )


def find_job_dirs(roots: Sequence[Path], filters: Optional[dict]) -> List[Path]:
    """Every matching job directory under ``roots``, oldest first.

    Walks the filesystem rather than consulting ``runs.csv``: the registry records
    absolute paths from the machine that trained the run and is not written at all
    under ``debug: true``, so a run copied in from a GPU box — the normal way
    results arrive — is invisible to it while sitting in plain sight on disk.
    """
    seen: Dict[Path, None] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for config in sorted(root.rglob(".hydra/config.yaml")):
            job_dir = config.parent.parent
            if is_job_dir(job_dir) and matches_filters(job_dir, filters):
                seen.setdefault(job_dir.resolve(), None)
    return sorted(seen, key=lambda p: p.stat().st_mtime)


def resolve_job_dirs(cfg: DictConfig) -> List[Path]:
    """Every Hydra job directory to analyse, oldest first.

    An explicit ``job_dir`` wins and may be a path or a list of them. Otherwise
    every run under ``search_dirs`` matching ``filters`` is returned; with
    ``all_jobs: false`` only the newest.
    """
    if cfg.get("job_dir"):
        given = cfg.job_dir
        return [Path(given)] if isinstance(given, str) else [Path(d) for d in given]

    # Relative roots resolve against the launch directory. `get_original_cwd`
    # needs an initialised HydraConfig -- true for the script, not for a REPL.
    try:
        base = Path(get_original_cwd())
    except ValueError:
        base = Path.cwd()
    roots = [Path(d) if Path(d).is_absolute() else base / d
             for d in cfg.get("search_dirs") or ["outputs"]]
    filters = dict(cfg.filters) if cfg.get("filters") else None
    found = find_job_dirs(roots, filters)
    if not found:
        raise SystemExit(
            f"No runs found under {[str(r) for r in roots]} matching {filters}.\n"
            "Train one first:\n"
            "  python run_experiment.py --config-name cbgm dataset=colormnist\n"
            "or point `search_dirs` elsewhere, or pass job_dir=<hydra job dir>."
        )
    logger.info("found %d matching run(s)", len(found))
    return found if cfg.get("all_jobs", True) else found[-1:]


def resolve_device(accelerator: Optional[str]) -> torch.device:
    """Device to analyse on. ``'auto'`` picks CUDA when present, else CPU.

    Deliberately never MPS: :mod:`torch_concepts.backbone` refuses it too, because
    torchvision preprocessing and HuggingFace models misbehave there.
    """
    if accelerator in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(accelerator)


def load_job(job_dir: Path) -> Tuple[DictConfig, Optional[Path]]:
    """The run's saved config and its first checkpoint (``None`` if there is none)."""
    cfg = OmegaConf.load(Path(job_dir) / ".hydra" / "config.yaml")
    ckpts = sorted((Path(job_dir) / "checkpoints").glob("*.ckpt"))
    return cfg, ckpts[0] if ckpts else None


def rebuild(job_cfg: DictConfig):
    """Datamodule and model for a finished job, wired as ``run_experiment`` did.

    Deliberately not ``evaluate.evaluate_job``: that helper reads the pre-
    ``model_cls`` config schema and never builds the latent encoder, so it cannot
    reconstruct a current model.
    """
    # Hydra writes .hydra/config.yaml *before* main() runs, so the saved config
    # has not been through setup_run_env and carries no data root -- without this
    # the datamodule falls back to a cwd-relative path and re-downloads.
    job_cfg = setup_run_env(job_cfg)

    datamodule = instantiate(job_cfg.dataset.datamodule, _convert_="all")
    backbone = instantiate(job_cfg.dataset.backbone, _convert_="all")
    datamodule.setup("fit")
    job_cfg = update_config_from_data(job_cfg, datamodule)

    model = instantiate(job_cfg.model.model_cls, _convert_="all", _partial_=True)(
        annotations=datamodule.annotations,
        graph=resolve_graph(datamodule.graph, datamodule.annotations,
                            job_cfg.dataset.default_task_names),
        backbone=attach_latent_encoder(job_cfg, backbone),
        loss=instantiate(job_cfg.loss, _convert_="all"),
        metrics=instantiate(job_cfg.metrics, _convert_="all", _partial_=True)(
            annotations=datamodule.annotations
        ),
    )
    return datamodule, model


def run_identity(job_cfg: DictConfig, job_dir: Path) -> Dict[str, str]:
    """The columns identifying a run: dataset, model, seed, and its directory.

    Read from the run's own saved config, so the CSV says what was trained rather
    than what this analysis was configured with.
    """
    target = OmegaConf.select(job_cfg, "model.model_cls._target_") or "unknown"
    return {
        "dataset_name": str(OmegaConf.select(job_cfg, "dataset.name") or "unknown"),
        "model_name": target.split(".")[-1],
        "seed": str(OmegaConf.select(job_cfg, "seed") or ""),
        "run_dir": str(job_dir),
    }


# ===========================================================================
# 2. Modalities
# ===========================================================================
class Modality(ABC):
    """What a generated sample *is*, and the two things metrics ask of it.

    Every metric below is written against this interface rather than against
    pixels, so supporting a generative model over discrete data (text, sequences)
    means adding a subclass here — a token-level classifier and a language-model
    feature extractor — and changing no metric.
    """

    #: Event shape of one sample, e.g. ``(3, 28, 28)``.
    shape: Tuple[int, ...]

    @abstractmethod
    def features(self, flat: torch.Tensor) -> torch.Tensor:
        """``(N, prod(shape)) -> (N, D)`` features for a distributional distance."""

    @abstractmethod
    def build_classifier(self, n_outputs: int, width: int) -> nn.Module:
        """A ``(N, prod(shape)) -> (N, n_outputs)`` concept classifier, untrained.

        ``width`` is the narrowest layer; the judge only has to separate binary
        concepts, so it can stay far smaller than the model under test.
        """


class ImageModality(Modality):
    """Images. Features from Inception-v3 (the standard FID extractor).

    The extractor is built lazily and cached: it downloads ImageNet weights on
    first use, which a run that asks for no distributional metric should not pay.
    Inputs are resized to 299x299 and expanded to 3 channels, as FID expects.
    """

    def __init__(self, shape: Sequence[int]) -> None:
        self.shape = tuple(int(s) for s in shape)
        self._inception: Optional[nn.Module] = None

    def _extractor(self, device) -> nn.Module:
        if self._inception is None:
            from torchvision.models import Inception_V3_Weights, inception_v3

            net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            net.fc = nn.Identity()  # 2048-d pool3 features, the FID convention
            self._inception = net.eval().to(device)
        return self._inception

    @torch.inference_mode()
    def features(self, flat: torch.Tensor) -> torch.Tensor:
        images = flat.reshape(-1, *self.shape).clamp(0, 1)
        if images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1)
        images = nn.functional.interpolate(
            images, size=(299, 299), mode="bilinear", align_corners=False
        )
        # ImageNet normalisation, which the pretrained weights expect.
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        return self._extractor(images.device)((images - mean) / std)

    def build_classifier(self, n_outputs: int, width: int = 16) -> nn.Module:
        """A small conv net: three stride-2 stages, then global pooling.

        Deliberately tiny. It only has to answer "is this binary concept present",
        which on these datasets is an easy problem — and it is built alongside a
        generative model that is already holding the GPU, so a large judge is the
        thing that runs the analysis out of memory. ``width`` doubles per stage,
        so the default is 16/32/64 channels: a few tens of thousands of
        parameters. Raise it if `classifier_accuracy` comes out below the 98% the
        metric assumes.

        Global average pooling, not a flatten, so the parameter count is
        independent of the image size — CelebA at 64x64 costs the same as
        Color-MNIST at 28x28.
        """
        channels = self.shape[0]
        return nn.Sequential(
            nn.Unflatten(-1, self.shape),
            nn.Conv2d(channels, width, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(4 * width, n_outputs),
        )


def modality_for(datamodule) -> Modality:
    """The modality of a datamodule's samples.

    One branch today. A dataset of token sequences would be recognised here and
    return a ``TextModality``; nothing else in this file would change.
    """
    shape = tuple(datamodule.n_features)
    if len(shape) == 3:
        return ImageModality(shape)
    raise NotImplementedError(
        f"No Modality for samples of shape {shape}. Images are (C, H, W); add a "
        "Modality subclass for other data and register it here."
    )


# ===========================================================================
# 3. Metrics
# ===========================================================================
def concept_variables(model) -> List:
    """The model's concept variables, in graph order."""
    return [v for v in model.pgm.variables.values() if v.variable_type == "concept"]


def concept_states(variable, device=None) -> List[Tuple[str, torch.Tensor]]:
    """Every state of ``variable`` as ``(label, value)``, in the PGM's layout.

    A binary concept is one column that is 0 or 1; a categorical one is a one-hot
    row of width ``size``. Both are what the CPD's value looks like, so either can
    be forced straight into a query.
    """
    size = variable.size
    values = ([("0", torch.zeros(1, 1)), ("1", torch.ones(1, 1))] if size == 1
              else [(str(s), torch.eye(size)[s].unsqueeze(0)) for s in range(size)])
    return values if device is None else [(k, v.to(device)) for k, v in values]


def observation_of(model, out, name: str = "input") -> torch.Tensor:
    """The reconstructed observation, whichever quantity its family reports.

    ``probs`` for a Bernoulli, ``loc`` for a Normal, ``value`` for a Delta — read
    off the variable, so this survives a change of observation family.
    """
    return getattr(out, spec_for(model.pgm.variables[name].distribution).primary_param)[name]


@dataclass
class EvalContext:
    """Everything a metric may need, with the expensive parts computed once.

    Generation and classifier training are shared: FID and steerability both draw
    from the prior, and re-drawing per metric would both waste time and make the
    two numbers describe different samples.
    """

    model: nn.Module
    engine: object
    datamodule: object
    modality: Modality
    device: torch.device
    cfg: DictConfig
    _generated: Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor]]] = None
    _classifier: Optional[nn.Module] = None
    _classifier_accuracy: float = field(default=float("nan"))

    @property
    def concepts(self) -> List:
        return concept_variables(self.model)

    @property
    def query(self) -> List[str]:
        """A decoding query: the observation, ``z`` and the concepts.

        The embedding and bottleneck variables are left out — their events have
        differing widths and cannot share one annotated tensor. They are ancestors
        of ``input``, so they are still computed, just not reported.
        """
        return ["input", "z", *(v.name for v in self.concepts)]

    def real_samples(self, n: int) -> torch.Tensor:
        """``n`` flattened test images."""
        loader = self.datamodule.test_dataloader() or self.datamodule.val_dataloader()
        out, taken = [], 0
        for batch in loader:
            x = batch["inputs"]["x"]
            out.append(x.reshape(x.shape[0], -1)[: n - taken])
            taken += out[-1].shape[0]
            if taken >= n:
                break
        return torch.cat(out).to(self.device)

    def generate(self, n: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """``(samples, drawn)`` from ``n`` prior draws, computed once and cached.

        ``drawn`` maps each variable to its sampled value — notably ``z`` and each
        concept — which the interventions replay so only the target concept moves.
        """
        if self._generated is None or self._generated[0].shape[0] < n:
            # Batched: one query for a few thousand draws holds the decoder's
            # activations for all of them at once.
            batches = []
            with torch.no_grad():
                for start in range(0, n, self.batch_size):
                    out = self.engine.query(
                        query=self.query, evidence={},
                        n_samples=min(self.batch_size, n - start),
                    )
                    batches.append((
                        observation_of(self.model, out),
                        {name: getattr(out.samples[name], "tensor", out.samples[name])
                         for name in out.samples.annotation.labels},
                    ))
            self._generated = (
                torch.cat([s for s, _ in batches]),
                {key: torch.cat([d[key] for _, d in batches]) for key in batches[0][1]},
            )
        samples, drawn = self._generated
        return samples[:n], {k: v[:n] for k, v in drawn.items()}

    @property
    def batch_size(self) -> int:
        """Rows per generative pass. Caps peak memory, not the sample count."""
        return int(self.cfg.get("eval_batch_size", 128))

    def decode(self, evidence: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode the observation under ``evidence`` (an intervention).

        Chunked: steerability decodes every candidate sample, so on CelebA a
        single pass would allocate the decoder activations for the whole draw
        at once.
        """
        n = next(iter(evidence.values())).shape[0]
        chunks = []
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                piece = {k: v[start:start + self.batch_size] for k, v in evidence.items()}
                chunks.append(observation_of(
                    self.model, self.engine.query(query=["input"], evidence=piece)
                ))
        return torch.cat(chunks)

    def classifier(self) -> Tuple[nn.Module, float]:
        """The concept classifier trained on *real* data, and its test accuracy.

        The paper's steerability metric is defined against an external judge, not
        the model under test — using the model's own concept head would ask it to
        grade its own homework. Trained once and cached.
        """
        if self._classifier is None:
            self._classifier, self._classifier_accuracy = train_concept_classifier(
                self.datamodule, self.modality, self.device,
                epochs=int(self.cfg.get("classifier_epochs", 5)),
                lr=float(self.cfg.get("classifier_lr", 1e-3)),
                width=int(self.cfg.get("classifier_width", 16)),
            )
        return self._classifier, self._classifier_accuracy


# --- the metric interface --------------------------------------------------
class Metric(ABC):
    """One quantitative metric. Subclass, set ``name``, register in :data:`METRICS`."""

    name: str

    @abstractmethod
    def compute(self, ctx: EvalContext) -> Dict[str, float]:
        """``{column_name: value}`` — one metric may report several columns."""


class _ModalityFeatures(nn.Module):
    """Adapts :meth:`Modality.features` to the ``nn.Module`` torchmetrics expects.

    ``FrechetInceptionDistance(feature=<Module>)`` is the documented hook for a
    custom extractor. Going through it — rather than letting torchmetrics build
    its own Inception — is what keeps the metric modality-agnostic *and* avoids
    the ``torch-fidelity`` dependency its integer ``feature`` argument requires.
    """

    def __init__(self, modality: Modality) -> None:
        super().__init__()
        self.modality = modality

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        return self.modality.features(flat)


class FID(Metric):
    """Fréchet distance between generated samples and real data (Table 2).

    ``|mu_r - mu_f|^2 + tr(C_r + C_f - 2 (C_r C_f)^{1/2})``, computed by
    :class:`torchmetrics.image.fid.FrechetInceptionDistance` over the features
    :meth:`Modality.features` supplies. For images those are Inception-v3 pool3
    activations, so this is FID exactly. Lower is better.

    torchmetrics accumulates sums and covariance sums per ``update``, so the
    samples are streamed in ``eval_batch_size`` chunks and the full feature matrix
    is never held at once.

    ``n_fid_samples`` sets both sides. The published number uses 40k; the default
    here is far smaller, and FID is biased upward at small sample counts — so
    compare runs at the same setting rather than against the paper.
    """

    name = "fid"

    def compute(self, ctx: EvalContext) -> Dict[str, float]:
        from torchmetrics.image.fid import FrechetInceptionDistance

        n = int(ctx.cfg.get("n_fid_samples", 2048))
        generated, _ = ctx.generate(n)
        real = ctx.real_samples(n)
        if min(real.shape[0], generated.shape[0]) < 2:
            return {"fid": float("nan")}

        extractor = _ModalityFeatures(ctx.modality)
        # Declaring the width skips torchmetrics' probe, which would call the
        # extractor with a dummy *image* -- our features take a flat sample. One
        # forward on a single row, which also pins the lazily-built extractor to
        # the right device before any real data reaches it.
        extractor.num_features = int(
            extractor(torch.zeros(1, real.shape[1], device=ctx.device)).shape[-1]
        )
        # `normalize` is ignored for a custom extractor (torchmetrics only rescales
        # to bytes for its own Inception), so the features are used as produced.
        metric = FrechetInceptionDistance(feature=extractor, normalize=False).to(ctx.device)
        for source, is_real in ((real, True), (generated, False)):
            for start in range(0, source.shape[0], ctx.batch_size):
                metric.update(source[start:start + ctx.batch_size], real=is_real)
        return {"fid": float(metric.compute())}


class Steerability(Metric):
    """Does intervening on a concept put that concept into the output? (Table 1)

    The paper's procedure, followed exactly: generate from the prior, keep the
    samples an external classifier says do *not* have the concept, intervene to
    turn it **on**, and report the fraction the classifier now says do. Reported
    as a percentage, so the numbers are directly comparable to the paper's
    Table 1.

    **Binary concepts only.** "Turn the concept on" has no single meaning for a
    k-way categorical one, so non-binary concepts are skipped with a warning
    rather than scored under some invented convention. If a model has no binary
    concepts the metric reports ``nan``.

    Every *other* concept is clamped to its own drawn value, so the intervention
    is the only thing that differs; left free they would be redrawn and confound
    the measurement with fresh sampling noise.

    Reports ``steerability`` (the mean over binary concepts), one
    ``steerability_<concept>`` column each, and ``classifier_accuracy`` — the
    judge's own test accuracy, which the paper requires to be at least 98%. Treat
    a steerability number sitting under a low classifier accuracy as unreadable.
    """

    name = "steerability"

    def compute(self, ctx: EvalContext) -> Dict[str, float]:
        annotations = ctx.datamodule.annotations
        names = [name for name, _ in binary_concepts(annotations)]
        skipped = [n for n, kind in zip(annotations.labels, annotations.types)
                   if kind != "binary"]
        if skipped:
            logger.warning(
                "steerability is defined for binary concepts only; skipping %s",
                skipped,
            )
        if not names:
            logger.warning("no binary concepts: steerability is not defined here")
            return {"steerability": float("nan")}

        classifier, accuracy = ctx.classifier()
        batch = ctx.batch_size
        generated, drawn = ctx.generate(int(ctx.cfg.get("n_steering_samples", 1000)))
        predicted = predict_binary(classifier, generated, names, batch)
        by_name = {v.name: v for v in ctx.concepts}

        results: Dict[str, float] = {"classifier_accuracy": accuracy}
        per_concept = []
        for name in names:
            variable = by_name.get(name)
            if variable is None:  # annotated in the data, absent from the graph
                continue
            # The paper's selection step: only samples that do NOT already show
            # the concept can demonstrate that steering added it.
            candidates = (predicted[name] == 0).nonzero().flatten()
            if candidates.numel() == 0:
                results[f"steerability_{name}"] = float("nan")
                continue
            evidence = {"z": drawn["z"][candidates]}
            for other in ctx.concepts:
                if other.name != name:
                    evidence[other.name] = drawn[other.name][candidates]
            evidence[name] = torch.ones(candidates.numel(), variable.size,
                                        device=ctx.device)
            steered = predict_binary(classifier, ctx.decode(evidence), names, batch)[name]
            score = 100.0 * (steered == 1).float().mean().item()
            results[f"steerability_{name}"] = score
            per_concept.append(score)
        results["steerability"] = (
            sum(per_concept) / len(per_concept) if per_concept else float("nan")
        )
        return results


class ReconstructionNLL(Metric):
    """Negative log-likelihood of the test set under the model's own observation CPD.

    Not from the paper — kept because it is the objective's own reconstruction
    term, and a steerability number is only meaningful for a model that fits.
    """

    name = "reconstruction_nll"

    def compute(self, ctx: EvalContext) -> Dict[str, float]:
        loader = ctx.datamodule.test_dataloader() or ctx.datamodule.val_dataloader()
        reconstruction = ReconstructionLoss(variable="input")
        max_batches = ctx.cfg.get("max_eval_batches")
        vi_query = list(ctx.model.pgm.variables)
        total, nll = 0, 0.0
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                if max_batches is not None and index >= max_batches:
                    break
                x = batch["inputs"]["x"].to(ctx.device)
                out = ctx.model(query=vi_query, input=x)
                out.extra = {"evidence": {"input": x}}
                nll += reconstruction(out) * x.shape[0]
                total += x.shape[0]
        return {"reconstruction_nll": float(nll) / max(total, 1),
                "n_test_samples": float(total)}


class ConceptAccuracy(Metric):
    """The model's own concept metrics on the test set.

    Reuses ``model.test_metrics`` — the collection ``conf/metrics`` configured and
    training logged — so binary and categorical concepts are routed correctly
    without this file re-deriving the rule.
    """

    name = "concept_accuracy"

    def compute(self, ctx: EvalContext) -> Dict[str, float]:
        loader = ctx.datamodule.test_dataloader() or ctx.datamodule.val_dataloader()
        max_batches = ctx.cfg.get("max_eval_batches")
        vi_query = list(ctx.model.pgm.variables)
        metrics = ctx.model.test_metrics
        metrics.reset()
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                if max_batches is not None and index >= max_batches:
                    break
                x = batch["inputs"]["x"].to(ctx.device)
                c = batch["concepts"]["c"].to(ctx.device)
                metrics.update(ctx.model(query=vi_query, input=x),
                               ctx.model.prepare_target(c))
        return {k: float(v) for k, v in metrics.compute().items()}


#: Every available metric, by the name a config uses to request it. Adding one is
#: a :class:`Metric` subclass and a line here.
METRICS: Dict[str, Metric] = {
    m.name: m() for m in (FID, Steerability, ReconstructionNLL, ConceptAccuracy)
}


# --- the external judge ----------------------------------------------------
def binary_concepts(annotations) -> List[Tuple[str, int]]:
    """``[(name, target_column)]`` for the **binary** concepts only.

    ``target_column`` indexes the ground-truth tensor ``c``. Non-binary concepts
    are left out: :class:`Steerability` is defined for binary concepts (the
    paper's "turn the concept on" has no single meaning for a k-way one), so the
    judge is not asked to predict them either — which also keeps it small.

    Returned as a list of names rather than positions because the classifier is
    trained from the *data's* annotations while the metric addresses the *model's*
    concept variables, and ``ConceptDataset`` reorders concepts by type, so the
    two orders need not agree.
    """
    return [(name, index)
            for index, (name, kind) in enumerate(zip(annotations.labels, annotations.types))
            if kind == "binary"]


def predict_binary(classifier: nn.Module, flat: torch.Tensor, names: Sequence[str],
                   batch_size: int = 128) -> Dict[str, torch.Tensor]:
    """Each binary concept's predicted ``0``/``1``, per sample.

    Batched: the metric classifies every generated sample at once otherwise, which
    on a GPU already holding the generative model is where the analysis runs out
    of memory.
    """
    chunks = []
    with torch.inference_mode():
        for start in range(0, flat.shape[0], batch_size):
            logits = classifier(flat[start:start + batch_size].clamp(0, 1))
            chunks.append((logits > 0).long())
    predicted = torch.cat(chunks) if chunks else torch.zeros(0, len(names), dtype=torch.long)
    return {name: predicted[:, i] for i, name in enumerate(names)}


def train_concept_classifier(datamodule, modality: Modality, device,
                             epochs=5, lr=1e-3, width=16):
    """Train the external judge on real data; return ``(classifier, test accuracy)``.

    One binary head per binary concept, scored by
    ``binary_cross_entropy_with_logits``. Non-binary concepts are not predicted at
    all — :class:`Steerability` does not score them, so a head for them would be
    parameters trained for nothing.

    Accuracy is the mean over concepts of per-concept accuracy on the test split:
    the number the paper requires to be at least 98% before the steerability
    figure means anything.
    """
    columns = binary_concepts(datamodule.annotations)
    if not columns:
        raise ValueError(
            "The steerability judge needs at least one binary concept; this "
            f"dataset annotates types {list(datamodule.annotations.types)}."
        )
    targets = torch.tensor([index for _, index in columns], device=device)
    classifier = modality.build_classifier(len(columns), width).to(device)
    optimiser = torch.optim.AdamW(classifier.parameters(), lr=lr)

    classifier.train()
    for _ in range(epochs):
        for batch in datamodule.train_dataloader():
            x = batch["inputs"]["x"].to(device)
            c = batch["concepts"]["c"].to(device)
            logits = classifier(x.reshape(x.shape[0], -1))
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, c[:, targets].float()
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    classifier.eval()
    correct, total = torch.zeros(len(columns)), 0
    loader = datamodule.test_dataloader() or datamodule.val_dataloader()
    with torch.inference_mode():
        for batch in loader:
            x = batch["inputs"]["x"].to(device)
            c = batch["concepts"]["c"].to(device)
            predicted = (classifier(x.reshape(x.shape[0], -1)) > 0).long()
            correct += (predicted == c[:, targets].long()).sum(0).cpu()
            total += x.shape[0]
    accuracy = float((correct / max(total, 1)).mean())
    if accuracy < 0.98:
        logger.warning(
            "concept classifier reached %.1f%% accuracy, below the 98%% the "
            "steerability metric assumes; raise `classifier_epochs` or "
            "`classifier_width`.", 100 * accuracy
        )
    return classifier, accuracy


# ===========================================================================
# 4. Results
# ===========================================================================
def write_results(rows: List[Dict[str, object]], out_root: Path) -> Tuple[Path, Path]:
    """Write ``results.csv`` and its seed-averaged companion; return both paths.

    Columns are the union over rows, so a run missing a metric leaves a blank cell
    rather than shifting everyone else's columns — which is what lets a new metric
    be added without re-running the old experiments.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    identity = ["model_name", "dataset_name", "seed", "run_dir"]
    metrics = sorted({k for row in rows for k in row} - set(identity))

    per_run = out_root / "results.csv"
    with open(per_run, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=identity + metrics, restval="")
        writer.writeheader()
        writer.writerows(rows)

    aggregated = out_root / "results_by_model.csv"
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for row in rows:
        groups.setdefault((row["dataset_name"], row["model_name"]), []).append(row)

    fields = ["model_name", "dataset_name", "n_seeds"]
    for metric in metrics:
        fields += [f"{metric}_mean", f"{metric}_std"]
    with open(aggregated, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, restval="")
        writer.writeheader()
        for (dataset, model), group in sorted(groups.items()):
            record = {"model_name": model, "dataset_name": dataset,
                      "n_seeds": len(group)}
            for metric in metrics:
                reported = [float(r[metric]) for r in group
                            if isinstance(r.get(metric), (int, float))]
                if not reported:
                    continue  # metric not computed for this model: blank cells
                # A metric can legitimately report nan for a run -- steerability
                # on a model with no binary concepts, say -- and averaging must
                # survive it. Non-finite values are excluded from the statistics
                # (`statistics.stdev` cannot even accept them: it evaluates its
                # sums as exact Fractions and raises AttributeError on nan/inf),
                # and a group with nothing finite left keeps its nan rather than
                # silently becoming a blank cell.
                values = [v for v in reported if math.isfinite(v)]
                if not values:
                    record[f"{metric}_mean"] = record[f"{metric}_std"] = float("nan")
                    continue
                record[f"{metric}_mean"] = statistics.fmean(values)
                # A single seed has no spread; report 0.0 rather than blank, so a
                # reader is not left wondering whether the run failed.
                record[f"{metric}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
            writer.writerow(record)
    return per_run, aggregated


def qualitative_dir(out_root: Path, identity: Dict[str, str]) -> Path:
    """``results/qualitative/<dataset>/<model>/<run>`` for one run's figures.

    The trailing run component is not in the requested layout but is necessary:
    a sweep has several seeds per (dataset, model), and without it each would
    overwrite the previous one's figures and config.
    """
    run = Path(identity["run_dir"]).name
    seed = identity.get("seed")
    leaf = f"seed{seed}_{run}" if seed else run
    path = out_root / "qualitative" / identity["dataset_name"] / identity["model_name"] / leaf
    path.mkdir(parents=True, exist_ok=True)
    return path


# ===========================================================================
# Figures
# ===========================================================================
def save_grid(rows: List[torch.Tensor], shape, path: Path,
              row_labels=None, col_labels=None) -> None:
    """Write a grid of images, one supplied tensor batch per row.

    Rows may be of different lengths — the grid is as wide as the longest and
    short rows leave their tail blank. ``col_labels`` is one list per row
    (``None`` for an unlabelled row). The backend is forced to Agg so this runs on
    a headless GPU box.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping %s", path.name)
        return

    n_rows, n_cols = len(rows), max(r.shape[0] for r in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows + 0.4), squeeze=False)
    for r, batch in enumerate(rows):
        images = batch.detach().reshape(-1, *shape).cpu()
        for c in range(n_cols):
            ax = axes[r][c]
            ax.axis("off")
            if c >= images.shape[0]:
                continue
            # (C, H, W) -> (H, W, C); a single channel drops to greyscale.
            ax.imshow(images[c].permute(1, 2, 0).squeeze().numpy().clip(0, 1))
            titles = col_labels[r] if col_labels is not None else None
            if titles is not None and c < len(titles):
                ax.set_title(titles[c], fontsize=6)
        if row_labels is not None:
            # Re-enable the axis only to carry the label, without drawing a box.
            ax = axes[r][0]
            ax.axis("on")
            ax.set_ylabel(row_labels[r], fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", path)


def figure_overview(ctx: EvalContext, out_dir: Path, n: int) -> None:
    """Originals, their reconstructions, and unconditional prior samples.

    The generations are *not* reconstructions of the originals above them — they
    are independent draws from ``p(z)``. The rows sit together because a
    generative model is judged on both at once: sharp reconstructions beside
    incoherent samples means the posterior has drifted off the prior.
    """
    images = ctx.real_samples(n)
    generated, _ = ctx.generate(n)
    with torch.no_grad():
        # The guide is the only route from an image to a posterior z, so encoding
        # needs the variational engine (the model's own eval one), which requires
        # every variable in its query. The posterior *mean*, not a draw, so the
        # row shows the model's best reconstruction.
        encoded = ctx.model(query=list(ctx.model.pgm.variables),
                            input=images.reshape(-1, *ctx.modality.shape))
        recon = ctx.decode({"z": encoded.guide_params["loc"]["z"].tensor})
    save_grid([images, recon, generated], ctx.modality.shape, out_dir / "overview.png",
              row_labels=["original", "reconstruction", "generation"])


def figure_steering(ctx: EvalContext, variable, out_dir: Path) -> None:
    """One figure per concept: a generated sample, then that sample at every state.

    The picture behind the :class:`Steerability` number, for one draw.
    """
    generated, drawn = ctx.generate(max(1, int(ctx.cfg.get("n_samples", 10))))
    states = concept_states(variable, device=ctx.device)
    n = len(states)
    evidence = {"z": drawn["z"][:1].expand(n, -1)}
    for other in ctx.concepts:
        if other.name != variable.name:
            evidence[other.name] = drawn[other.name][:1].expand(n, -1)
    evidence[variable.name] = torch.cat([value for _, value in states], dim=0)
    save_grid([generated[:1], ctx.decode(evidence)], ctx.modality.shape,
              out_dir / f"steering_{variable.name}.png",
              row_labels=["generated", f"do({variable.name})"],
              col_labels=[None, [label for label, _ in states]])
