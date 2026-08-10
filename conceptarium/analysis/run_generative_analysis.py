"""Quantitative and qualitative analysis of trained generative concept models.

Post-hoc counterpart to ``run_experiment.py`` for the generative branch. It finds
every finished run matching ``filters``, rebuilds each from its saved config and
checkpoint, and writes one results tree:

    results/
      results.csv                                  one row per run
      results_by_model.csv                         mean/std over seeds
      qualitative/<dataset>/<model>/<run>/
          config.yaml                              the run's own config
          overview.png                             original / reconstruction / generation
          steering_<concept>.png                   one sample at every state of a concept

The metrics implement Ismail et al., "Concept Bottleneck Generative Models"
(ICLR 2024): **steerability** (Table 1) — does intervening on a concept put that
concept into the output, judged by a classifier trained on real data — and
**FID** (Table 2). Both live in :data:`~conceptarium.generative_utils.METRICS`;
adding another is one class and one registry entry, and needs no change here.

Everything modality-specific sits behind
:class:`~conceptarium.generative_utils.Modality`, so a generative model over
discrete data needs a subclass there and nothing in this file.

Runs are discovered by scanning ``search_dirs`` for directories holding both
``.hydra/config.yaml`` and a checkpoint. Notably *not* via ``runs.csv``: the
registry records absolute paths from the machine that trained the run and is
skipped entirely under ``debug: true``, so results copied in from a GPU box are
invisible to it. A run whose checkpoint will not load — an architecture change
since it was trained, usually — is reported and skipped rather than aborting the
rest.

Two engines are used, for different jobs: the model's own
:class:`~torch_concepts.nn.VariationalInference` is the only one that consults
the guide, so it does the encoding; an ancestral engine does every decode. They
share the same PGM by reference, so no weights are copied.

On speed: the guide runs the image backbone live and torchvision preprocessing
resizes to 224x224, so the test-set passes dominate. ``accelerator`` picks the
device; ``max_eval_batches`` caps those passes, and ``n_fid_samples`` /
``n_steering_samples`` the generative ones.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Running this file as a script puts `analysis/` on sys.path, not `conceptarium/`,
# so `conceptarium.*`, `env` and `analysis.*` would all be unreachable. Adding the
# parent makes every invocation work the same way:
#     python conceptarium/analysis/run_generative_analysis.py   (from the repo root)
#     python analysis/run_generative_analysis.py                (from conceptarium/)
#     python -m analysis.run_generative_analysis                (from conceptarium/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra  # noqa: E402
import torch  # noqa: E402
from hydra.utils import get_original_cwd  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from conceptarium.resolvers import register_custom_resolvers  # noqa: E402
from conceptarium.utils import seed_everything  # noqa: E402
from analysis.generative_utils import (  # noqa: E402
    METRICS,
    EvalContext,
    figure_overview,
    figure_steering,
    load_job,
    modality_for,
    qualitative_dir,
    rebuild,
    resolve_device,
    resolve_job_dirs,
    run_identity,
    write_results,
)
from torch_concepts.nn import AncestralSamplingInference  # noqa: E402

logger = logging.getLogger(__name__)


def build_context(cfg: DictConfig, job_dir: Path, device: torch.device) -> Tuple[EvalContext, DictConfig]:
    """Load a run and wrap it in the context the metrics and figures share."""
    job_cfg, ckpt_path = load_job(job_dir)
    if ckpt_path is None:
        # A plain exception, not SystemExit: the caller analyses a *list* of runs
        # and catches Exception to skip broken ones, while SystemExit is a
        # BaseException that would sail past that and kill the whole sweep.
        raise FileNotFoundError(f"No checkpoint under {job_dir / 'checkpoints'}.")

    datamodule, model = rebuild(job_cfg)
    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    )
    model.eval()
    # Moves the backbone with it: `Backbone` is an nn.Module holding its
    # torchvision model as a submodule and reads `.device` off its parameters.
    model.to(device)

    # Decoding engine: ancestral sampling resolves every root through its own
    # prior, so an unconditioned query really does draw z ~ p(z).
    #
    # Soft vs hard discrete draws needs no setting: it follows from the concepts'
    # declared family, which the rebuilt PGM already carries from training.
    # `p_int` is inert here -- generation supplies no ground truth in the query,
    # and interventions are replayed as *evidence*.
    #
    # The temperature does have to be carried over: the relaxation is annealed
    # during training, so decoding at the default 1.0 would sample far softer
    # codes than the trained decoder ever saw. The checkpoint restores the
    # training engine's temperature buffer, so this reads what training ended on.
    temperature = float(model.train_inference.temperature)
    engine = AncestralSamplingInference(
        model.pgm, p_int=1.0, initial_temperature=temperature, annealing="constant"
    )
    logger.info("decoding %s at temperature %.4f", job_dir.name, temperature)

    ctx = EvalContext(
        model=model, engine=engine, datamodule=datamodule,
        modality=modality_for(datamodule), device=device, cfg=cfg,
    )
    return ctx, job_cfg


def analyse_job(cfg: DictConfig, job_dir: Path, device: torch.device,
                out_root: Path) -> Dict[str, object]:
    """Metrics and figures for one trained model. Returns its results row."""
    seed_everything(cfg.get("seed", 42))
    logger.info("analysing %s", job_dir)

    ctx, job_cfg = build_context(cfg, job_dir, device)
    identity = run_identity(job_cfg, job_dir)

    # --- qualitative: config beside the figures, so a picture is always
    # traceable to the run that produced it ---
    out_dir = qualitative_dir(out_root, identity)
    OmegaConf.save(job_cfg, out_dir / "config.yaml")
    figure_overview(ctx, out_dir, int(cfg.get("n_samples", 10)))
    wanted = cfg.get("steer_concepts")
    for variable in ctx.concepts:
        if not wanted or variable.name in set(wanted):
            figure_steering(ctx, variable, out_dir)

    # --- quantitative ---
    row: Dict[str, object] = dict(identity)
    for name in cfg.get("compute_metrics") or list(METRICS):
        if name not in METRICS:
            raise SystemExit(
                f"Unknown metric {name!r}. Available: {sorted(METRICS)}."
            )
        row.update(METRICS[name].compute(ctx))
    return row


@hydra.main(config_path="../conf", config_name="generative_analysis", version_base="1.3")
def main(cfg: DictConfig) -> None:
    job_dirs = resolve_job_dirs(cfg)
    device = resolve_device(cfg.get("accelerator"))
    try:
        base = Path(get_original_cwd())
    except ValueError:
        base = Path.cwd()
    results_dir = Path(cfg.get("results_dir", "results"))
    out_root = results_dir if results_dir.is_absolute() else base / results_dir
    logger.info("analysing %d run(s) on %s -> %s", len(job_dirs), device, out_root)

    rows: List[Dict[str, object]] = []
    failures: List[Tuple[Path, Exception]] = []
    for job_dir in job_dirs:
        print(f"\n=== {job_dir} ===")
        try:
            row = analyse_job(cfg, job_dir, device, out_root)
        except Exception as error:
            # One unreadable run must not cost the rest of a sweep its results.
            # The common cause is a checkpoint predating an architecture change,
            # which surfaces here as a state_dict shape mismatch.
            logger.exception("skipping %s: %s", job_dir, error)
            failures.append((job_dir, error))
            continue
        rows.append(row)
        for key, value in row.items():
            print(f"{key:>28}: {value}")

    if rows:
        per_run, aggregated = write_results(rows, out_root)
        print(f"\nresults:    {per_run}\naveraged:   {aggregated}"
              f"\nqualitative: {out_root / 'qualitative'}")

    if failures:
        print(f"\n{len(failures)} of {len(job_dirs)} run(s) failed:")
        for job_dir, error in failures:
            print(f"  {job_dir}: {type(error).__name__}: {error}")
        # Nothing at all came out: fail the process rather than exiting 0 on an
        # empty result, which a caller would read as success.
        if len(failures) == len(job_dirs):
            raise SystemExit(1)


if __name__ == "__main__":
    register_custom_resolvers()
    main()
