"""Versioned experiment runner for the JEPA smoke test.

Each configuration gets a `run_id` derived from its hyperparameters, and every
artifact produced from that run -- metrics, probe results, figures -- is filed
under that id. Nothing is overwritten anonymously, so a figure can always be
traced back to the run that produced it.

    runs/baselines.json          non-JEPA baselines (config-independent, cached)
    runs/<run_id>/metrics.json   history + probe results for one run
    runs/manifest.json           every run_id, newest last
    figures/<run_id>/*.png       figures for that run
    figures/sweep/*.png          cross-run comparisons

Usage:
    uv run python -m src.experiment            # run the full sweep
    uv run python -m src.experiment --baselines-only
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"


@dataclass(frozen=True)
class Config:
    """One training configuration. The run_id is derived from these fields, so
    two runs with different hyperparameters can never collide."""

    inv_coeff: float = 25.0
    var_coeff: float = 15.0
    cov_coeff: float = 1.0
    epochs: int = 30
    seed: int = 0
    label: str = ""

    @property
    def run_id(self) -> str:
        return (
            f"inv{self.inv_coeff:g}_var{self.var_coeff:g}_cov{self.cov_coeff:g}"
            f"_e{self.epochs}_s{self.seed}"
        )


# The sweep. The diagnosis from the first run was that the covariance term
# supplied 90.6% of the total loss reduction while the prediction term supplied
# 3.1%: at init the RAW terms are inv 0.96, var 0.19, cov 65.4, so cov_coeff=1
# still leaves covariance dominating the gradient by ~68x. This sweep walks
# cov_coeff down by decades, including a full ablation, to find out whether the
# prediction objective starts driving optimization once it is not drowned out.
SWEEP = [
    Config(cov_coeff=1.0, label="baseline (VICReg defaults)"),
    Config(cov_coeff=0.1, label="cov x0.1"),
    Config(cov_coeff=0.01, label="cov x0.01"),
    Config(cov_coeff=0.0, label="cov ablated"),
]


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def baselines(refresh: bool = False) -> dict:
    """Non-JEPA baselines. Independent of any training config, so computed once
    and cached -- the HMM forward pass over every test window is the slow part."""
    path = RUNS / "baselines.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    from .probes import run as probe_run

    rows = probe_run(include_jepa=False)
    _write(path, rows)
    return rows


def run_one(cfg: Config, refresh: bool = False) -> dict:
    """Train one configuration, probe it, and persist everything under its run_id."""
    path = RUNS / cfg.run_id / "metrics.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    from .HMM_data import X_test, X_train
    from .jepa import build_encoder, embed, train_jepa
    from .probes import probe

    # Null control: same architecture, never trained. Depends only on the seed,
    # but it is cheap and belongs with the run it is the control for.
    rnd = build_encoder(seed=cfg.seed)
    null = probe(embed(rnd, X_train), embed(rnd, X_test))

    enc, history = train_jepa(
        epochs=cfg.epochs,
        seed=cfg.seed,
        verbose=True,
        inv_coeff=cfg.inv_coeff,
        var_coeff=cfg.var_coeff,
        cov_coeff=cfg.cov_coeff,
    )
    trained = probe(embed(enc, X_train), embed(enc, X_test))

    record = {
        "run_id": cfg.run_id,
        "label": cfg.label,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "history": history,
        "probe": {"random_init_null": null, "trained": trained},
        # Raw prediction MSE, recovered by undoing the coefficient. This is the
        # number that says whether the predictor learned anything: the variance
        # term drives targets to unit std, so MSE 1.0 == predicting the mean.
        "final_pred_mse": (
            history[-1]["inv_loss"] / cfg.inv_coeff if cfg.inv_coeff else None
        ),
    }
    _write(path, record)

    manifest_path = RUNS / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    if cfg.run_id not in manifest:
        manifest.append(cfg.run_id)
    _write(manifest_path, manifest)
    return record


def main(argv: list[str]) -> None:
    refresh = "--refresh" in argv
    print("=== baselines ===")
    base = baselines(refresh=refresh)
    from .probes import print_table

    print_table(base)
    if "--baselines-only" in argv:
        return

    records = []
    for cfg in SWEEP:
        print(f"\n=== {cfg.label}  [{cfg.run_id}] ===")
        records.append(run_one(cfg, refresh=refresh))

    h0 = base["raw flattened window"]
    print(f"\n{'run':<34}{'pred MSE':>10}{'acc':>8}{'b-v-k':>8}{'vs null':>9}")
    print("-" * 69)
    print(f"{'H0 bar: raw flattened window':<34}{'':>10}{h0['acc']:>8.3f}{h0['bear_vs_kang']:>8.3f}")
    for r in records:
        t, n = r["probe"]["trained"], r["probe"]["random_init_null"]
        print(
            f"{r['label']:<34}{r['final_pred_mse']:>10.3f}"
            f"{t['acc']:>8.3f}{t['bear_vs_kang']:>8.3f}{t['acc'] - n['acc']:>+9.3f}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
