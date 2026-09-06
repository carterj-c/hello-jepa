"""Figure generation for the JEPA infrastructure smoke test.

Reads run metrics and baseline results written under ``runs/`` and renders
PNG figures under ``figures/`` for the project README. Never trains or
recomputes anything — pure read-and-plot.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
FIGURES_DIR = REPO_ROOT / "figures"

# Fixed-order categorical palette. Assign by slot, never cycle.
SLOT1_BLUE = "#2a78d6"
SLOT2_ORANGE = "#eb6834"
SLOT3_AQUA = "#1baf7a"
SLOT4_YELLOW = "#eda100"
SLOT5_MAGENTA = "#e87ba4"
SLOT6_GREEN = "#008300"
SLOT7_VIOLET = "#4a3aa7"
SLOT8_RED = "#e34948"

PALETTE = [
    SLOT1_BLUE,
    SLOT2_ORANGE,
    SLOT3_AQUA,
    SLOT4_YELLOW,
    SLOT5_MAGENTA,
    SLOT6_GREEN,
    SLOT7_VIOLET,
    SLOT8_RED,
]

INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID_COLOR = "#dddad4"


def _style_axes(ax) -> None:
    """Apply the shared recessive-grid, no-top/right-spine look."""
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_MUTED)


def _footer(fig, text: str) -> None:
    """Muted, small provenance line, bottom-left. Not a headline."""
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=7, color=INK_MUTED)


def _run_footer(run: dict) -> str:
    return f"{run['run_id']}  |  created {run['created_utc']}"


def _sweep_footer(runs: list[dict]) -> str:
    epochs = {r["config"]["epochs"] for r in runs}
    seeds = {r["config"]["seed"] for r in runs}
    epochs_s = str(next(iter(epochs))) if len(epochs) == 1 else "mixed"
    seeds_s = str(next(iter(seeds))) if len(seeds) == 1 else "mixed"
    return f"{len(runs)} runs  |  epochs={epochs_s}  seed={seeds_s}"


def _load_runs() -> list[dict]:
    manifest = json.loads((RUNS_DIR / "manifest.json").read_text())
    runs = []
    for run_id in manifest:
        metrics_path = RUNS_DIR / run_id / "metrics.json"
        runs.append(json.loads(metrics_path.read_text()))
    return runs


def _load_baselines() -> dict:
    return json.loads((RUNS_DIR / "baselines.json").read_text())


def plot_loss_components(run: dict, out_dir: Path) -> Path:
    history = run["history"]
    epochs = [h["epoch"] for h in history]
    inv = [h["inv_loss"] for h in history]
    var = [h["var_loss"] for h in history]
    cov = [h["cov_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    ax.stackplot(
        epochs,
        inv,
        var,
        cov,
        labels=["inv_loss", "var_loss", "cov_loss"],
        colors=[SLOT1_BLUE, SLOT3_AQUA, SLOT2_ORANGE],
        edgecolor=SURFACE,
        linewidth=0.5,
    )
    _style_axes(ax)
    ax.set_xlabel("epoch", color=INK_MUTED)
    ax.set_ylabel("loss", color=INK_MUTED)
    ax.set_title(
        "Covariance term dominates the loss, not prediction",
        color=INK_PRIMARY,
        fontsize=12,
        loc="left",
    )
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_MUTED)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _footer(fig, _run_footer(run))
    out_path = out_dir / "loss_components.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_prediction_mse(run: dict, out_dir: Path) -> Path:
    history = run["history"]
    inv_coeff = run["config"]["inv_coeff"]
    epochs = [h["epoch"] for h in history]
    if inv_coeff:
        mse = [h["inv_loss"] / inv_coeff for h in history]
    else:
        mse = [float("nan") for _ in history]

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    ax.plot(epochs, mse, color=SLOT1_BLUE, linewidth=2)
    ax.axhline(
        1.0,
        color=INK_MUTED,
        linewidth=1.2,
        linestyle="--",
    )
    ax.annotate(
        "predicting the mean",
        xy=(epochs[-1] if epochs else 0, 1.0),
        xytext=(0, 6),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK_MUTED,
    )
    _style_axes(ax)
    ax.set_xlabel("epoch", color=INK_MUTED)
    ax.set_ylabel("prediction MSE", color=INK_MUTED)
    ax.set_title(
        "Did the predictor beat a constant guess?",
        color=INK_PRIMARY,
        fontsize=12,
        loc="left",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _footer(fig, _run_footer(run))
    out_path = out_dir / "prediction_mse.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_collapse(run: dict, out_dir: Path) -> Path:
    history = run["history"]
    epochs = [h["epoch"] for h in history]
    emb_std = [h["emb_std"] for h in history]
    eff_rank = [h["eff_rank"] for h in history]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 6), facecolor=SURFACE, sharex=True
    )
    ax1.plot(epochs, emb_std, color=SLOT1_BLUE, linewidth=2)
    _style_axes(ax1)
    ax1.set_ylabel("emb_std", color=INK_MUTED)
    ax1.set_title(
        "Are the embeddings collapsing to a point?",
        color=INK_PRIMARY,
        fontsize=12,
        loc="left",
    )

    ax2.plot(epochs, eff_rank, color=SLOT3_AQUA, linewidth=2)
    ax2.axhline(128, color=INK_MUTED, linewidth=1.2, linestyle="--")
    ax2.annotate(
        "max possible = 128 (embedding dim)",
        xy=(epochs[-1] if epochs else 0, 128),
        xytext=(0, -12),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK_MUTED,
    )
    _style_axes(ax2)
    ax2.set_xlabel("epoch", color=INK_MUTED)
    ax2.set_ylabel("eff_rank", color=INK_MUTED)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _footer(fig, _run_footer(run))
    out_path = out_dir / "collapse.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_sweep_prediction_mse(runs: list[dict], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    for i, run in enumerate(runs):
        history = run["history"]
        inv_coeff = run["config"]["inv_coeff"]
        epochs = [h["epoch"] for h in history]
        if inv_coeff:
            mse = [h["inv_loss"] / inv_coeff for h in history]
        else:
            mse = [float("nan") for _ in history]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(epochs, mse, color=color, linewidth=2, label=run["label"])

    ax.axhline(1.0, color=INK_MUTED, linewidth=1.2, linestyle="--")
    ax.annotate(
        "predicting the mean",
        xy=(1.0, 1.0),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK_MUTED,
    )
    _style_axes(ax)
    ax.set_xlabel("epoch", color=INK_MUTED)
    ax.set_ylabel("prediction MSE", color=INK_MUTED)
    ax.set_title(
        "Did shrinking cov_coeff make the predictor learn?",
        color=INK_PRIMARY,
        fontsize=12,
        loc="left",
    )
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(len(runs), 4),
        frameon=False,
        fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color(INK_MUTED)

    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _footer(fig, _sweep_footer(runs))
    out_path = out_dir / "sweep_prediction_mse.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


X_MIN, X_MAX = 0.45, 0.85


def plot_results(runs: list[dict], baselines: dict, out_dir: Path) -> Path:
    from matplotlib.lines import Line2D

    chance_acc = baselines["majority class (chance)"]["acc"]
    chance_bvk = baselines["majority class (chance)"]["bear_vs_kang"]
    h0_bar = baselines["raw flattened window"]["acc"]

    # rows: (label, acc, bvk, category) with category in {"baseline", "trained", "null"}
    rows = []
    for name, vals in baselines.items():
        rows.append((name, vals["acc"], vals["bear_vs_kang"], "baseline"))
    for run in runs:
        label = run["label"]
        trained = run["probe"]["trained"]
        null = run["probe"]["random_init_null"]
        rows.append((f"{label} (trained)", trained["acc"], trained["bear_vs_kang"], "trained"))
        rows.append((f"{label} (random init)", null["acc"], null["bear_vs_kang"], "null"))

    category_style = {
        # (marker facecolor, marker edgecolor)
        "baseline": (SLOT1_BLUE, SLOT1_BLUE),
        "trained": (SLOT2_ORANGE, SLOT2_ORANGE),
        "null": ("none", SLOT2_ORANGE),
    }

    fig, (ax_acc, ax_bvk) = plt.subplots(
        1, 2, figsize=(13, max(4, 0.4 * len(rows) + 1.5)), facecolor=SURFACE
    )

    for ax, metric_idx, ref_lines, title in (
        (
            ax_acc,
            1,
            [(chance_acc, "chance"), (h0_bar, "H0 bar")],
            "No trained run beats the raw-window baseline",
        ),
        (
            ax_bvk,
            2,
            [(chance_bvk, "chance")],
            "Trained runs separate bear/kangaroo worse than their own null",
        ),
    ):
        sorted_rows = sorted(rows, key=lambda r: r[metric_idx])
        labels = [r[0] for r in sorted_rows]
        values = [r[metric_idx] for r in sorted_rows]
        cats = [r[3] for r in sorted_rows]
        y_pos = list(range(len(labels)))

        # thin connecting stems from the left edge of the visible range to each mark
        ax.hlines(y_pos, X_MIN, values, color=GRID_COLOR, linewidth=1.2, zorder=1)

        for y, val, cat in zip(y_pos, values, cats):
            face, edge = category_style[cat]
            ax.scatter(
                [val],
                [y],
                s=90,
                facecolors=face,
                edgecolors=edge,
                linewidths=2,
                zorder=3,
            )
            label_ha = "left" if val < X_MAX - 0.05 else "right"
            offset = 6 if label_ha == "left" else -6
            ax.annotate(
                f"{val:.3f}",
                xy=(val, y),
                xytext=(offset, 0),
                textcoords="offset points",
                ha=label_ha,
                va="center",
                fontsize=7.5,
                color=INK_PRIMARY,
                zorder=4,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(-0.7, len(labels) - 0.3)
        for line_val, line_label in ref_lines:
            ax.axvline(line_val, color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=2)
            ax.annotate(
                line_label,
                xy=(line_val, len(labels) - 1),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color=INK_MUTED,
                ha="left",
            )
        _style_axes(ax)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left")

    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=9,
            markerfacecolor=SLOT1_BLUE, markeredgecolor=SLOT1_BLUE,
            label="baseline",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=9,
            markerfacecolor=SLOT2_ORANGE, markeredgecolor=SLOT2_ORANGE,
            label="JEPA, trained",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=9,
            markerfacecolor="none", markeredgecolor=SLOT2_ORANGE, markeredgewidth=2,
            label="JEPA, random-init null",
        ),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.0),
    )
    for text in legend.get_texts():
        text.set_color(INK_MUTED)

    fig.suptitle(
        "JEPA probe accuracy vs simple baselines",
        color=INK_PRIMARY,
        fontsize=13,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    _footer(fig, _sweep_footer(runs))
    out_path = out_dir / "results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def make_all() -> list[Path]:
    runs = _load_runs()
    baselines = _load_baselines()
    written: list[Path] = []

    for run in runs:
        run_dir = FIGURES_DIR / run["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        written.append(plot_loss_components(run, run_dir))
        written.append(plot_prediction_mse(run, run_dir))
        written.append(plot_collapse(run, run_dir))

    sweep_dir = FIGURES_DIR / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    written.append(plot_sweep_prediction_mse(runs, sweep_dir))
    written.append(plot_results(runs, baselines, sweep_dir))

    return written


if __name__ == "__main__":
    paths = make_all()
    for p in paths:
        print(p)
