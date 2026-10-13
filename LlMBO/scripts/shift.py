"""Figures fig_shift (gain against u, both metrics) and fig_shift_fit (training-set vs MATH-500 accuracy), SimpleRL tiers vs MATH-only training: `uv run --no-project --with matplotlib --with numpy overleaf/scripts/shift.py` from the repo root."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    BASE_STYLE,
    BENCHMARKS,
    COLORS,
    LOW_U,
    MARKERS,
    METRIC_LINESTYLES,
    METRIC_OBJECTIVE,
    OUT,
    PERCENT,
    TEXT_WIDTH,
    SHIFT_MODELS,
    dodge,
    finish,
    gains,
    grid,
    handle,
    label_grid,
    label_of,
    load_data,
    log2_axis,
    mean_se,
    metric_color,
    metric_handle,
    value,
)


TRAIN_FILLED = {"": True, "-math": False}  # training set tag -> filled marker
TRAIN_STYLES = {"": "-", "-math": "--"}  # training set tag -> line style, in the fit figure
METRIC_NUDGE = {4: -0.02, 1: 0.02}  # x nudge in octaves per metric, on top of the method nudge


def shift_figure(
    baselines: dict,
    cells: dict,
    models: list[str],
    benches: list[str],
    path: Path,
) -> None:
    """Gain over base against u for GRPO and LlMBO trained on each metric (line style), training on the SimpleRL tiers (filled) vs MATH only (open); one row per model, one column per benchmark."""
    fig, axes = grid(models, benches, 1.3, sharex=True)
    for row, bench in zip(axes, benches):
        for ax, model in zip(row, models):
            for (k, bo), (tag, filled) in ((kb, tf) for kb in METRIC_OBJECTIVE.items() for tf in TRAIN_FILLED.items()):
                for m, shift in (("grpo", -0.05), (bo, 0.05)):
                    stats = [
                        (u, *mean_se(g))
                        for u in LOW_U
                        if (g := gains(baselines, cells, model, m + tag, k, [u], bench))
                    ]
                    if not stats:
                        continue
                    u, mean, se = map(np.array, zip(*stats))
                    ax.errorbar(
                        dodge(u, shift + METRIC_NUDGE[k]),
                        mean,
                        yerr=se,
                        color=metric_color(COLORS[m], k),
                        marker=MARKERS[m],
                        mfc=metric_color(COLORS[m], k) if filled else "white",
                        ls=METRIC_LINESTYLES[k],
                        ms=3,
                        capsize=1.2,
                        elinewidth=0.7,
                        lw=1,
                    )
            ax.axhline(0, **BASE_STYLE)
            log2_axis(ax, LOW_U)
            ax.tick_params(labelsize=6, pad=1)
            ax.yaxis.set_major_formatter(PERCENT)
    axes[0][0].set_xlim(dodge(1, -0.3), dodge(4, 0.3))
    label_grid(axes, models, benches, "gain", "$u$")
    handles = [
        handle("grpo"),
        handle("bo"),
        *(metric_handle(k) for k in METRIC_OBJECTIVE),
        plt.Line2D([], [], color="0.3", ls="", marker="o", ms=3, label="SimpleRL tiers"),
        plt.Line2D([], [], color="0.3", ls="", marker="o", mfc="white", ms=3, label="MATH only"),
    ]
    finish(fig, handles, path)


def fit_figure(baselines: dict, cells: dict, models: list[str], path: Path) -> None:
    """avg@4 (T=1) of the final adapter on each training set and on MATH-500, same seeds in every column; GRPO and LlMBO_T=1, tiers (solid) vs MATH only (dashed)."""
    k = 4
    columns = {"": 0, "-math": 1}  # training set tag -> column of its own training-set accuracy; MATH-500 is column 2
    fig, axes = plt.subplots(len(models), 3, figsize=(0.75 * TEXT_WIDTH, 0.5 + 1.3 * len(models)), sharex=True, squeeze=False)
    for row, model in zip(axes, models):
        for tag, ls in TRAIN_STYLES.items():
            task = None
            for m, shift in (("grpo", -0.05), ("bo", 0.05)):
                train, test = [], []
                for u in LOW_U:
                    records = [r for r in cells[model][m + tag].get(u, []) if r["train"].get(k) is not None]
                    if not records:
                        continue
                    task = records[0]["run"]["task"]
                    train.append((u, *mean_se([value(r, k, train=True) for r in records])))
                    test.append((u, *mean_se([value(r, k, "math500") for r in records])))
                for ax, stats in ((row[columns[tag]], train), (row[2], test)):
                    if not stats:
                        continue
                    u, mean, se = map(np.array, zip(*stats))
                    ax.errorbar(dodge(u, shift), mean, yerr=se, color=COLORS[m], marker=MARKERS[m], mfc=COLORS[m] if not tag else "white", ls=ls, ms=3, capsize=1.2, elinewidth=0.7, lw=1)
            # base model on this training set
            if task and (base := baselines[model]["train"][k].get(task)) is not None:
                row[columns[tag]].axhline(base, **BASE_STYLE)
        row[2].axhline(baselines[model]["bench"][k]["math500"], **BASE_STYLE)
        row[0].set_ylabel(f"{label_of(model)}\navg@4 acc.")
        for ax in row:
            log2_axis(ax, LOW_U)
            ax.tick_params(labelsize=6, pad=1)
            ax.yaxis.set_major_formatter(PERCENT)
    axes[0][0].set_xlim(dodge(1, -0.3), dodge(4, 0.3))
    for ax, title in zip(axes[0], ("Tier train set", "MATH train set", "MATH-500")):
        ax.set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("$u$")
    handles = [
        handle("grpo", "GRPO"),
        handle("bo", "LlMBO"),
        plt.Line2D([], [], color="0.3", ls="-", label="trained on tiers"),
        plt.Line2D([], [], color="0.3", ls="--", mfc="white", marker="o", ms=3, label="trained on MATH only"),
    ]
    finish(fig, handles, path)


def fit_coverage(cells: dict, models: list[str]) -> None:
    """Print how many seeds per cell have a avg@4 training-set eval, out of those trained."""
    for model in models:
        for m in ("grpo", "bo"):
            for tag in TRAIN_STYLES:
                runs = [cells[model][m + tag].get(u, []) for u in LOW_U]
                print(f"train eval {model:18s} {m + tag:9s}", " ".join(f"u{u}:{sum(r['train'].get(4) is not None for r in rs)}/{len(rs)}" for u, rs in zip(LOW_U, runs)))


def main(baselines: dict, cells: dict) -> None:
    shift_figure(baselines, cells, SHIFT_MODELS, BENCHMARKS, OUT / "fig_shift.pdf")
    fit_figure(baselines, cells, SHIFT_MODELS, OUT / "fig_shift_fit.pdf")
    fit_coverage(cells, SHIFT_MODELS)


if __name__ == "__main__":
    main(*load_data())
