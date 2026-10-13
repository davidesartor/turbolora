"""Figures fig_gap_<group>: paired gap LlMBO - GRPO against u, both metrics in each panel: `uv run --no-project --with matplotlib --with numpy overleaf/scripts/gap.py` from the repo root."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    BATCH_STYLES,
    BENCHMARKS,
    BO_MAX_U,
    COLORS,
    GROUPS,
    LABELS,
    MARKERS,
    METRIC_OBJECTIVE,
    METRICS,
    OUT,
    PERCENT,
    PRIMARY,
    TEXT_WIDTH,
    U_ALL,
    bottom_legend,
    dodge,
    load_data,
    log2_axis,
    mean_se,
    metric_color,
    paired,
    save,
)
from scatter import label_rows, mark_ood, split_subplots

METRIC_NUDGE = {4: -0.04, 1: 0.04}  # x nudge in octaves per metric, on top of the batch nudge


def gap_figure(cells: dict, blocks: list[list[str]], benches: list[str], path: Path) -> None:
    """Paired gap LlMBO − GRPO on each metric (saturation) against u ≤ BO_MAX_U, LlMBO trained on that metric, both batch sizes; one row per model, blocks of models set apart, one column per benchmark (in-distribution first), each panel on its own gap axis."""
    width, left, right, top, bottom, row_height, row_gap = TEXT_WIDTH, 0.72, 0.1, 0.55, 0.62, 0.7, 0.25
    models = [model for block in blocks for model in block]
    height = top + bottom + row_height * (len(models) + row_gap * (len(blocks) - 1))
    fig, axes = split_subplots(
        len(models), len(PRIMARY), len(benches) - len(PRIMARY), (width, height),
        row_groups=[len(block) for block in blocks], row_gap=row_gap, sharex=True,
    )
    max_u = 1
    for row, model in zip(axes, models):
        for ax, g in zip(row, benches):
            ax.axhline(0, color=COLORS["grpo"], lw=1, ls=(0, (4, 2)))
            for (k, bo), (batch, (suffix, ls, shift)) in ((kb, bs) for kb in METRIC_OBJECTIVE.items() for bs in BATCH_STYLES.items()):
                stats = [
                    (u, *mean_se(d))
                    for u in U_ALL
                    if u <= BO_MAX_U
                    if (d := paired(cells, bo + suffix, "grpo", k, [u], [model], g))
                ]
                if not stats:
                    continue
                u, mean, se = map(np.array, zip(*stats))
                max_u = max(max_u, u.max())
                ax.errorbar(
                    dodge(u, shift + METRIC_NUDGE[k]),
                    mean,
                    yerr=se,
                    color=metric_color(COLORS[bo], k),
                    marker=MARKERS[bo],
                    mfc=metric_color(COLORS[bo], k) if batch == 1 else "white",
                    ls=ls,
                    ms=3,
                    capsize=1,
                    elinewidth=0.6,
                    lw=0.9,
                )
            log2_axis(ax, [u for u in (1, 2, 4, 8, 16) if u <= max_u])
            ax.tick_params(labelsize=5.5, pad=1)
            ax.yaxis.set_major_formatter(PERCENT)
    axes[0][0].set_xlim(dodge(1, -0.25), dodge(max_u, 0.25))
    label_rows(fig, axes, models, benches, "LlMBO $-$ GRPO", "$u$")
    fig.subplots_adjust(
        left=left / width, right=1 - right / width, top=1 - top / height, bottom=bottom / height, hspace=0.35, wspace=0.3
    )
    mark_ood(fig, axes, len(PRIMARY))
    handles = [
        plt.Line2D([], [], color=metric_color(COLORS[bo], k), marker=MARKERS[bo], ls="", ms=3, label=f"{LABELS[bo].split(', ')[1]}, {METRICS[k].split(',')[0]}")
        for k, bo in METRIC_OBJECTIVE.items()
    ] + [
        plt.Line2D([], [], color="0.3", marker="o", ls=ls, ms=3, mfc="0.3" if batch == 1 else "white", label=f"$b{{=}}{batch}$")
        for batch, (_, ls, _) in BATCH_STYLES.items()
    ] + [plt.Line2D([], [], color=COLORS["grpo"], lw=1, ls=(0, (4, 2)), label="GRPO")]
    bottom_legend(fig, handles, columnspacing=1.0)
    save(fig, path)


def main(baselines: dict, cells: dict) -> None:
    for group, blocks in GROUPS.items():
        gap_figure(cells, blocks, BENCHMARKS, OUT / f"fig_gap_{group}.pdf")


if __name__ == "__main__":
    main(*load_data())
