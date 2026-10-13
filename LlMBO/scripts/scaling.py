"""Figures fig_scaling_{all,qwen}: greedy and avg@4 accuracy against trainable parameters under GRPO: `uv run --no-project --with matplotlib --with numpy overleaf/scripts/scaling.py` from the repo root."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    METRIC_LINESTYLES,
    BASE_STYLE,
    BENCHMARKS,
    COLORS,
    SCALING_BLOCKS,
    MARKERS,
    OUT,
    PERCENT,
    PRIMARY,
    TEXT_WIDTH,
    base_value,
    bench_label,
    bottom_legend,
    handle,
    label_of,
    load_data,
    mean_se,
    metric_color,
    metric_handle,
    save,
    value,
)
from scatter import label_rows, mark_ood, split_subplots


SCALING_METHODS = ["lora", "loraxs", "grpo"]  # every adapter under GRPO; LlMBO is compared in its own figures


def scaling_panel(ax: plt.Axes, baselines: dict, cells: dict, model: str, bench: str) -> None:
    """One model's greedy and avg@4 accuracy (saturation and line style) against trainable parameters under GRPO: LoRA, LoRA-XS and TinyLoRA."""
    for k, m in ((k, m) for k in METRIC_LINESTYLES for m in SCALING_METHODS):
        stats = sorted(
            (
                np.median([r["run"]["params"] for r in records]),
                *mean_se([value(r, k, bench) for r in records]),
            )
            for records in cells[model][m].values()
        )
        if not stats:
            continue
        x, mean, se = map(np.array, zip(*stats))
        ax.errorbar(
            x,
            mean,
            yerr=se,
            color=metric_color(COLORS[m], k),
            ls=METRIC_LINESTYLES[k],
            marker=MARKERS[m],
            mew=0.7,
            ms=3,
            capsize=1,
            elinewidth=0.6,
            lw=1,
        )
    for k, ls in METRIC_LINESTYLES.items():
        ax.axhline(base_value(baselines, model, k, bench), **{**BASE_STYLE, "ls": ls, "lw": 0.7})
    ax.set_xscale("log", base=2)
    ax.set_xticks([2**e for e in (0, 8, 16, 24)], [f"$2^{{{e}}}$" for e in (0, 8, 16, 24)])
    ax.minorticks_off()
    ax.tick_params(labelsize=6, pad=1)
    ax.yaxis.set_major_formatter(PERCENT)


def scaling_handles() -> list:
    base_model = plt.Line2D([], [], **{**BASE_STYLE, "ls": "-", "lw": 0.7}, label="base model")
    return [handle(m) for m in SCALING_METHODS] + [base_model] + [metric_handle(k) for k in METRIC_LINESTYLES]


def scaling_figure(
    baselines: dict, cells: dict, groups: list[list[str]], benches: list[str], path: Path, row_height: float = 0.62
) -> None:
    """One row per model, groups of models set apart, one column per benchmark (in-distribution first), each panel on its own accuracy axis."""
    width, left, right, top, bottom, row_gap = TEXT_WIDTH, 0.7, 0.1, 0.55, 0.62, 0.25
    models = [model for group in groups for model in group]
    height = top + bottom + row_height * (len(models) + row_gap * (len(groups) - 1))
    fig, axes = split_subplots(
        len(models), len(PRIMARY), len(benches) - len(PRIMARY), (width, height),
        row_groups=[len(group) for group in groups], row_gap=row_gap, sharex=True,
    )
    for row, model in zip(axes, models):
        for ax, bench in zip(row, benches):
            scaling_panel(ax, baselines, cells, model, bench)
    label_rows(fig, axes, models, benches, "accuracy", "parameters")
    for row, model in zip(axes, models):  # short rows: model name over two lines
        row[0].set_ylabel("-\n".join(label_of(model).split("-", 1)) + "\naccuracy", labelpad=2, fontsize=6.5, linespacing=1.0)
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(
        left=left / width, right=1 - right / width, top=1 - top / height, bottom=bottom / height, hspace=0.35, wspace=0.3
    )
    mark_ood(fig, axes, len(PRIMARY))
    handles = scaling_handles()
    bottom_legend(fig, handles, columnspacing=1.0)
    save(fig, path)


def scaling_grid_figure(baselines: dict, cells: dict, model: str, path: Path) -> None:
    """One model's scaling on a 2x3 grid of wide panels: in-distribution benchmarks in the first column, out-of-distribution in the other two."""
    width, left, right, top, bottom, panel_height, row_gap = TEXT_WIDTH, 0.5, 0.1, 0.5, 0.62, 0.75, 0.3
    height = top + bottom + 2 * panel_height + row_gap
    fig, axes = split_subplots(2, 1, 2, (width, height), sharex=True)
    panels = [*axes[:, 0], *axes[:, 1:].flat]  # in-distribution down the first column, then out-of-distribution row by row
    for ax, bench in zip(panels, BENCHMARKS):
        scaling_panel(ax, baselines, cells, model, bench)
        ax.set_title(bench_label(bench), pad=3)
    for ax in axes[:, :2].flat:  # left column of each block
        ax.set_ylabel("accuracy", labelpad=2, fontsize=6.5)
    for ax in axes[-1]:
        ax.set_xlabel("parameters", labelpad=1)
    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    fig.subplots_adjust(
        left=left / width, right=1 - right / width, top=1 - top / height, bottom=bottom / height,
        hspace=row_gap / panel_height, wspace=0.18,
    )
    mark_ood(fig, axes, 1, show_divider=False)
    model_entry = plt.Line2D([], [], ls="", label=label_of(model))  # no marker: the logo stands in for it
    bottom_legend(fig, [model_entry, *scaling_handles()], columnspacing=1.0)
    save(fig, path)


def main(baselines: dict, cells: dict) -> None:
    scaling_figure(baselines, cells, SCALING_BLOCKS, BENCHMARKS, OUT / "fig_scaling_all.pdf", row_height=0.8)
    # main-text excerpt: Qwen2.5-7B
    scaling_grid_figure(baselines, cells, "qwen2.5-7b", OUT / "fig_scaling_qwen.pdf")


if __name__ == "__main__":
    main(*load_data())
