"""Figures fig_scatter_<group>: LlMBO gain against GRPO gain per model and benchmark, both metrics in each panel: `uv run --no-project --with matplotlib --with numpy overleaf/scripts/scatter.py` from the repo root."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from headline import FAMILIES, legend_rows
from common import (
    BASE_STYLE,
    BASES,
    BENCHMARKS,
    COLORS,
    GROUPS,
    LOW_U,
    METRIC_OBJECTIVE,
    OUT,
    PERCENT,
    PRIMARY,
    TEXT_WIDTH,
    bench_label,
    bottom_legend,
    gains,
    label_of,
    load_data,
    mean_se,
    metric_color,
    metric_handle,
    save,
)

U_MARKERS = {1: "o", 2: "^", 4: "s"}  # as in fig_headline
GREEDY = 1  # filled marker; avg@4 open, as in fig_headline


def scatter_panel(ax: plt.Axes, baselines: dict, cells: dict, model: str, bench: str, color: str | None = None) -> list[float]:
    """Gain over base of LlMBO trained on each metric against GRPO's on that metric (saturation; greedy filled, avg@4 open), one point per u <= 4 (marker).

    A given color (fig_headline style) replaces the per-metric saturation. Returns the error-bar extents, so the caller can set the panel's limits."""
    points = []
    # avg@4 drawn last, its open markers over the filled greedy ones
    for (k, bo), u in ((kb, u) for kb in sorted(METRIC_OBJECTIVE.items()) for u in LOW_U):
        grpo, llmbo = (gains(baselines, cells, model, method, k, [u], bench) for method in ("grpo", bo))
        if not (grpo and llmbo):
            continue
        (x, x_se), (y, y_se) = mean_se(grpo), mean_se(llmbo)
        point_color = color or metric_color(COLORS[bo], k)
        ax.errorbar(
            x,
            y,
            xerr=x_se,
            yerr=y_se,
            color=point_color,
            marker=U_MARKERS[u],
            ms=4,
            mfc=point_color if k == GREEDY else "white",
            mec="white" if k == GREEDY else point_color,
            mew=0.4 if k == GREEDY else 0.7,
            elinewidth=0.5,
            capsize=0,
            alpha=0.85 if color else 1,
            zorder=3,
        )
        points += [x - x_se, x + x_se, y - y_se, y + y_se]
    return points


def finish_panel(ax: plt.Axes, lo: float, hi: float, formatter=PERCENT) -> None:
    """Square limits [lo, hi] on both axes, the LlMBO-better half shaded, diagonal and zero lines."""
    ax.fill_between([lo, hi], [lo, hi], hi, color="0.5", alpha=0.07, lw=0)
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.8)
    ax.axhline(0, **BASE_STYLE)
    ax.axvline(0, **BASE_STYLE)
    ax.set(xlim=(lo, hi), ylim=(lo, hi))
    ax.set_box_aspect(1)
    ax.locator_params(nbins=4)
    ax.tick_params(labelsize=6, pad=1)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)


def split_subplots(
    n_rows: int, n_id: int, n_ood: int, figsize: tuple, row_groups: list[int] = (), row_gap: float = 0.2, **share
) -> tuple:
    """Subplot grid with an empty spacer column between the in- and out-of-distribution blocks, so the divider clears the tick labels.

    row_groups (sizes summing to n_rows) adds a spacer row of relative height row_gap between consecutive groups of rows."""
    spacer_rows = list(np.cumsum([size + 1 for size in row_groups])[:-1] - 1)
    heights = [row_gap if i in spacer_rows else 1 for i in range(n_rows + len(spacer_rows))]
    fig, axes = plt.subplots(
        len(heights), n_id + 1 + n_ood, figsize=figsize, squeeze=False,
        gridspec_kw={"width_ratios": [1] * n_id + [0.05] + [1] * n_ood, "height_ratios": heights}, **share,
    )
    for ax in {*axes[:, n_id], *axes[spacer_rows].flat}:
        ax.remove()
    return fig, np.delete(np.delete(axes, n_id, axis=1), spacer_rows, axis=0)


def mark_ood(fig, axes, n_id: int, show_divider: bool = True) -> None:
    """Divider right of the in-distribution columns and a bracketed group label above each block of columns."""
    position = lambda column: axes[0][column].get_position()
    y = position(0).y1 + 0.24 / fig.get_figheight()
    divider = position(n_id - 1).x1 + 0.1 / fig.get_figwidth()  # close to the ID block, clear of the OOD tick labels
    if show_divider:
        fig.add_artist(plt.Line2D([divider] * 2, [axes[-1][0].get_position().y0, position(0).y1], color="0.35", lw=0.6, ls=(0, (2, 2))))
    for label, left, right in (
        ("in-distribution", position(0).x0, position(n_id - 1).x1),
        ("out-of-distribution", position(n_id).x0, position(-1).x1),
    ):
        fig.add_artist(plt.Line2D([left, right], [y] * 2, color="0.35", lw=0.8))
        fig.text((left + right) / 2, y + 0.03 / fig.get_figheight(), label, ha="center", va="bottom", style="italic")


def label_rows(fig, axes, models: list[str], benches: list[str], ylabel: str, xlabel: str) -> None:
    """Benchmark on each column, model name and ylabel left of each row, xlabel under each bottom panel."""
    for ax, bench in zip(axes[0], benches):
        ax.set_title(bench_label(bench))
    for row, model in zip(axes, models):
        label = label_of(model)
        # long names break after their first hyphen, so they stay within a square panel's height
        name = label.replace("-", "-\n", 1) if len(label) > 12 else label
        row[0].set_ylabel(f"{name}\n{ylabel}", labelpad=2, fontsize=6.5)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, labelpad=1)
    fig.align_ylabels(axes[:, 0])


def scatter_figure(
    baselines: dict, cells: dict, blocks: list[list[str]], benches: list[str], path: Path
) -> None:
    """One row per model, blocks of models set apart, one column per benchmark (in-distribution first), each panel on its own axes."""
    width, left, right, top, bottom, row_height, row_gap = TEXT_WIDTH, 0.6, 0.1, 0.55, 0.62, 0.8, 0.25
    models = [model for block in blocks for model in block]
    height = top + bottom + row_height * (len(models) + row_gap * (len(blocks) - 1))
    fig, axes = split_subplots(
        len(models), len(PRIMARY), len(benches) - len(PRIMARY), (width, height),
        row_groups=[len(block) for block in blocks], row_gap=row_gap,
    )
    for row, model in zip(axes, models):
        for ax, bench in zip(row, benches):
            points = scatter_panel(ax, baselines, cells, model, bench)
            finish_panel(ax, min(points + [0]) - 0.6, max(points + [0]) + 0.6)
    label_rows(fig, axes, models, benches, "LlMBO", "GRPO")
    fig.subplots_adjust(
        left=left / width, right=1 - right / width, top=1 - top / height, bottom=bottom / height, hspace=0.35, wspace=0.3
    )
    mark_ood(fig, axes, len(PRIMARY))
    handles = [
        metric_handle(k, ls="", marker="o", mfc=None if k == GREEDY else "white", ms=4)
        for k in METRIC_OBJECTIVE
    ] + [
        plt.Line2D([], [], ls="", marker=U_MARKERS[u], color="0.3", ms=4, label=f"$u={u}$") for u in LOW_U
    ] + [plt.Rectangle((0, 0), 1, 1, fc="0.5", alpha=0.15, label="LlMBO better")]
    bottom_legend(fig, handles)
    save(fig, path)


def scatter_grid(baselines: dict, cells: dict, models: list[str], path: Path) -> None:
    """Selection of scatter_figure in the style of fig_headline, narrower than the text: one column per model, one row per in-distribution benchmark."""
    panel, left, right, top, bottom, gap = 1.08, 0.62, 0.05, 0.25, 0.52, 0.22
    width = left + right + panel * (len(models) + gap * (len(models) - 1))
    height = top + bottom + panel * (len(PRIMARY) + gap * (len(PRIMARY) - 1))
    family_color = {m.split("-")[0]: color for family, color in FAMILIES.values() for m in family}
    fig, axes = plt.subplots(len(PRIMARY), len(models), figsize=(width, height), squeeze=False)
    for row, bench in zip(axes, PRIMARY):
        for ax, model in zip(row, models):
            points = scatter_panel(ax, baselines, cells, model, bench, family_color[model.split("-")[0]])
            finish_panel(ax, min(points + [0]) - 0.6, max(points + [0]) + 0.6)
        row[0].set_ylabel(f"{bench_label(bench)}\nLlMBO", labelpad=2)
    for ax in axes[-1]:
        ax.set_xlabel("GRPO", labelpad=1)
    fig.subplots_adjust(left=left / width, right=1 - right / width, top=1 - top / height, bottom=bottom / height, hspace=gap, wspace=gap)
    for ax, model in zip(axes[0], models):
        position = ax.get_position()
        fig.text((position.x0 + position.x1) / 2, position.y1 + 0.06 / height, label_of(model), ha="center", va="bottom")
    _, handles = legend_rows()
    bottom_legend(fig, [h for h in handles if h.get_label() != "$u=8$"], columnspacing=0.8, handletextpad=0.3, handlelength=1.2)
    save(fig, path)


def main(baselines: dict, cells: dict) -> None:
    benches = PRIMARY + [b for b in BENCHMARKS if b not in PRIMARY]
    for group, blocks in GROUPS.items():
        scatter_figure(baselines, cells, blocks, benches, OUT / f"fig_scatter_{group}.pdf")
    scatter_grid(baselines, cells, BASES, OUT / "fig_scatter_bases_id.pdf")


if __name__ == "__main__":
    main(*load_data())
