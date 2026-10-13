"""Figures fig_budget{,_main,_mathtrain,_batch}: gain over the base model against completions generated: `uv run --no-project --with matplotlib --with numpy overleaf/scripts/budget.py` from the repo root."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLUMN_WIDTH = 3.25  # AISTATS \columnwidth, inches

from common import (
    BASE_STYLE,
    COLORS,
    LABELS,
    LOW_U,
    MARKERS,
    METRIC_OBJECTIVE,
    METRICS,
    OUT,
    PERCENT,
    PRIMARY,
    SHIFT_MODELS,
    SMALL_BASES,
    TABLE_BLOCKS,
    TEXT_WIDTH,
    base_value,
    bench_label,
    bottom_legend,
    budget_curve,
    label_of,
    load_data,
    metric_color,
    save,
)
from headline import FAMILIES


BUDGET_MODELS = [m for block in TABLE_BLOCKS for m in block] + SMALL_BASES
LARGE_GROUPS = [models for models, _ in FAMILIES.values()]  # the 7-8B families of fig_headline, each counted once
SMALL_GROUPS = [[m] for m in SMALL_BASES]
MIN_POINTS = 3  # a metric evaluated at the final snapshot only is not a curve
SOLID = dict(ls="-", filled=True)
OPEN = dict(ls=(0, (3, 1.5)), filled=False)  # avg@4, b = 4 or MATH only: dashed, open, as in fig_scaling and fig_gap

# series: (legend label of its style, method, training set tag, metric k, style); colour is the optimizer
METRIC_SERIES = [
    (METRICS[k].split(",")[0], m, "", k, SOLID if k == 1 else OPEN) for k in (1, 4) for m in ("grpo", METRIC_OBJECTIVE[k])
]
BATCH_SERIES = [("$b=1$", m, "", 1, SOLID) for m in ("grpo", "bo-greedy")] + [("$b=4$", "bo-greedy-b4", "", 1, OPEN)]
TRAIN_SERIES = [
    (label, m, tag, 1, style) for label, tag, style in (("SimpleRL tiers", "", SOLID), ("MATH only", "-math", OPEN)) for m in ("grpo", "bo-greedy")
]


def optimizer(method: str) -> str:
    return "grpo" if method == "grpo" else "bo-greedy"


def pooled_curve(
    baselines: dict, cells: dict, groups: list[list[str]], method: str, tag: str, k: int, us: list[int], bench: str
) -> np.ndarray | None:
    """(completions, mean, se) per snapshot step of the gain on metric k, averaged over groups (each counted once) and, within one, over its models and us."""
    stats = []
    for models in groups:
        curves = [
            np.array(c)
            for m in models
            if method + tag in cells[m]
            for u in us
            if len(c := budget_curve(cells, m, method, tag, u, bench, base=base_value(baselines, m, k, bench), k=k)) >= MIN_POINTS
        ]
        if curves:
            curves = np.stack([c[: min(map(len, curves))] for c in curves])  # curve, step, (completions, mean, se)
            stats.append(np.stack([curves[..., 0].mean(0), curves[..., 1].mean(0), np.sqrt((curves[..., 2] ** 2).sum(0)) / len(curves)]))
    if not stats:
        return None
    steps = min(s.shape[1] for s in stats)
    n, mean, se = np.stack([s[:, :steps] for s in stats], 1)  # stat, group, step
    return np.stack([n.mean(0), mean.mean(0), np.sqrt((se**2).sum(0)) / len(stats)])


def budget_panel(
    ax: plt.Axes, baselines: dict, cells: dict, groups: list[list[str]], series: list, us: list[int], bench: str
) -> set[str]:
    """Each series against completions generated; dotted: GRPO's final gain, unless the series span training sets. Returns the style labels drawn."""
    ax.axhline(0, **BASE_STYLE)
    one_training_set = len({tag for _, _, tag, _, _ in series}) == 1
    drawn = set()
    for label, method, tag, k, style in series:
        if (curve := pooled_curve(baselines, cells, groups, method, tag, k, us, bench)) is None:
            continue
        n, mean, se = curve
        color, marker = metric_color(COLORS[optimizer(method)], k), MARKERS[optimizer(method)]
        ax.fill_between(n, mean - se, mean + se, color=color, alpha=0.12, lw=0)
        ax.plot(n, mean, color=color, ls=style["ls"], marker=marker, ms=2.2, lw=0.9, mfc=color if style["filled"] else "white", mec=color, mew=0.6)
        if method == "grpo" and one_training_set:
            ax.axhline(mean[-1], color=color, ls=":", lw=0.8)
        drawn.add(label)
    ax.set_xscale("log")
    ax.set_xticks([1e3, 1e5], ["1k", "100k"])
    ax.minorticks_off()
    ax.tick_params(labelsize=6, pad=1)
    ax.yaxis.set_major_formatter(PERCENT)
    return drawn


def budget_handles(series: list, drawn: set[str]) -> list:
    """Optimizers by colour, then each drawn style, GRPO's final gain and the base model."""
    optimizers = [plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], ms=4, label=LABELS[m]) for m in ("grpo", "bo-greedy")]
    styles = {label: (k, style) for label, _, _, k, style in series if label in drawn}
    styles = [
        plt.Line2D([], [], color=(c := metric_color("0.4", k)), ls=style["ls"], marker="o", ms=3, mfc=c if style["filled"] else "white", label=label)
        for label, (k, style) in styles.items()
    ]
    end = [] if series is TRAIN_SERIES else [plt.Line2D([], [], color=COLORS["grpo"], ls=":", lw=0.8, label="GRPO, end of training")]
    return [*optimizers, *styles, *end, plt.Line2D([], [], **BASE_STYLE, label="base model")]


def budget_figure(baselines: dict, cells: dict, rows: list[tuple[str, list[list[str]]]], series: list, path: Path) -> None:
    """Gain over the base model against completions generated; one row per (label, groups pooled), one column per primary task and u."""
    columns = [(bench, u) for bench in PRIMARY for u in LOW_U]
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(TEXT_WIDTH, 0.65 + 0.98 * len(rows)), sharex=True, squeeze=False)
    drawn = set()
    for row, (label, groups) in zip(axes, rows):
        for ax, (bench, u) in zip(row, columns):
            drawn |= budget_panel(ax, baselines, cells, groups, series, [u], bench)
        row[0].set_ylabel(f"{label}\ngain")
    for ax, (bench, u) in zip(axes[0], columns):
        ax.set_title(f"{bench_label(bench)}\n$u={u}$")
    for ax in axes[-1]:
        ax.set_xlabel("completions", labelpad=1)
    bottom_legend(fig, budget_handles(series, drawn))
    fig.tight_layout(rect=(0, 0.2 / fig.get_figheight(), 1, 1), h_pad=0.5, w_pad=0.3)
    save(fig, path)


def budget_main_figure(baselines: dict, cells: dict, path: Path) -> None:
    """Main-text summary: both primary tasks on one axis, pooled over the 7-8B families and u <= 4, SimpleRL tiers, b = 1, greedy."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.2))
    ax.axhline(0, **BASE_STYLE)
    task_styles = dict(zip(PRIMARY, (SOLID, OPEN)))
    reaches = []
    for bench in PRIMARY:
        # each curve starts from the base model: no completions, no gain
        curves = {
            m: np.concatenate([np.zeros((3, 1)), pooled_curve(baselines, cells, LARGE_GROUPS, m, "", 1, LOW_U, bench)], axis=1)
            for m in ("grpo", "bo-greedy")
        }
        grpo_final = curves["grpo"][1][-1]
        ax.hlines(grpo_final, 0, curves["grpo"][0][-1], color=COLORS["grpo"], ls=":", lw=0.8)
        for method, (n, mean, se) in curves.items():
            ax.fill_between(n, mean - se, mean + se, color=COLORS[method], alpha=0.12, lw=0)
            style, color = task_styles[bench], COLORS[method]
            ax.plot(n, mean, color=color, ls=style["ls"], marker=MARKERS[method], ms=3, lw=1.2, mfc=color if style["filled"] else "white", mec=color, mew=0.6)
        # first step from which LlMBO stays above GRPO's final gain, ringed
        n, mean, _ = curves["bo-greedy"]
        i = next(i for i in range(len(mean)) if (mean[i:] >= grpo_final).all())
        ax.plot(n[i], mean[i], "o", ms=7, mfc="none", mec=COLORS["bo-greedy"], mew=0.8)
        reaches.append((n[i], curves["grpo"][0][-1]))

    # LlMBO's budget to pass GRPO's final gain against GRPO's whole run
    reach_n, grpo_n = max(r for r, _ in reaches), min(g for _, g in reaches)
    top = ax.get_ylim()[1] + 0.25
    ax.annotate("", (reach_n, top), (grpo_n, top), arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.7, shrinkA=0, shrinkB=0))
    ax.text(np.sqrt(reach_n * grpo_n), top, f"{grpo_n / reach_n:.0f}$\\times$ fewer completions", ha="center", va="bottom", fontsize=6.5, color="0.2")
    ax.set_ylim(top=top + 0.35)

    # log scale past the first batch, linear below it so the base model sits at 0
    ax.set_xscale("symlog", linthresh=256, linscale=0.4)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.1)
    exponents = range(8, 18)
    ax.set_xticks([0, *(2**e for e in exponents)], ["0", *(f"$2^{{{e}}}$" for e in exponents)])
    ax.minorticks_off()
    ax.tick_params(axis="x", labelsize=6)
    ax.yaxis.set_major_formatter(PERCENT)
    ax.set_xlabel("completions generated")
    handles = [plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], ms=4, label=LABELS[m].removeprefix("TinyLoRA, ")) for m in ("grpo", "bo-greedy")]
    handles += [
        plt.Line2D([], [], color=COLORS["grpo"], ls=":", lw=0.8, label="GRPO, end"),
        *(
            plt.Line2D([], [], color="0.4", ls=style["ls"], marker="o", ms=3, mfc="0.4" if style["filled"] else "white", label=bench_label(bench))
            for bench, style in task_styles.items()
        ),
        plt.Line2D([], [], **BASE_STYLE, label="base model"),
    ]
    # column-major fill: methods, then tasks, then the reference lines
    handles = [handles[i] for i in (0, 3, 1, 4, 2, 5)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6, columnspacing=1.2, handlelength=2.2)
    fig.tight_layout(pad=0.4, rect=(0, 0.3 / fig.get_figheight(), 1, 1))
    save(fig, path)


def main(baselines: dict, cells: dict) -> None:
    per_model = lambda models: [(label_of(m), [[m]]) for m in models]
    budget_main_figure(baselines, cells, OUT / "fig_budget_main.pdf")
    budget_figure(baselines, cells, per_model(BUDGET_MODELS), METRIC_SERIES, OUT / "fig_budget.pdf")
    budget_figure(baselines, cells, per_model(SHIFT_MODELS), TRAIN_SERIES, OUT / "fig_budget_mathtrain.pdf")
    budget_figure(baselines, cells, [("7–8B, pooled", LARGE_GROUPS), ("1.5B, pooled", SMALL_GROUPS)], BATCH_SERIES, OUT / "fig_budget_batch.pdf")


if __name__ == "__main__":
    main(*load_data())
