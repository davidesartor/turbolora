"""Figure fig_headline: mean ± s.e. of seed-matched LlMBO vs GRPO gain on GSM8K and MATH-500, LlMBO trained on each metric, greedy (filled) and avg@4 (open) overlaid, one colour per 7-8B family (base and Math pooled), one marker per u <= 8: `uv run --no-project --with matplotlib --with numpy overleaf/scripts/headline.py` from the repo root."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import BASE_STYLE, METRIC_OBJECTIVE, METRICS, OUT, PERCENT, PRIMARY, base_value, bench_label, load_data, mean_se, save, value

COLUMN_WIDTH = 3.25  # AISTATS \columnwidth, inches
METRIC_FILLED = {4: False, 1: True}  # greedy filled, avg@4 open; each scored for LlMBO trained on it
U_MARKERS = {1: "o", 2: "^", 4: "s", 8: "D"}
FAMILIES = {  # label (a model name, for its logo; Qwen2.5-7B and DeepSeek-7B pool base and Math) -> (models pooled with each counted once, colour)
    "Qwen2.5-7B": (["qwen2.5-7b", "qwen2.5-7b-math"], "#7b3fbf"),
    "Llama3.1-8B": (["llama3.1-8b"], "#7f7f7f"),
    "Mistral-7B": (["mistral-7b"], "#f28c28"),
    "DeepSeek-7B": (["deepseek-7b", "deepseek-7b-math"], "#3b6fd8"),
}


def seed_gains(
    baselines: dict, cells: dict, model: str, bo: str, k: int, u: int, bench: str
) -> list[tuple[float, float]]:
    """(GRPO, LlMBO) gain over the base model on metric k at u, for every seed both ran."""
    base = base_value(baselines, model, k, bench)
    grpo = {r["seed"]: value(r, k, bench) for r in cells[model]["grpo"].get(u, [])}
    return [
        (grpo[r["seed"]] - base, value(r, k, bench) - base)
        for r in cells[model][bo].get(u, [])
        if r["seed"] in grpo
    ]


def family_gain(baselines: dict, cells: dict, models: list[str], k: int, u: int, bench: str) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """(GRPO, LlMBO) mean ± s.e. gain, averaged over models with each model's seed pairs counted once."""
    per_model = [np.array(pairs).T for m in models if (pairs := seed_gains(baselines, cells, m, METRIC_OBJECTIVE[k], k, u, bench))]
    if not per_model:
        return None
    stats = np.array([[mean_se(x), mean_se(y)] for x, y in per_model])  # model, axis, (mean, se)
    means = stats[..., 0].mean(0)
    ses = np.sqrt((stats[..., 1] ** 2).sum(0)) / len(per_model)
    return (means[0], ses[0]), (means[1], ses[1])


def headline_panel(ax: plt.Axes, baselines: dict, cells: dict, bench: str) -> None:
    """Mean ± s.e. per family, u and metric."""
    extent = [0]
    # later draws land on top: Qwen over every other family, greedy under avg@4, large u under small u
    for models, color in reversed(FAMILIES.values()):
        for k in reversed(METRIC_FILLED):
            for u in reversed(U_MARKERS):
                if not (gain := family_gain(baselines, cells, models, k, u, bench)):
                    continue
                (x_mid, x_se), (y_mid, y_se) = gain
                extent += [x_mid - x_se, x_mid + x_se, y_mid - y_se, y_mid + y_se]
                ax.errorbar(
                    x_mid,
                    y_mid,
                    xerr=x_se,
                    yerr=y_se,
                    color=color,
                    marker=U_MARKERS[u],
                    ms=3.5,
                    mfc=color if METRIC_FILLED[k] else "white",
                    mec="white" if METRIC_FILLED[k] else color,
                    mew=0.4 if METRIC_FILLED[k] else 0.7,
                    elinewidth=0.5,
                    capsize=0,
                    alpha=0.85,
                    zorder=3,
                )
    margin = 0.05 * (max(extent) - min(extent))
    lo, hi = min(extent) - margin, max(extent) + margin
    ax.fill_between([lo, hi], [lo, hi], hi, color="0.5", alpha=0.07, lw=0)
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.8)
    ax.axhline(0, **BASE_STYLE)
    ax.axvline(0, **BASE_STYLE)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
    ax.locator_params(nbins=4)
    ax.tick_params(labelsize=6, pad=1)
    ax.xaxis.set_major_formatter(PERCENT)
    ax.yaxis.set_major_formatter(PERCENT)


def legend_rows() -> tuple[list, list]:
    """Families (colour swatch) on the first row; u (marker), metrics (fill) and the shaded half on the second."""
    families = [plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.85, lw=0, label=label) for label, (_, color) in FAMILIES.items()]
    us = [plt.Line2D([], [], ls="", marker=marker, color="0.3", ms=3.5, label=f"$u={u}$") for u, marker in U_MARKERS.items()]
    metrics = [
        plt.Line2D([], [], ls="", marker="o", color="0.3", mfc="0.3" if filled else "white", ms=3.5, label=METRICS[k].split(",")[0])
        for k, filled in METRIC_FILLED.items()
    ]
    better = plt.Rectangle((0, 0), 1, 1, fc="0.5", alpha=0.15, label="LlMBO better")
    return families, us + metrics + [better]


def headline_figure(baselines: dict, cells: dict, path: Path) -> None:
    """Square panels at \\columnwidth, one per benchmark (GSM8K, MATH-500), both metrics overlaid."""
    fig, axes = plt.subplots(1, len(PRIMARY), figsize=(COLUMN_WIDTH, 2.15))
    for ax, bench in zip(axes, PRIMARY):
        headline_panel(ax, baselines, cells, bench)
        ax.set_title(bench_label(bench))
        ax.set_xlabel("GRPO gain", fontsize=6.5, labelpad=1.5)
    axes[0].set_ylabel("LlMBO gain", fontsize=6.5, labelpad=1.5)
    families, markers = legend_rows()
    legend = dict(loc="lower center", frameon=False, fontsize=6, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.2, handlelength=1.2)
    fig.legend(handles=families, ncol=len(families), bbox_to_anchor=(0.5, 0.06), **legend | dict(handlelength=0.7, handleheight=0.7))  # square swatches
    fig.legend(handles=markers, ncol=len(markers), bbox_to_anchor=(0.5, 0), **legend)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.935, bottom=0.255, wspace=0.17)
    save(fig, path)


def main(baselines: dict, cells: dict) -> None:
    headline_figure(baselines, cells, OUT / "fig_headline.pdf")


if __name__ == "__main__":
    main(*load_data())
