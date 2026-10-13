"""Shared loading, statistics and plot style for the paper scripts in this folder."""

import colorsys
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

RUNS = Path("outputs/runs")
BASE = Path(
    "tmp/runs/base-1024"
)  # base evals re-graded at the 1024-token training budget (tmp/tools/truncate_base.py)
PAPER = Path(__file__).parents[1]
CACHE = PAPER / "results-cache.json"
OUT = PAPER / "results"
LOGO_DIR = PAPER / "logos"
LOGOS = {
    "qwen2.5": "qwen",
    "llama": "llama",
    "mistral": "mistral",
    "deepseek": "deepseek",
}  # family -> logos/<name>.png
LOGO_GAP = "     "  # room left in front of a model name in a figure for its logo
BENCHMARKS = ["gsm8k", "math500", "aime24", "amc23", "minerva", "olympiad"]
BENCHMARK_LABELS = ["GSM8K", "MATH-500", "AIME24", "AMC23", "Minerva", "Olympiad"]
PRIMARY = ["gsm8k", "math500"]  # the in-distribution tasks the text leads with
COMPLETIONS_PER_STEP = 256  # 64 prompts x 4 rollouts per GRPO step = one BO batch
MODELS = {  # name -> (family dir, label, billions of weights, macro name)
    "qwen2.5-1.5b": ("qwen2.5", "Qwen2.5-1.5B", 1.54, "QwenSmall"),
    "qwen2.5-7b": ("qwen2.5", "Qwen2.5-7B", 7.62, "Qwen"),
    "qwen2.5-1.5b-instruct": (
        "qwen2.5-instruct",
        "Qwen2.5-1.5B-Inst.",
        1.54,
        "InstSmall",
    ),
    "qwen2.5-7b-instruct": ("qwen2.5-instruct", "Qwen2.5-7B-Inst.", 7.62, "Inst"),
    "qwen2.5-1.5b-math": ("qwen2.5-math", "Qwen2.5-Math-1.5B", 1.54, "MathSmall"),
    "qwen2.5-7b-math": ("qwen2.5-math", "Qwen2.5-Math-7B", 7.62, "Math"),
    "llama3.1-8b": ("llama", "Llama3.1-8B", 8.03, "Llama"),
    "mistral-7b": ("mistral", "Mistral-7B", 7.24, "Mistral"),
    "deepseek-7b": ("deepseek", "DeepSeek-7B", 6.91, "DeepseekBase"),
    "deepseek-7b-math": ("deepseek", "DeepSeek-Math-7B", 6.91, "Deepseek"),
}
QWEN = ["qwen2.5-7b", "qwen2.5-7b-math"]
OTHERS = ["llama3.1-8b", "mistral-7b", "deepseek-7b"]  # the non-Qwen base models
LARGE = (
    QWEN + OTHERS + ["deepseek-7b-math"]
)  # 7-8B: pooled comparisons, Instruct left out
SMALL = ["qwen2.5-1.5b", "qwen2.5-1.5b-math"]
BASES = [
    "qwen2.5-7b",
    "llama3.1-8b",
    "mistral-7b",
    "deepseek-7b",
]  # one 7-8B base model per family
MATHS = ["qwen2.5-7b-math", "deepseek-7b-math"]
SMALL_BASES = ["qwen2.5-1.5b", "qwen2.5-1.5b-math"]
INSTRUCTS = [
    "qwen2.5-7b-instruct",
    "qwen2.5-1.5b-instruct",
]  # appendix: no headroom at u <= 4
GROUPS = {
    "main": [BASES, MATHS],
    "small": [SMALL_BASES],
    "instruct": [INSTRUCTS],
}  # one table and figure set each, as row blocks; 1.5B and Instruct: own appendices
TABLE_BLOCKS = [  # row blocks of the 7-8B per-model tables: the vanilla base models, then the math-specialized ones
    ["llama3.1-8b", "mistral-7b", "deepseek-7b", "qwen2.5-7b"],
    ["deepseek-7b-math", "qwen2.5-7b-math"],
]
SCALING_BLOCKS = [
    BASES,
    MATHS,
    ["qwen2.5-7b-instruct", *SMALL, "qwen2.5-1.5b-instruct"],
]  # row blocks of the one scaling figure
SHIFT_MODELS = ["qwen2.5-7b", "qwen2.5-7b-math", "qwen2.5-1.5b", "qwen2.5-1.5b-math"]
LOW_U = [1, 2, 4]  # the 8-seed cells
U_ALL = [1, 2, 4, 8, 16, 64, 256]
LARGE_U = [64, 256]  # LlMBO is not robust here: appendix only
BO_MAX_U = 16  # largest u where LlMBO enters the main results
CFG_METHOD = {  # adapter dir -> cfg regex (size, batch, t1, math)
    "lora-grpo": r"r(?P<size>\d+)",
    "loraxs-grpo": r"r(?P<size>\d+)",
    "tinylora-grpo": r"r2-u(?P<size>\d+)(?P<math>-math)?",
    "tinylora-turbo": r"r2-u(?P<size>\d+)-b(?P<batch>\d)(?P<t1>-t1)?(?P<math>-math)?",
}
LABELS = {
    "lora": "LoRA",
    "loraxs": "LoRA-XS",
    "grpo": "TinyLoRA, GRPO",
    "bo": "TinyLoRA, LlMBO",
    "bo-greedy": "TinyLoRA, LlMBO",  # one method: each metric is reported for LlMBO trained on it
}
METRICS = {
    1: "greedy",
    4: "avg@4, $T{=}1$",
}
OBJECTIVE_METRIC = {
    "bo": 4,
    "bo-greedy": 1,
}  # each LlMBO objective is scored on the metric it optimizes
METRIC_OBJECTIVE = {k: m for m, k in OBJECTIVE_METRIC.items()}
METRIC_LINESTYLES = {
    1: "-",
    4: (0, (3, 1.5)),
}  # greedy solid, avg@4 dashed, in every figure that overlays the metrics
METRIC_SATURATION = {
    1: 1.0,
    4: 0.5,
}  # greedy saturated, avg@4 washed out: hue is the method, saturation the metric
TRAIN_SETS = {"": "SimpleRL tiers", "-math": "MATH only"}

TEXT_WIDTH = 6.75  # AISTATS \textwidth, inches: figures render at native size
COLORS = {
    "lora": "#8a4fc4",
    "loraxs": "#d6457a",
    "grpo": "#2a78d6",
    "bo": "#eb6834",
    "bo-greedy": "#eb6834",  # one method: LlMBO is one hue whichever metric it optimizes
}
MARKERS = {"lora": "o", "loraxs": "s", "grpo": "D", "bo": "p", "bo-greedy": "p"}
SCATTER_US = [u for u in U_ALL if u <= BO_MAX_U]  # one point per u in the gain scatters
U_CMAP = LinearSegmentedColormap.from_list(
    "u", plt.cm.viridis_r(np.linspace(0.1, 0.95, 256))
)
U_NORM = Normalize(0, np.log2(SCATTER_US[-1]))  # colour is continuous in log2 u
U_COLORS = {u: U_CMAP(U_NORM(np.log2(u))) for u in SCATTER_US}
BATCH_STYLES = {
    1: ("", "-", -0.08),
    4: ("-b4", (0, (3, 1.5)), 0.08),
}  # batch -> (method suffix, line, x nudge in octaves); b=1 is the default LlMBO
PERCENT = FuncFormatter(
    lambda v, _: f"{v:g}%".replace("-", "\u2212")
)  # tick labels for accuracies, gains and gaps
BASE_STYLE = dict(color="0.45", ls=(0, (4, 2)), lw=1)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "0.35",
        "axes.linewidth": 0.6,
        "xtick.color": "0.35",
        "ytick.color": "0.35",
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "0.9",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "pdf.fonttype": 42,
    }
)


# ---------------------------------------------------------------- loading


def method_of(adapter: str, match: re.Match) -> str:
    g = match.groupdict()
    name = {"lora-grpo": "lora", "loraxs-grpo": "loraxs", "tinylora-grpo": "grpo"}.get(
        adapter
    )
    if name is None:
        name = (
            "bo" + ("" if g["t1"] else "-greedy") + ("-b4" if g["batch"] == "4" else "")
        )
    return name + ("-math" if g.get("math") else "")


def summary(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def bench_avg(s: dict | None) -> float | None:
    if not s or not all(b in s for b in BENCHMARKS):
        return None
    return float(np.mean([100 * s[b]["accuracy"] for b in BENCHMARKS]))


def step_of(p: Path) -> int:
    return int(p.name.split("-")[1])


def completions_at(seed_dir: Path, run: dict, steps: list[int]) -> dict[int, int]:
    """Snapshot step -> completions generated up to it; BO skips its initial design but counts restart designs."""
    if run["loss"] != "turbo":
        return {s: s * COMPLETIONS_PER_STEP for s in steps}
    trials = json.loads((seed_dir / "trials.json").read_text())
    design = {t["batch"]: t["design"] for t in trials}
    initial = next((b for b in sorted(design) if not design[b]), len(design))
    counts, guided = {}, 0
    for batch in sorted(design):
        guided += not design[batch]
        counts.setdefault(guided, (batch + 1 - initial) * COMPLETIONS_PER_STEP)
    return {
        s: counts.get(s, (len(design) - initial) * COMPLETIONS_PER_STEP) for s in steps
    }


def search_scores(seed_dir: Path, run: dict) -> dict[str, float] | None:
    """LlMBO's in-search pass rate (%) of its final pick and of θ = 0, from the minibatch scores it searched on."""
    if (
        run["loss"] != "turbo" or "batch" not in run
    ):  # GRPO, or a search from before batches were logged
        return None
    trials = json.loads((seed_dir / "trials.json").read_text())
    n = COMPLETIONS_PER_STEP // (run["batch"] * run["k_rollouts"])
    # a trial value is the Beta(1, 1) posterior logit mean ψ(1+s) − ψ(1+n−s) = H_s − H_{n−s} of s passes in n
    harmonic = np.concatenate([[0.0], np.cumsum(1 / np.arange(1, n + 1))])
    rate = lambda v: 100 * int(np.argmin(abs(harmonic - harmonic[::-1] - v))) / n
    pick = next(t for t in trials if t["theta"] == run["theta"])
    return dict(
        pick=rate(pick["value"]),
        baseline=float(np.mean([rate(t["value"]) for t in trials if t["baseline"]])),
    )


def read_seed(seed_dir: Path) -> dict | None:
    """Final per-benchmark @1/@4 accuracy, train-set accuracy, cost fields and @1/@4 snapshot curves of one finished seed."""
    run = json.loads((seed_dir / "run.json").read_text())
    snaps = sorted(seed_dir.glob("snapshots/step-*"), key=step_of)
    if "steps" not in run or not snaps:
        return None
    final = {k: summary(snaps[-1] / f"eval@{k}" / "summary.json") for k in (1, 4)}
    if bench_avg(final[1]) is None or bench_avg(final[4]) is None:
        return None
    curve = {
        k: {step_of(p): summary(p / f"eval@{k}" / "summary.json") for p in snaps}
        for k in (1, 4)
    }
    curve = {
        k: {
            s: {b: 100 * c[b]["accuracy"] for b in BENCHMARKS}
            for s, c in by_step.items()
            if bench_avg(c) is not None
        }
        for k, by_step in curve.items()
    }
    total = completions_at(seed_dir, run, list(curve[1]))
    train_key = f"{run['task']}-train"
    fields = (
        "model",
        "task",
        "adapter",
        "loss",
        "params",
        "steps",
        "train_hours",
        "peak_vram_gb",
        "gpu",
    )
    return dict(
        run={k: run.get(k) for k in fields},
        bench={
            k: {b: 100 * final[k][b]["accuracy"] for b in BENCHMARKS} for k in (1, 4)
        },
        acc={k: bench_avg(final[k]) for k in (1, 4)},
        train={
            k: 100 * final[k][train_key]["accuracy"]
            for k in (1, 4)
            if train_key in final[k]
        },
        curve={
            k: {s: (total[s], a) for s, a in by_step.items() if s in total}
            for k, by_step in curve.items()
        },
        completions=max(total.values()),
        search=search_scores(seed_dir, run),
    )


def base_train(family: str, model: str, k: int) -> dict[str, float]:
    """Base model's accuracy on each training set it was evaluated on: task -> %."""
    s = summary(RUNS / family / model / "base" / f"eval@{k}" / "summary.json") or {}
    return {
        t.removesuffix("-train"): 100 * v["accuracy"]
        for t, v in s.items()
        if t.endswith("-train")
    }


def stamp(seed_dir: Path) -> str:
    """Cheap change detector: run.json and last-snapshot summary mtimes, and the number of snapshot avg@4 evals."""
    parts = [
        (seed_dir / "run.json").stat().st_mtime,
        len(list(seed_dir.glob("snapshots/step-*/eval@4/summary.json"))),
    ]
    snaps = sorted(seed_dir.glob("snapshots/step-*"), key=step_of)
    if snaps:
        parts += [snaps[-1].name] + [
            p.stat().st_mtime
            for k in (1, 4)
            if (p := snaps[-1] / f"eval@{k}" / "summary.json").is_file()
        ]
    return json.dumps(parts)


def int_keys(record: dict) -> dict:
    """JSON round-trip turns int keys into strings; undo it."""
    for field in ("acc", "bench", "train", "curve"):
        record[field] = {int(k): v for k, v in record[field].items()}
    record["curve"] = {
        k: {int(s): tuple(v) for s, v in c.items()} for k, c in record["curve"].items()
    }
    return record


def load() -> tuple[dict, dict]:
    """(baselines[model] -> {acc, bench}, cells[model][method][size] -> list of seed records), cached on file stats."""
    cache = json.loads(CACHE.read_text()) if CACHE.is_file() else {}
    fresh, baselines, seeds = {}, {}, []
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for model, (family, *_) in MODELS.items():
        base = {
            k: json.loads((BASE / model / f"eval@{k}" / "summary.json").read_text())
            for k in (1, 4)
        }
        baselines[model] = dict(
            acc={k: bench_avg(base[k]) for k in (1, 4)},
            bench={
                k: {b: 100 * base[k][b]["accuracy"] for b in BENCHMARKS} for k in (1, 4)
            },
            train={k: base_train(family, model, k) for k in (1, 4)},
        )
        for adapter, pattern in CFG_METHOD.items():
            for cfg_dir in sorted((RUNS / family / model / adapter).glob("*")):
                if not (match := re.fullmatch(pattern, cfg_dir.name)):
                    continue
                method, size = method_of(adapter, match), int(match["size"])
                seeds += [
                    (model, method, size, d) for d in sorted(cfg_dir.glob("seed*"))
                ]
    for model, method, size, seed_dir in tqdm(seeds, desc="seeds", unit="run"):
        if not (seed_dir / "run.json").is_file():
            continue
        key, st = str(seed_dir), stamp(seed_dir)
        record = (
            cache[key][1]
            if key in cache and cache[key][0] == st
            else read_seed(seed_dir)
        )
        fresh[key] = (st, record)
        if record:
            record = int_keys(json.loads(json.dumps(record)))
            record["seed"] = int(seed_dir.name[4:])
            cells[model][method][size].append(record)
    # atomic swap: the scripts may run in parallel
    partial = CACHE.with_suffix(f".{os.getpid()}.tmp")
    partial.write_text(json.dumps(fresh))
    partial.replace(CACHE)
    return baselines, cells


def load_data() -> tuple[dict, dict]:
    """load(), with the output folder created."""
    baselines, cells = load()
    OUT.mkdir(exist_ok=True)
    return baselines, cells


# ---------------------------------------------------------------- statistics


def label_of(model: str) -> str:
    return MODELS[model][1]


def logo_of(model: str) -> str:
    return LOGOS[MODELS[model][0].split("-")[0]]


def logo_tex(model: str) -> str:
    return rf"\raisebox{{-0.15em}}{{\includegraphics[height=0.9em]{{logos/{logo_of(model)}.png}}}}"


def logo_label(model: str) -> str:
    """Model label with its family logo in front, for a table."""
    return rf"{logo_tex(model)}~{label_of(model)}"


def bench_value(accuracy: dict, bench: str) -> float:
    return float(accuracy[bench])


def value(
    record: dict, k: int, bench: str | None = None, train: bool = False
) -> float | None:
    if train:
        return record["train"].get(k)
    return bench_value(record["bench"][k], bench) if bench else record["acc"][k]


def base_value(baselines: dict, model: str, k: int, bench: str | None = None) -> float:
    if bench:
        return bench_value(baselines[model]["bench"][k], bench)
    return baselines[model]["acc"][k]


def bench_label(bench: str) -> str:
    return BENCHMARK_LABELS[BENCHMARKS.index(bench)]


def pooled(cells: dict, model: str, method: str, us: list[int]) -> list[dict]:
    return [r for u in us for r in cells[model][method].get(u, [])]


def gains(
    baselines: dict,
    cells: dict,
    model: str,
    method: str,
    k: int,
    us: list[int],
    bench: str | None = None,
) -> list[float]:
    """Accuracy minus the base model's, one entry per seed."""
    return [
        value(r, k, bench) - base_value(baselines, model, k, bench)
        for r in pooled(cells, model, method, us)
    ]


def paired(
    cells: dict,
    a: str,
    b: str,
    k: int,
    us: list[int],
    models: list[str],
    bench: str | None = None,
    train: bool = False,
) -> list[float]:
    """Per (model, u, seed) difference a − b; BO seed s searches the bases of GRPO seed s."""
    diffs = []
    for model in models:
        for u in us:
            left = {
                r["seed"]: value(r, k, bench, train) for r in cells[model][a].get(u, [])
            }
            right = {
                r["seed"]: value(r, k, bench, train) for r in cells[model][b].get(u, [])
            }
            diffs += [
                left[s] - right[s]
                for s in left.keys() & right.keys()
                if left[s] is not None and right[s] is not None
            ]
    return diffs


def u_colorbar(fig: plt.Figure, rect: list[float], fontsize: float = 6) -> None:
    """Continuous log2 u colorbar in the figure-fraction rect [x0, y0, w, h], ticks 2^0, 2^1, ..."""
    cbar = fig.colorbar(ScalarMappable(U_NORM, U_CMAP), cax=fig.add_axes(rect))
    cbar.set_ticks(
        np.log2(SCATTER_US), labels=[f"$2^{{{int(np.log2(u))}}}$" for u in SCATTER_US]
    )
    cbar.ax.tick_params(labelsize=fontsize, length=1.5, width=0.4, pad=1)
    cbar.outline.set_linewidth(0.4)
    cbar.set_label("trainable parameters", fontsize=fontsize, labelpad=2)


def mean_se(values: list[float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(values.mean()), (
        float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    )


def budget_curve(
    cells: dict,
    model: str,
    method: str,
    tag: str,
    u: int,
    bench: str,
    base: float | None = None,
    k: int = 1,
) -> list[tuple[float, float, float]]:
    """Per snapshot step at one u: (mean completions, mean, se) of accuracy on metric k minus base, or else minus the paired GRPO seed's final."""
    by_step = defaultdict(list)
    final = {
        r["seed"]: value(r, k, bench) for r in cells[model]["grpo" + tag].get(u, [])
    }
    for r in cells[model][method + tag].get(u, []):
        if r["seed"] in final:
            reference = final[r["seed"]] if base is None else base
            for step, (n, a) in r["curve"][k].items():
                by_step[step].append((n, bench_value(a, bench) - reference))
    return [
        (float(np.mean(n)), *mean_se(a))
        for _, v in sorted(by_step.items())
        if len(v) >= 3
        for n, a in [zip(*v)]
    ]


# ---------------------------------------------------------------- figures


def metric_color(color, k: int) -> tuple[float, float, float]:
    """color at metric k's saturation, lightened in step so a washed-out hue stays clear of gray."""
    h, l, s = colorsys.rgb_to_hls(*to_rgb(color))
    fade = 1 - METRIC_SATURATION[k]
    return colorsys.hls_to_rgb(h, l + (0.85 - l) * fade * 0.6, s * METRIC_SATURATION[k])


def metric_handle(k: int, **style) -> plt.Line2D:
    """Neutral legend entry for metric k: dark for greedy, light for avg@4."""
    return plt.Line2D(
        [],
        [],
        color=metric_color("0.25", k),
        label=METRICS[k].split(",")[0],
        **{"ls": METRIC_LINESTYLES[k], "lw": 1, **style},
    )


def handle(method: str, label: str | None = None) -> plt.Line2D:
    return plt.Line2D(
        [],
        [],
        color=COLORS[method],
        marker=MARKERS[method],
        ms=4,
        label=label or LABELS[method],
    )


def add_logos(fig) -> None:
    """Family logo in front of every model name that starts a y-label, figure text or legend entry."""
    logo_by_label = {label_of(m): logo_of(m) for m in MODELS}
    legends = [*fig.legends, *(ax.get_legend() for ax in fig.axes)]
    texts = (
        [ax.yaxis.label for ax in fig.axes]
        + list(fig.texts)
        + [t for lg in legends if lg for t in lg.get_texts()]
    )
    tagged = [
        (t, logo_by_label[name])
        for t in texts
        if (name := t.get_text().replace("-\n", "-").split("\n")[0]) in logo_by_label
    ]
    if not tagged:
        return
    for text, _ in tagged:
        text.set_text(LOGO_GAP + text.get_text())
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # logo centred in the gap: at the bottom of the first line of a rotated y-label, else left of the text, as a fraction of the text's box
    for text, logo in tagged:
        box = text.get_window_extent(renderer)
        lines = text.get_text().split("\n")
        width = lambda s: renderer.get_text_width_height_descent(
            s, text.get_fontproperties(), ismath=False
        )[0]
        gap = width(lines[0]) - width(lines[0][len(LOGO_GAP) :])
        if text.get_rotation() == 90:
            xy = (0.5 / len(lines), 0.5 + (gap - width(lines[0])) / 2 / box.height)
        else:
            xy = (gap / 2 / box.width, 0.5)
        image = OffsetImage(
            plt.imread(LOGO_DIR / f"{logo}.png"), zoom=0.85 * text.get_fontsize() / 640
        )
        fig.add_artist(
            AnnotationBbox(image, xy, xycoords=text, frameon=False, pad=0, zorder=10)
        )  # zorder: after legends, which place their texts


def save(fig, path: Path) -> None:
    add_logos(fig)
    fig.savefig(path, dpi=600)  # dpi: sharp logos and colorbars
    plt.close(fig)


def dodge(x, offset: float):
    """Shift x by a fixed distance (in octaves) on the log2 axis."""
    return np.asarray(x, dtype=float) * 2.0**offset


def log2_axis(ax: plt.Axes, ticks: list[int]) -> None:
    """x on log2, so u = 1, 2, 4, 8, 16 are evenly spaced."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()


def grid(models: list[str], benches: list[str], row: float, **share) -> tuple:
    """One row per model, one column per benchmark (sharex/sharey="col" share per benchmark); axes indexed [bench][model]."""
    fig, axes = plt.subplots(
        len(models),
        len(benches),
        figsize=(TEXT_WIDTH, 0.5 + row * len(models)),
        squeeze=False,
        **share,
    )
    return fig, axes.T


def bottom_legend(fig, handles: list, **kwargs) -> None:
    """One-row legend centred along the bottom edge of the figure."""
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=6.5,
        **kwargs,
    )


def finish(fig, handles: list, path: Path) -> None:
    """Legend across the bottom, then save."""
    bottom_legend(fig, handles)
    fig.tight_layout(rect=(0, 0.2 / fig.get_figheight(), 1, 1), h_pad=0.5, w_pad=0.3)
    save(fig, path)


def label_grid(
    axes, models: list[str], benches: list[str], ylabel: str, xlabel: str
) -> None:
    """Model name (split over two lines at its first hyphen when alone) and ylabel on each row, benchmark on each column; axes indexed [bench][model]."""
    for ax, model in zip(axes[0], models):
        ax.set_ylabel(
            f"{label_of(model)}\n{ylabel}"
            if ylabel
            else "-\n".join(label_of(model).split("-", 1))
        )
    for column, bench in zip(axes, benches):
        column[0].set_title(bench_label(bench))
        column[-1].set_xlabel(xlabel)
