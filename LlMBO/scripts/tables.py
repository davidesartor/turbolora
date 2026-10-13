"""Tables tab_*.tex and number macros numbers.tex: `uv run --no-project --with matplotlib --with numpy overleaf/scripts/tables.py` from the repo root."""

import json
from collections import defaultdict

import numpy as np

from budget import LARGE_GROUPS, pooled_curve
from common import (
    BASES,
    BENCHMARKS,
    GROUPS,
    INSTRUCTS,
    LABELS,
    LARGE,
    LARGE_U,
    LOW_U,
    MATHS,
    METRICS,
    METRIC_OBJECTIVE,
    MODELS,
    OBJECTIVE_METRIC,
    OTHERS,
    OUT,
    RUNS,
    PRIMARY,
    QWEN,
    SHIFT_MODELS,
    SMALL,
    SMALL_BASES,
    TABLE_BLOCKS,
    U_ALL,
    base_value,
    bench_label,
    budget_curve,
    gains,
    load_data,
    logo_label,
    label_of,
    logo_tex,
    mean_se,
    paired,
    pooled,
)


TASK_MACROS = {"gsm8k": "Gsm", "math500": "MathFive"}
GRPO_PASSES = 4  # forward-pass equivalents per completion token: generate, rescore, forward, backward (frozen weights: activation grads only)
KV_SHAPES = {  # (layers, KV heads, head dim), from each HF config
    "qwen2.5-1.5b": (28, 2, 128),
    "qwen2.5-7b": (28, 4, 128),
    "llama3.1-8b": (32, 8, 128),
    "mistral-7b": (32, 8, 128),
    "deepseek-7b-math": (30, 32, 128),
}
KV_BYTES_PER_TOKEN = {  # 2 (K,V) x layers x KV heads x head dim x 2 bytes
    kv: 2 * layers * heads * head_dim * 2 for kv, (layers, heads, head_dim) in KV_SHAPES.items()
}
ARCHITECTURES = {  # memory-table row -> (KV key, models pooled)
    "Qwen2.5-1.5B": (
        "qwen2.5-1.5b",
        ["qwen2.5-1.5b", "qwen2.5-1.5b-instruct", "qwen2.5-1.5b-math"],
    ),
    "Qwen2.5-7B": (
        "qwen2.5-7b",
        ["qwen2.5-7b", "qwen2.5-7b-instruct", "qwen2.5-7b-math"],
    ),
    "Llama3.1-8B": ("llama3.1-8b", ["llama3.1-8b"]),
    "Mistral-7B": ("mistral-7b", ["mistral-7b"]),
    "DeepSeek-7B": ("deepseek-7b-math", ["deepseek-7b-math"]),  # DeepSeek-Math-7B shares DeepSeek-7B's architecture; add deepseek-7b once its runs log peak memory
}
ENGINE_GIB = 2.0  # vLLM activation profile + CUDA graphs, ~1.9 GiB for the 7B engines in the job logs
SEQUENCE_TOKENS = 512 + 1024  # max prompt + max completion
GRPO_GROUP = 4  # completions GRPO samples per prompt
GRPO_MICRO_BATCH = GRPO_GROUP  # sequences GRPO passes forward and backward at a time (one group); its working set is measured at this size
DESIGN_STEPS = 20  # GRPO steps at the start of training, near the base model, for the prompt-correlation fit
DESIGN_BLOCKS = [TABLE_BLOCKS[0], MATHS, SMALL, INSTRUCTS]  # rows of the prompt-correlation table; DeepSeek-7B from its own runs
MIN_SEQUENCES = GRPO_MICRO_BATCH  # fewer sequences in flight than one GRPO micro-batch counts as not fitting
CARDS = {"24 GB": 22.03, "40 GB": 39.39, "48 GB": 44.39, "80 GB": 79.19}  # L4, A100, L40S, H100 usable GiB
CARD_GPUS = {"24 GB": "L4", "40 GB": "A100", "48 GB": "L40S", "80 GB": "H100"}
GPU_GIB = {
    "NVIDIA A100 80GB PCIe": 79.25,
    "NVIDIA A100-SXM4-80GB": 79.25,
    "NVIDIA H100 80GB HBM3": 79.19,
    "NVIDIA A100-PCIE-40GB": 39.39,
    "NVIDIA L40S": 44.39,
    "NVIDIA A40": 44.34,
    "NVIDIA L4": 22.03,
    "NVIDIA A16": 14.63,
    "NVIDIA RTX A4000": 15.72,
}
U_WORDS = dict(
    zip(U_ALL, ("One", "Two", "Four", "Eight", "Sixteen", "SixtyFour", "TwoFiftySix"))
)
GAP_TINT = 28  # % colour at full saturation in the gap tables
GAP_SATURATION = 2.0  # pt: paired gap at which the tint saturates
RESPONSIVE_GAIN = (
    3.0  # pt: TinyLoRA "moves" a model if GRPO gains this much over base at u <= 4
)


def responsive(baselines: dict, cells: dict) -> list[str]:
    """Models where GRPO gains RESPONSIVE_GAIN points on MATH-500 or GSM8K, either metric, at some u <= 4; GRPO only, so LlMBO does not pick its models."""
    return [
        model
        for model in MODELS
        if any(
            np.mean(gains(baselines, cells, model, "grpo", k, [u], g))
            >= RESPONSIVE_GAIN
            for k in (4, 1)
            for u in LOW_U
            for g in PRIMARY
            if cells[model]["grpo"].get(u)
        )
    ]


def reach(curve: list[tuple[float, float, float]]) -> float | None:
    """Completions from which the pooled curve stays at or above GRPO's final accuracy."""
    return next(
        (
            n
            for i, (n, _, _) in enumerate(curve)
            if all(mean >= 0 for _, mean, _ in curve[i:])
        ),
        None,
    )


# ---------------------------------------------------------------- tables


def pm(values: list[float], sign: bool = True) -> str:
    if not values:
        return "--"
    mean, se = mean_se(values)
    number = f"{round(mean, 1) + 0.0:+.1f}" if sign else f"{mean:.1f}"
    return rf"${number}$\,{{\scriptsize$\pm${se:.1f}}}"


def bold_if(text: str, condition: bool) -> str:
    return rf"\textbf{{\boldmath {text}}}" if condition else text


def tabular(spec: str, header: list[str], rows: list[str]) -> str:
    return "\n".join(
        [
            rf"\begin{{tabular}}{{{spec}}}",
            r"\toprule",
            *header,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def metric_table(
    baselines: dict,
    cells: dict,
    models: list[str],
    k: int,
    us: list[int],
    tag: str = "",
    block_starts: tuple[str, ...] = (),
    pair: tuple[str, str] | None = None,
    names: tuple[str, str] = ("GRPO", "LlMBO"),
) -> str:
    """Per model a base-accuracy row, then per u the gain over base of methods b and a (default GRPO and LlMBO trained on metric k) and their paired gap a − b, on every benchmark in metric k; a heavier rule above each of block_starts."""
    a, b = pair or (METRIC_OBJECTIVE[k], "grpo")
    rows = []
    for i, model in enumerate(models):
        if i:
            rows.append(r"\midrule[\heavyrulewidth]" if model in block_starts else r"\midrule")
        base = [rf"\multicolumn{{3}}{{c}}{{{base_value(baselines, model, k, g):.1f}}}" for g in BENCHMARKS]
        family, size = label_of(model).split("-", 1)
        stacked = rf"\scriptsize\begin{{tabular}}{{@{{}}c@{{}}}}{logo_tex(model)}~{family}\\ {size}\end{{tabular}}"
        vertical = rf"\multirow{{{1 + len(us)}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{stacked}}}}}"
        name = vertical if len(us) >= MIN_VERTICAL_U else logo_label(model)
        rows.append(" & ".join([name, "base", *base]) + r" \\")
        for u in us:
            row = ["", str(u)]
            for g in BENCHMARKS:
                means = [gains(baselines, cells, model, m + tag, k, [u], bench=g) for m in [b, a]]
                shown = [round(np.mean(v), 1) if v else None for v in means]
                row += [bold_if(signed(v), x == max(shown) and shown.count(x) == 1) if v else "--" for v, x in zip(means, shown)]
                row.append(shade(paired(cells, a + tag, b + tag, k, [u], [model], bench=g)))
            rows.append(" & ".join(row) + r" \\")
    header = [line.replace("Model & &", r" & $u$ &") for line in bench_metric_header([*names, r"$\Delta$"])]
    return tabular(bench_metric_spec(3).replace("l l", "c l", 1), header, rows)


def shade(diffs: list[float], chip: str = "gapchip", se: bool = True) -> str:
    """Paired gap cell in a rounded chip, tinted green (LlMBO ahead) or red by its mean, saturating at GAP_SATURATION points."""
    if not diffs:
        return "--"
    mean = np.mean(diffs)
    strength = round(GAP_TINT * min(abs(mean) / GAP_SATURATION, 1))
    color = f"green!{strength}" if mean > 0 else f"red!{strength}"
    text = pm(diffs) if se else f"${round(mean, 1) + 0.0:+.1f}$"
    return rf"\{chip}{{{color if strength else 'none'}}}{{{text}}}"


def bench_metric_header(metric_labels: list[str]) -> list[str]:
    """Header rows over two label columns: in-/out-of-distribution, then each benchmark split into one sub-column per metric."""
    n_k, n_id, n_ood = len(metric_labels), len(PRIMARY), len(BENCHMARKS) - len(PRIMARY)
    bench_cols = " & ".join(rf"\multicolumn{{{n_k}}}{{c}}{{{bench_label(g)}}}" for g in BENCHMARKS)
    bench_rules = "".join(rf"\cmidrule(lr){{{3 + n_k * b}-{2 + n_k * (b + 1)}}}" for b in range(len(BENCHMARKS)))
    return [
        rf" & & \multicolumn{{{n_k * n_id}}}{{c}}{{in-distribution}} & \multicolumn{{{n_k * n_ood}}}{{c}}{{out-of-distribution}} \\",
        rf"\cmidrule(lr){{3-{2 + n_k * n_id}}}\cmidrule(lr){{{3 + n_k * n_id}-{2 + n_k * (n_id + n_ood)}}}",
        *(
            [rf" & & {bench_cols} \\", bench_rules, rf"Model & & {' & '.join(metric_labels * len(BENCHMARKS))} \\"]
            if n_k > 1
            else [rf"Model & & {' & '.join(bench_label(g) for g in BENCHMARKS)} \\"]
        ),
    ]


def bench_metric_spec(n_k: int) -> str:
    return "l l " + "c" * (n_k * len(PRIMARY)) + r"@{\hspace{1.2em}}" + "c" * (n_k * (len(BENCHMARKS) - len(PRIMARY)))


METRIC_SUFFIX = {4: "avg4", 1: "greedy"}  # one results table per metric
MIN_VERTICAL_U = 4  # u rows a model block needs for its name to fit rotated
HEADLINE = (("bo", "grpo", 4), ("bo-greedy", "grpo", 1))  # LlMBO trained on each metric against GRPO


def headline_table(cells: dict, blocks: list[list[str]], comparisons: tuple[tuple[str, str, int], ...] = HEADLINE) -> str:
    """Paired gap a − b on metric k, one row block per u ≤ 4 with every model, on every benchmark and comparison (sub-columns); each group of models ends in its average row."""
    rows = []
    entries = [(i, label, group) for i, models in enumerate(blocks) for label, group in [*((logo_label(m), [m]) for m in models), (r"\emph{Avg}", models)]]
    for j, u in enumerate(LOW_U):
        if j:
            rows.append(r"\addlinespace[0.6em]\midrule[\heavyrulewidth]")
        for n, (i, label, group) in enumerate(entries):
            if n and i != entries[n - 1][0]:
                rows.append(rf"\cmidrule(l){{2-{2 + len(BENCHMARKS) * len(comparisons)}}}")
            elif label == r"\emph{Avg}":
                rows.append(r"\addlinespace[0.25em]")
            gaps = [
                shade(paired(cells, a, b, k, [u], group, bench=g), "gapdot", se=False)
                for g in BENCHMARKS
                for a, b, k in comparisons
            ]
            rows.append(" & ".join([rf"\multirow{{{len(entries)}}}{{*}}{{{u}}}" if n == 0 else "", label, *gaps]) + r" \\")
    labels = [METRICS[k].split(",")[0] for _, _, k in comparisons]
    header = [line.replace("Model & &", "$u$ & Model &") for line in bench_metric_header(labels)]
    return tabular(bench_metric_spec(len(labels)), header, rows)


def large_u_models(cells: dict, models: list[str]) -> list[str]:
    return [m for m in models if all(cells[m]["grpo"].get(u) for u in LARGE_U)]


def ablation_table(cells: dict, pools: dict[str, list[str]]) -> str:
    """Paired gaps per u pooled over each pool of models (row block): LlMBO − GRPO and LlMBO b=4 − b=1 (column groups), per task and metric, LlMBO trained on the metric."""
    metrics = METRIC_OBJECTIVE.items()
    vs_grpo = [(bo, "grpo", k, g) for g in PRIMARY for k, bo in metrics] + [("bo-greedy", "grpo", 1, None)]
    vs_b1 = [(f"{bo}-b4", bo, k, g) for g in PRIMARY for k, bo in metrics]
    rows = []
    for i, (pool, models) in enumerate(pools.items()):
        if i:
            rows.append(r"\midrule")
        for j, u in enumerate(LOW_U):
            gaps = [
                shade(paired(cells, a, b, k, [u], models, bench=g, train=g is None))
                for a, b, k, g in vs_grpo + vs_b1
            ]
            rows.append(" & ".join([pool if j == 0 else "", f"$u={u}$", *gaps]) + r" \\")

    # comparison groups, then tasks (the training tier only in greedy), then metrics
    n_grpo, n_b1, width = len(vs_grpo), len(vs_b1), len(METRIC_OBJECTIVE)
    labels = [METRICS[k].split(",")[0] for k in METRIC_OBJECTIVE]
    tasks = [rf"\multicolumn{{{width}}}{{c}}{{{bench_label(g)}}}" for g in PRIMARY]
    task_starts = [offset + width * t for offset in (3, 3 + n_grpo) for t in range(len(PRIMARY))]
    header = [
        rf" & & \multicolumn{{{n_grpo}}}{{c}}{{\llmbo{{}} $-$ GRPO}} & \multicolumn{{{n_b1}}}{{c}}{{\llmbo{{}}: $b{{=}}4$ $-$ $b{{=}}1$}} \\",
        rf"\cmidrule(lr){{3-{2 + n_grpo}}}\cmidrule(lr){{{3 + n_grpo}-{2 + n_grpo + n_b1}}}",
        " & ".join(["", "", *tasks, "train tier", *tasks]) + r" \\",
        "".join(rf"\cmidrule(lr){{{c}-{c + width - 1}}}" for c in task_starts) + rf"\cmidrule(lr){{{2 + n_grpo}-{2 + n_grpo}}}",
        " & ".join(["Models", "", *labels * len(PRIMARY), labels[-1], *labels * len(PRIMARY)]) + r" \\",
    ]
    return tabular("l l " + "c" * n_grpo + r"@{\hspace{1.2em}}" + "c" * n_b1, header, rows)


def memory(cells: dict) -> dict:
    """Per architecture: weights (GiB) and the GRPO/BO working set beyond vLLM's fixed reserve, from peak_vram_gb − share × card."""
    rows = {}
    for label, (kv, models) in ARCHITECTURES.items():
        overhead = defaultdict(list)
        for model in models:
            for method in ("lora", "loraxs", "grpo", "bo", "bo-greedy", "bo-b4"):
                for records in cells[model][method].values():
                    for r in records:
                        run = r["run"]
                        card = GPU_GIB.get(run.get("gpu"))
                        if card is None or run.get("peak_vram_gb") is None:
                            continue
                        share = (
                            0.85
                            if run["loss"] == "turbo"
                            else (0.5 if card < 60 else 0.45)
                        )
                        overhead["bo" if run["loss"] == "turbo" else "grpo"].append(
                            run["peak_vram_gb"] - share * card
                        )
        rows[label] = dict(
            kv=kv,
            weights=MODELS[models[0]][2] * 1e9 * 2 / 2**30,
            grpo=float(np.median(overhead["grpo"])),
            bo=max(float(np.median(overhead["bo"])), 0.0),
        )
    return rows


def sequences_in_flight(row: dict, card_gib: float, working_set_gib: float) -> int:
    """Full-length sequences that fit after weights and engine, each costing its KV cache plus its linear share of the working set; ≤0 = does not fit."""
    free_gib = card_gib - row["weights"] - ENGINE_GIB
    per_sequence = SEQUENCE_TOKENS * KV_BYTES_PER_TOKEN[row["kv"]] + working_set_gib / GRPO_MICRO_BATCH * 2**30
    return int(free_gib * 2**30 // per_sequence)


def equal_rewards(pass_rate: float, concentration: float) -> float:
    """P(all GRPO_GROUP completions of a prompt agree) when its pass rate is Beta(pass_rate·c, (1 − pass_rate)·c)."""
    a, b = pass_rate * concentration, (1 - pass_rate) * concentration
    rising = lambda x: np.prod([x + i for i in range(GRPO_GROUP)])
    return (rising(a) + rising(b)) / rising(concentration)


def prompt_correlation(model: str) -> dict:
    """Pass rate and share of equal-reward groups over GRPO's first steps at u ≤ 4; ρ = 1/(c + 1) of the beta-binomial matching both."""
    family = MODELS[model][0]
    per_run = []
    for u in LOW_U:
        for curves in (RUNS / family / model / "tinylora-grpo" / f"r2-u{u}").glob("seed*/curves.jsonl"):
            steps = [row for line in curves.open() if "frac_reward_zero_std" in (row := json.loads(line))][:DESIGN_STEPS]
            per_run.append([np.mean([row[key] for row in steps]) for key in ("reward", "frac_reward_zero_std")])
    pass_rate, equal = np.mean(per_run, axis=0)
    independent = pass_rate**GRPO_GROUP + (1 - pass_rate) ** GRPO_GROUP
    # equal_rewards falls from 1 (c → 0) to `independent` (c → ∞): bisect log c
    low, high = -10.0, 10.0
    for _ in range(100):
        mid = (low + high) / 2
        low, high = (mid, high) if equal_rewards(pass_rate, np.exp(mid)) > equal else (low, mid)
    rho = 1 / (np.exp(mid) + 1) if equal > independent else 0.0
    design = 1 + (GRPO_GROUP - 1) * rho
    return dict(pass_rate=pass_rate, equal=equal, independent=independent, rho=rho, design=design,
                effective=GRPO_GROUP / design, runs=len(per_run))


def design_table(rows: dict) -> str:
    """Per model: pass rate, equal-reward groups observed and if independent, ρ, design effect and effective completions per group."""
    lines = []
    for block in DESIGN_BLOCKS:
        if lines:
            lines.append(r"\midrule")
        for model in block:
            r = rows[model]
            lines.append(" & ".join([
                logo_label(model), f"{100 * r['pass_rate']:.0f}", f"{100 * r['equal']:.0f}", f"{100 * r['independent']:.0f}",
                f"{r['rho']:.2f}", f"{r['design']:.2f}", f"{r['effective']:.1f}",
            ]) + r" \\")
    return tabular("l c cc c c c", [
        r" & & \multicolumn{2}{c}{equal-reward groups (\%)} & & & \\",
        r"\cmidrule(lr){3-4}",
        rf"Model & pass rate (\%) & observed & independent & $\rho$ & $1 + {GRPO_GROUP - 1}\rho$ & $k_\mathrm{{eff}}$ \\",
    ], lines)


def memory_table(rows: dict) -> str:
    """Per architecture: KV shape, weights, KV cache and working set per method, sequences in flight per card; bold: the better method, ✗: does not fit."""
    lines = []
    for label, r in rows.items():
        layers, heads, _ = KV_SHAPES[r["kv"]]
        kv_mib = SEQUENCE_TOKENS * KV_BYTES_PER_TOKEN[r["kv"]] / 2**20
        overheads = [bold_if(f"{r[m]:.1f}", r[m] == min(r["grpo"], r["bo"])) for m in ("grpo", "bo")]
        fits = []
        for card in CARDS.values():
            seqs = {m: sequences_in_flight(r, card, r[m]) for m in ("grpo", "bo")}
            fits += [bold_if(str(n), n == max(seqs.values())) if n >= MIN_SEQUENCES else r"\ding{55}" for n in seqs.values()]
        lines.append(
            " & ".join([
                f"{logo_tex(ARCHITECTURES[label][1][0])}~{label}", rf"{layers}\,$\times$\,{heads}",
                f"{r['weights']:.1f}", f"{kv_mib:.0f}", *overheads, *fits,
            ])
            + r" \\"
        )
    n_cards = len(CARDS)
    card_header = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{\begin{{tabular}}{{@{{}}c@{{}}}}{CARD_GPUS[name]}\\{name}\end{{tabular}}}}" for name in CARDS
    )
    card_rules = "".join(rf"\cmidrule(lr){{{7 + 2 * i}-{8 + 2 * i}}}" for i in range(n_cards))
    return tabular(
        "l c cc cc" + " cc" * n_cards,
        [
            rf" & & & & \multicolumn{{2}}{{c}}{{beyond vLLM}} & \multicolumn{{{2 * n_cards}}}{{c}}{{sequences in flight}} \\",
            rf"\cmidrule(lr){{5-6}}\cmidrule(lr){{7-{6 + 2 * n_cards}}}",
            rf" & layers $\times$ & weights & KV/seq. & \multicolumn{{2}}{{c}}{{(GiB)}} & {card_header} \\",
            card_rules,
            r"Architecture & KV heads & (GiB) & (MiB) & GRPO & LlMBO" + " & GRPO & LlMBO" * n_cards + r" \\",
        ],
        lines,
    )



CHIP_CARDS = [  # (GPU, GB on the label, usable GiB)
    ("L4", "24", CARDS["24 GB"]),
    ("L40S", "48", CARDS["48 GB"]),
    ("A100", "40", CARDS["40 GB"]),
    ("A100", "80", GPU_GIB["NVIDIA A100-SXM4-80GB"]),
    ("H100", "80", CARDS["80 GB"]),
]


def fit_cell(bo: int, grpo: int) -> str:
    """"LlMBO / GRPO" cell; LlMBO chip tinted by |log2| of the ratio (saturating at 32×), colours baked in so Overleaf needs only \\fitdot."""
    if grpo < MIN_SEQUENCES:
        return rf"\fitdot{{green!15}}{{{bo}}}\,/\,\textcolor{{red!70}}{{\ding{{55}}}}"
    tint = round(15 * min(abs(np.log(bo / grpo)) / np.log(32), 1))
    color = "none" if tint == 0 else f"{'green' if bo > grpo else 'red'}!{tint}"
    return rf"\fitdot{{{color}}}{{{bo}}}\,/\,{grpo}"


def chips_table(rows: dict) -> str:
    """Table 2: per 7-8B architecture (rows) and GPU (columns), cells "LlMBO / GRPO" sequences in flight."""
    lines = []
    for label, r in rows.items():
        if label == "Qwen2.5-1.5B":
            continue
        cells = []
        for _, _, card in CHIP_CARDS:
            bo, grpo = (sequences_in_flight(r, card, r[m]) for m in ("bo", "grpo"))
            cells.append(fit_cell(bo, grpo))
        lines.append(" & ".join([f"{logo_tex(ARCHITECTURES[label][1][0])}~{label}", *cells]) + r" \\")
    heads = " & ".join(rf"\textbf{{{gpu}}} {gb}\,GB" for gpu, gb, _ in CHIP_CARDS)
    return tabular("@{}l" + "c" * len(CHIP_CARDS) + "@{}", [rf"Architecture & {heads} \\"], lines)


# ---------------------------------------------------------------- numbers


def signed(values: list[float]) -> str:
    return rf"${round(np.mean(values), 1) + 0.0:+.1f}$"


def with_se(values: list[float]) -> str:
    mean, se = mean_se(values)
    return rf"${round(mean, 1) + 0.0:+.1f}\pm{se:.1f}$"


def delta_with_se(a: list[float], b: list[float]) -> str:
    """Difference of two independent means, a − b, with the s.e. of the difference."""
    (mean_a, se_a), (mean_b, se_b) = mean_se(a), mean_se(b)
    return rf"${round(mean_a - mean_b, 1) + 0.0:+.1f}\pm{np.hypot(se_a, se_b):.1f}$"


def macros(baselines: dict, cells: dict, mem: dict, design: dict) -> dict[str, str]:
    """Numbers quoted in the text, as \\newcommand names without the backslash; every per-cell number is per u and task group."""
    out = {}
    objectives = {"bo": "Tone", "bo-greedy": "Greedy", "grpo": "Grpo"}
    metric_names = {4: "Mean", 1: "Greedy"}
    trainings = {"": "", "-math": "MathTrain"}
    out["nRuns"] = str(
        sum(len(v) for m in MODELS for c in cells[m].values() for v in c.values())
    )
    out["nModels"] = str(len(MODELS))
    out["nLargeModels"] = str(len(large_u_models(cells, QWEN)))
    for model in MODELS:
        name = MODELS[model][3]
        for g, group in TASK_MACROS.items():
            for k, metric in metric_names.items():
                out[f"base{metric}{name}{group}"] = (
                    f"{base_value(baselines, model, k, g):.1f}"
                )
                for u in U_ALL:
                    cell = f"{name}U{U_WORDS[u]}{group}"
                    for tag, train in trainings.items():
                        # gain over base: \gainToneMeanQwenUTwoMathFive
                        for m, obj in objectives.items():
                            if gain := gains(
                                baselines, cells, model, m + tag, k, [u], g
                            ):
                                out[f"gain{obj}{metric}{cell}{train}"] = signed(gain)
                        # paired gap to GRPO: \gapToneMeanQwenUTwoMathFive
                        for m in ("bo", "bo-greedy"):
                            diffs = paired(
                                cells, m + tag, "grpo" + tag, k, [u], [model], g
                            )
                            if diffs:
                                obj = f"{objectives[m]}{metric}{cell}{train}"
                                out[f"gap{obj}"] = with_se(diffs)
                        # batch-4 paired gap to GRPO: \gapBFourToneMeanQwenUTwoMathFive
                        for m in ("bo", "bo-greedy"):
                            if not tag and (
                                diffs := paired(
                                    cells, m + "-b4", "grpo", k, [u], [model], g
                                )
                            ):
                                out[f"gapBFour{objectives[m]}{metric}{cell}"] = with_se(
                                    diffs
                                )
        # greedy accuracy on the training problems: \trainGapGreedyMathSmallUTwo
        for u in LOW_U:
            if train := paired(cells, "bo-greedy", "grpo", 1, [u], [model], train=True):
                out[f"trainGapGreedy{name}U{U_WORDS[u]}"] = with_se(train)
    # completion budget per u and task: where each LlMBO curve first reaches GRPO's final greedy accuracy, and where it ends
    for model in SHIFT_MODELS:
        for tag, train in trainings.items():
            for m in ("bo", "bo-greedy"):
                for u in LOW_U:
                    for g, group in TASK_MACROS.items():
                        curve = budget_curve(cells, model, m, tag, u, g)
                        if not curve:
                            continue
                        name = f"{objectives[m]}{MODELS[model][3]}U{U_WORDS[u]}{group}{train}"
                        if (n := reach(curve)) is not None:
                            out[f"reach{name}"] = (
                                f"{n:.0f}" if n < 1e3 else f"{n / 1e3:.0f}k"
                            )
                        out[f"budgetEnd{name}"] = rf"${curve[-1][1]:+.1f}$"
    # fig_budget_main, pooled over the 7-8B families and u <= 4: \reachPoolGreedyGsm, \poolEndGreedyGsm, \poolEndGrpoGsm
    for g, group in TASK_MACROS.items():
        bo_curve = pooled_curve(baselines, cells, LARGE_GROUPS, "bo-greedy", "", 1, LOW_U, g)
        grpo_curve = pooled_curve(baselines, cells, LARGE_GROUPS, "grpo", "", 1, LOW_U, g)
        grpo_end = grpo_curve[1][-1]
        if (n := reach([(n, mean - grpo_end, se) for n, mean, se in bo_curve.T])) is not None:
            out[f"reachPoolGreedy{group}"] = f"{n:.0f}" if n < 1e3 else f"{n / 1e3:.0f}k"
        out[f"poolEndGreedy{group}"] = rf"${bo_curve[1][-1]:+.1f}$"
        out[f"poolEndGrpo{group}"] = rf"${grpo_end:+.1f}$"
    grpo_n = [r["completions"] for m in MODELS for r in pooled(cells, m, "grpo", LOW_U)]
    bo_n = [r["completions"] for m in MODELS for r in pooled(cells, m, "bo", LOW_U)]
    out["grpoCompletions"] = f"{np.median(grpo_n) / 1e3:.0f}k"
    out["boCompletions"] = f"{np.median(bo_n) / 1e3:.0f}k"
    # training FLOPs of LlMBO as a share of GRPO's: LlMBO only generates
    out["grpoPasses"] = str(GRPO_PASSES)
    out["boComputeShare"] = (
        f"{100 * np.median(bo_n) / (GRPO_PASSES * np.median(grpo_n)):.0f}"
    )
    # ablations per u, pooled over a model group: \batchFourToneMeanUTwoMathFive, \crossGreedyMeanUTwoMathFiveSmall
    for suffix, group_models in (("", LARGE), ("Small", SMALL)):
        for k, metric in metric_names.items():
            for u in LOW_U:
                for g, group in TASK_MACROS.items():
                    cell = f"{metric}U{U_WORDS[u]}{group}{suffix}"
                    for name, a, b in (
                        ("batchFourTone", "bo-b4", "bo"),
                        ("batchFourGreedy", "bo-greedy-b4", "bo-greedy"),
                        ("crossTone", "bo", "grpo"),
                        ("crossGreedy", "bo-greedy", "grpo"),
                    ):
                        if diffs := paired(cells, a, b, k, [u], group_models, g):
                            out[f"{name}{cell}"] = signed(diffs)
    # paired gap per u and task pooled over a model group: \\poolToneMeanUTwoMathFive, \\poolToneMeanUTwoMathFiveSmall
    pools = {
        "": ("", LARGE),
        "Base": ("", BASES),
        "MathVariants": ("", MATHS),
        "Small": ("", SMALL),
        "ShiftLarge": ("", SHIFT_MODELS[:2]),
        "ShiftSmall": ("", SHIFT_MODELS[2:]),
        "ShiftLargeMathTrain": ("-math", SHIFT_MODELS[:2]),
        "ShiftSmallMathTrain": ("-math", SHIFT_MODELS[2:]),
        "LargeU": ("", large_u_models(cells, QWEN)),
        "LargeUSmall": ("", large_u_models(cells, SMALL)),
    }
    for suffix, group_models in (("", LARGE), ("Small", SMALL)):
        for u in LOW_U:
            if train := paired(
                cells, "bo-greedy", "grpo", 1, [u], group_models, train=True
            ):
                out[f"trainGapGreedyU{U_WORDS[u]}{suffix}"] = with_se(train)
    # LlMBO greedy objective − T=1 objective, greedy accuracy, same seeds: \objectiveUTwoGsmBase
    for suffix, group_models in (("", LARGE), ("Base", BASES), ("MathVariants", MATHS), ("Small", SMALL)):
        for u in LOW_U:
            for g, group in TASK_MACROS.items():
                if diffs := paired(cells, "bo-greedy", "bo", 1, [u], group_models, g):
                    out[f"objectiveU{U_WORDS[u]}{group}{suffix}"] = with_se(diffs)
    for model in MODELS:
        for u in LOW_U:
            for g, group in TASK_MACROS.items():
                if diffs := paired(cells, "bo-greedy", "bo", 1, [u], [model], g):
                    out[f"objective{MODELS[model][3]}U{U_WORDS[u]}{group}"] = signed(diffs)
    pool_methods = {"bo": "Tone", "bo-greedy": "Greedy", "bo-b4": "BFourTone"}
    for suffix, (tag, group_models) in pools.items():
        for k, metric in metric_names.items():
            for m, obj in pool_methods.items():
                for u in U_ALL:
                    for g, group in TASK_MACROS.items():
                        diffs = paired(
                            cells, m + tag, "grpo" + tag, k, [u], group_models, g
                        )
                        if diffs:
                            cell = f"{obj}{metric}U{U_WORDS[u]}{group}{suffix}"
                            out[f"pool{cell}"] = with_se(diffs)
    # paired gap on MATH alone minus on the tiers: \shiftDeltaToneMeanUFourMathFiveLarge
    for suffix, group_models in (("Large", SHIFT_MODELS[:2]), ("Small", SHIFT_MODELS[2:])):
        for k, metric in metric_names.items():
            for m, obj in pool_methods.items():
                for u in LOW_U:
                    for g, group in TASK_MACROS.items():
                        tiers = paired(cells, m, "grpo", k, [u], group_models, g)
                        alone = paired(
                            cells, m + "-math", "grpo-math", k, [u], group_models, g
                        )
                        if tiers and alone:
                            cell = f"{obj}{metric}U{U_WORDS[u]}{group}{suffix}"
                            out[f"shiftDelta{cell}"] = delta_with_se(alone, tiers)
    # LlMBO's in-search score of its pick minus its held-out training-set accuracy, u <= 4, models with both b = 1 and b = 4 evaluated: \searchBiasBFourTone
    for b1, b4, k, obj in (("bo", "bo-b4", 4, "Tone"), ("bo-greedy", "bo-greedy-b4", 1, "Greedy")):
        bias = {
            m: {
                model: [
                    r["search"]["pick"] - r["train"][k]
                    for r in pooled(cells, model, m, LOW_U)
                    if r["search"] and k in r["train"]
                ]
                for model in MODELS
            }
            for m in (b1, b4)
        }
        both = [model for model in MODELS if bias[b1][model] and bias[b4][model]]
        for m, name in ((b1, obj), (b4, f"BFour{obj}")):
            out[f"searchBias{name}"] = with_se([d for model in both for d in bias[m][model]])
        out[f"searchBiasModels{obj}"] = str(len(both))
    # u in LARGE_U: the worst seed pair per task over those models and sizes
    for suffix, group_models in (("", QWEN), ("Small", SMALL)):
        for k, metric in metric_names.items():
            for m in ("bo", "bo-greedy"):
                for g, group in TASK_MACROS.items():
                    diffs = paired(
                        cells,
                        m,
                        "grpo",
                        k,
                        LARGE_U,
                        large_u_models(cells, group_models),
                        g,
                    )
                    out[f"worst{objectives[m]}{metric}Large{group}{suffix}"] = (
                        f"{min(diffs):+.1f}"
                    )
    # other families: best GRPO gain at any u <= 4 on MATH-500 or GSM8K, against LoRA r=32
    out["othersTinyBest"] = (
        f"{max(np.mean(gains(baselines, cells, model, 'grpo', k, [u], g)) for model in OTHERS for k in (4, 1) for u in LOW_U for g in PRIMARY):.1f}"
    )
    for g, group in TASK_MACROS.items():
        lora = [np.mean(gains(baselines, cells, m, "lora", 1, [32], g)) for m in OTHERS]
        out[f"othersLora{group}"] = f"{min(lora):.0f}--{max(lora):.0f}"
    # LoRA r=32 under GRPO, the top of the scaling figures: \loraGainMeanQwenMathFive
    for model in MODELS:
        for g, group in TASK_MACROS.items():
            for k, metric in metric_names.items():
                if gain := gains(baselines, cells, model, "lora", k, [32], g):
                    out[f"loraGain{metric}{MODELS[model][3]}{group}"] = signed(gain)
    # memory
    small, large = (
        mem["Qwen2.5-1.5B"],
        [r for label, r in mem.items() if label != "Qwen2.5-1.5B"],
    )
    out["grpoOverheadSmall"] = f"{small['grpo']:.1f}"
    out["grpoOverheadLarge"] = (
        f"{min(r['grpo'] for r in large):.1f}--{max(r['grpo'] for r in large):.1f}"
    )
    out["boOverheadMax"] = f"{max(r['bo'] for r in mem.values()):.1f}"
    out["grpoOverheadLargeMax"] = f"{max(r['grpo'] for r in large):.1f}"
    out["grpoOverheadDeepseek"] = f"{mem['DeepSeek-7B']['grpo']:.1f}"
    # worked example of the sequences column: Qwen2.5-7B on the 24 GB card
    qwen = mem["Qwen2.5-7B"]
    out["grpoOverheadQwen"] = f"{qwen['grpo']:.1f}"
    out["memExampleFree"] = f"{CARDS['24 GB'] - qwen['weights'] - ENGINE_GIB:.1f}"
    out["memExampleSeqMiB"] = f"{SEQUENCE_TOKENS * KV_BYTES_PER_TOKEN[qwen['kv']] / 2**20:.0f}"
    out["memExampleSeqs"] = str(sequences_in_flight(qwen, CARDS["24 GB"], qwen["bo"]))
    out["memExampleGrpoPerSeq"] = f"{qwen['grpo'] / GRPO_MICRO_BATCH:.1f}"
    out["memExampleGrpoSeqs"] = str(sequences_in_flight(qwen, CARDS["24 GB"], qwen["grpo"]))
    out["grpoMicroBatch"] = str(GRPO_MICRO_BATCH)
    for word, name in (("Small", "24 GB"), ("Forty", "40 GB"), ("Large", "48 GB"), ("Eighty", "80 GB")):
        out[f"card{word}GiB"] = f"{CARDS[name]:.1f}"
    out["engineGiB"] = f"{ENGINE_GIB:.0f}"
    # one completion per prompt: k_eff of GRPO's groups on the Qwen2.5 base and Math models, equal-reward groups elsewhere
    effective = [design[m]["effective"] for m in QWEN + SMALL]
    out["designEffectiveQwen"] = f"{min(effective):.1f}--{max(effective):.1f}"
    out["designEffectMax"] = f"{max(r['design'] for r in design.values()):.1f}"
    equal = [100 * design[m]["equal"] for m in ["llama3.1-8b", "mistral-7b", "deepseek-7b", "deepseek-7b-math"]]
    out["designEqualOthers"] = f"{min(equal):.0f}--{max(equal):.0f}"
    out["designSteps"] = str(DESIGN_STEPS)
    out["grpoGroup"] = str(GRPO_GROUP)
    out["nLargeArch"] = str(len(large))
    out["grpoNoFitSmallCard"] = str(
        sum(sequences_in_flight(r, CARDS["24 GB"], r["grpo"]) < MIN_SEQUENCES for r in large)
    )
    out["grpoSequencesSmallCard"] = str(
        max(n for r in large if (n := sequences_in_flight(r, CARDS["24 GB"], r["grpo"])) < MIN_SEQUENCES)
    )
    small_card = [sequences_in_flight(r, CARDS["24 GB"], r["bo"]) for r in large]
    out["boSequencesSmallCard"] = f"{min(small_card)}--{max(small_card)}"
    ratio = [
        sequences_in_flight(r, CARDS["48 GB"], r["bo"])
        / sequences_in_flight(r, CARDS["48 GB"], r["grpo"])
        for r in large
    ]
    out["boSequenceRatio"] = f"{min(ratio):.1f}--{max(ratio):.0f}"
    return out


def main(baselines: dict, cells: dict) -> None:
    print("TinyLoRA moves:", responsive(baselines, cells))
    mem = memory(cells)
    design = {model: prompt_correlation(model) for block in DESIGN_BLOCKS for model in block}
    tables = {
        "tab_memory": memory_table(mem),
        "tab_chips": chips_table(mem),
        "tab_design": design_table(design),
        "tab_ablations": ablation_table(cells, {"7-8B models": LARGE, "1.5B models": SMALL}),
        "tab_headline": headline_table(cells, TABLE_BLOCKS),
        "tab_objective": metric_table(
            baselines, cells, [m for block in [*TABLE_BLOCKS, SMALL] for m in block], 1, [u for u in U_ALL if u not in LARGE_U],
            block_starts=(TABLE_BLOCKS[1][0], SMALL[0]), pair=("bo-greedy", "bo"), names=("avg@4", "greedy"),
        ),
    }
    mid_u = [u for u in U_ALL if u not in LARGE_U]
    for k, suffix in METRIC_SUFFIX.items():
        for name, blocks in GROUPS.items():
            if name == "main":
                blocks = TABLE_BLOCKS
            models = [model for block in blocks for model in block]
            starts = tuple(block[0] for block in blocks[1:])
            tables[f"tab_{name}_{suffix}"] = metric_table(baselines, cells, models, k, mid_u, block_starts=starts)
        tables[f"tab_shift_{suffix}"] = metric_table(baselines, cells, SHIFT_MODELS, k, mid_u, tag="-math", block_starts=(SHIFT_MODELS[2],))
        tables[f"tab_large_u_{suffix}"] = metric_table(baselines, cells, large_u_models(cells, QWEN), k, LARGE_U)
    for name, table in tables.items():
        (OUT / f"{name}.tex").write_text(table + "\n")
    numbers = macros(baselines, cells, mem, design)
    (OUT / "numbers.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in numbers.items())
    )
    print(len(numbers), "macros")


if __name__ == "__main__":
    main(*load_data())
