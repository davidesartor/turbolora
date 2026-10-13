"""Table tab_snr.tex and macros numbers_snr.tex: what the GP believes at its pick vs held-out, per LlMBO_T=1 run at u <= 4.

Refits each run's final GP, so it needs the project env: `uv run --with matplotlib overleaf/scripts/snr.py` from the repo root.
"""

import json
import math

import numpy as np
import torch

from common import INSTRUCTS, LOW_U, MODELS, OUT, PRIMARY, RUNS, bench_label, gains, load_data, logo_label, mean_se
from tables import U_WORDS, pm, tabular, with_se
from turbolora.bo import fit_gp

CONTRAST = ["qwen2.5-7b"]  # a model TinyLoRA moves, where the GP's belief can be checked against real gains
N_BASELINE = 8  # theta = 0 replicates in every search design
PRECISION = [1.0, 0.5]  # pt: gaps a search would need to resolve


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def fraction(values: list[float]) -> str:
    mean, se = mean_se(values)
    return rf"${mean:.2f}$\,{{\scriptsize$\pm${se:.2f}}}"


def run_stats(seed_dir) -> dict | None:
    """Base-model train pass rate, per-evaluation s.e. and GP-believed gain at the pick (pt), and |theta|_inf / box half-width."""
    run_path, trials_path, state_path = (seed_dir / f for f in ("run.json", "trials.json", "turbo.json"))
    if not all(p.is_file() for p in (run_path, trials_path, state_path)):
        return None
    run, trials, state = (json.loads(p.read_text()) for p in (run_path, trials_path, state_path))
    if "theta" not in run:
        return None
    half_width, dim = run["theta_range"], len(run["theta"])

    # the region-scaled GP over all trials in unit-cube coordinates, as turbo.region_pick refits it
    unit_trials = [dict(t, theta=[(x + half_width) / (2 * half_width) for x in t["theta"]]) for t in trials]
    unit_bounds = torch.tensor([[0.0] * dim, [1.0] * dim], dtype=torch.float64)
    gp = fit_gp(unit_trials, unit_bounds, state["length"])
    with torch.no_grad():
        means = gp.posterior(torch.tensor([t["theta"] for t in unit_trials], dtype=torch.float64)).mean.squeeze(-1)
        mean_at_zero = gp.posterior(torch.full((1, dim), 0.5, dtype=torch.float64)).mean.item()
    pick = int(means.argmax())

    p0 = sigmoid(run["baseline_logit"])
    return dict(
        seed=int(seed_dir.name[4:]),
        p0=100 * p0,
        sem=100 * float(np.median([t["sem"] for t in trials])) * p0 * (1 - p0),
        believed=100 * (sigmoid(means[pick].item()) - sigmoid(mean_at_zero)),
        theta=max(abs(x) for x in trials[pick]["theta"]) / half_width,
        trials=len(trials),
    )


def main(baselines: dict, cells: dict) -> None:
    models = INSTRUCTS + CONTRAST
    stats = {
        (model, u): [
            s
            for d in sorted((RUNS / MODELS[model][0] / model / "tinylora-turbo" / f"r2-u{u}-b1-t1").glob("seed*"))
            if (s := run_stats(d))
        ]
        for model in models
        for u in LOW_U
    }

    # one row per model and u: train-side noise and belief, then held-out gain of the same runs
    rows = []
    for i, model in enumerate(models):
        if i:
            rows.append(r"\midrule")
        for u in LOW_U:
            runs = stats[model, u]
            seeds = {s["seed"] for s in runs}
            held = [
                [g for r, g in zip(cells[model]["bo"][u], gains(baselines, cells, model, "bo", 4, [u], b)) if r["seed"] in seeds]
                for b in PRIMARY
            ]
            name = logo_label(model) if u == LOW_U[0] else ""
            rows.append(
                " & ".join(
                    [name, str(u)]
                    + [pm([s["p0"] for s in runs], sign=False), f"${np.mean([s['sem'] for s in runs]):.1f}$"]
                    + [pm([s["believed"] for s in runs])]
                    + [fraction([s["theta"] for s in runs])]
                    + [pm(h) for h in held]
                )
                + r" \\"
            )
    table = tabular(
        "l c cccc cc",
        [
            r" & & \multicolumn{4}{c}{training distribution} & \multicolumn{2}{c}{held-out gain} \\",
            r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
            r"Model & $u$ & base (\%) & s.e./eval. & GP gain & $\|\theta\|_\infty / R$ & "
            + " & ".join(bench_label(b) for b in PRIMARY)
            + r" \\",
        ],
        rows,
    )
    (OUT / "tab_snr.tex").write_text(table + "\n")

    # macros quoted in the instruct appendix
    instruct_runs = [s for m in INSTRUCTS for u in LOW_U for s in stats[m, u]]
    sems = [float(np.median([s["sem"] for u in LOW_U for s in stats[m, u]])) for m in INSTRUCTS]
    sigma = float(np.median([s["sem"] for s in instruct_runs]))
    out = {
        "snrSemInstruct": f"{min(sems):.1f}--{max(sems):.1f}",
        "snrReplicateSem": f"{sigma / math.sqrt(N_BASELINE):.1f}",
        "snrTrials": f"{np.median([s['trials'] for s in instruct_runs]):.0f}",
        "grpoBestInstruct": f"{max(np.mean(gains(baselines, cells, m, 'grpo', 4, [u], g)) for m in INSTRUCTS for u in LOW_U for g in PRIMARY):.1f}",
    }
    for delta, word in zip(PRECISION, ("One", "Half")):
        out[f"snrEvals{word}"] = f"{8 * sigma**2 / delta**2:.0f}"  # replicates per point so that 2 s.e. of a paired difference = delta
    for model in models:
        name = MODELS[model][3]
        for u in LOW_U:
            runs = stats[model, u]
            out[f"snrBelieved{name}U{U_WORDS[u]}"] = with_se([s["believed"] for s in runs])
            out[f"snrTheta{name}U{U_WORDS[u]}"] = f"{mean_se([s['theta'] for s in runs])[0]:.2f}"
    (OUT / "numbers_snr.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in out.items()))
    print(out)


if __name__ == "__main__":
    main(*load_data())
