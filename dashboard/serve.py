"""Serve the dashboard with live data from baseline evals (outputs/baselines/<model>/<task>.json) and training runs (any outputs/**/run.json or checkpoint-*/, incl. still-running ones): `uv run dashboard/serve.py`, then open http://localhost:8000."""

import argparse
import gzip
import hashlib
import http.server
import json
import re
import threading
import time
from pathlib import Path

from turbolora.models import MODELS
from turbolora.tasks import TASKS

TEMPLATE = Path(__file__).with_name("template.html")

# SimpleRL-Zoo Table 2, base models before RL (greedy pass@1)
PAPER = {
    "qwen2.5-0.5b": dict(
        gsm8k=36.7, math500=16.0, minerva=4.4, olympiad=2.8, aime24=0.0, amc23=5.0
    ),
    "qwen2.5-7b": dict(
        gsm8k=88.2, math500=64.6, minerva=25.7, olympiad=30.1, aime24=3.3, amc23=30.0
    ),
}

# TinyLoRA Table 1: accuracy by trainable parameters (0 = untrained, last row = full fine-tune); columns gsm8k, math500, minerva, olympiad, aime24, amc23
PAPER_TABLE = {
    "qwen2.5-3b-instruct": [
        (0, 76.0, 55.0, 18.5, 21.3, 2.1, 23.4),
        (16, 80.9, 64.0, 19.9, 23.0, 3.0, 31.5),
        (63, 85.3, 64.1, 20.1, 26.6, 7.3, 36.0),
        (252, 85.4, 66.4, 28.3, 29.3, 13.3, 47.5),
        (504, 86.1, 66.6, 28.7, 30.8, 16.7, 47.5),
        (8_064, 87.2, 67.8, 28.3, 30.7, 10.0, 47.5),
        (129_024, 86.7, 67.8, 29.4, 32.3, 10.0, 55.0),
        (3_085_846_528, 87.0, 69.0, 31.7, 33.1, 15.0, 52.2),
    ],
    "qwen2.5-7b-instruct": [
        (0, 88.2, 64.6, 25.7, 30.1, 3.3, 30.0),
        (13, 91.8, 74.6, 27.1, 36.3, 16.0, 54.5),
        (49, 91.5, 74.2, 26.6, 37.2, 12.6, 55.5),
        (196, 92.2, 76.6, 37.1, 38.8, 16.7, 57.5),
        (392, 92.2, 77.0, 35.7, 40.1, 16.7, 65.0),
        (6_272, 91.9, 78.0, 37.5, 41.0, 16.7, 57.5),
        (100_352, 92.8, 78.0, 37.1, 43.3, 16.7, 60.0),
        (7_615_487_488, 91.7, 78.2, 38.6, 40.4, 20.0, 62.5),
    ],
    "qwen2.5-7b-math": [
        (0, 65.5, 63.6, 12.5, 25.8, 13.3, 42.5),
        (196, 72.5, 62.0, 26.5, 31.4, 20.0, 50.0),
        (392, 86.0, 74.8, 31.6, 37.9, 26.7, 60.0),
        (6_272, 87.0, 77.4, 28.7, 40.0, 16.7, 67.5),
        (100_352, 87.0, 78.6, 32.7, 39.0, 30.0, 62.5),
        (7_615_487_488, 90.2, 80.2, 37.5, 39.0, 40.0, 70.0),
    ],
}
TABLE_TASKS = ["gsm8k", "math500", "minerva", "olympiad", "aime24", "amc23"]
EVAL_TASKS = [t for t in TASKS if not t.startswith(("easy", "medium", "hard"))]

# TinyLoRA Fig. 1: Qwen2.5-7B on GSM8K, accuracy vs trainable parameters (digitized; TinyLoRA <1k params, LoRA-XS <1M, LoRA above)
PAPER_CURVES = {
    "qwen2.5-7b": dict(
        gsm8k=[
            (1.00, 79.90),
            (1.81, 84.26),
            (3.22, 85.37),
            (5.16, 84.32),
            (8.60, 90.70),
            (14.79, 92.22),
            (25.95, 92.92),
            (32.67, 92.14),
            (46.60, 93.59),
            (82.96, 94.86),
            (469.9, 95.38),
            (1044.7, 95.00),
            (2656.6, 96.01),
            (3317.3, 95.59),
            (15286, 96.21),
            (33338, 96.63),
            (84551, 96.42),
            (107591, 96.37),
            (221406, 97.22),
            (382271, 96.72),
            (497148, 96.78),
            (1264202, 96.74),
            (2702661, 97.44),
            (3976230, 96.22),
            (7056871, 96.37),
            (12532409, 96.79),
        ]
    ),
}
# adapter used at each point on the paper curve, by parameter count
PAPER_ADAPTER = lambda params: (
    "tinylora"
    if params < 1e3
    else "loraxs" if params < 1e6 else "lora" if params < 1e9 else "full"
)

# Table 1 rows feed the same structures: 0-param rows are baselines, the rest are curve points
for model, rows in PAPER_TABLE.items():
    for params, *scores in rows:
        if params == 0:
            PAPER[model] = dict(zip(TABLE_TASKS, scores))
        else:
            for task, score in zip(TABLE_TASKS, scores):
                PAPER_CURVES.setdefault(model, {}).setdefault(task, []).append(
                    (params, score)
                )


# metrics no card renders: the min/max envelopes, the raw reward mirrors and the duplicate length series
CURVE_DROP = re.compile(
    r"^(completion_length|completions/(min|max)_|clip_ratio/(high|low)_|rewards/)"
)


def load_task(path: Path) -> dict:
    """Stats for one <task>.json.gz plus wrong-or-unparsed examples (questions truncated, completions dropped)."""
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    misses = [
        dict(question=r["question"][:240], predicted=r["predicted"], answer=r["answer"])
        for r in data["records"]
        if not r["correct"]
    ]
    return {k: v for k, v in data.items() if k != "records"} | dict(misses=misses)


def load_curves(run_dir: Path) -> list[dict]:
    """Per-step training metrics from curves.jsonl, or the newest checkpoint's log_history while a run is still training."""
    curves = run_dir / "curves.jsonl"
    if curves.exists():
        history = [json.loads(line) for line in curves.read_text().splitlines() if line]
    else:
        checkpoints = sorted(
            run_dir.glob("checkpoint-*/trainer_state.json"),
            key=lambda p: int(p.parent.name.split("-")[1]),
        )
        if not checkpoints:
            return []
        history = json.loads(checkpoints[-1].read_text())["log_history"]
    return [
        {
            k: (float(f"{v:.4g}") if isinstance(v, float) else v)
            for k, v in row.items()
            if isinstance(v, (int, float)) and not CURVE_DROP.match(k)
        }
        for row in history
        if "loss" in row
    ]


def snapshots(run_dir: Path) -> list[Path]:
    """Snapshot dirs in step order."""
    return sorted(
        run_dir.glob("snapshots/step-*"), key=lambda p: int(p.name.split("-")[1])
    )


def last_full_eval(run_dir: Path) -> list[Path]:
    """The newest snapshot evaluated on every task any snapshot of this run has (a running job writes evals one task at a time)."""
    evaluated = {s: {p.name for p in s.glob("*.json.gz")} for s in snapshots(run_dir)}
    full = set().union(*evaluated.values()) if evaluated else set()
    return [s for s, done in evaluated.items() if done == full][-1:]


def eval_curves(run_dir: Path) -> list[dict]:
    """One eval_<task> row per evaluated snapshot, in the same shape as the training rows."""
    rows = []
    for snapshot in snapshots(run_dir):
        summary = snapshot / "eval.json"
        if not summary.exists():
            continue
        row = {"step": int(snapshot.name.split("-")[1])}
        for task, stats in json.loads(summary.read_text()).items():
            row[f"eval_{task}"] = float(f"{stats['accuracy']:.4g}")
            row[f"eval_{task}_unparsed"] = stats["unparsed"]
        rows.append(row)
    return rows


def thin(curves: list[dict], every: int) -> list[dict]:
    """Keep every Nth step plus the endpoints and every mid-train eval, which land off the grid."""
    if every <= 1:
        return curves
    return [
        row
        for i, row in enumerate(curves)
        if row.get("step", 0) % every == 0
        or i in (0, len(curves) - 1)
        or any(k.startswith("eval_") for k in row)
    ]


def parse_run_path(rel: Path) -> dict:
    """Config from <model>/<task>/<adapter>-<loss>[-lr<lr>][-<cfg>]/seed<N> for runs that have not written run.json yet (BO writes it last)."""
    m = re.search(
        r"(?:^|/)([^/]+)/([^/]+)/([^/-]+)-([^/-]+)(?:-lr(\d[\d.]*(?:e[+-]?\d+)?))?(?:-[^/]+)?/seed(\d+)$",
        str(rel),
    )
    if not m:
        return {}
    model, task, adapter, loss, lr, seed = m.groups()
    config = dict(model=model, task=task, adapter=adapter, loss=loss, seed=int(seed))
    return config | dict(lr=float(lr)) if lr else config


def hours_per_step(curves: list[dict]) -> float | None:
    """Training rate over the newest stretch of the clock, which restarts at zero every requeue."""
    by_step = {c["step"]: c["elapsed_hours"] for c in curves if "elapsed_hours" in c}
    points = sorted(by_step.items())
    if len(points) < 2:
        return None
    start = len(points) - 1
    while start and points[start - 1][1] < points[start][1]:
        start -= 1
    if start == len(points) - 1:
        return None
    (first_step, first_hours), (last_step, last_hours) = points[start], points[-1]
    return (last_hours - first_hours) / (last_step - first_step)


def checkpoint_rate(checkpoints: list[Path]) -> float | None:
    """Fallback rate from checkpoint mtimes, for runs logged before elapsed_hours existed."""
    step_of = lambda p: int(p.parent.name.split("-")[1])
    if len(checkpoints) < 2 or step_of(checkpoints[-1]) <= step_of(checkpoints[0]):
        return None
    span = checkpoints[-1].stat().st_mtime - checkpoints[0].stat().st_mtime
    return span / 3600 / (step_of(checkpoints[-1]) - step_of(checkpoints[0]))


GP_CACHE_DIR = Path(__file__).with_name(".gp-cache")
GP_CACHE: dict[str, dict | None] = {}
GP_PENDING: set[str] = set()
GP_LOCK = threading.Lock()


def fit_frames(trials: list[dict], design: int, lo: float, hi: float) -> dict | None:
    """GP posterior mean/sd on a grid over the search box after GP-guided batch 1, 2, 4, ... and the last, i.e. the snapshot steps (1-2 dim searches only)."""
    dim = len(trials[0]["theta"])
    if dim > 2:
        return None
    import torch
    from turbolora.bo import fit_gp

    n = 200 if dim == 1 else 40
    axis = torch.linspace(lo, hi, n, dtype=torch.float64)
    grid = axis[:, None] if dim == 1 else torch.cartesian_prod(axis, axis)
    rnd = lambda xs: [float(f"{v:.4g}") for v in xs.tolist()]
    # a frame per snapshot: the trials through GP-guided batch 1, 2, 4, ..., and everything so far
    batch_of = lambda t: t.get("batch", t["trial"])
    last_step = batch_of(trials[-1]) + 1 - design
    steps = sorted({s for s in (2**i for i in range(0, 20)) if s <= last_step} | {last_step})
    frames = []
    for step in steps:
        head = [t for t in trials if batch_of(t) < design + step]
        k = len(head)
        if len({tuple(t["theta"]) for t in head}) < 2:
            continue
        gp = fit_gp(head)
        with torch.no_grad():
            post = gp.posterior(grid)
            observed = gp.posterior(torch.tensor([t["theta"] for t in head], dtype=torch.float64)).mean.squeeze(-1)
        frames.append(dict(step=step, k=k, mean=rnd(post.mean.squeeze(-1)), sd=rnd(post.variance.sqrt().squeeze(-1)), pick=int(observed.argmax())))
    return dict(axis=rnd(axis), frames=frames) if frames else None


def gp_frames(trials_path: Path, trials: list[dict], design: int, lo: float, hi: float, wait: bool = False) -> dict | None:
    """Frames for this trials.json, from memory or the disk cache; otherwise fitted in a background thread (or inline when `wait`), the page reloads when they land."""
    run_key = hashlib.md5(str(trials_path.resolve()).encode()).hexdigest()
    key = f"{run_key}-{trials_path.stat().st_mtime_ns}"
    cache_file = GP_CACHE_DIR / f"{key}.json"
    with GP_LOCK:
        if key in GP_CACHE:
            return GP_CACHE[key]
        if cache_file.exists():
            GP_CACHE[key] = json.loads(cache_file.read_text())
            return GP_CACHE[key]
        if key in GP_PENDING:
            return None
        GP_PENDING.add(key)

    def work():
        try:
            result = fit_frames(trials, design, lo, hi)
        except Exception as e:
            print(f"GP fit failed for {trials_path}: {e}", flush=True)
            result = None
        # a running search rewrites trials.json every batch: keep only the newest fit per run
        GP_CACHE_DIR.mkdir(exist_ok=True)
        for stale in GP_CACHE_DIR.glob(f"{run_key}-*.json"):
            stale.unlink()
        cache_file.write_text(json.dumps(result))
        with GP_LOCK:
            GP_CACHE[key] = result
            GP_PENDING.discard(key)
        return result

    if wait:
        return work()
    threading.Thread(target=work, daemon=True).start()
    return None


def load_bo(run_dir: Path, summary: dict, wait_gp: bool = False) -> dict:
    """Trial log of a BO search plus the GP posterior's evolution, for the run page's search plots."""
    trials_path = run_dir / "trials.json"
    if not trials_path.exists():
        return {}
    trials = json.loads(trials_path.read_text())
    if not trials:
        return {}
    rng = summary.get("theta_range") or max(abs(x) for t in trials for x in t["theta"]) or 1.0
    rows = [
        dict(trial=t["trial"], batch=t.get("batch", t["trial"]), baseline=t["baseline"], theta=[float(f"{x:.4g}") for x in t["theta"]], value=float(f"{t['value']:.4g}"), sem=float(f"{t['sem']:.3g}"))
        for t in trials
    ]
    # `design` counts batches; the final run.json drops it (8 θ=0 replicates + 8 Sobol by default)
    design = summary.get("design", 16)
    return dict(bo=dict(trials=rows, range=rng, design=design, gp=gp_frames(trials_path, trials, design, -rng, rng, wait_gp)))


def load_progress(run_dir: Path, summary: dict, curves: list[dict]) -> dict:
    """Step count and, for runs still training, resources so far from the latest logged step (or BO batch)."""
    checkpoints = sorted(
        run_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-")[1]),
    )
    status = "done" if "steps" in summary else "running"
    progress = dict(status=status)
    if checkpoints:
        state = json.loads(checkpoints[-1].read_text())
        progress |= dict(step=state["global_step"], max_steps=state["max_steps"])
    if status == "running" and curves:
        last = curves[-1]
        progress |= {
            k: last[src]
            for k, src in [
                ("train_hours", "elapsed_hours"),
                ("peak_vram_gb", "peak_vram_gib"),
            ]
            if src in last
        }
    # a BO search logs batches to trials.json instead of checkpoints; its GP-guided batches are the steps
    trials_path = run_dir / "trials.json"
    if status == "running" and not checkpoints and trials_path.exists() and "design" in summary:
        trials = json.loads(trials_path.read_text())
        batches = trials[-1]["batch"] + 1 if trials else 0
        step = max(0, batches - summary["design"])
        started, last = (run_dir / "run.json").stat().st_mtime, trials_path.stat().st_mtime
        idle_hours = (time.time() - last) / 3600
        rate = (last - started) / 3600 / batches if batches else None
        progress |= dict(step=step, checkpoint_time=round(last), idle_hours=round(idle_hours, 3))
        if rate and idle_hours < max(2 * rate, 0.5):
            progress |= dict(eta_time=round(time.time() + 3600 * rate * (summary["max_steps"] - step)))
    # a queued or preempted job keeps its checkpoints but stops advancing, so ETA needs both a rate and proof of life
    if status == "running" and checkpoints:
        newest = checkpoints[-1]
        idle_hours = (time.time() - newest.stat().st_mtime) / 3600
        steps = [int(p.parent.name.split("-")[1]) for p in checkpoints]
        save_every = steps[-1] - steps[-2] if len(steps) > 1 else steps[-1]
        rate = hours_per_step(curves) or checkpoint_rate(checkpoints)
        progress |= dict(
            checkpoint_time=round(newest.stat().st_mtime),
            idle_hours=round(idle_hours, 3),
        )
        alive = rate and idle_hours < max(2 * rate * save_every, 0.5)
        if alive:
            left = rate * (state["max_steps"] - state["global_step"])
            progress |= dict(eta_time=round(time.time() + 3600 * left))
    return progress


def collect(baselines_dir: Path, runs_dir: Path, curve_every: int = 1, wait_gp: bool = False) -> dict:
    """Baselines keyed by model name, training runs keyed by path relative to runs_dir."""
    models = {}
    for path in sorted(baselines_dir.glob("*/*.json.gz")):
        name, task = path.parent.name, path.name.removesuffix(".json.gz")
        if name not in MODELS or task not in EVAL_TASKS:
            continue
        models.setdefault(name, dict(hf_id=MODELS[name].hf_id, tasks={}))
        models[name]["tasks"][task] = load_task(path)

    # a run is a training output dir with run.json and/or checkpoints; its latest fully evaluated snapshot is the headline accuracy
    run_dirs = (
        {p.parent for p in runs_dir.glob("**/run.json")}
        | {p.parent.parent for p in runs_dir.glob("**/checkpoint-*/trainer_state.json")}
        | {p.parent.parent for p in runs_dir.glob("**/snapshots/step-*")}
    )
    runs = {}
    for run_dir in sorted(run_dirs):
        rel = run_dir.relative_to(runs_dir)
        summary = (
            json.loads((run_dir / "run.json").read_text())
            if (run_dir / "run.json").exists()
            else parse_run_path(rel)
        )
        if summary.get("model") not in MODELS:
            continue
        # the same adapter searched by BO is its own family on the plot, not a TinyLoRA-GRPO point
        if summary.get("loss") == "bo":
            summary = summary | dict(adapter=f"{summary['adapter']}-bo")
        # a BO search writes run.json only at the end; until then its θ vector length is the parameter count
        if "params" not in summary and (run_dir / "trials.json").exists():
            trials = json.loads((run_dir / "trials.json").read_text())
            if trials:
                summary = summary | dict(params=len(trials[0]["theta"]))
        last = last_full_eval(run_dir)
        tasks = {
            p.name.removesuffix(".json.gz"): load_task(p)
            for snapshot in last
            for p in sorted(snapshot.glob("*.json.gz"))
            if p.name.removesuffix(".json.gz") in EVAL_TASKS
        }
        if last:
            summary = summary | dict(eval_step=int(last[0].name.split("-")[1]))
        curves = thin(load_curves(run_dir), curve_every)
        curves = sorted(
            curves + eval_curves(run_dir), key=lambda row: row.get("step", 0)
        )
        runs[str(rel)] = (
            summary
            | load_progress(run_dir, summary, curves)
            | load_bo(run_dir, summary, wait_gp)
            | dict(tasks=tasks, curves=curves)
        )

    # keep MODELS' declaration order so families stay grouped
    ordered = {name: models[name] for name in MODELS if name in models}

    # the same benchmark question is missed by hundreds of runs: store each once and index into the pool
    questions: dict[str, int] = {}
    for stats in [t for r in runs.values() for t in r["tasks"].values()] + [
        t for m in ordered.values() for t in m["tasks"].values()
    ]:
        for miss in stats["misses"]:
            miss["q"] = questions.setdefault(miss.pop("question"), len(questions))

    return dict(
        models=ordered,
        runs=runs,
        questions=list(questions),
        tasks=EVAL_TASKS,
        paper=PAPER,
        paper_curves={
            m: {t: [(x, y, PAPER_ADAPTER(x)) for x, y in pts] for t, pts in c.items()}
            for m, c in PAPER_CURVES.items()
        },
    )


class Dashboard(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        payload = json.dumps(collect(args.baselines_dir, args.runs_dir)).replace(
            "</", "<\\/"
        )
        version = hashlib.md5(payload.encode()).hexdigest()
        if self.path == "/version":
            body, ctype = version.encode(), "text/plain"
        else:
            # ride the template's `const DATA = /*__DATA__*/null;` statement to also define a poller that reloads the page when the data changes
            inject = (
                f"{payload};\nconst VERSION = {version!r};\n"
                'setInterval(async () => { try { if (await (await fetch("version", { cache: "no-store" })).text() !== VERSION) location.reload(); } catch {} }, 30000)'
            )
            html = TEMPLATE.read_text().replace("/*__DATA__*/null", inject)
            body, ctype = html.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    log_message = lambda *a, **kw: None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines-dir", type=Path, default="outputs/baselines")
    parser.add_argument("--runs-dir", type=Path, default="outputs")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"serving dashboard at http://localhost:{args.port}", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Dashboard).serve_forever()
