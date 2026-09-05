"""Serve the dashboard with live data from baseline evals (outputs/baselines/<model>/<task>.json.gz) and training runs (any outputs/**/run.json, incl. still-running ones): `uv run dashboard/serve.py`, then open http://localhost:8000."""

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
EVAL_TASKS = [t for t in TASKS if not t.startswith(("easy", "medium", "hard"))]

# metrics no card renders: the min/max envelopes, the raw reward mirrors and the duplicate length series
CURVE_DROP = re.compile(
    r"^(completion_length|completions/(min|max)_|clip_ratio/(high|low)_|rewards/)"
)

step_of = lambda p: int(p.name.split("-")[1])
by_step = lambda paths: sorted(paths, key=step_of)
rnd = lambda v, digits=4: float(f"{v:.{digits}g}")


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
    """Per-step training metrics from curves.jsonl, rewritten by the trainer on every logged step."""
    curves = run_dir / "curves.jsonl"
    if not curves.exists():
        return []
    return [
        {
            k: (rnd(v) if isinstance(v, float) else v)
            for k, v in row.items()
            if isinstance(v, (int, float)) and not CURVE_DROP.match(k)
        }
        for row in map(json.loads, curves.read_text().splitlines())
        if "loss" in row
    ]


def last_full_eval(run_dir: Path) -> Path | None:
    """The newest snapshot evaluated on every task any snapshot of this run has (a running job writes evals one task at a time)."""
    evaluated = {s: {p.name for p in s.glob("*.json.gz")} for s in by_step(run_dir.glob("snapshots/step-*"))}
    full = set().union(*evaluated.values()) if evaluated else set()
    complete = [s for s, done in evaluated.items() if done == full]
    return complete[-1] if complete else None


def eval_curves(run_dir: Path) -> list[dict]:
    """One eval_<task> row per evaluated snapshot, in the same shape as the training rows."""
    rows = []
    for snapshot in by_step(run_dir.glob("snapshots/step-*")):
        if not (snapshot / "eval.json").exists():
            continue
        row = {"step": step_of(snapshot)}
        for task, stats in json.loads((snapshot / "eval.json").read_text()).items():
            row[f"eval_{task}"] = rnd(stats["accuracy"])
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


def hours_per_step(curves: list[dict]) -> float | None:
    """Training rate over the newest stretch of the clock, which restarts at zero every requeue."""
    points = sorted({c["step"]: c["elapsed_hours"] for c in curves if "elapsed_hours" in c}.items())
    start = len(points) - 1
    while start and points[start - 1][1] < points[start][1]:
        start -= 1
    if start == len(points) - 1:
        return None
    (first_step, first_hours), (last_step, last_hours) = points[start], points[-1]
    return (last_hours - first_hours) / (last_step - first_step)


GP_CACHE_DIR = Path(__file__).with_name(".gp-cache")
GP_CACHE: dict[str, dict | None] = {}
GP_PENDING: set[str] = set()
GP_LOCK = threading.Lock()


def fit_frames(trials: list[dict], design: int, lo: float, hi: float) -> dict | None:
    """GP posterior mean/sd on a grid over the search box after GP-guided batch 1, 2, 4, ... and the last, i.e. the snapshot steps (1-dim searches only)."""
    if len(trials[0]["theta"]) != 1:
        return None
    import torch
    from turbolora.bo import fit_gp

    axis = torch.linspace(lo, hi, 200, dtype=torch.float64)
    grid = axis[:, None]
    # a frame per snapshot: the trials through GP-guided batch 1, 2, 4, ..., and everything so far
    last_step = trials[-1]["batch"] + 1 - design
    steps = sorted({s for s in (2**i for i in range(20)) if s <= last_step} | {last_step})
    frames = []
    for step in steps:
        head = [t for t in trials if t["batch"] < design + step]
        if len({tuple(t["theta"]) for t in head}) < 2:
            continue
        gp = fit_gp(head)
        with torch.no_grad():
            post = gp.posterior(grid)
            observed = gp.posterior(torch.tensor([t["theta"] for t in head], dtype=torch.float64)).mean.squeeze(-1)
        frames.append(dict(step=step, k=len(head), mean=[rnd(v) for v in post.mean.squeeze(-1).tolist()], sd=[rnd(v) for v in post.variance.sqrt().squeeze(-1).tolist()], pick=int(observed.argmax())))
    return dict(axis=[rnd(v) for v in axis.tolist()], frames=frames) if frames else None


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
    trials = json.loads(trials_path.read_text()) if trials_path.exists() else []
    if not trials:
        return {}
    rng, design = summary["theta_range"], summary["design"]
    # bo.py rows carry no design flag: its initial design is the first `design` batches
    rows = [
        dict(trial=t["trial"], design=bool(t.get("design", t["batch"] < design)), baseline=t["baseline"], theta=[rnd(x) for x in t["theta"]], value=rnd(t["value"]), sem=rnd(t["sem"], 3))
        for t in trials
    ]
    return dict(bo=dict(trials=rows, range=rng, gp=gp_frames(trials_path, trials, design, -rng, rng, wait_gp)))


def eta(step: int, max_steps: int, rate: float | None, last_activity: float, grace_hours: float) -> dict:
    """Only a run still advancing gets an ETA: a queued or preempted job keeps its files but stops touching them."""
    idle_hours = (time.time() - last_activity) / 3600
    progress = dict(step=step, checkpoint_time=round(last_activity), idle_hours=round(idle_hours, 3))
    if rate and idle_hours < max(grace_hours, 0.5):
        progress |= dict(eta_time=round(time.time() + 3600 * rate * (max_steps - step)))
    return progress


def load_progress(run_dir: Path, summary: dict, curves: list[dict]) -> dict:
    """Status, step count and, for runs still going, resources so far and an ETA."""
    if "steps" in summary:
        return dict(status="done")
    progress = dict(status="running")
    if curves:
        last = curves[-1]
        progress |= {k: last[src] for k, src in [("train_hours", "elapsed_hours"), ("peak_vram_gb", "peak_vram_gib")] if src in last}
    # a BO search logs batches to trials.json; the GP-guided ones are its steps
    trials_path = run_dir / "trials.json"
    if trials_path.exists():
        batches = len({t["batch"] for t in json.loads(trials_path.read_text())})
        step = max(0, batches - summary["design"])
        started, last = (run_dir / "run.json").stat().st_mtime, trials_path.stat().st_mtime
        rate = (last - started) / 3600 / batches if batches else None
        return progress | eta(step, summary["max_steps"], rate, last, 2 * (rate or 0))
    checkpoints = by_step(p.parent for p in run_dir.glob("checkpoint-*/trainer_state.json"))
    if checkpoints:
        state = json.loads((checkpoints[-1] / "trainer_state.json").read_text())
        steps = [step_of(c) for c in checkpoints]
        save_every = steps[-1] - steps[-2] if len(steps) > 1 else steps[-1]
        rate = hours_per_step(curves)
        return progress | dict(max_steps=state["max_steps"]) | eta(state["global_step"], state["max_steps"], rate, checkpoints[-1].stat().st_mtime, 2 * (rate or 0) * save_every)
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

    # every trainer writes the config half of run.json before its first step; the latest fully evaluated snapshot is the headline accuracy
    runs = {}
    for run_json in sorted(runs_dir.glob("**/run.json")):
        run_dir = run_json.parent
        summary = json.loads(run_json.read_text())
        if summary.get("model") not in MODELS:
            continue
        # the same adapter searched by BO/TuRBO is its own family on the plot, not a TinyLoRA-GRPO point
        if summary["loss"] in ("bo", "turbo"):
            summary |= dict(adapter=f"{summary['adapter']}-{summary['loss']}")
        last = last_full_eval(run_dir)
        tasks = {}
        if last:
            summary |= dict(eval_step=step_of(last))
            tasks = {p.name.removesuffix(".json.gz"): load_task(p) for p in sorted(last.glob("*.json.gz")) if p.name.removesuffix(".json.gz") in EVAL_TASKS}
        curves = sorted(thin(load_curves(run_dir), curve_every) + eval_curves(run_dir), key=lambda row: row.get("step", 0))
        runs[str(run_dir.relative_to(runs_dir))] = summary | load_progress(run_dir, summary, curves) | load_bo(run_dir, summary, wait_gp) | dict(tasks=tasks, curves=curves)

    # keep MODELS' declaration order so families stay grouped
    ordered = {name: models[name] for name in MODELS if name in models}

    # the same benchmark question is missed by hundreds of runs: store each once and index into the pool
    questions: dict[str, int] = {}
    for stats in [t for r in runs.values() for t in r["tasks"].values()] + [t for m in ordered.values() for t in m["tasks"].values()]:
        for miss in stats["misses"]:
            miss["q"] = questions.setdefault(miss.pop("question"), len(questions))

    return dict(models=ordered, runs=runs, questions=list(questions), tasks=EVAL_TASKS)


class Dashboard(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        payload = json.dumps(collect(args.baselines_dir, args.runs_dir)).replace("</", "<\\/")
        version = hashlib.md5(payload.encode()).hexdigest()
        if self.path == "/version":
            body, ctype = version.encode(), "text/plain"
        else:
            # ride the template's `const DATA = /*__DATA__*/null;` statement to also define a poller that reloads the page when the data changes
            inject = (
                f"{payload};\nconst VERSION = {version!r};\n"
                'setInterval(async () => { try { if (await (await fetch("version", { cache: "no-store" })).text() !== VERSION) location.reload(); } catch {} }, 30000)'
            )
            body, ctype = TEMPLATE.read_text().replace("/*__DATA__*/null", inject).encode(), "text/html; charset=utf-8"
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
