"""Serve the dashboard with live data from baseline evals (outputs/runs/<family>/<model>/base/eval@K/<task>.json.gz) and training runs (any outputs/**/run.json, incl. still-running ones): `uv run dashboard/serve.py`, then open http://localhost:8000."""

import argparse
import ast
import gzip
import hashlib
import http.server
import json
import os
import re

import numpy as np
import orjson
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from turbolora.models import MODELS

TEMPLATE = Path(__file__).with_name("template.html")


def task_names() -> list[str]:
    """Keys of turbolora.tasks.TASKS read off the source: importing the module pulls in datasets, torch and math_verify (~15 s)."""
    tree = ast.parse((Path(__file__).resolve().parents[1] / "src/turbolora/tasks.py").read_text())
    tasks = next(n for n in tree.body if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == "TASKS")
    return [ast.literal_eval(k) for k in tasks.value.keys]


EVAL_TASKS = [t for t in task_names() if not t.startswith(("easy", "medium", "hard"))]
# a run's eval on its own training tier (eval.py --split train writes eval@K/<tier>-train.json.gz) shows as the pseudo-task `train` under objective K
task_of = lambda name: "train" if re.fullmatch(r"(easy|medium|hard)-train", name) else name if name in EVAL_TASKS else None
# eval objective -> eval dir of `turbolora.eval` (<snapshot>/eval@K/<task>.json.gz + summary.json); curve keys are <key>_<task>
OBJECTIVES = {"greedy": dict(dir="eval@1", key="eval", label="Greedy"), "sampled": dict(dir="eval@4", key="eval@4", label="4 samples · T=1")}

# metrics no card renders: the min/max envelopes, the raw reward mirrors and the duplicate length series
CURVE_DROP = re.compile(
    r"^(completion_length|completions/(min|max)_|clip_ratio/(high|low)_|rewards/)"
)

step_of = lambda p: int(p.name.split("-")[1])
by_step = lambda paths: sorted(paths, key=step_of)
rnd = lambda v, digits=4: float(f"{v:.{digits}g}")


def rnd_array(values, digits: int = 4) -> list:
    """rnd() over a whole array in numpy: BO trial θ vectors run to millions of floats."""
    a = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore"):
        scale = 10.0 ** (digits - 1 - np.where(a == 0, 0, np.floor(np.log10(np.abs(a)))))
    return (np.round(a * scale) / scale).tolist()


# append-only log of decoded eval files: ["q", id, question, answer] rows define the miss pool, ["t", "<path>@<mtime_ns>", stats] rows the per-file stats
TASK_CACHE_FILE = Path(__file__).with_name(".task-cache.jsonl")
TASK_CACHE: dict[str, dict | bytes] = {}  # a log row's bytes until first use: parsing all 6k rows costs seconds, each scan process needs a slice
TASK_CACHE_PENDING: list[list] = []
QUESTION_POOL: dict[str, list[str]] = {}


def load_caches() -> None:
    """Once per process: replay the task log (a re-evaluated file appears twice, the newer row wins) and index the curves cache."""
    if CURVES_CACHE_FILE.exists() and not CURVES_CACHE:
        for line in CURVES_CACHE_FILE.read_bytes().splitlines():
            CURVES_CACHE[line.split(b'"', 2)[1].decode()] = line
    if TASK_CACHE or QUESTION_POOL or not TASK_CACHE_FILE.exists():
        return
    for line in TASK_CACHE_FILE.read_bytes().splitlines():
        if line.startswith(b'["q"'):
            _, qid, question, answer = orjson.loads(line)
            QUESTION_POOL[qid] = [question, answer]
        else:
            TASK_CACHE[line.split(b'"', 4)[3].decode()] = line


def load_task(path: Path) -> dict:
    """Stats for one <task>.json.gz; gunzip + json of ~6k files is the bulk of a cold scan, so decoded once.

    A fresh decode still carries raw misses and its cache key: pool_misses() turns them into [question id, predicted].
    """
    key = f"{os.path.abspath(path)}@{path.stat().st_mtime_ns}"
    if key not in TASK_CACHE:
        TASK_CACHE[key] = read_task(path) | dict(key=key)
    elif isinstance(TASK_CACHE[key], bytes):
        TASK_CACHE[key] = orjson.loads(TASK_CACHE[key])[2]
    # slim() strips the misses from what it is handed: hand out a shallow copy
    return dict(TASK_CACHE[key])


def pool_misses(stats: dict) -> None:
    """Index a fresh decode's misses into the question pool and log it for the next build; ids are content hashes, so scan processes never share a counter."""
    key = stats.pop("key", None)
    if key is None:
        return
    misses = []
    for m in stats["misses"]:
        qid = hashlib.blake2b(f"{m['question']}\0{m['answer']}".encode(), digest_size=5).hexdigest()
        if qid not in QUESTION_POOL:
            QUESTION_POOL[qid] = [m["question"], m["answer"]]
            TASK_CACHE_PENDING.append(["q", qid, *QUESTION_POOL[qid]])
        misses.append([qid, m["predicted"]])
    stats["misses"] = misses
    TASK_CACHE[key] = dict(stats)
    TASK_CACHE_PENDING.append(["t", key, dict(stats)])


def save_task_cache() -> None:
    if TASK_CACHE_PENDING:
        with TASK_CACHE_FILE.open("ab") as f:
            f.writelines(orjson.dumps(row) + b"\n" for row in TASK_CACHE_PENDING)
        TASK_CACHE_PENDING.clear()


def read_task(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        data = orjson.loads(f.read())
    # no miss drawer for the train split: its ~8k questions would swell the pooled question index on every page
    misses = [
        dict(question=r["question"][:240], predicted=r["predicted"], answer=r["answer"])
        for r in (data["records"] if data.get("split", "test") == "test" else [])
        if not (all(r["correct"]) if isinstance(r["correct"], list) else r["correct"])
    ]
    stats = {k: v for k, v in data.items() if k != "records"}
    return stats | dict(misses=misses, wrong=stats["n"] * stats.get("samples", 1) - stats["n_correct"])


CURVES_CACHE_FILE = Path(__file__).with_name(".curves-cache.jsonl")
CURVES_CACHE: dict[str, list[dict] | bytes] = {}  # [key, rows] lines, parsed on first use
CURVES_SEEN: set[str] = set()
CURVES_NEW: set[str] = set()


def load_curves(run_dir: Path) -> list[dict]:
    """Per-step training metrics from curves.jsonl, rewritten by the trainer on every logged step; parsed once per mtime, cached across builds."""
    curves = run_dir / "curves.jsonl"
    try:
        key = f"{os.path.abspath(curves)}@{curves.stat().st_mtime_ns}"
    except FileNotFoundError:
        return []
    if key not in CURVES_CACHE:
        CURVES_CACHE[key] = parse_curves(curves)
        CURVES_NEW.add(key)
    elif isinstance(CURVES_CACHE[key], bytes):
        CURVES_CACHE[key] = orjson.loads(CURVES_CACHE[key])[1]
    CURVES_SEEN.add(key)
    return CURVES_CACHE[key]


def save_curves_cache() -> None:
    """Rewritten whenever a run logged new steps, keeping only the runs seen this scan."""
    if CURVES_NEW:
        for key in set(CURVES_CACHE) - CURVES_SEEN:
            del CURVES_CACHE[key]
        line = lambda key, rows: rows if isinstance(rows, bytes) else orjson.dumps([key, rows])
        CURVES_CACHE_FILE.write_bytes(b"".join(line(k, r) + b"\n" for k, r in CURVES_CACHE.items()))
    CURVES_SEEN.clear()
    CURVES_NEW.clear()


def parse_curves(curves: Path) -> list[dict]:
    return [
        {
            k: (rnd(v) if isinstance(v, float) else v)
            for k, v in row.items()
            if isinstance(v, (int, float)) and not CURVE_DROP.match(k)
        }
        for row in map(orjson.loads, curves.read_bytes().splitlines())
        if "loss" in row
    ]


def task_files(snapshot: Path, eval_dir: str) -> dict[str, Path]:
    """<task>.json.gz files of one eval objective (<snapshot>/<eval_dir>), keyed by task."""
    return {task: p for p in sorted((snapshot / eval_dir).glob("*.json.gz")) if (task := task_of(p.stem.removesuffix(".json")))}


def snapshot_summaries(run_dir: Path) -> dict[Path, dict[str, dict[str, dict]]]:
    """{snapshot: {eval_dir: {task: stats}}} in step order from each eval@K/summary.json, which eval.py rewrites after every task: no directory listings."""
    try:
        snapshots = by_step(Path(e.path) for e in os.scandir(run_dir / "snapshots") if e.name.startswith("step-"))
    except FileNotFoundError:
        return {}
    summaries = {}
    for snapshot in snapshots:
        summaries[snapshot] = {}
        for obj in OBJECTIVES.values():
            try:
                stats = orjson.loads((snapshot / obj["dir"] / "summary.json").read_bytes())
            except FileNotFoundError:
                stats = {}
            summaries[snapshot][obj["dir"]] = {task_of(name): st for name, st in stats.items() if task_of(name)}
    return summaries


def last_full_eval(evaluated: dict[Path, dict[str, dict]]) -> Path | None:
    """The newest snapshot evaluated on every benchmark any snapshot of this run has (a running job writes evals one task at a time); the train-set eval is optional."""
    full = set().union(*map(set, evaluated.values())) - {"train"} if evaluated else set()
    complete = [s for s, done in evaluated.items() if set(done) >= full]
    return complete[-1] if complete and full else None


def eval_curves(summaries: dict[Path, dict[str, dict[str, dict]]]) -> list[dict]:
    """One <key>_<task> row per evaluated snapshot and objective, in the same shape as the training rows."""
    rows = []
    for snapshot, per_obj in summaries.items():
        row = {"step": step_of(snapshot)}
        for obj in OBJECTIVES.values():
            for task, stats in per_obj[obj["dir"]].items():
                row[f"{obj['key']}_{task}"] = rnd(stats["accuracy"])
                row[f"{obj['key']}_{task}_unparsed"] = stats["unparsed"]
        if len(row) > 1:
            rows.append(row)
    return rows


def thin(curves: list[dict], every: int) -> list[dict]:
    """Keep every Nth step plus the first and last training row and every eval row, which land off the grid."""
    if every <= 1:
        return curves
    is_eval = lambda row: any(k.startswith("eval_") for k in row)
    train = [i for i, row in enumerate(curves) if not is_eval(row)]
    endpoints = set(train[:1] + train[-1:])
    return [row for i, row in enumerate(curves) if i in endpoints or row.get("step", 0) % every == 0 or is_eval(row)]


def slim(entry: dict, every: int) -> dict:
    """Shrink a run or baseline in place to what the charts draw: curves every Nth step, no misses, θ₀ per trial (the 1-D plot)."""
    if "curves" in entry:
        entry["curves"] = thin(entry["curves"], every)
    if "bo" in entry:
        entry["bo"]["trials"]["theta"] = [theta[:1] for theta in entry["bo"]["trials"]["theta"]]
    for stats in (s for per_task in entry["evals"].values() for s in per_task.values()):
        del stats["misses"]
    return entry


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


def fit_frames(trials: list[dict], design: int, lo: float, hi: float, known: dict[int, dict] | None = None) -> dict | None:
    """GP posterior mean/sd on a grid over the search box after GP-guided batch 1, 2, 4, ... and the last, i.e. the snapshot steps (1-dim searches only).

    `known` holds an earlier fit's frames by step: a frame whose trial prefix is unchanged is reused, so a running search refits only its newest frame.
    """
    if len(trials[0]["theta"]) != 1:
        return None
    import torch
    from turbolora.bo import fit_gp

    # one fit per thread: collect() fits every running search concurrently
    torch.set_num_threads(1)
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
        sig = hashlib.md5(json.dumps(head, sort_keys=True).encode()).hexdigest()
        if known and known.get(step, {}).get("sig") == sig:
            frames.append(known[step])
            continue
        gp = fit_gp(head)
        with torch.no_grad():
            post = gp.posterior(grid)
            observed = gp.posterior(torch.tensor([t["theta"] for t in head], dtype=torch.float64)).mean.squeeze(-1)
        frames.append(dict(step=step, k=len(head), sig=sig, mean=[rnd(v) for v in post.mean.squeeze(-1).tolist()], sd=[rnd(v) for v in post.variance.sqrt().squeeze(-1).tolist()], pick=int(observed.argmax())))
    return dict(axis=[rnd(v) for v in axis.tolist()], frames=frames) if frames else None


def gp_key(trials_path: Path) -> tuple[str, str]:
    """(run key, fit key): the fit key changes with every rewrite of trials.json."""
    run_key = hashlib.md5(str(trials_path.resolve()).encode()).hexdigest()
    return run_key, f"{run_key}-{trials_path.stat().st_mtime_ns}"


def cached_frames(trials_path: Path) -> tuple[bool, dict | None]:
    """Whether this trials.json has been fitted (in memory or on disk) and its frames, None for a search the plots cannot draw."""
    _, key = gp_key(trials_path)
    cache_file = GP_CACHE_DIR / f"{key}.json"
    if key not in GP_CACHE and cache_file.exists():
        GP_CACHE[key] = orjson.loads(cache_file.read_bytes())
    return key in GP_CACHE, GP_CACHE.get(key)


def fit_job(trials_path: Path, trials: list[dict], design: int, lo: float, hi: float) -> dict | None:
    """Fit the frames of one search and cache them on disk."""
    run_key, key = gp_key(trials_path)
    # the previous fit of this run, whose earlier frames still hold
    earlier = [orjson.loads(f.read_bytes()) or {} for f in GP_CACHE_DIR.glob(f"{run_key}-*.json")]
    known = {f["step"]: f for fit in earlier for f in fit.get("frames", [])}
    try:
        result = fit_frames(trials, design, lo, hi, known)
    except Exception as e:
        print(f"GP fit failed for {trials_path}: {e}", flush=True)
        result = None
    # a running search rewrites trials.json every batch: keep only the newest fit per run
    GP_CACHE_DIR.mkdir(exist_ok=True)
    for stale in GP_CACHE_DIR.glob(f"{run_key}-*.json"):
        stale.unlink()
    (GP_CACHE_DIR / f"{key}.json").write_bytes(orjson.dumps(result))
    GP_CACHE[key] = result
    return result


def fit_jobs(jobs: dict[str, tuple]) -> dict[str, dict | None]:
    """Frames by run name for the searches whose fit is missing, one fit per thread (a 2000-trial fit holds ~300 MB, so few threads)."""
    with ThreadPoolExecutor(min(4, os.cpu_count() or 1)) as pool:
        return dict(zip(jobs, pool.map(lambda job: fit_job(*job), jobs.values())))


def gp_server(conn) -> None:
    """A forked helper that pays the torch import while the scan runs, then fits whatever jobs arrive on the pipe."""
    import torch  # noqa: F401
    import turbolora.bo  # noqa: F401

    conn.send(fit_jobs(conn.recv()))


def load_bo(run_dir: Path, summary: dict) -> dict:
    """Trial log of a BO search plus the GP posterior's evolution, for the run page's search plots; `job` marks a fit still to run."""
    trials_path = run_dir / "trials.json"
    trials = orjson.loads(trials_path.read_bytes()) if trials_path.exists() else []
    if not trials:
        return {}
    rng, design = summary["theta_range"], summary["design"]
    # columnar (trial i = position i): a row-per-trial dict costs ~5x the bytes over 600+ searches
    flag = lambda name: [i for i, t in enumerate(trials) if t[name]]
    columns = dict(theta=rnd_array([t["theta"] for t in trials]), value=[rnd(t["value"]) for t in trials], sem=[rnd(t["sem"], 3) for t in trials], design=flag("design"), baseline=flag("baseline"), dim=len(trials[0]["theta"]))
    fitted, gp = cached_frames(trials_path)
    job = None if fitted else (trials_path, trials, design, -rng, rng)
    return dict(bo=dict(trials=columns, range=rng, gp=gp), job=job)


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
        batches = len({t["batch"] for t in orjson.loads(trials_path.read_bytes())})
        step = max(0, batches - summary["design"])
        started, last = (run_dir / "run.json").stat().st_mtime, trials_path.stat().st_mtime
        rate = (last - started) / 3600 / batches if batches else None
        return progress | eta(step, summary["max_steps"], rate, last, 2 * (rate or 0))
    checkpoints = by_step(p.parent for p in run_dir.glob("checkpoint-*/trainer_state.json"))
    if checkpoints:
        state = orjson.loads((checkpoints[-1] / "trainer_state.json").read_bytes())
        steps = [step_of(c) for c in checkpoints]
        save_every = steps[-1] - steps[-2] if len(steps) > 1 else steps[-1]
        rate = hours_per_step(curves)
        return progress | dict(max_steps=state["max_steps"]) | eta(state["global_step"], state["max_steps"], rate, checkpoints[-1].stat().st_mtime, 2 * (rate or 0) * save_every)
    return progress


def run_jsons(runs_dir: Path) -> list[Path]:
    """Every run.json below runs_dir without descending into snapshot, checkpoint or adapter trees (tens of thousands of NFS dirs)."""
    skip = lambda d: d.name in ("snapshots", "candidate") or d.name.startswith(("checkpoint-", "final_adapter", "grpo_trainer_lora_model"))
    found, level = [], [runs_dir]
    # one NFS round trip per directory: list each level of the tree in parallel
    with ThreadPoolExecutor(32) as pool:
        while level:
            listings = pool.map(lambda d: list(os.scandir(d)), level)
            level = []
            for entries in listings:
                found += [Path(e.path) for e in entries if e.name == "run.json"]
                level += [Path(e.path) for e in entries if e.is_dir() and not skip(e)]
    return sorted(found)


def load_baseline(base_dir: Path) -> dict | None:
    evals = {name: {task: load_task(p) for task, p in task_files(base_dir, obj["dir"]).items()} for name, obj in OBJECTIVES.items()}
    return dict(hf_id=MODELS[base_dir.parent.name].hf_id, evals=evals) if any(evals.values()) else None


def load_run(run_json: Path) -> dict | None:
    """One training run: config, progress, search log and the latest fully evaluated snapshot as the headline accuracy."""
    run_dir = run_json.parent
    summary = orjson.loads(run_json.read_bytes())
    if summary.get("model") not in MODELS:
        return None
    # the same adapter searched by BO/TuRBO is its own family on the plot, not a TinyLoRA-GRPO point
    if summary["loss"] in ("bo", "turbo"):
        summary |= dict(adapter=f"{summary['adapter']}-{summary['loss']}")
    summaries = snapshot_summaries(run_dir)
    file_of = lambda task: f"{summary['task']}-train" if task == "train" else task
    evals, eval_step = {}, {}
    for name, obj in OBJECTIVES.items():
        last = last_full_eval({s: per_obj[obj["dir"]] for s, per_obj in summaries.items()})
        evals[name] = {task: load_task(last / obj["dir"] / f"{file_of(task)}.json.gz") for task in summaries[last][obj["dir"]]} if last else {}
        if last:
            eval_step[name] = step_of(last)
    curves = sorted(load_curves(run_dir) + eval_curves(summaries), key=lambda row: row.get("step", 0))
    return summary | load_progress(run_dir, summary, curves) | load_bo(run_dir, summary) | dict(evals=evals, eval_step=eval_step, curves=curves)


def scan_slice(base_dirs: list[Path], jsons: list[Path], runs_dir: Path, every: int) -> dict:
    """Baselines and runs of one slice as (full, slim) JSON fragments by name, the GP fits they still need and what the slice added to the caches.

    Every run is a few dozen NFS round trips, so they overlap in threads; the fragments keep the pickling back to the parent to a memcpy.
    """
    with ThreadPoolExecutor(16) as pool:
        baselines = zip((d.parent.name for d in base_dirs), pool.map(load_baseline, base_dirs))
        runs = zip((str(p.parent.relative_to(runs_dir)) for p in jsons), pool.map(load_run, jsons))
    pending = len(TASK_CACHE_PENDING)
    fragments, jobs = {}, {}
    for name, entry in (*baselines, *runs):
        if entry:
            if job := entry.pop("job", None):
                jobs[name] = job
            for stats in (s for per_task in entry["evals"].values() for s in per_task.values()):
                pool_misses(stats)
            # slim() shrinks the entry in place, so the full fragment goes first
            full = orjson.dumps(entry)
            fragments[name] = tuple(f.replace(b"</", b"<\\/") for f in (full, orjson.dumps(slim(entry, every))))
    rows, TASK_CACHE_PENDING[pending:] = TASK_CACHE_PENDING[pending:], []
    return dict(
        fragments=fragments,
        jobs=jobs,
        rows=rows,
        curves_new={k: CURVES_CACHE[k] for k in CURVES_NEW},
        curves_seen=set(CURVES_SEEN),
    )


def assemble(fragments: dict[str, bytes], models: list[str], questions: dict) -> bytes:
    """The page payload from per-entry JSON fragments (already safe inside a <script> tag)."""
    obj = lambda names: b"{" + b",".join(orjson.dumps(n) + b":" + fragments[n] for n in names) + b"}"
    head = orjson.dumps(dict(tasks=EVAL_TASKS + ["train"], objectives=OBJECTIVES, questions=questions))[:-1].replace(b"</", b"<\\/")
    runs = [n for n in fragments if n not in models]
    return head + b',"models":' + obj(models) + b',"runs":' + obj(runs) + b"}"


def collect(baselines_dir: Path, runs_dir: Path, workers: int = 1, every: int = 25) -> tuple[bytes, bytes]:
    """The full and the slim page payload: baselines keyed by model name, runs by path relative to runs_dir, misses indexing the pooled `questions`.

    The warm scan is CPU-bound in Python: `workers` forked processes (sharing the loaded caches) each take a slice of the runs, and a
    forked GP helper imports torch meanwhile. Single-process mode (the live server, whose threads rule out forking) does it all inline.
    """
    if workers > 1:
        ctx = get_context("fork")
        gp_conn, child_conn = ctx.Pipe()
        gp_proc = ctx.Process(target=gp_server, args=(child_conn,), daemon=True)
        gp_proc.start()
        child_conn.close()  # else a killed helper never shows as EOF on gp_conn
    # the run.json walk is NFS-bound and the cache replay CPU-bound: overlap them
    with ThreadPoolExecutor(1) as pool:
        jsons = pool.submit(run_jsons, runs_dir)
        load_caches()
        base_dirs = [d for d in sorted(baselines_dir.glob("*/*/base")) if d.parent.name in MODELS]
        jsons = jsons.result()
    slices = [(base_dirs[i::workers], jsons[i::workers], runs_dir, every) for i in range(workers)]
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=ctx) as procs:
            parts = list(procs.map(scan_slice, *zip(*slices)))
    else:
        parts = [scan_slice(*slices[0])]
    fragments, jobs = {}, {}
    for part in parts:
        fragments |= part["fragments"]
        jobs |= part["jobs"]
        for row in part["rows"]:
            if row[0] == "q":
                QUESTION_POOL[row[1]] = row[2:]
            else:
                TASK_CACHE[row[1]] = row[2]
        TASK_CACHE_PENDING.extend(part["rows"])
        CURVES_CACHE.update(part["curves_new"])
        CURVES_NEW.update(part["curves_new"])
        CURVES_SEEN.update(part["curves_seen"])
    save_task_cache()
    save_curves_cache()

    # searches whose newest batch had no fit yet: splice the frames into their fragments
    if workers > 1:
        # an OOM-killed helper (12 GB cgroups on interactive nodes) surfaces as EOF: fit inline instead
        try:
            gp_conn.send(jobs)
            frames = gp_conn.recv()
        except (EOFError, BrokenPipeError):
            print(f"GP helper died (exit {gp_proc.exitcode}); fitting {len(jobs)} searches inline", flush=True)
            frames = fit_jobs(jobs)
        gp_proc.join()
    else:
        frames = fit_jobs(jobs)
    for name, gp in frames.items():
        assert all(f.count(b'"gp":null') == 1 for f in fragments[name])
        fragments[name] = tuple(f.replace(b'"gp":null', b'"gp":' + orjson.dumps(gp).replace(b"</", b"<\\/")) for f in fragments[name])

    # keep MODELS' declaration order so families stay grouped
    models = [name for name in MODELS if name in fragments]
    full = {n: f[0] for n, f in fragments.items()}
    slimmed = {n: f[1] for n, f in fragments.items()}
    return assemble(full, models, QUESTION_POOL), assemble(slimmed, models, {})


# a full scan costs minutes, and every open tab polls /version: serve one scan to all requests and refresh it at most this often
SCAN_TTL = 300
scan_lock = threading.Lock()


def page(data: bytes) -> bytes:
    """The template with a payload bound in place of `const DATA = /*__DATA__*/null;`."""
    return TEMPLATE.read_bytes().replace(b"/*__DATA__*/null", data)


scan = dict(at=0.0, page=b"", version="")


def scanned() -> tuple[bytes, str]:
    """The rendered page and its version hash, rescanned only once the cached one is older than SCAN_TTL."""
    with scan_lock:
        if time.time() - scan["at"] > SCAN_TTL:
            data, _ = collect(args.baselines_dir, args.runs_dir)
            version = hashlib.md5(data).hexdigest()
            # ride the template's DATA statement to also define a poller that reloads the page when the data changes
            poller = (
                f";\nconst VERSION = {version!r};\n"
                'setInterval(async () => { try { if (await (await fetch("version", { cache: "no-store" })).text() !== VERSION) location.reload(); } catch {} }, 30000)'
            )
            scan.update(at=time.time(), page=page(data + poller.encode()), version=version)
        return scan["page"], scan["version"]


class Dashboard(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        html, version = scanned()
        body, ctype = (version.encode(), "text/plain") if self.path == "/version" else (html, "text/html; charset=utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    log_message = lambda *a, **kw: None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines-dir", type=Path, default="outputs/runs", help="holds <family>/<model>/base/eval@K")
    parser.add_argument("--runs-dir", type=Path, default="outputs")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"serving dashboard at http://localhost:{args.port}", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Dashboard).serve_forever()
