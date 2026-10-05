"""Paper-grid sweep driver: `status` prints coverage per cell, `submit` (dry-run unless --go) launches, resumes, stamps and backfills whatever is short.

Usage: uv run slurm/sweep.py status | submit [--go] [--only <model/adapter/cfg regex>] [--stagger 45] [--time 2:00:00]
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from turbolora.models import MODELS

RUNS = Path("outputs/runs")
IDLE_HOURS = 3  # a running job whose progress file is older than this is stuck (math-verify hang)
U = (1, 2, 4, 8, 16, 64, 256)
TURBO_MAX_U = 16  # turbo u64/u256 left out of the grid since 2026-09-12; GRPO runs every u
GRID = {  # model -> (train tier, turbo/tinylora u's); ranks and batches are the same everywhere
    "qwen2.5-1.5b": ("easy", U),
    "qwen2.5-1.5b-instruct": ("easy", U),
    "qwen2.5-1.5b-math": ("easy", U),
    "qwen2.5-7b": ("hard", U),
    "qwen2.5-7b-instruct": ("hard", U),
    "qwen2.5-7b-math": ("hard", U),
    "llama3.1-8b": ("hard", U),
    "mistral-7b": ("hard", U),
    "deepseek-7b-math": ("hard", U),
    "deepseek-7b": ("hard", U),
}
# distribution-shift check (2026-09-17): the TinyLoRA cells again on a single-source train set, cfg tagged `-<task>`
EXTRA_TASKS = {
    m: ("math", (1, 2, 4, 8, 16))
    for m in ("qwen2.5-1.5b", "qwen2.5-1.5b-math", "qwen2.5-7b", "qwen2.5-7b-math")
}
DEEP_SEEDS, DEEP_US = 8, (
    1,
    2,
    4,
)  # tinylora-grpo r2-u{1,2,4} and their noisy turbo b4-t1 (2026-09-15) and b1-t1 (2026-10-01) twins get seeds 0-7, the rest 0-2
SCRIPT = {
    "lora-grpo": "train_lora",
    "loraxs-grpo": "train_loraxs",
    "tinylora-grpo": "train_tinylora",
    "tinylora-turbo": "train_turbo",
}
CARDS_40G = "a40|l40s|a100-40g|a100-80g|h100"  # 7B GRPO and turbo (turbo on an L4 takes ~18 min/trial)
CARDS_24G = "l4|" + CARDS_40G
EVAL_TASKS = (
    "gsm8k",
    "math500",
    "aime24",
    "amc23",
    "minerva",
    "olympiad",
)  # = grpo.EVAL_TASKS, which imports torch


def cells(us: tuple[int, ...], tag: str = "") -> list[tuple[str, str, int]]:
    """(adapter-loss dir, cfg, seeds) for every cell of one model's row, in submission order (noisy turbo twins before greedy).

    A `tag` (a non-default train set) names TinyLoRA-only cells `<cfg>-<tag>`.
    """
    turbo_us = tuple(u for u in us if u <= TURBO_MAX_U)
    turbo = [f"r2-u{u}-b{b}{tag}" for u in turbo_us for b in (1, 4)]
    deep = lambda u: DEEP_SEEDS if u in DEEP_US else 3
    return (
        []
        if tag
        else [("lora-grpo", f"r{r}", 3) for r in (1, 2, 8, 32)]
        + [("loraxs-grpo", f"r{r}", 3) for r in (1, 2, 8, 32)]
    ) + (
        [("tinylora-grpo", f"r2-u{u}{tag}", deep(u)) for u in us]
        + [("tinylora-turbo", f"r2-u{u}-b4-t1{tag}", deep(u)) for u in turbo_us]
        + [("tinylora-turbo", f"r2-u{u}-b1-t1{tag}", deep(u)) for u in turbo_us]
        + [("tinylora-turbo", cfg, 3) for cfg in turbo]
    )


def twin_cfg(turbo_cfg: str) -> str:
    """tinylora-grpo cfg a turbo cfg searches from: drop `-b<batch>[-t1]`, keep any train-set tag (`r2-u8-b4-t1-math` -> `r2-u8-math`)."""
    return re.sub(r"-b\d+(-t1)?", "", turbo_cfg)


def queued_jobs() -> dict[tuple[str, int], dict]:
    """(cfg dir, seed) -> job id/state/hours running; seed from the array index, else the comment."""
    out = subprocess.run(
        ["squeue", "--me", "-r", "-h", "-o", "%i %k %K %T %M"],
        capture_output=True,
        text=True,
    ).stdout
    jobs = {}
    for line in out.splitlines():
        job, comment, index, state, elapsed = line.split()
        if not comment.startswith("outputs/"):
            continue
        # seed = array index: `scontrol update job=$SLURM_JOB_ID` from one task used to retag the whole array with its seed dir;
        # non-array jobs (eval.sh backfills) carry the seed dir in the comment
        cfg_dir, seed = re.sub(r"/seed\d+$", "", comment), re.search(
            r"/seed(\d+)$", comment
        )
        days, clock = elapsed.split("-") if "-" in elapsed else ("0", elapsed)
        parts = [int(x) for x in clock.split(":")]
        hours = (
            int(days) * 24 + sum(x / 60**i for i, x in enumerate(reversed(parts))) / 60
        )
        jobs[cfg_dir, int(index) if index != "N/A" else int(seed.group(1))] = dict(
            job=job, state=state, hours=hours
        )
    return jobs


def last_snapshot(seed_dir: Path) -> tuple[int, Path | None]:
    snapshots = sorted(
        seed_dir.glob("snapshots/step-*"), key=lambda p: int(p.name.split("-")[1])
    )
    return (
        (int(snapshots[-1].name.split("-")[1]), snapshots[-1])
        if snapshots
        else (0, None)
    )


def eval_tasks_done(summary: Path) -> set[str]:
    try:
        return set(json.loads(summary.read_text()))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def idle_hours(path: Path) -> float:
    return (time.time() - path.stat().st_mtime) / 3600 if path.is_file() else 0.0


def seed_status(seed_dir: Path, adapter: str, queued: dict | None) -> tuple[str, str]:
    """(state, detail): done | no-eval4 | unstamped | running | queued | hung | partial | missing."""
    if not seed_dir.is_dir():
        return ("queued" if queued else "missing"), ""
    try:  # a seed dir archived for a rerun can vanish mid-scan
        run = json.loads((seed_dir / "run.json").read_text())
    except FileNotFoundError:
        run = {}
    step, snapshot = last_snapshot(seed_dir)
    # an eval@4 killed mid-way leaves a summary of only the tasks it finished
    has_eval4 = snapshot is not None and set(EVAL_TASKS) <= eval_tasks_done(
        snapshot / "eval@4" / "summary.json"
    )
    if "steps" in run and has_eval4:
        return "done", f"step-{step}"
    if "steps" in run:
        return ("queued" if queued else "no-eval4"), f"step-{step}"

    # progress of an unfinished run: trials for turbo, last curve step for GRPO; a just-started run has the file but no record yet
    turbo = adapter == "tinylora-turbo"
    progress = seed_dir / ("trials.json" if turbo else "curves.jsonl")
    text = progress.read_text() if progress.is_file() else ""
    try:
        detail = (
            f"{len(json.loads(text))}t"
            if turbo
            else f"{json.loads(text.splitlines()[-1])['step']}s"
        )
    except (json.JSONDecodeError, IndexError):
        detail = "0t" if turbo else "0s"
    if queued and queued["state"] == "RUNNING":
        stuck = queued["hours"] > IDLE_HOURS and idle_hours(progress) > IDLE_HOURS
        return ("hung" if stuck else "running"), f"{detail},{queued['job']}"
    if queued:
        return "queued", detail
    # GRPO killed between the final eval@4 and the run.json stamp: nothing left to train
    live_checkpoints = [
        c
        for c in seed_dir.glob("checkpoint-*")
        if (c / "adapter_model.safetensors").is_file()
    ]
    if adapter != "tinylora-turbo" and has_eval4 and not live_checkpoints:
        return "unstamped", f"step-{step}"
    return "partial", detail


def scan(only: str | None) -> list[dict]:
    """Every grid seed with its state, in grid order."""
    queued = queued_jobs()
    rows = []
    grid = [(model, task, cells(us)) for model, (task, us) in GRID.items()]
    grid += [
        (model, task, cells(us, f"-{task}"))
        for model, (task, us) in EXTRA_TASKS.items()
    ]
    for model, task, model_cells in grid:
        for adapter, cfg, n_seeds in model_cells:
            if only and not re.search(only, f"{model}/{adapter}/{cfg}"):
                continue
            cfg_dir = RUNS / MODELS[model].family / model / adapter / cfg
            for seed in range(n_seeds):
                state, detail = seed_status(
                    cfg_dir / f"seed{seed}", adapter, queued.get((str(cfg_dir), seed))
                )
                rows.append(
                    dict(
                        model=model,
                        task=task,
                        adapter=adapter,
                        cfg=cfg,
                        cfg_dir=cfg_dir,
                        seed=seed,
                        state=state,
                        detail=detail,
                    )
                )
    return rows


def status(rows: list[dict]) -> None:
    """One line per model x adapter; a cell is `cfg:done/seeds` plus the seeds that are not done, grouped by state."""
    for (model, adapter), row in groupby(rows, ("model", "adapter")).items():
        tags = []
        for cfg, seeds in groupby(row, ("cfg",)).items():
            done = sum(s["state"] == "done" for s in seeds)
            rest = ";".join(
                f"{state}:"
                + ",".join(
                    f"s{s['seed']}" + (f"({s['detail']})" if s["detail"] else "")
                    for s in group
                )
                for (state,), group in groupby(
                    [s for s in seeds if s["state"] != "done"], ("state",)
                ).items()
            )
            tags.append(f"{cfg[0]}:{done}/{len(seeds)}" + (f"[{rest}]" if rest else ""))
        print(f"{model:24} {adapter:15} " + "  ".join(tags))
    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def groupby(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    return groups


def stamp(seed_dir: Path, go: bool) -> None:
    """Write the finish stamp grpo.run could not: steps from the last snapshot, hours/VRAM from curves.jsonl."""
    step, _ = last_snapshot(seed_dir)
    curves = [
        json.loads(l) for l in (seed_dir / "curves.jsonl").read_text().splitlines()
    ]
    summary = json.loads((seed_dir / "run.json").read_text()) | dict(
        steps=step,
        train_hours=round(max(r.get("elapsed_hours", 0) for r in curves), 3),
        peak_vram_gb=round(max(r.get("peak_vram_gib", 0) for r in curves), 2),
    )
    print(f"stamp {seed_dir} steps={step}")
    if go:
        (seed_dir / "run.json").write_text(json.dumps(summary, indent=1))
        for husk in seed_dir.glob("checkpoint-*"):
            for f in husk.iterdir():
                f.unlink()
            husk.rmdir()


def sbatch(env: dict[str, str], args: list[str], go: bool) -> None:
    print(" ".join(f"{k}={v}" for k, v in env.items()), "sbatch", *args)
    if go:
        subprocess.run(["sbatch", *args], env=os.environ | env, check=True)


def twin_ready(grpo_seed: Path) -> bool:
    has_bases = (grpo_seed / "final_adapter" / "adapter_model.safetensors").is_file()
    has_v = any(grpo_seed.glob("snapshots/step-*/trainable.safetensors")) or any(
        grpo_seed.glob("checkpoint-*/adapter_model.safetensors")
    )
    return has_bases and has_v


def submit(rows: list[dict], go: bool, stagger: int, time_limit: str | None) -> None:
    """Stamp unstamped finishes, backfill missing eval@4, resume/launch every partial or missing seed (one array per cfg)."""
    clock = ["-t", time_limit] if time_limit else []
    # SBATCH_EXCLUDE is not a slurm input variable, so the node list has to go on the command line
    off = ["-x", os.environ["EXCLUDE"]] if os.environ.get("EXCLUDE") else []
    for r in rows:
        if r["state"] == "hung":
            print(
                f"hung: {r['cfg_dir']}/seed{r['seed']} ({r['detail']}); scancel it and resubmit"
            )
        elif r["state"] == "unstamped":
            stamp(r["cfg_dir"] / f"seed{r['seed']}", go)
        elif r["state"] == "no-eval4":
            _, snapshot = last_snapshot(r["cfg_dir"] / f"seed{r['seed']}")
            opts = [
                "-p",
                "gpu,gpu-preempt",
                "-q",
                "normal",
                "--requeue",
                *clock,
                *off,
                f"--comment={r['cfg_dir']}/seed{r['seed']}",
            ]
            sbatch(
                dict(SAMPLES="4", ADAPTERS=str(snapshot), TASKS=" ".join(EVAL_TASKS)),
                [*opts, "slurm/eval_tasks.sh"],
                go,
            )

    delay, waiting_on_twin, queued = 0, 0, queued_jobs()
    for (model, task, adapter, cfg, cfg_dir), seeds in groupby(
        rows, ("model", "task", "adapter", "cfg", "cfg_dir")
    ).items():
        seeds = [s for s in seeds if s["state"] in ("partial", "missing")]
        small = model.startswith("qwen2.5-1.5b")
        constraint = CARDS_24G if small else CARDS_40G

        def launch(seeds: list[dict], extra: list[str] = []) -> None:
            nonlocal delay
            # HF 429s and node black-holes when many jobs start at once: stagger the starts (chained jobs are spread by their twins)
            opts = [
                "-a",
                ",".join(str(s["seed"]) for s in seeds),
                f"--comment={cfg_dir}",
                f"--constraint={constraint}",
                *clock,
                *off,
                *extra,
            ]
            if delay and not extra:
                opts.append(f"--begin=now+{delay}seconds")
            sbatch(
                dict(MODEL=model, TASK=task, CFG=cfg),
                [*opts, f"slurm/{SCRIPT[adapter]}.sh"],
                go,
            )
            delay += stagger if not extra else 0

        # turbo takes its SVD bases from the tinylora-grpo twin's export (written at its first start) and lora_v from a
        # snapshot or checkpoint of it (first at step 1): a seed whose twin is queued or running is chained to it with
        # `after:<twin>+30` (starts 30 min after the twin first starts, when both files exist; a twin killed before that
        # releases it too, the turbo then fails fast and comes back as `partial` for the next submit), the rest wait
        if adapter == "tinylora-turbo":
            twin = cfg_dir.parent.parent / "tinylora-grpo" / twin_cfg(cfg)
            ready = [s for s in seeds if twin_ready(twin / f"seed{s['seed']}")]
            for s in seeds:
                twin_job = queued.get((str(twin), s["seed"]))
                if s not in ready and twin_job:
                    launch([s], [f"--dependency=after:{twin_job['job']}+30"])
                elif s not in ready:
                    waiting_on_twin += 1
            seeds = ready
        if seeds:
            launch(seeds)
    if waiting_on_twin:
        print(
            f"{waiting_on_twin} turbo seeds wait for their tinylora-grpo twin's final_adapter"
        )
    fill_short(go)


def fill_short(go: bool) -> None:
    """Move one pending training seed into the short QOS (1 job per user, 4 h, +1000 priority) whenever it is free."""
    out = subprocess.run(
        ["squeue", "--me", "-r", "-h", "-o", "%i %q %T %r %j %k"],
        capture_output=True,
        text=True,
    ).stdout
    jobs = [line.split() for line in out.splitlines()]
    if any(qos == "short" for _, qos, *_ in jobs):
        return
    pending = [
        (job, comment)
        for job, _, state, reason, name, comment in jobs
        if state == "PENDING"
        and reason in ("Priority", "Resources")
        and name in ("train_turbo", "train_tinylora")
    ]
    if not pending:
        return
    # 1.5B seeds can finish inside 4 h; among them the deep u<=4 b1-t1 seeds the paper is waiting on
    job, comment = min(
        pending, key=lambda p: ("qwen2.5-1.5b" not in p[1], "-b1-t1" not in p[1])
    )
    print(f"short QOS: {job} ({comment})")
    if go:
        subprocess.run(
            ["scontrol", "update", f"JobId={job}", "QOS=short", "TimeLimit=04:00:00"],
            check=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["status", "submit"])
    parser.add_argument(
        "--go",
        action="store_true",
        help="submit for real (default: print the sbatch lines)",
    )
    parser.add_argument("--only", help="regex on <model>/<adapter-loss>/<cfg>")
    parser.add_argument(
        "--stagger", type=int, default=45, help="seconds between array starts"
    )
    parser.add_argument(
        "--time",
        help="sbatch -t for the launched jobs; short slices survive maintenance reservations, the watcher resubmits what times out",
    )
    args = parser.parse_args()
    rows = scan(args.only)
    (
        status(rows)
        if args.command == "status"
        else submit(rows, args.go, args.stagger, args.time)
    )
