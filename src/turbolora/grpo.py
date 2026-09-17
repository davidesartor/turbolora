"""Shared GRPO/GSPO loop: train_<adapter>.py scripts add adapter args, then call `run`."""

import argparse
import json
import os
import shutil
import signal
import time
from pathlib import Path

from unsloth import FastLanguageModel  # must import before trl/transformers

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

from turbolora.adapters import Adapter
from turbolora.eval import evaluate, summarize, write_result
from turbolora.models import MODELS, Model
from turbolora.tasks import TASKS, reward

# target per-step ‖ΔR‖_F: Adam moves each param ~lr/step, so lr = R_STEP_NORM / (r·√u)
R_STEP_NORM = 1e-3
PROMPTS_PER_STEP = 64
ROLLOUTS_PER_PROMPT = 4
MAX_PROMPT_LENGTH = 512  # 75 of 8521 hard prompts exceed it and are dropped
EVAL_TASKS = ["gsm8k", "math500", "aime24", "amc23", "minerva", "olympiad"]


def patch_config_pad_token(hf_id: str, pad_token_id: int):
    """Unsloth synthesizes (and then rejects) a pad token unless the model config already declares one."""
    from_pretrained = AutoConfig.from_pretrained

    def from_pretrained_with_pad(name, *args, **kwargs):
        config = from_pretrained(name, *args, **kwargs)
        if name == hf_id:
            config.pad_token_id = pad_token_id
        return config

    AutoConfig.from_pretrained = from_pretrained_with_pad


def load_model(
    spec: Model,
    adapter: type[Adapter],
    rank: int,
    seed: int,
    max_completion: int,
    vllm_share: float | None = None,
    max_loras: int = 1,
    **adapter_kwargs,
):
    """Base weights + colocated vLLM (one weight copy), with `adapter` attached; shared by the GRPO and BO trainers."""
    # GRPO leaves half the card for training state; smaller GPUs (L40S 48G, A100 40G) give vLLM a bit more for KV cache
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    if vllm_share is None:
        vllm_share = 0.5 if vram_gb < 60 else 0.45
    if spec.pad_token_id is not None:
        patch_config_pad_token(spec.hf_id, spec.pad_token_id)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=spec.hf_id,
        max_seq_length=MAX_PROMPT_LENGTH + max_completion,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=max(rank, 8),  # vLLM accepts only {1, 8, 16, ...}; pads smaller adapters
        gpu_memory_utilization=vllm_share,
        max_loras=max_loras,  # adapters vLLM can serve in one batch (forwarded to unsloth's load_vllm)
    )
    return adapter.attach(model, rank, seed, **adapter_kwargs), tokenizer


class SaveOnPreempt(TrainerCallback):
    """Slurm signals before preemption/wall limit; checkpoint at the next step so the requeue resumes from it."""

    def __init__(self):
        self.requested = False
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(sig, lambda *_: setattr(self, "requested", True))

    def on_step_end(self, args, state, control, **kwargs):
        if self.requested:
            control.should_save = True
            self.requested = False


class CheckResume(TrainerCallback):
    """Once the checkpoint is restored, assert the rebuilt adapter reproduces its materialized lora_B (same SVD signs and P).

    Tolerance: bf16 storage costs ~0.3%; near-degenerate singular pairs (e.g. Qwen2.5-7B layer-0 up_proj, σ₂/σ₃ = 4.711/4.705)
    give solver-dependent directions worth up to ~2%; a flipped sign is ≥13%.
    """

    def __init__(self, model, checkpoint: str):
        self.model, self.checkpoint = model, checkpoint

    def on_train_begin(self, args, state, control, **kwargs):
        saved = load_file(f"{self.checkpoint}/adapter_model.safetensors")
        live = self.model.state_dict()
        for key, expected in saved.items():
            if not key.endswith(".lora_B.weight"):
                continue
            actual = live[key.replace(".lora_B.weight", ".lora_B.default.weight")].detach().float().cpu()
            error = (actual - expected.float()).norm() / max(expected.float().norm(), 1e-12)
            if error > 5e-2:
                raise RuntimeError(f"resumed adapter differs from {self.checkpoint} at {key} (rel err {error:.3f}): SVD bases or P mismatch")
        print(f"resumed adapter matches {self.checkpoint}")


class Snapshot(TrainerCallback):
    """At steps 1, 2, 4, ... and the last: save the trainable tensors to snapshots/step-N and eval greedily on the full test sets.

    The last snapshot is also evaluated sampled (K=4, T=1, GRPO's rollout setting), written under eval@4 like eval.py --samples 4.

    Each snapshot also refreshes the run's final_adapter PEFT export, which eval.py loads standalone; an earlier snapshot is that
    export with its trainable tensors swapped in (SVD adapters: `attach(..., bases=<its lora_A>)` + trainable.safetensors).
    """

    def __init__(
        self,
        model,
        adapter: type[Adapter],
        spec: Model,
        tasks: list[str],
        max_tokens: int,
        root: Path,
        export_dir: Path,
    ):
        self.model, self.adapter, self.spec, self.root, self.export_dir = model, adapter, spec, root, export_dir
        self.trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        self.datasets = {task: TASKS[task]("test") for task in tasks}
        stop = list(spec.prompt.stop)
        self.greedy = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop=stop)
        self.sampled = SamplingParams(n=4, temperature=1.0, max_tokens=max_tokens, stop=stop)

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step & (step - 1) and step != state.max_steps:  # not a power of two
            return
        self.save_and_eval(step, last=step == state.max_steps)

    def save_and_eval(self, step: int, last: bool) -> Path:
        """Write snapshots/step-N (trainable tensors), refresh the final_adapter export, and eval on every task."""
        out_dir = self.root / f"step-{step:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        params = dict(self.model.named_parameters())
        trainable = {n: params[n].detach().cpu().contiguous() for n in self.trainable}
        save_file(trainable, out_dir / "trainable.safetensors")
        self.adapter.export(self.model, str(self.export_dir))
        # same call the rollout path uses: a LoRARequest built from the live state_dict, sharing Unsloth's config-only
        # placeholder dir (relative to the run dir, its name is <trainer file>_lora_model_<CUDA_VISIBLE_DEVICES>)
        placeholder = "grpo_trainer_lora_model_" + os.environ.get("CUDA_VISIBLE_DEVICES", "0").replace(",", "")
        request = self.model.load_lora(placeholder, load_tensors=True)
        for sampling in [self.greedy, self.sampled] if last else [self.greedy]:
            generate = lambda prompts: [
                [c.text for c in o.outputs]
                for o in self.model.fast_generate(
                    prompts, sampling, use_tqdm=False, lora_request=request
                )
            ]
            for task, dataset in self.datasets.items():
                records = evaluate(generate, self.spec, dataset)
                stats = summarize(records)
                print(
                    f"[step {step} {task}@{sampling.n}] accuracy: {stats['accuracy']:.4f} "
                    f"({stats['n_correct']}/{stats['n'] * sampling.n})"
                )
                write_result(
                    out_dir, task, stats, records, sampling.n, step=step, temperature=sampling.temperature
                )
        return out_dir


class CurveLogger(TrainerCallback):
    """Stamps peak VRAM and elapsed wall time on every logged step and rewrites curves.jsonl from log_history."""

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        extra = dict(
            peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
            elapsed_hours=(time.time() - self.start) / 3600,
        )
        logs.update(extra)
        # Trainer.log copies into log_history before on_log, so stamp the checkpointed copy too
        if state.log_history and state.log_history[-1].get("step") == state.global_step:
            state.log_history[-1].update(extra)
        # rewritten whole every step: resume-safe, and the curve outlives the rotating checkpoints
        with (Path(args.output_dir) / "curves.jsonl").open("w") as f:
            f.writelines(json.dumps(row) + "\n" for row in state.log_history)


def argument_parser() -> argparse.ArgumentParser:
    """Arguments shared by every train_<adapter>.py; adapter-specific ones get added on top."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--out", required=True, help="output dir (unique per run)")
    parser.add_argument("--loss", choices=["grpo", "gspo"], default="grpo")
    parser.add_argument(
        "--lr", type=float, default=None, help="override the per-adapter default"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-completion", type=int, default=1024)
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="cap optimizer steps (smoke runs)"
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="snapshot adapters only, skip their evals",
    )
    parser.add_argument(
        "--eval-tasks",
        nargs="+",
        choices=TASKS,
        default=EVAL_TASKS,
    )
    return parser


def run(
    args: argparse.Namespace, adapter: type[Adapter], rank: int, **adapter_kwargs
) -> None:
    """Load the base, attach `adapter`, train with GRPO/GSPO, snapshot+eval at steps 1, 2, 4, ..., write run.json."""
    args.out = str(Path(args.out).resolve())
    outputs_dir = Path("outputs").resolve()
    run_name = (
        str(Path(args.out).relative_to(outputs_dir))
        if Path(args.out).is_relative_to(outputs_dir)
        else Path(args.out).name
    )
    spec = MODELS[args.model]

    # final_adapter is the PEFT export of the current state, written at first start and refreshed at every snapshot; every
    # restart (and BO) reuses its lora_A as the SVD bases, since solvers on different cards can flip singular-pair signs.
    # A resume also passes the checkpoint's lora_B and lora_v, which pin U's signs to the ones the run actually trained with.
    final_adapter = Path(args.out) / "final_adapter"
    # a finish that could not delete its last checkpoint (NFS .nfs* stub) leaves an empty checkpoint-N: never resume from it
    for husk in Path(args.out).glob("checkpoint-*"):
        if not (husk / "adapter_model.safetensors").is_file():
            shutil.rmtree(husk, ignore_errors=True)
    last_checkpoint = get_last_checkpoint(args.out) if Path(args.out).is_dir() else None
    # a kill between the final eval@4 (written only at the last step) and the stamp leaves a finished run with no
    # checkpoint: stamp it from the snapshot and curves instead of retraining from scratch
    snapshots = sorted(Path(args.out).glob("snapshots/step-*"), key=lambda p: int(p.name.split("-")[1]))
    if not last_checkpoint and snapshots and (snapshots[-1] / "eval@4" / "summary.json").is_file():
        curves = [json.loads(line) for line in (Path(args.out) / "curves.jsonl").read_text().splitlines()]
        run_json = Path(args.out) / "run.json"
        summary = json.loads(run_json.read_text()) | dict(
            steps=int(snapshots[-1].name.split("-")[1]),
            train_hours=round(max(r.get("elapsed_hours", 0) for r in curves), 3),
            peak_vram_gb=round(max(r.get("peak_vram_gib", 0) for r in curves), 2),
        )
        run_json.write_text(json.dumps(summary, indent=1))
        print(f"already finished at step {summary['steps']}: stamped run.json, nothing to train")
        return
    bases_export = final_adapter / "adapter_model.safetensors"
    bases = None
    if last_checkpoint:
        bases = {k: v for k, v in load_file(f"{last_checkpoint}/adapter_model.safetensors").items() if ".lora_" in k}
    elif bases_export.is_file():
        # a fresh start over an export (e.g. a twin's copied final_adapter) pins U to that export's lora_B too; the export
        # strips lora_v, so a run killed before its first checkpoint (step 25) takes v from the snapshot the export was made at
        bases = {k: v for k, v in load_file(bases_export).items() if ".lora_" in k}
        if snapshots and (snapshots[-1] / "trainable.safetensors").is_file():
            vs = {k.removesuffix(".default"): v for k, v in load_file(snapshots[-1] / "trainable.safetensors").items()}
            for name in [k.removesuffix(".lora_B.weight") for k in bases if k.endswith(".lora_B.weight")]:
                bases[f"{name}.lora_v"] = vs[f"{name}.lora_v"] if len(vs) > 1 else next(iter(vs.values()))
    model, tokenizer = load_model(spec, adapter, rank, args.seed, args.max_completion, bases=bases, **adapter_kwargs)
    if not (final_adapter / "adapter_model.safetensors").is_file():
        adapter.export(model, str(final_adapter))

    # raw-text prompts as in SimpleRL-Zoo: TRL then skips the tokenizer's chat template
    dataset = TASKS[args.task]("train").map(
        lambda r: {"prompt": spec.prompt(r["question"])}
    )
    # the colocated vLLM path never truncates prompts, and an over-long batch crashes Unsloth's compiled loss
    dataset = dataset.filter(
        lambda r: len(tokenizer(r["prompt"]).input_ids) <= MAX_PROMPT_LENGTH
    )

    # paper setup: 64 problems x 4 generations = 256 completions per optimizer step
    config = GRPOConfig(
        output_dir=args.out,
        run_name=run_name,
        use_vllm=True,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        optim="adamw_8bit",
        num_generations=ROLLOUTS_PER_PROMPT,
        per_device_train_batch_size=ROLLOUTS_PER_PROMPT,
        gradient_accumulation_steps=PROMPTS_PER_STEP,
        beta=0.0,
        # dual-clip PPO: caps ratio·|A| for A<0 tokens, whose bf16 old-logp noise otherwise gives unbounded grad spikes
        delta=2.0,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=args.max_completion,
        generation_kwargs={"stop": list(spec.prompt.stop)},
        seed=args.seed,
        # GSPO (arXiv 2507.18071) = GRPO with sequence-level importance ratios and tighter clipping
        importance_sampling_level="sequence" if args.loss == "gspo" else "token",
        epsilon=3e-4 if args.loss == "gspo" else 0.2,
        epsilon_high=4e-4 if args.loss == "gspo" else None,
        # force 256/64=4-row scoring chunks; unsloth's autotuner sizes chunks from
        # free VRAM at first call and OOMs when vLLM's share is resident
        unsloth_grpo_mini_batch=64,
        logging_steps=1,
        save_steps=25,
        save_total_limit=2,
        report_to="none",
    )
    callbacks: list[TrainerCallback] = [
        SaveOnPreempt(),
        *([CheckResume(model, last_checkpoint)] if last_checkpoint else []),
        Snapshot(
            model,
            adapter,
            spec,
            [] if args.no_eval else args.eval_tasks,
            args.max_completion,
            Path(args.out) / "snapshots",
            final_adapter,
        ),
    ]
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward],
        args=config,
        train_dataset=dataset,
        callbacks=callbacks,
    )
    # runs before the other callbacks so its extra keys are in the log dict they see
    trainer.callback_handler.callbacks.insert(0, CurveLogger())

    # config half of run.json goes out before training so the dashboard can show the run while it trains
    summary = dict(
        model=args.model,
        task=args.task,
        adapter=adapter.__name__.lower(),
        loss=args.loss,
        rank=rank,
        **adapter_kwargs,
        lr=args.lr,
        seed=args.seed,
        params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        gpu=torch.cuda.get_device_name(0),
    )
    (Path(args.out) / "run.json").write_text(json.dumps(summary, indent=1))

    start = time.time()
    os.chdir(args.out)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # resource summary the dashboard plots accuracy against; stamped before any cleanup so a cleanup error can't hide a finished run
    peak_vram_gb = torch.cuda.max_memory_allocated() / 2**30
    summary |= dict(
        steps=trainer.state.global_step,
        train_hours=round((time.time() - start) / 3600, 3),
        peak_vram_gb=round(peak_vram_gb, 2),
    )
    (Path(args.out) / "run.json").write_text(json.dumps(summary, indent=1))

    # the last snapshot holds the final adapter; the resume checkpoints have nothing else. Best effort: a file still
    # open in-process becomes an NFS .nfs* stub that only vanishes at exit, and the next start prunes the empty dir
    for checkpoint in Path(args.out).glob("checkpoint-*"):
        shutil.rmtree(checkpoint, ignore_errors=True)
    print(
        f"peak VRAM: {peak_vram_gb:.1f} GiB, {summary['params']} trainable params"
    )
