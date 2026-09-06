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
from safetensors.torch import save_file
from transformers import TrainerCallback
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


class Snapshot(TrainerCallback):
    """At steps 1, 2, 4, ... and the last: save the trainable tensors to snapshots/step-N and eval greedily on the full test sets.

    The last snapshot is also evaluated sampled (K=4, T=1, GRPO's rollout setting), written as <task>@4 like eval.py --samples 4.

    Only the last snapshot also gets the full PEFT export, which eval.py loads standalone; earlier ones
    are rebuilt by `Adapter.attach(model, rank, seed)` + loading trainable.safetensors (the frozen bases are seeded).
    """

    def __init__(
        self,
        model,
        adapter: type[Adapter],
        spec: Model,
        tasks: list[str],
        max_tokens: int,
        root: Path,
    ):
        self.model, self.adapter, self.spec, self.root = model, adapter, spec, root
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
        """Write snapshots/step-N (trainable tensors, PEFT export if `last`) and eval it on every task."""
        out_dir = self.root / f"step-{step:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        params = dict(self.model.named_parameters())
        trainable = {n: params[n].detach().cpu().contiguous() for n in self.trainable}
        save_file(trainable, out_dir / "trainable.safetensors")
        if last:
            self.adapter.export(self.model, str(out_dir))
        # same call the rollout path uses: a LoRARequest built from the live state_dict
        request = self.model.load_lora(str(self.root / "eval_lora"), load_tensors=True)
        for sampling in [self.greedy, self.sampled] if last else [self.greedy]:
            generate = lambda prompts: [
                [c.text for c in o.outputs]
                for o in self.model.fast_generate(
                    prompts, sampling, use_tqdm=False, lora_request=request
                )
            ]
            suffix = f"@{sampling.n}" if sampling.n > 1 else ""
            for task, dataset in self.datasets.items():
                records = evaluate(generate, self.spec, dataset)
                stats = summarize(records)
                print(
                    f"[step {step} {task}{suffix}] accuracy: {stats['accuracy']:.4f} "
                    f"({stats['n_correct']}/{stats['n'] * sampling.n})"
                )
                write_result(
                    out_dir, task, stats, records, suffix, step=step, temperature=sampling.temperature
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
        default=["gsm8k", "math500", "aime24", "amc23", "minerva", "olympiad"],
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

    model, tokenizer = load_model(spec, adapter, rank, args.seed, args.max_completion, **adapter_kwargs)

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
        Snapshot(
            model,
            adapter,
            spec,
            [] if args.no_eval else args.eval_tasks,
            args.max_completion,
            Path(args.out) / "snapshots",
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

    last_checkpoint = get_last_checkpoint(args.out) if Path(args.out).is_dir() else None
    start = time.time()
    os.chdir(args.out)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # the last snapshot holds the final adapter; the resume checkpoints have nothing else
    for checkpoint in Path(args.out).glob("checkpoint-*"):
        shutil.rmtree(checkpoint)

    # resource summary the dashboard plots accuracy against
    peak_vram_gb = torch.cuda.max_memory_allocated() / 2**30
    summary |= dict(
        steps=trainer.state.global_step,
        train_hours=round((time.time() - start) / 3600, 3),
        peak_vram_gb=round(peak_vram_gb, 2),
    )
    (Path(args.out) / "run.json").write_text(json.dumps(summary, indent=1))
    print(
        f"peak VRAM: {peak_vram_gb:.1f} GiB, {summary['params']} trainable params"
    )
