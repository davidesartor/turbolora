"""Shared GRPO/GSPO loop: train_<adapter>.py scripts add adapter args, then call `run`."""

import argparse
import json
import os
import signal
import time
from pathlib import Path

from unsloth import FastLanguageModel  # noqa: F401  must import before trl/transformers

import torch
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

from turbolora.adapters import Adapter
from turbolora.models import MODELS
from turbolora.tasks import TASKS, reward



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


class ResourceLogger(TrainerCallback):
    """Adds peak VRAM and elapsed wall time to every logged step."""

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
        logs["elapsed_hours"] = (time.time() - self.start) / 3600


def argument_parser() -> argparse.ArgumentParser:
    """Arguments shared by every train_<adapter>.py; adapter-specific ones get added on top."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--out", required=True, help="output dir (unique per run)")
    parser.add_argument("--loss", choices=["grpo", "gspo"], default="grpo")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-completion", type=int, default=1024)
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="cap optimizer steps (smoke runs)"
    )
    return parser


def run(args: argparse.Namespace, adapter: type[Adapter], rank: int, **adapter_kwargs) -> None:
    """Load the base, attach `adapter`, train with GRPO/GSPO, export to `<out>/final_adapter`, write run.json."""
    args.out = str(Path(args.out).resolve())
    spec = MODELS[args.model]

    max_prompt_length = 512  # 75 of 8521 hard prompts exceed it and are dropped below

    # smaller GPUs (L40S 48G, A100 40G): give vLLM a larger share so its KV cache stays usable
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    vllm_share = 0.5 if vram_gb < 60 else 0.45

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=spec.hf_id,
        max_seq_length=max_prompt_length + args.max_completion,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=max(rank, 8),  # vLLM accepts only {1, 8, 16, ...}; pads smaller adapters
        gpu_memory_utilization=vllm_share,
    )
    model = adapter.attach(model, rank, args.seed, **adapter_kwargs)

    # raw-text prompts as in SimpleRL-Zoo: TRL then skips the tokenizer's chat template
    dataset = TASKS[args.task]("train").map(
        lambda r: {"prompt": spec.prompt(r["question"])}
    )
    # the colocated vLLM path never truncates prompts, and an over-long batch crashes Unsloth's compiled loss
    dataset = dataset.filter(lambda r: len(tokenizer(r["prompt"]).input_ids) <= max_prompt_length)

    # paper setup: 64 problems x 4 generations = 256 completions per optimizer step
    config = GRPOConfig(
        output_dir=args.out,
        run_name=Path(args.out).name,
        use_vllm=True,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        optim="adamw_8bit",
        num_generations=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=64,
        beta=0.0,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        max_prompt_length=max_prompt_length,
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
        report_to="wandb",
        log_completions=True,
        num_completions_to_print=16,
        wandb_log_unique_prompts=True,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward],
        args=config,
        train_dataset=dataset,
        callbacks=[ResourceLogger(), SaveOnPreempt()],
    )

    last_checkpoint = get_last_checkpoint(args.out) if Path(args.out).is_dir() else None
    start = time.time()
    os.chdir(args.out)  # Unsloth writes its vLLM LoRA stub (grpo_trainer_lora_model_*) to cwd
    trainer.train(resume_from_checkpoint=last_checkpoint)

    adapter_dir = Path(args.out) / "final_adapter"
    adapter.export(model, str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"saved adapter to {adapter_dir}")

    # resource summary the dashboard plots accuracy against
    peak_vram_gb = torch.cuda.max_memory_allocated() / 2**30
    summary = dict(
        model=args.model,
        task=args.task,
        adapter=adapter.__name__.lower(),
        loss=args.loss,
        rank=rank,
        **adapter_kwargs,
        lr=args.lr,
        seed=args.seed,
        steps=trainer.state.global_step,
        params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        train_hours=round((time.time() - start) / 3600, 3),
        peak_vram_gb=round(peak_vram_gb, 2),
        gpu=torch.cuda.get_device_name(0),
    )
    (Path(args.out) / "run.json").write_text(json.dumps(summary, indent=1))
    print(f"peak VRAM: {peak_vram_gb:.1f} GiB of {vram_gb:.0f}, {summary['params']} trainable params")
