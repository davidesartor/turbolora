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
from vllm import SamplingParams

from turbolora.adapters import Adapter
from turbolora.eval import evaluate, summarize
from turbolora.models import MODELS, Model
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


class FullEval(TrainerCallback):
    """Greedy full-test-set eval every `eval_steps` steps through the colocated vLLM engine, at the training completion budget."""

    def __init__(
        self, model, spec: Model, tasks: list[str], eval_steps: int, max_tokens: int
    ):
        self.model, self.spec, self.eval_steps = model, spec, eval_steps
        self.datasets = {task: TASKS[task]("test") for task in tasks}
        self.sampling = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, stop=list(spec.prompt.stop)
        )
        self.trainer = (
            None  # set after construction; .log routes metrics to wandb and log_history
        )

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps:
            return
        # same call the rollout path uses: a LoRARequest built from the live state_dict
        request = self.model.load_lora("eval_lora", load_tensors=True)
        generate = lambda prompts: [
            o.outputs[0].text
            for o in self.model.fast_generate(
                prompts, self.sampling, use_tqdm=False, lora_request=request
            )
        ]
        for task, dataset in self.datasets.items():
            stats = summarize(evaluate(generate, self.spec, dataset))
            print(
                f"[step {state.global_step} {task}] accuracy: {stats['accuracy']:.4f} ({stats['n_correct']}/{stats['n']})"
            )
            self.trainer.log(
                {
                    f"eval_{task}": stats["accuracy"],
                    f"eval_{task}_unparsed": stats["unparsed"],
                }
            )


class ResourceLogger(TrainerCallback):
    """Adds peak VRAM and elapsed wall time to every logged step."""

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
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=25,
        help="full-bench eval through the training engine every N steps (0 = off)",
    )
    parser.add_argument(
        "--eval-tasks", nargs="+", choices=TASKS, default=["gsm8k", "math500"]
    )
    return parser


def run(
    args: argparse.Namespace, adapter: type[Adapter], rank: int, **adapter_kwargs
) -> None:
    """Load the base, attach `adapter`, train with GRPO/GSPO, export to `<out>/final_adapter`, write run.json."""
    args.out = str(Path(args.out).resolve())
    outputs_dir = Path("outputs").resolve()
    run_name = (
        str(Path(args.out).relative_to(outputs_dir))
        if Path(args.out).is_relative_to(outputs_dir)
        else Path(args.out).name
    )
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
        max_lora_rank=max(rank, 8),
        gpu_memory_utilization=vllm_share,
    )
    model = adapter.attach(model, rank, args.seed, **adapter_kwargs)

    # raw-text prompts as in SimpleRL-Zoo: TRL then skips the tokenizer's chat template
    dataset = TASKS[args.task]("train").map(
        lambda r: {"prompt": spec.prompt(r["question"])}
    )
    # the colocated vLLM path never truncates prompts, and an over-long batch crashes Unsloth's compiled loss
    dataset = dataset.filter(
        lambda r: len(tokenizer(r["prompt"]).input_ids) <= max_prompt_length
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
    callbacks: list[TrainerCallback] = [SaveOnPreempt()]
    if args.eval_steps:
        callbacks.append(
            FullEval(model, spec, args.eval_tasks, args.eval_steps, args.max_completion)
        )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward],
        args=config,
        train_dataset=dataset,
        callbacks=callbacks,
    )
    if args.eval_steps:
        callbacks[-1].trainer = trainer
    # must precede the WandbCallback (which Trainer puts before user callbacks) so wandb sees the extra keys
    trainer.callback_handler.callbacks.insert(0, ResourceLogger())

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

    adapter_dir = Path(args.out) / "final_adapter"
    adapter.export(model, str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"saved adapter to {adapter_dir}")

    # resource summary the dashboard plots accuracy against
    peak_vram_gb = torch.cuda.max_memory_allocated() / 2**30
    summary |= dict(
        steps=trainer.state.global_step,
        train_hours=round((time.time() - start) / 3600, 3),
        peak_vram_gb=round(peak_vram_gb, 2),
    )
    (Path(args.out) / "run.json").write_text(json.dumps(summary, indent=1))
    print(
        f"peak VRAM: {peak_vram_gb:.1f} GiB of {vram_gb:.0f}, {summary['params']} trainable params"
    )
