"""BO training of TinyLoRA: θ = every v concatenated (one global v by default), each trial scored by vLLM pass rate on a random train subset."""

import argparse
import json
import math
import time
from pathlib import Path

import unsloth  # noqa: F401  must import before peft/transformers

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn.utils import vector_to_parameters

from turbolora import bo, grpo
from turbolora.adapters import Adapter, TinyLoRA
from turbolora.models import MODELS
from turbolora.tasks import TASKS, extract, grade
from vllm import SamplingParams


def argument_parser() -> argparse.ArgumentParser:
    parser = bo.argument_parser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--out", required=True, help="output dir (unique per run)")
    parser.add_argument("--max-completion", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=2, help="frozen truncated-SVD rank")
    parser.add_argument("--proj-dim", type=int, default=1, help="u: entries per v")
    parser.add_argument(
        "--tie",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="one global v shared by every module (--no-tie: one per module, θ ∈ ℝ^{u·modules})",
    )
    # objective: mean per-question pass rate on a fresh random train subset each trial (one GRPO step of samples by default)
    parser.add_argument("--n-questions", type=int, default=grpo.PROMPTS_PER_STEP)
    parser.add_argument("--k-rollouts", type=int, default=grpo.ROLLOUTS_PER_PROMPT)
    return parser


def max_grpo_displacement(n_prompts: int, rank: int, proj_dim: int) -> float:
    """How far Adam can carry one entry of v on train_tinylora's default schedule: ~lr per step, every step of every epoch."""
    lr = grpo.R_STEP_NORM / (rank * proj_dim**0.5)
    epochs = grpo.argument_parser().get_default("epochs")
    return lr * epochs * math.ceil(n_prompts / grpo.PROMPTS_PER_STEP)


def run(args: argparse.Namespace, adapter: type[Adapter] = TinyLoRA) -> None:
    """Load the base, attach `adapter`, BO-search its v's, export the pick to `<out>/final_adapter`, write run.json."""
    from vllm import SamplingParams

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    spec = MODELS[args.model]
    start = time.time()

    model, tokenizer = grpo.load_model(
        spec,
        adapter,
        args.rank,
        args.seed,
        args.max_completion,
        proj_dim=args.proj_dim,
        tie=0 if args.tie else 1,
    )
    vs = [p for p in model.parameters() if p.requires_grad]  # θ = every v, concatenated
    dim = sum(v.numel() for v in vs)
    model.requires_grad_(False)  # search only: no autograd graph anywhere
    print(f"{adapter.__name__}: {len(vs)} v's, θ ∈ ℝ^{dim}")

    dataset = TASKS[args.task]("train")
    candidate_dir = out / "candidate"
    if args.theta_range is None:
        args.theta_range = max_grpo_displacement(len(dataset), args.rank, args.proj_dim)
        print(f"theta range ±{args.theta_range:.4f} (GRPO max displacement)")

    def objective(theta: Float[Tensor, "T"], trial: int) -> tuple[float, float]:
        """Mean over questions of the K-rollout pass rate; SEM across questions, floored."""
        rng = np.random.default_rng(args.seed + trial)
        batch = dataset.select(
            rng.choice(len(dataset), args.n_questions, replace=False)
        )
        vector_to_parameters(theta.to(vs[0]), vs)
        # same call grpo.Snapshot uses: a fresh-id LoRARequest built from the live state_dict, no export
        request = model.load_lora(str(candidate_dir), load_tensors=True)
        outputs = model.fast_generate(
            [spec.prompt(q) for q in batch["question"]],
            lora_request=request,
            use_tqdm=False,
            sampling_params=SamplingParams(
                n=args.k_rollouts,
                temperature=1.0,  # GRPO's rollout sampler TRL default
                max_tokens=args.max_completion,
                stop=list(spec.prompt.stop),
                seed=args.seed + trial,
            ),
        )
        scores = np.array(
            [
                np.mean([grade(extract(o.text), a) for o in r.outputs])
                for r, a in zip(outputs, batch["answer"])
            ]
        )
        # noise from the Beta(½,½) posterior on the pass rate: never 0, even when every question scores the same
        n, m = len(scores), (scores.sum() + 0.5) / (len(scores) + 1)
        return float(scores.mean()), float(math.sqrt(m * (1 - m) / (n + 2)))

    chosen = bo.search(args, objective, dim, args.n_questions, out)
    if chosen is None:
        return

    adapter_dir = out / "final_adapter"
    vector_to_parameters(torch.tensor(chosen["theta"]).to(vs[0]), vs)
    adapter.export(model, str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"saved adapter to {adapter_dir}")

    # resource summary the dashboard plots accuracy against (same keys as grpo.run)
    summary = dict(
        model=args.model,
        task=args.task,
        adapter=adapter.__name__.lower(),
        loss="bo",
        rank=args.rank,
        proj_dim=args.proj_dim,
        tie=0 if args.tie else 1,
        seed=args.seed,
        steps=chosen["steps"],
        params=dim,
        theta=chosen["theta"],
        theta_range=args.theta_range,
        baseline=chosen["baseline"],
        baseline_sem=chosen["baseline_sem"],
        train_hours=round((time.time() - start) / 3600, 3),
        peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        gpu=torch.cuda.get_device_name(0),
    )
    (out / "run.json").write_text(json.dumps(summary, indent=1))


def main() -> None:
    run(argument_parser().parse_args())


if __name__ == "__main__":
    main()
