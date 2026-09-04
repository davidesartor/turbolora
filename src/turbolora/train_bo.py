"""BO training of TinyLoRA: θ = every v concatenated (one global v unless --untie), each trial scored by the logit vLLM pass rate on a random train subset."""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import unsloth  # noqa: F401  must import before peft/transformers

import numpy as np
import torch
from scipy.special import digamma, polygamma
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
        "--untie",
        action="store_true",
        help="one v per module, θ ∈ ℝ^{u·modules} (default: one global v)",
    )
    # objective: mean per-question pass rate on a fresh random train subset each trial (one GRPO step of samples by default)
    parser.add_argument("--n-questions", type=int, default=grpo.PROMPTS_PER_STEP)
    parser.add_argument("--k-rollouts", type=int, default=grpo.ROLLOUTS_PER_PROMPT)
    parser.add_argument("--no-eval", action="store_true", help="snapshot the pick only, skip its evals")
    parser.add_argument(
        "--eval-tasks",
        nargs="+",
        choices=TASKS,
        default=grpo.argument_parser().get_default("eval_tasks"),
    )
    return parser


def grpo_steps(n_prompts: int) -> int:
    """Optimizer steps train_tinylora takes on its default schedule: every step of every epoch."""
    epochs = grpo.argument_parser().get_default("epochs")
    return epochs * math.ceil(n_prompts / grpo.PROMPTS_PER_STEP)


def grpo_budget_trials(n_prompts: int, n_questions: int, k_rollouts: int) -> int:
    """GP-guided trials that sample as many completions as train_tinylora's default schedule (initial design not counted)."""
    epochs = grpo.argument_parser().get_default("epochs")
    completions = epochs * n_prompts * grpo.ROLLOUTS_PER_PROMPT
    return math.ceil(completions / (n_questions * k_rollouts))


def max_grpo_displacement(n_prompts: int, rank: int, proj_dim: int) -> float:
    """How far Adam can carry one entry of v on train_tinylora's default schedule: ~lr per step."""
    lr = grpo.R_STEP_NORM / (rank * proj_dim**0.5)
    return lr * grpo_steps(n_prompts)


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
        tie=1 if args.untie else 0,
    )
    vs = [
        p for p in model.parameters() if p.requires_grad
    ]  # the distinct v's, module order
    dim = sum(v.numel() for v in vs)
    snapshot = grpo.Snapshot(
        model,
        adapter,
        spec,
        [] if args.no_eval else args.eval_tasks,
        args.max_completion,
        out / "snapshots",
    )
    model.requires_grad_(False)  # search only: no autograd graph anywhere
    print(f"{adapter.__name__}: {len(vs)} v's, θ ∈ ℝ^{dim}")

    dataset = TASKS[args.task]("train")
    candidate_dir = out / "candidate"
    if args.theta_range is None:
        args.theta_range = max_grpo_displacement(len(dataset), args.rank, args.proj_dim)
        print(f"theta range ±{args.theta_range:.4f} (GRPO max displacement)")
    if args.n_evals is None:
        args.n_evals = grpo_budget_trials(len(dataset), args.n_questions, args.k_rollouts)
        print(f"n_evals {args.n_evals} (GRPO completion budget)")

    def objective(theta: Float[Tensor, "T"], trial: int) -> tuple[float, float]:
        """Logit of the K-rollout pass rate over questions, with its posterior std."""
        rng = np.random.default_rng(args.seed + trial)
        batch = dataset.select(
            rng.choice(len(dataset), args.n_questions, replace=False)
        )
        vector_to_parameters(theta.to(vs[0]), vs)  # θ = every v concatenated
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
        scores = [
            np.mean([grade(extract(o.text), a) for o in r.outputs])
            for r, a in zip(outputs, batch["answer"])
        ]
        scores = np.array(scores)

        # Jeffreys Beta(½,½) posterior on the pass rate, reported to the GP as its exact moment-matched Gaussian in logit space:
        # logit(X) has cumulants ψ(a)−ψ(b), ψ₁(a)+ψ₁(b) (mean/variance exact; shape is skewed while a shape stays <1)
        a, b = 0.5 + scores.sum(), 0.5 + len(scores) - scores.sum()
        return float(digamma(a) - digamma(b)), float(math.sqrt(polygamma(1, a) + polygamma(1, b)))

    last_snapshot: dict = {}

    def on_snapshot(step: int, current: dict) -> None:
        """Snapshot+eval the GP's current pick at GP-guided trial 1, 2, 4, ...; an unchanged pick just copies the previous snapshot."""
        last = step == args.n_evals
        if last_snapshot.get("theta") == current["theta"] and not last:
            shutil.copytree(last_snapshot["dir"], out / "snapshots" / f"step-{step:06d}", dirs_exist_ok=True)
            return
        vector_to_parameters(torch.tensor(current["theta"]).to(vs[0]), vs)
        last_snapshot.update(theta=current["theta"], dir=snapshot.save_and_eval(step, last))

    chosen = bo.search(args, objective, dim, out, on_snapshot)
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
        tie=1 if args.untie else 0,
        seed=args.seed,
        steps=chosen["steps"],
        params=dim,
        theta=chosen["theta"],
        theta_range=args.theta_range,
        # search values are logit pass rates; report the θ=0 baseline back on the accuracy scale
        baseline=torch.sigmoid(torch.tensor(chosen["baseline"])).item(),
        baseline_logit=chosen["baseline"],
        baseline_logit_sem=chosen["baseline_sem"],
        train_hours=round((time.time() - start) / 3600, 3),
        peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        gpu=torch.cuda.get_device_name(0),
    )
    (out / "run.json").write_text(json.dumps(summary, indent=1))


def main() -> None:
    run(argument_parser().parse_args())


if __name__ == "__main__":
    main()
