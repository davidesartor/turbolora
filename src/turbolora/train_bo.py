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
    # objective: mean per-question pass rate on a fresh random train subset each call (one GRPO step of samples by default), split over --batch θ's
    parser.add_argument("--n-questions", type=int, default=grpo.PROMPTS_PER_STEP, help="prompts per objective call, shared by the batch")
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
        vllm_share=0.85,  # no training state: the card is vLLM's, so small cards (L4 24G) work at a lower --n-questions
        max_loras=args.batch,
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

    # same prompt cut as the GRPO trainer: vLLM never truncates, and a prompt over the 1536-token context kills the job
    dataset = TASKS[args.task]("train").filter(
        lambda r: len(tokenizer(spec.prompt(r["question"])).input_ids) <= grpo.MAX_PROMPT_LENGTH
    )
    candidate_dir = out / "candidate"
    if args.theta_range is None:
        args.theta_range = max_grpo_displacement(len(dataset), args.rank, args.proj_dim)
        print(f"theta range ±{args.theta_range:.4f} (GRPO max displacement)")
    if args.n_evals is None:
        args.n_evals = grpo_budget_trials(len(dataset), args.n_questions, args.k_rollouts)
        print(f"n_evals {args.n_evals} batches (GRPO completion budget)")
    if args.n_questions % args.batch:
        raise ValueError("n_questions must be a multiple of batch")
    questions_per_theta = args.n_questions // args.batch

    # config half of run.json goes out before the search so the dashboard can show the run while it runs
    config = dict(
        model=args.model,
        task=args.task,
        adapter=adapter.__name__.lower(),
        loss="bo",
        rank=args.rank,
        proj_dim=args.proj_dim,
        tie=1 if args.untie else 0,
        seed=args.seed,
        params=dim,
        theta_range=args.theta_range,
        max_steps=args.n_evals,
        design=math.ceil((args.n_baseline + args.n_sobol) / args.batch),  # batches before the GP-guided ones
        gpu=torch.cuda.get_device_name(0),
    )
    (out / "run.json").write_text(json.dumps(config, indent=1))

    def objective(thetas: Float[Tensor, "B T"], batch: int) -> list[tuple[float, float]]:
        """Per θ: logit of the K-rollout pass rate over its questions, with its posterior std; one LoRA per θ, one vLLM call."""
        rng = np.random.default_rng(args.seed + batch)
        prompts, answers, requests = [], [], []
        for theta in thetas:
            vector_to_parameters(theta.to(vs[0]), vs)  # θ = every v concatenated
            # same call grpo.Snapshot uses: a fresh-id LoRARequest built from the live state_dict, no export
            request = model.load_lora(str(candidate_dir), load_tensors=True)
            subset = dataset.select(rng.choice(len(dataset), questions_per_theta, replace=False))
            prompts += [spec.prompt(q) for q in subset["question"]]
            answers += subset["answer"]
            requests += [request] * questions_per_theta
        outputs = model.fast_generate(
            prompts,
            lora_request=requests,
            use_tqdm=False,
            sampling_params=SamplingParams(
                n=args.k_rollouts,
                temperature=1.0,  # GRPO's rollout sampler TRL default
                max_tokens=args.max_completion,
                stop=list(spec.prompt.stop),
                seed=args.seed + batch,
            ),
        )
        scores = np.array(
            [np.mean([grade(extract(o.text), a) for o in r.outputs]) for r, a in zip(outputs, answers)]
        ).reshape(len(thetas), questions_per_theta)

        # Jeffreys Beta(½,½) posterior on each θ's pass rate, reported to the GP as its exact moment-matched Gaussian in logit space:
        # logit(X) has cumulants ψ(a)−ψ(b), ψ₁(a)+ψ₁(b) (mean/variance exact; shape is skewed while a shape stays <1)
        a, b = 0.5 + scores.sum(1), 0.5 + questions_per_theta - scores.sum(1)
        return [(float(digamma(ai) - digamma(bi)), float(math.sqrt(polygamma(1, ai) + polygamma(1, bi)))) for ai, bi in zip(a, b)]

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
    summary = config | dict(
        steps=chosen["steps"],
        theta=chosen["theta"],
        # search values are logit pass rates; report the θ=0 baseline back on the accuracy scale
        baseline=torch.sigmoid(torch.tensor(chosen["baseline"])).item(),
        baseline_logit=chosen["baseline"],
        baseline_logit_sem=chosen["baseline_sem"],
        train_hours=round((time.time() - start) / 3600, 3),
        peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
    )
    (out / "run.json").write_text(json.dumps(summary, indent=1))


def main() -> None:
    run(argument_parser().parse_args())


if __name__ == "__main__":
    main()
