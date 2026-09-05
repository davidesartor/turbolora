"""BO training of TinyLoRA: θ = every v concatenated (one global v unless --untie), each trial scored by the logit vLLM pass rate on a random train subset (greedy by default)."""

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
from safetensors import safe_open
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
    # objective: pass rate on a fresh random train subset; one GRPO step of completions per vLLM call, allocated as (--batch θ's, prompts each, --k-rollouts each)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=True, help="greedy completions (the eval sampler); --no-greedy samples at GRPO's T=1")
    parser.add_argument("--k-rollouts", type=int, default=1, help="completions per prompt (needs --no-greedy if >1)")
    parser.add_argument("--no-eval", action="store_true", help="snapshot the pick only, skip its evals")
    parser.add_argument("--bases", type=Path, default=None, help="LoRA export whose lora_A fixes the SVD signs; default: the tinylora-grpo run with the same model/task/rank/u/seed")
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


def questions_per_theta(batch: int, k_rollouts: int) -> int:
    """Prompts per θ so one objective call generates exactly one GRPO step of completions."""
    completions = grpo.PROMPTS_PER_STEP * grpo.ROLLOUTS_PER_PROMPT
    if completions % (batch * k_rollouts):
        raise ValueError(f"batch · k_rollouts must divide {completions} completions per call")
    return completions // (batch * k_rollouts)


def grpo_bases(args: argparse.Namespace) -> tuple[Path, dict[str, Tensor]]:
    """lora_A of the matching tinylora-grpo export (`--bases` or the same model/task/rank/u/seed under outputs/runs), so BO searches in that run's SVD signs."""
    path = args.bases
    if path is None:
        run = Path("outputs/runs") / args.model / args.task / f"tinylora-grpo-r{args.rank}-u{args.proj_dim}" / f"seed{args.seed}"
        exports = sorted(run.glob("snapshots/step-*/adapter_model.safetensors"))
        if not exports:
            raise FileNotFoundError(f"BO needs a reference export to fix the SVD signs: none under {run}; pass --bases")
        path = exports[-1]
    if not path.is_file():
        raise FileNotFoundError(f"reference export {path} not found")
    with safe_open(str(path), "pt") as f:
        return path, {k: f.get_tensor(k) for k in f.keys() if k.endswith(".lora_A.weight")}


def max_grpo_displacement(n_prompts: int, rank: int, proj_dim: int) -> float:
    """How far Adam can carry one entry of v on train_tinylora's default schedule: ~lr per step."""
    lr = grpo.R_STEP_NORM / (rank * proj_dim**0.5)
    return lr * grpo_steps(n_prompts)


def run(args: argparse.Namespace, adapter: type[Adapter] = TinyLoRA, search: bo.Search = bo.search, loss: str = "bo") -> None:
    """Load the base, attach `adapter`, `search` its v's, export the pick to `<out>/final_adapter`, write run.json."""
    from vllm import SamplingParams

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    spec = MODELS[args.model]
    start = time.time()

    bases_path, bases = grpo_bases(args)
    print(f"SVD bases from {bases_path}")
    model, tokenizer = grpo.load_model(
        spec,
        adapter,
        args.rank,
        args.seed,
        args.max_completion,
        vllm_share=0.85,  # no training state: the card is vLLM's, so small cards (L4 24G) work with a smaller --batch
        max_loras=args.batch,
        proj_dim=args.proj_dim,
        tie=1 if args.untie else 0,
        bases=bases,
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
        args.n_evals = grpo_steps(len(dataset))
        print(f"n_evals {args.n_evals} batches (one GRPO step of completions each)")
    if args.greedy and args.k_rollouts > 1:
        raise ValueError("greedy completions are identical: pass --no-greedy to use several rollouts per prompt")
    n_questions = questions_per_theta(args.batch, args.k_rollouts)

    # config half of run.json goes out before the search so the dashboard can show the run while it runs
    config = dict(
        model=args.model,
        task=args.task,
        adapter=adapter.__name__.lower(),
        loss=loss,
        rank=args.rank,
        proj_dim=args.proj_dim,
        tie=1 if args.untie else 0,
        seed=args.seed,
        params=dim,
        theta_range=args.theta_range,
        max_steps=args.n_evals,
        design=math.ceil((args.n_baseline + args.n_sobol) / args.batch),  # batches before the GP-guided ones
        bases=str(bases_path),
        batch=args.batch,
        k_rollouts=args.k_rollouts,
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
            subset = dataset.select(rng.choice(len(dataset), n_questions, replace=False))
            prompts += [spec.prompt(q) for q in subset["question"]]
            answers += subset["answer"]
            requests += [request] * n_questions
        outputs = model.fast_generate(
            prompts,
            lora_request=requests,
            use_tqdm=False,
            sampling_params=SamplingParams(
                n=args.k_rollouts,
                temperature=0.0 if args.greedy else 1.0,  # eval sampler vs GRPO's rollout sampler (TRL default)
                max_tokens=args.max_completion,
                stop=list(spec.prompt.stop),
                seed=args.seed + batch,
            ),
        )
        scores = np.array(
            [np.mean([grade(extract(o.text), a) for o in r.outputs]) for r, a in zip(outputs, answers)]
        ).reshape(len(thetas), n_questions)

        # uniform Beta(1,1) posterior on each θ's pass rate, reported to the GP as its exact moment-matched Gaussian in logit space:
        # logit(X) has cumulants ψ(a)−ψ(b), ψ₁(a)+ψ₁(b); a shape <1 (Jeffreys) makes the all-pass/all-fail corners badly skewed, 1 keeps them near-Gaussian
        a, b = 1 + scores.sum(1), 1 + n_questions - scores.sum(1)
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

    chosen = search(args, objective, dim, out, on_snapshot)
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
