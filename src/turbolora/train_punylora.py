"""PunyLoRA: TinyLoRA searched by BO on raw hit counts, a batch of θ's per vLLM call (one LoRA each), down to one completion per θ."""

import argparse
import json
import shutil
import time
from pathlib import Path

import unsloth  # noqa: F401  must import before peft/transformers

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn.utils import vector_to_parameters

from turbolora import grpo, puny_lora
from turbolora.adapters import Adapter, TinyLoRA
from turbolora.models import MODELS
from turbolora.tasks import TASKS, extract, grade
from turbolora.train_bo import grpo_budget_trials, max_grpo_displacement


def argument_parser() -> argparse.ArgumentParser:
    parser = puny_lora.argument_parser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--out", required=True, help="output dir (unique per run)")
    parser.add_argument("--max-completion", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=2, help="frozen truncated-SVD rank")
    parser.add_argument("--proj-dim", type=int, default=1, help="u: entries per v")
    parser.add_argument("--untie", action="store_true", help="one v per module, θ ∈ ℝ^{u·modules} (default: one global v)")
    # objective: hits over a fresh random train subset; n_questions · k_rollouts completions per vLLM call, split over --batch θ's
    parser.add_argument("--n-questions", type=int, default=16, help="prompts per objective call, shared by the batch")
    parser.add_argument("--k-rollouts", type=int, default=1, help="completions per prompt")
    parser.add_argument("--vllm-share", type=float, default=0.85, help="gpu_memory_utilization; no training state, so the card is vLLM's")
    parser.add_argument("--no-eval", action="store_true", help="snapshot the pick only, skip its evals")
    parser.add_argument("--eval-tasks", nargs="+", choices=TASKS, default=grpo.argument_parser().get_default("eval_tasks"))
    return parser


def run(args: argparse.Namespace, adapter: type[Adapter] = TinyLoRA) -> None:
    """Load the base, attach `adapter`, BO-search its v's on hit counts, export the pick to `<out>/final_adapter`, write run.json."""
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
        vllm_share=args.vllm_share,
        max_loras=args.batch,
        proj_dim=args.proj_dim,
        tie=1 if args.untie else 0,
    )
    vs = [p for p in model.parameters() if p.requires_grad]  # the distinct v's, module order
    dim = sum(v.numel() for v in vs)
    snapshot = grpo.Snapshot(model, adapter, spec, [] if args.no_eval else args.eval_tasks, args.max_completion, out / "snapshots")
    model.requires_grad_(False)  # search only: no autograd graph anywhere
    print(f"{adapter.__name__}: {len(vs)} v's, θ ∈ ℝ^{dim}")

    dataset = TASKS[args.task]("train")
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

    def objective(thetas: Float[Tensor, "B T"], batch: int) -> list[tuple[int, int]]:
        """Hits and completions per θ: one LoRA per θ, every θ's prompts in a single vLLM call."""
        rng = np.random.default_rng(args.seed + batch)
        prompts, answers, requests = [], [], []
        for theta in thetas:
            vector_to_parameters(theta.to(vs[0]), vs)  # θ = every v concatenated
            # a fresh-id LoRARequest holding this θ's materialized lora_B (the state_dict hook builds a new tensor)
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
        hits = np.array([sum(grade(extract(o.text), a) for o in r.outputs) for r, a in zip(outputs, answers)])
        per_theta = hits.reshape(len(thetas), questions_per_theta).sum(1)
        return [(int(h), questions_per_theta * args.k_rollouts) for h in per_theta]

    last_snapshot: dict = {}

    def on_snapshot(step: int, theta: list[float]) -> None:
        """Snapshot+eval the GP's current pick at GP-guided batch 1, 2, 4, ...; an unchanged pick just copies the previous snapshot."""
        last = step == args.n_evals
        if last_snapshot.get("theta") == theta and not last:
            shutil.copytree(last_snapshot["dir"], out / "snapshots" / f"step-{step:06d}", dirs_exist_ok=True)
            return
        vector_to_parameters(torch.tensor(theta).to(vs[0]), vs)
        last_snapshot.update(theta=theta, dir=snapshot.save_and_eval(step, last))

    chosen = puny_lora.search(args, objective, dim, out, on_snapshot)
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
        loss="puny_lora",
        rank=args.rank,
        proj_dim=args.proj_dim,
        tie=1 if args.untie else 0,
        seed=args.seed,
        steps=chosen["steps"],
        params=dim,
        theta=chosen["theta"],
        theta_range=args.theta_range,
        posterior=chosen["posterior"],
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
