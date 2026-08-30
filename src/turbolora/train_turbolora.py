"""BO training of TurboLoRA: the shared v's are the search vector, each trial scored by vLLM pass rate on a random train subset."""

import argparse
import json
import math
import time
from pathlib import Path

from unsloth import FastLanguageModel  # must import before peft/transformers

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from turbolora import bo
from turbolora.adapters import Adapter, TurboLoRA
from turbolora.models import MODELS
from turbolora.tasks import TASKS, extract, grade


def argument_parser() -> argparse.ArgumentParser:
    parser = bo.argument_parser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--out", required=True, help="output dir (unique per run)")
    parser.add_argument("--max-completion", type=int, default=1024)
    # objective: mean per-question pass rate on a fresh random train subset each trial
    parser.add_argument("--n-questions", type=int, default=32)
    parser.add_argument("--k-rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--sem-floor", type=float, default=0.01)
    return parser


def run(args: argparse.Namespace, adapter: type[Adapter], rank: int, **adapter_kwargs) -> None:
    """Load the base, attach `adapter`, BO-search its trainable vector, export the pick to `<out>/final_adapter`, write run.json."""
    from vllm import SamplingParams

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    spec = MODELS[args.model]
    start = time.time()

    # same load as grpo.run: in-process vLLM next to the HF weights the adapter wraps
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=spec.hf_id,
        max_seq_length=512 + args.max_completion,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=max(rank, 8),  # vLLM accepts only {1, 8, 16, ...}; pads smaller adapters
        gpu_memory_utilization=0.5 if vram_gb < 60 else 0.45,
    )
    model = adapter.attach(model, rank, args.seed, **adapter_kwargs)
    # θ = the adapter's trainable parameters (shared ones appear once), whatever their structure
    params = [p for p in model.parameters() if p.requires_grad]
    dim = sum(p.numel() for p in params)
    print(f"{adapter.__name__}: θ ∈ ℝ^{dim}")

    def set_theta(theta: Float[Tensor, "T"]) -> None:
        with torch.no_grad():
            for p, chunk in zip(params, theta.split([p.numel() for p in params])):
                p.copy_(chunk.view_as(p))

    dataset = TASKS[args.task]("train")
    candidate_dir = out / "candidate"

    def objective(theta: Float[Tensor, "T"], trial: int) -> tuple[float, float]:
        """Mean over questions of the K-rollout pass rate; SEM across questions, floored."""
        rng = np.random.default_rng(args.seed + trial)
        batch = dataset.select(rng.choice(len(dataset), args.n_questions, replace=False))
        set_theta(theta)
        adapter.export(model, str(candidate_dir))
        sampling = SamplingParams(
            n=args.k_rollouts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_completion,
            stop=list(spec.prompt.stop),
            seed=args.seed + trial,
        )
        outputs = model.fast_generate(
            [spec.prompt(q) for q in batch["question"]],
            sampling_params=sampling,
            lora_request=model.load_lora(str(candidate_dir)),  # fresh vLLM adapter id each call
            use_tqdm=False,
        )
        scores = np.array([np.mean([grade(extract(o.text), a) for o in r.outputs]) for r, a in zip(outputs, batch["answer"])])
        return float(scores.mean()), max(float(scores.std(ddof=1) / math.sqrt(len(scores))), args.sem_floor)

    chosen = bo.search(args, objective, dim, args.n_questions, out)
    if chosen is None:
        return

    adapter_dir = out / "final_adapter"
    set_theta(torch.tensor(chosen["theta"]))
    adapter.export(model, str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"saved adapter to {adapter_dir}")

    # resource summary the dashboard plots accuracy against (same keys as grpo.run)
    summary = dict(
        model=args.model,
        task=args.task,
        adapter=adapter.__name__.lower(),
        loss="bo",
        rank=rank,
        **adapter_kwargs,
        seed=args.seed,
        steps=chosen["steps"],
        params=dim,
        theta=chosen["theta"],
        train_hours=round((time.time() - start) / 3600, 3),
        peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        gpu=torch.cuda.get_device_name(0),
    )
    (out / "run.json").write_text(json.dumps(summary, indent=1))


def main() -> None:
    parser = argument_parser()
    parser.add_argument("--rank", type=int, default=2, help="frozen truncated-SVD rank")
    parser.add_argument("--proj-dim", type=int, default=1, help="u: entries of v per module group")
    parser.add_argument("--tie", type=int, default=98, help="consecutive modules sharing one v (PoC: 2 groups on a 28-layer model)")
    args = parser.parse_args()
    run(args, TurboLoRA, rank=args.rank, proj_dim=args.proj_dim, tie=args.tie)


if __name__ == "__main__":
    main()
