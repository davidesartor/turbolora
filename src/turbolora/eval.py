"""Greedy (pass@1) test-set eval of a model, optionally with a trained adapter, on one or more tasks with vLLM."""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from datasets import Dataset
from huggingface_hub import hf_hub_download

from turbolora.models import MODELS, Model
from turbolora.tasks import TASKS, extract, grade


def evaluate(
    generate: Callable[[list[str]], list[str]],
    spec: Model,
    dataset: Dataset,
) -> list[dict]:
    """One record per example: completion, extracted prediction, answer, and whether it was graded correct."""
    completions = generate([spec.prompt(q) for q in dataset["question"]])
    records = []
    for example, completion in zip(dataset.to_list(), completions):
        predicted = extract(completion)
        correct = grade(predicted, example["answer"])
        graded = dict(predicted=predicted, correct=correct, completion=completion)
        records.append(example | graded)
    return records


def summarize(records: list[dict]) -> dict:
    return {
        "accuracy": sum(r["correct"] for r in records) / len(records),
        "n_correct": sum(r["correct"] for r in records),
        "n": len(records),
        "unparsed": sum(r["predicted"] is None for r in records),
    }


def resolve_run(model: str | None, adapter: str | None, out_dir: str | None) -> tuple[str, Path]:
    """Adapter evals live next to their run.json (model read from it); baselines go under out-dir/<model>."""
    if adapter:
        run_dir = Path(adapter).parent
        model = model or json.loads((run_dir / "run.json").read_text())["model"]
        return model, Path(out_dir) if out_dir else run_dir / "eval"
    if not model:
        raise SystemExit("--model is required without --adapter")
    return model, Path(out_dir or "outputs/baselines") / model


def lora_engine_args(adapters: list[str]) -> dict:
    """vLLM's LoRA kernels need max_lora_rank >= 8 even for a rank-2 adapter."""
    ranks = [
        json.loads((Path(a) / "adapter_config.json").read_text())["r"] for a in adapters
    ]
    return dict(enable_lora=True, max_lora_rank=max(8, *ranks))


def usage(seconds: float) -> dict:
    """Wall time, host RAM peak, and the GPU's name / VRAM as reported by nvidia-smi."""
    query = "--query-gpu=name,memory.used,memory.total"
    gpu = (
        subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .split("\n")[0]
    )
    name, used_mib, total_mib = (v.strip() for v in gpu.split(","))
    return {
        "seconds": round(seconds, 1),
        "host_ram_gb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2
        ),
        "gpu": name,
        "vram_used_gb": round(int(used_mib) / 1024, 2),
        "vram_total_gb": round(int(total_mib) / 1024, 2),
    }


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, help="defaults to the adapter's run.json")
    parser.add_argument(
        "--adapters",
        nargs="+",
        help="PEFT adapter dirs (<run>/final_adapter); each one's results go to its own <run>/eval/",
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size (GPUs)")
    parser.add_argument(
        "--out-dir", help="baselines: <out-dir>/<model>/<task>.json (default outputs/baselines)"
    )
    parser.add_argument(
        "--show", type=int, default=0, help="print first N completions per task"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="leave already-written <task>.json alone"
    )
    args = parser.parse_args()

    # one engine load serves every adapter, so they must all sit on the same base model
    adapters = args.adapters or [None]
    runs = [resolve_run(args.model, adapter, args.out_dir) for adapter in adapters]
    models = {model for model, _ in runs}
    if len(models) > 1:
        raise SystemExit(f"adapters span several base models: {sorted(models)}")
    args.model = models.pop()
    spec = MODELS[args.model]
    config = json.loads(Path(hf_hub_download(spec.hf_id, "config.json")).read_text())
    text_config = config.get("text_config", config)

    # 4k-context models (Qwen2.5-Math, DeepSeek) can't take prompt + 4096 new tokens; shrink the budget
    max_model_len = min(args.max_tokens + 1024, text_config["max_position_embeddings"])
    max_tokens = max_model_len - 1024

    # multimodal bases (Ministral 3): skip the vision tower, whose xformers kernels are missing here
    multimodal: dict = (
        {"limit_mm_per_prompt": {"image": 0}} if "vision_config" in config else {}
    )

    lora = lora_engine_args(args.adapters) if args.adapters else {}

    llm = LLM(
        model=spec.hf_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=args.tp,
        **multimodal,
        **lora,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop=list(spec.prompt.stop),
    )
    datasets = {task: TASKS[task]("test") for task in args.tasks}

    for index, (adapter, (_, out_dir)) in enumerate(zip(adapters, runs)):
        lora_request = (
            LoRARequest(f"adapter{index}", index + 1, str(Path(adapter).resolve()))
            if adapter
            else None
        )
        generate = lambda prompts: [
            o.outputs[0].text
            for o in llm.generate(prompts, sampling, lora_request=lora_request)
        ]

        for task in args.tasks:
            out_path = out_dir / f"{task}.json"
            if args.skip_existing and out_path.exists():
                print(f"skipping {out_path}")
                continue

            start = time.perf_counter()
            records = evaluate(generate, spec, datasets[task])
            stats = summarize(records) | usage(time.perf_counter() - start)

            for record in records[: args.show]:
                print("=" * 80)
                print(f"question:  {record['question']}")
                print(f"completion:\n{record['completion']}")
                print(f"predicted: {record['predicted']}   answer: {record['answer']}")
            print(
                f"[{args.model}/{task}] accuracy: {stats['accuracy']:.4f} "
                f"({stats['n_correct']}/{stats['n']}), unparsed: {stats['unparsed']}, "
                f"{stats['seconds']}s, {stats['vram_used_gb']}/{stats['vram_total_gb']} GB on {stats['gpu']}"
            )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(
                    {"model": spec.hf_id, "adapter": adapter, "task": task, **stats, "records": records},
                    indent=2,
                )
            )
            print(f"wrote {out_path}")

    # vLLM's engine teardown can hang the interpreter at exit; skip it
    sys.stdout.flush()
    os._exit(0)
