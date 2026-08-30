"""Greedy (pass@1) test-set eval of a bare model on one or more tasks with vLLM."""

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size (GPUs)")
    parser.add_argument(
        "--out-dir", default="outputs/eval", help="writes <out-dir>/<model>/<task>.json"
    )
    parser.add_argument(
        "--show", type=int, default=0, help="print first N completions per task"
    )
    args = parser.parse_args()

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

    llm = LLM(
        model=spec.hf_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=args.tp,
        **multimodal,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop=list(spec.prompt.stop),
    )
    generate = lambda prompts: [
        o.outputs[0].text for o in llm.generate(prompts, sampling)
    ]

    for task in args.tasks:
        start = time.perf_counter()
        records = evaluate(generate, spec, TASKS[task]("test"))
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

        out_path = Path(args.out_dir) / args.model / f"{task}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {"model": spec.hf_id, "task": task, **stats, "records": records},
                indent=2,
            )
        )
        print(f"wrote {out_path}")

    # vLLM's engine teardown can hang the interpreter at exit; skip it
    sys.stdout.flush()
    os._exit(0)
