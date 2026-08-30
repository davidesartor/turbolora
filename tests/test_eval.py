from datasets import Dataset

import subprocess

import turbolora.eval
from turbolora.eval import evaluate, summarize, usage
from turbolora.models import MODELS

DATASET = Dataset.from_dict({"question": ["1+1?", "2+2?", "3+3?"], "answer": ["2", "4", "6"]})


def test_evaluate_grades_each_prompt():
    spec = MODELS["qwen2.5-7b"]
    seen = []

    def generate(prompts):
        seen.extend(prompts)
        return ["\\boxed{2}", "\\boxed{5}", "no answer"]

    records = evaluate(generate, spec, DATASET)
    assert seen == [spec.prompt(q) for q in DATASET["question"]]
    assert [r["correct"] for r in records] == [True, False, False]
    assert [r["predicted"] for r in records] == ["2", "5", None]
    assert [r["answer"] for r in records] == ["2", "4", "6"]


def test_summarize():
    records = evaluate(lambda ps: ["\\boxed{2}", "\\boxed{5}", "no answer"], MODELS["llama3.1-8b"], DATASET)
    assert summarize(records) == {"accuracy": 1 / 3, "n_correct": 1, "n": 3, "unparsed": 1}


def test_usage_parses_nvidia_smi(monkeypatch):
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout="NVIDIA L40S, 41865, 46068\n")
    monkeypatch.setattr(turbolora.eval.subprocess, "run", lambda *a, **k: fake)

    stats = usage(12.345)
    assert stats["seconds"] == 12.3
    assert stats["gpu"] == "NVIDIA L40S"
    assert stats["vram_used_gb"] == round(41865 / 1024, 2)
    assert stats["vram_total_gb"] == round(46068 / 1024, 2)
    assert stats["host_ram_gb"] > 0
