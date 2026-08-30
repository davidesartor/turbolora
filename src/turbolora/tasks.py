"""Benchmark tasks: math-verify grading plus one callable `Task` per benchmark."""

import ast
import re
from typing import Protocol

from datasets import Dataset, load_dataset
from math_verify import parse, verify


def extract(completion: str) -> str | None:
    """Final answer as math-verify extracts it: last \\boxed{}, else last math expression."""
    parsed = parse(completion.replace("\\boxed{}", ""))
    return parsed[-1] if parsed else None


def grade(prediction: str | None, reference: str) -> bool:
    """Symbolic/numeric equivalence via math-verify (`reference` is raw answer text, not a completion)."""
    if prediction is None:
        return False
    # spell out `4.5e33` as `4.5\times10^{33}`; math-verify would otherwise read it as 4.5*e*33
    sci = re.compile(r"(\d)[eE]([-+]?\d+)\b")
    return verify(
        parse(f"${sci.sub(r'\1\\times10^{\2}', reference)}$"),
        parse(f"${sci.sub(r'\1\\times10^{\2}', prediction)}$"),
    )


def reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    """TRL reward function over raw-text completions: 1.0 if the final answer matches gold, else 0.0."""
    return [float(grade(extract(c), a)) for c, a in zip(completions, answer)]


def load_from_hf(path: str, name: str | None = None, split: str = "test") -> Dataset:
    """`load_dataset` narrowed to `Dataset`: with a `split` it never returns a dict or iterable."""
    ds = load_dataset(path, name, split=split)
    assert isinstance(ds, Dataset)
    return ds


class Task(Protocol):
    def __call__(self, split: str = "test") -> Dataset:
        """Load a dataset with columns `question` and `answer`."""
        ...


class SimpleRLZoo(Task):
    """SimpleRL-Zoo difficulty tier (GSM8K + MATH train, split by MATH level); its `test` split is MATH-500."""

    def __init__(self, config: str):
        self.config = config

    def __call__(self, split: str = "test") -> Dataset:
        ds = load_dataset(
            "hkust-nlp/SimpleRL-Zoo-Data",
            data_files=f"{self.config}/{split}.parquet",
            split="train",
        )
        assert isinstance(ds, Dataset)
        ds = ds.map(
            lambda r: {
                "question": r["extra_info"]["question"],
                "answer": r["extra_info"]["answer"],
            }
        )
        # two level-5 problems ship without a gold answer; drop them
        return ds.select_columns(["question", "answer"]).filter(
            lambda r: bool(r["answer"])
        )


class GSM8K(Task):
    @staticmethod
    def format_answer(answer: str) -> str:
        return answer.split("####")[-1].strip()

    def __call__(self, split: str = "test") -> Dataset:
        ds = load_from_hf("openai/gsm8k", "main", split)
        return ds.map(lambda r: {"answer": self.format_answer(r["answer"])})


class Math500(Task):
    def __call__(self, split: str = "test") -> Dataset:
        ds = load_from_hf("HuggingFaceH4/MATH-500", split=split)
        ds = ds.rename_column("problem", "question")
        return ds.select_columns(["question", "answer"])


class AIME24(Task):
    @staticmethod
    def format_answer(solution: str) -> str:
        return extract(solution) or ""

    def __call__(self, split: str = "test") -> Dataset:
        ds = load_from_hf("math-ai/aime24", split=split)
        ds = ds.rename_columns({"problem": "question", "solution": "answer"})
        ds = ds.select_columns(["question", "answer"])
        return ds.map(lambda r: {"answer": self.format_answer(r["answer"])})


class AMC23(Task):
    def __call__(self, split: str = "test") -> Dataset:
        ds = load_from_hf("math-ai/amc23", split=split)
        return ds.select_columns(["question", "answer"])


class Minerva(Task):
    def __call__(self, split: str = "test") -> Dataset:
        ds = load_from_hf("math-ai/minervamath", split=split)
        return ds.select_columns(["question", "answer"])


class OlympiadBench(Task):
    @staticmethod
    def format_answer(final_answer: str | list[str]) -> str:
        """`final_answer` is a (possibly stringified) list like ['$2^{1009}$']."""
        return ast.literal_eval(str(final_answer))[0]

    def __call__(self, split: str = "test") -> Dataset:
        ds = load_from_hf("math-ai/olympiadbench", split=split)
        ds = ds.rename_column("final_answer", "answer")
        ds = ds.select_columns(["question", "answer"])
        return ds.map(lambda r: {"answer": self.format_answer(r["answer"])})


TASKS: dict[str, Task] = {
    "easy": SimpleRLZoo("simplelr_qwen_gsm8k_level1"),  # GSM8K + MATH level 1
    "medium": SimpleRLZoo("simplelr_qwen_level1to4"),
    "hard": SimpleRLZoo("simplelr_qwen_level3to5"),
    "gsm8k": GSM8K(),
    "math500": Math500(),
    "aime24": AIME24(),
    "amc23": AMC23(),
    "minerva": Minerva(),
    "olympiad": OlympiadBench(),
}
