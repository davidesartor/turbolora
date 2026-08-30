"""Benchmark tasks: math-verify grading plus one callable `Task` per benchmark."""

import ast
import re
from typing import Protocol

from datasets import Dataset, concatenate_datasets, load_dataset
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


class GSM8K(Task):
    @staticmethod
    def format_answer(answer: str) -> str:
        return answer.split("####")[-1].strip()

    def __call__(self, split: str = "test") -> Dataset:
        ds = load_from_hf("openai/gsm8k", "main", split)
        return ds.map(lambda r: {"answer": self.format_answer(r["answer"])})


class HendrycksMath(Task):
    subjects = (
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    )

    @staticmethod
    def format_answer(solution: str) -> str:
        return extract(solution) or ""

    def __call__(self, split: str = "test") -> Dataset:
        ds = concatenate_datasets(
            [load_from_hf("EleutherAI/hendrycks_math", s, split) for s in self.subjects]
        )
        ds = ds.rename_columns({"problem": "question", "solution": "answer"})
        ds = ds.select_columns(["question", "answer"])
        ds = ds.map(lambda r: {"answer": self.format_answer(r["answer"])})
        # a couple of train problems have no \boxed{} gold; drop them
        return ds.filter(lambda r: bool(r["answer"]))


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
    "gsm8k": GSM8K(),
    "math": HendrycksMath(),
    "math500": Math500(),
    "aime24": AIME24(),
    "amc23": AMC23(),
    "minerva": Minerva(),
    "olympiad": OlympiadBench(),
}
