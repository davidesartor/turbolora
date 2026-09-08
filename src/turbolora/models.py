"""SimpleRL-Zoo base models with the raw-text prompt the paper trained each one with."""

from typing import NamedTuple, Protocol


class Prompt(Protocol):
    stop: tuple[str, ...]  # generation terminators beyond the tokenizer's EOS

    def __call__(self, question: str) -> str:
        """Raw-text prompt (no chat template) for a benchmark question."""
        ...


class SimplePrompt(Prompt):
    """Abel's plain prompt for weak instruction-followers (paper App. B.3)."""

    stop = ("\nQuestion:",)

    def __call__(self, question: str) -> str:
        return f"Question:\n{question}\nAnswer:\nLet's think step by step.\n"


class ChatMLBoxedPrompt(Prompt):
    """Qwen ChatML 'boxed' prompt, used verbatim (as raw text) for every family that can follow it."""

    stop = ("<|im_end|>",)

    def __call__(self, question: str) -> str:
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


class Model(NamedTuple):
    hf_id: str
    prompt: Prompt
    family: str  # dir grouping under outputs/runs
    pad_token_id: int | None = None  # for tokenizers offering neither a pad nor an unk token



MODELS: dict[str, Model] = {
    # QWEN
    "qwen2.5-0.5b": Model("Qwen/Qwen2.5-0.5B", SimplePrompt(), "qwen2.5"),
    "qwen2.5-1.5b": Model("Qwen/Qwen2.5-1.5B", SimplePrompt(), "qwen2.5"),
    "qwen2.5-3b": Model("Qwen/Qwen2.5-3B", ChatMLBoxedPrompt(), "qwen2.5"),
    "qwen2.5-7b": Model("Qwen/Qwen2.5-7B", ChatMLBoxedPrompt(), "qwen2.5"),
    "qwen2.5-14b": Model("Qwen/Qwen2.5-14B", ChatMLBoxedPrompt(), "qwen2.5"),
    "qwen2.5-32b": Model("Qwen/Qwen2.5-32B", ChatMLBoxedPrompt(), "qwen2.5"),
    # QWEN (instruct)
    "qwen2.5-0.5b-instruct": Model("Qwen/Qwen2.5-0.5B-Instruct", ChatMLBoxedPrompt(), "qwen2.5-instruct"),
    "qwen2.5-1.5b-instruct": Model("Qwen/Qwen2.5-1.5B-Instruct", ChatMLBoxedPrompt(), "qwen2.5-instruct"),
    "qwen2.5-3b-instruct": Model("Qwen/Qwen2.5-3B-Instruct", ChatMLBoxedPrompt(), "qwen2.5-instruct"),
    "qwen2.5-7b-instruct": Model("Qwen/Qwen2.5-7B-Instruct", ChatMLBoxedPrompt(), "qwen2.5-instruct"),
    "qwen2.5-14b-instruct": Model("Qwen/Qwen2.5-14B-Instruct", ChatMLBoxedPrompt(), "qwen2.5-instruct"),
    "qwen2.5-32b-instruct": Model("Qwen/Qwen2.5-32B-Instruct", ChatMLBoxedPrompt(), "qwen2.5-instruct"),
    # QWEN (math)
    "qwen2.5-1.5b-math": Model("Qwen/Qwen2.5-Math-1.5B", ChatMLBoxedPrompt(), "qwen2.5-math"),
    "qwen2.5-7b-math": Model("Qwen/Qwen2.5-Math-7B", ChatMLBoxedPrompt(), "qwen2.5-math"),
    # LLaMA
    "llama3.2-1b": Model("meta-llama/Llama-3.2-1B", SimplePrompt(), "llama"),
    "llama3.2-3b": Model("meta-llama/Llama-3.2-3B", SimplePrompt(), "llama"),
    "llama3.1-8b": Model("meta-llama/Llama-3.1-8B", SimplePrompt(), "llama"),
    # MISTRAL
    "mistral-7b": Model("mistralai/Mistral-7B-v0.1", SimplePrompt(), "mistral"),
    "ministral3-3b": Model("mistralai/Ministral-3-3B-Base-2512", SimplePrompt(), "mistral"),
    "ministral3-8b": Model("mistralai/Ministral-3-8B-Base-2512", ChatMLBoxedPrompt(), "mistral"),
    "ministral3-14b": Model("mistralai/Ministral-3-14B-Base-2512", ChatMLBoxedPrompt(), "mistral"),
    "mixtral-8x7b": Model("mistralai/Mixtral-8x7B-v0.1", ChatMLBoxedPrompt(), "mistral"),
    # DEEPSEEK
    "deepseek-7b": Model("deepseek-ai/deepseek-llm-7b-base", ChatMLBoxedPrompt(), "deepseek", 100000),
    "deepseek-7b-math": Model("deepseek-ai/deepseek-math-7b-base", ChatMLBoxedPrompt(), "deepseek", 100000),
}
