import pytest

from turbolora.models import MODELS

from turbolora.tasks import TASKS, extract, grade, reward


def test_prompt():
    assert MODELS["qwen2.5-7b"].prompt("Q?") == (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nQ?\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert MODELS["llama3.1-8b"].prompt("Q?") == "Question:\nQ?\nAnswer:\nLet's think step by step.\n"


def test_qwen_prompt_matches_chat_template():
    """For Qwen the raw paper prompt is exactly what the tokenizer's own chat template renders."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Q?\nPlease reason step by step, and put your final answer within \\boxed{}."},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert rendered == MODELS["qwen2.5-7b"].prompt("Q?")


@pytest.mark.parametrize(
    "completion, expected",
    [
        ("Thus \\boxed{18}.", "18"),
        ("\\boxed{\\frac{3}{4}}", "\\frac{3}{4}"),
        ("So the answer is 42.", "42"),
        ("The final answer is: $12$", "12"),
        ("She has 5 apples and 3 pears, so 8 fruits", "8"),
        ("no digits at all", None),
        ("answer within \\boxed{}:\n\n\\boxed{495}", "495"),
    ],
)
def test_extract_answer(completion, expected):
    assert extract(completion) == expected


@pytest.mark.parametrize(
    "pred, gold, expected",
    [
        ("18", "18", True),
        ("18.0", "18", True),
        ("0.5", "50", False),
        ("3.83\\times10^{35}", "3.83e35", True),
        ("3.89e-10", "3.89\\times10^{-10}", True),
        ("3.83\\times10^{35}", "3.9\\times10^{35}", False),
        ("19", "18", False),
        (None, "18", False),
        ("\\frac{1}{2}", "0.5", True),
        ("\\frac{1}{2}", "\\frac{2}{4}", True),
        ("(3,\\frac{\\pi}{2})", "(3,\\pi/2)", True),
        ("x^2+2x+1", "(x+1)^2", True),
        ("\\sqrt{2}", "1.5", False),
        # units and formatting are tolerated
        ("50", "50\\%", True),
        ("\\$18", "18", True),
        ("90^\\circ", "90", True),
        ("1050", "1,050", True),
        ("18", "18.00", True),
        ("18 dollars", "18", False),
        # symbolic vs numeric
        ("\\pi", "3.14159", False),
        ("2\\sqrt{2}", "\\sqrt{8}", True),
        ("x=3", "3", True),
        # tuples are ordered, bare comma lists are compared as sets
        ("(1,2)", "(2,1)", False),
        ("1,2", "2,1", True),
        # multiple choice
        ("\\text{(A)}", "A", True),
        ("A", "(A)", True),
    ],
)
def test_math_equal(pred, gold, expected):
    assert grade(pred, gold) is expected


def test_multiple_boxes_are_merged_not_last():
    """math-verify merges several \\boxed into one set, unlike SimpleRL's last-box rule; merged value fails vs scalar gold."""
    merged = extract("\\boxed{1} and \\boxed{2}")
    assert merged == "1,2"
    assert extract("\\boxed{1}\nso \\boxed{2}") == "1,2"
    assert grade(merged, "2") is False


def test_extract_strips_units_keeps_percent():
    assert extract("\\boxed{18 \\text{ dollars}}") == "18"
    assert extract("\\boxed{50\\%}") == "50\\%"
    assert extract("\\boxed{1,050}") == "1,050"
    assert extract("\\boxed{\\text{A}}") == "\\text{A}"
    assert extract("$x = 5$ so answer 5") == "5"


def test_format_answer():
    assert TASKS["gsm8k"].format_answer("... #### 1,050") == "1,050"
    assert TASKS["math"].format_answer("so \\boxed{\\dfrac{1}{2}}") == "\\frac{1}{2}"
    assert TASKS["aime24"].format_answer("\\boxed{204}") == "204"
    assert TASKS["olympiad"].format_answer(["$\\frac{1}{2 n+2}$"]) == "$\\frac{1}{2 n+2}$"
    assert TASKS["olympiad"].format_answer("['$2^{1009}$']") == "$2^{1009}$"


def test_reward():
    assert reward(["\\boxed{18}", "\\boxed{19}"], ["18", "18"]) == [1.0, 0.0]


TEST_SIZES = {
    "gsm8k": 1319,
    "math": 5000,
    "math500": 500,
    "aime24": 30,
    "amc23": 40,
    "minerva": 272,
    "olympiad": 674,
}


@pytest.mark.parametrize("task", TASKS)
def test_load_from_cache(task):
    ds = TASKS[task]("test")
    assert set(ds.column_names) == {"question", "answer"}
    assert len(ds) == TEST_SIZES[task]
    assert all(ds["answer"])


UNGRADEABLE_GOLD = {
    "minerva": ["x_{0} \\cos (\\omega t)+$ $\\dot{x}_{0} \\sin (\\omega t) / \\omega"],
    "olympiad": ["$(-\\infty, 0) \\cup\\{1\\}$."],
}


@pytest.mark.parametrize("task", TASKS)
def test_gold_grades_against_itself(task):
    """Every gold answer, boxed verbatim by the model, is graded correct (except the pinned malformed ones)."""
    bad = [a for a in TASKS[task]("test")["answer"] if not grade(extract(f"\\boxed{{{a}}}"), a)]
    assert bad == UNGRADEABLE_GOLD.get(task, [])


@pytest.mark.parametrize("task", ["gsm8k", "math"])
def test_train_split_loads(task):
    ds = TASKS[task]("train")
    assert set(ds.column_names) == {"question", "answer"}
    assert len(ds) > 5000
    assert all(ds["answer"])
