import pytest

from turbolora.models import MODELS


@pytest.mark.parametrize("name", MODELS)
def test_prompt_contains_question_verbatim(name):
    question = "What is 2+2 in {braces} and \\frac{1}{2}?"
    prompt = MODELS[name].prompt(question)
    assert question in prompt
    assert "{question}" not in prompt


@pytest.mark.parametrize("name", MODELS)
def test_prompt_ends_at_generation_point(name):
    """Prompt must end where the model speaks, not with a stop string (which would be the last turn's closer)."""
    spec = MODELS[name]
    prompt = spec.prompt("Q?")
    assert spec.prompt.stop
    assert prompt.endswith(("assistant\n", "Let's think step by step.\n"))
    assert not prompt.endswith(spec.prompt.stop)


def test_boxed_braces_survive_formatting():
    assert "\\boxed{}" in MODELS["qwen2.5-7b"].prompt("Q?")


def test_every_qwen_base_size_has_an_instruct_variant():
    bases = [n for n in MODELS if n.startswith("qwen2.5-") and n.count("-") == 1]
    assert bases
    for base in bases:
        assert f"{base}-instruct" in MODELS
        assert MODELS[f"{base}-instruct"].hf_id == MODELS[base].hf_id + "-Instruct"
