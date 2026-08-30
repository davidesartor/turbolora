"""Stub `unsloth` (GPU-only) so the package imports on CPU; PEFT stands in for get_peft_model."""

import importlib.machinery
import sys
import types

from peft import LoraConfig, get_peft_model


class FakeFastLanguageModel:
    @staticmethod
    def get_peft_model(model, r, lora_alpha, lora_dropout, target_modules, **_):
        return get_peft_model(
            model, LoraConfig(r=r, lora_alpha=lora_alpha, target_modules=target_modules)
        )

    @staticmethod
    def from_pretrained(**_):
        raise NotImplementedError("monkeypatch FakeFastLanguageModel.from_pretrained in the test")


unsloth = types.ModuleType("unsloth")
unsloth.__spec__ = importlib.machinery.ModuleSpec("unsloth", None)  # trl probes find_spec("unsloth")
unsloth.FastLanguageModel = FakeFastLanguageModel  # type: ignore[attr-defined]
sys.modules["unsloth"] = unsloth
