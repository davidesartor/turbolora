"""Stub `unsloth` (GPU-only) and `trl` (needs a real vllm) so the package imports on CPU; PEFT stands in for get_peft_model."""

import importlib.machinery
import sys
import types
from pathlib import Path

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


class FakeGRPOConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeGRPOTrainer:
    instances: list["FakeGRPOTrainer"] = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.state = types.SimpleNamespace(global_step=7)
        # as transformers.Trainer: user callbacks live on callback_handler, after the built-in ones
        self.callback_handler = types.SimpleNamespace(callbacks=[object(), *kwargs.get("callbacks", [])])
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)  # as transformers.Trainer.__init__ does
        FakeGRPOTrainer.instances.append(self)

    def train(self, resume_from_checkpoint=None):
        self.resume_from_checkpoint = resume_from_checkpoint


trl = types.ModuleType("trl")
trl.__spec__ = importlib.machinery.ModuleSpec("trl", None)
trl.GRPOConfig = FakeGRPOConfig  # type: ignore[attr-defined]
trl.GRPOTrainer = FakeGRPOTrainer  # type: ignore[attr-defined]
sys.modules["trl"] = trl
