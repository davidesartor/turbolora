"""Wiring tests for grpo.run with unsloth (conftest), trl and CUDA stubbed; nothing here trains."""

import importlib.machinery
import json
import os
import runpy
import signal
import sys
import types
from pathlib import Path

import pytest
from types import SimpleNamespace
import torch
from datasets import Dataset


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

import unsloth  # noqa: E402  conftest stub
from turbolora import grpo  # noqa: E402

DATASET = Dataset.from_dict({"question": ["1+1?", "2+2?"], "answer": ["2", "4"]})


class FakeAdapter:
    calls: dict = {}

    @staticmethod
    def attach(model, rank, seed, **kwargs):
        FakeAdapter.calls = dict(rank=rank, seed=seed, **kwargs)
        model.linear.weight.requires_grad_(False)
        return model

    @staticmethod
    def export(model, out_dir):
        FakeAdapter.calls["export"] = out_dir


class FakeTokenizer:
    def __call__(self, text):
        return types.SimpleNamespace(input_ids=text.split())

    def save_pretrained(self, out_dir):
        pass


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """Route model load, task, and CUDA queries to fakes; cwd is restored after `run` chdirs into out."""
    model = torch.nn.Module()
    model.linear = torch.nn.Linear(3, 5, bias=False)  # 15 params
    model.head = torch.nn.Linear(5, 1, bias=False)  # 5 params
    loads: dict = {}

    def from_pretrained(**kwargs):
        loads.update(kwargs)
        return model, FakeTokenizer()

    monkeypatch.setattr(unsloth.FastLanguageModel, "from_pretrained", from_pretrained)
    monkeypatch.setitem(grpo.TASKS, "gsm8k", lambda split: DATASET)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: types.SimpleNamespace(total_memory=80 * 2**30))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 3 * 2**30)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "FakeGPU")
    monkeypatch.chdir(tmp_path)
    FakeGRPOTrainer.instances.clear()
    return loads


def parse(*extra: str, out: str):
    return grpo.argument_parser().parse_args(["--model", "qwen2.5-7b", "--task", "gsm8k", "--out", out, *extra])


def test_argument_parser_defaults(tmp_path):
    args = parse(out=str(tmp_path))
    assert (args.loss, args.lr, args.seed, args.epochs, args.max_completion, args.max_steps) == (
        "grpo", 5e-6, 0, 3, 1024, -1)


def test_argument_parser_rejects_unknown_model():
    with pytest.raises(SystemExit):
        grpo.argument_parser().parse_args(["--model", "gpt-9", "--task", "gsm8k", "--out", "o"])


def test_resource_logger_adds_vram_and_time(monkeypatch):
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 2**30)
    logger = grpo.ResourceLogger()
    logger.on_train_begin(None, None, None)
    logs = {"loss": 0.1}
    state = SimpleNamespace(global_step=7, log_history=[{"loss": 0.1, "step": 7}])
    logger.on_log(None, state, None, logs=logs)
    assert logs["peak_vram_gib"] == 1.0
    assert 0 <= logs["elapsed_hours"] < 1e-3
    # Trainer copies logs into log_history before on_log; the checkpointed copy must get the keys too
    assert state.log_history[-1]["peak_vram_gib"] == 1.0


def test_save_on_preempt_checkpoints_once_after_signal():
    callback = grpo.SaveOnPreempt()
    control = types.SimpleNamespace(should_save=False)
    callback.on_step_end(None, None, control)
    assert not control.should_save

    # slurm/train.sh forwards both the wall-limit USR1 and the preemption TERM
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        os.kill(os.getpid(), sig)
        control.should_save = False
        callback.on_step_end(None, None, control)
        assert control.should_save
        control.should_save = False
        callback.on_step_end(None, None, control)
        assert not control.should_save


def test_run_wires_model_adapter_data_and_outputs(stubbed, tmp_path):
    out = tmp_path / "run"
    grpo.run(parse("--max-steps", "3", out=str(out)), FakeAdapter, rank=2, proj_dim=3)
    spec = grpo.MODELS["qwen2.5-7b"]

    # base load: seq length = prompt + completion, vLLM pads rank to 8, big GPU gets 0.45 share
    assert stubbed["model_name"] == spec.hf_id
    assert stubbed["max_seq_length"] == 512 + 1024
    assert stubbed["max_lora_rank"] == 8
    assert stubbed["gpu_memory_utilization"] == 0.45
    assert FakeAdapter.calls == dict(rank=2, seed=0, proj_dim=3, export=str(out / "final_adapter"))

    # trainer gets raw-text prompts (no chat template), our reward and the resource logger
    (trainer,) = FakeGRPOTrainer.instances
    assert trainer.train_dataset["prompt"] == [spec.prompt(q) for q in DATASET["question"]]
    assert trainer.reward_funcs == [grpo.reward]
    callbacks = trainer.callback_handler.callbacks
    assert type(callbacks[0]) is grpo.ResourceLogger  # before WandbCallback so wandb sees its keys
    assert {type(c) for c in callbacks[1:] if not type(c) is object} == {grpo.SaveOnPreempt}
    assert trainer.resume_from_checkpoint is None
    config = trainer.args
    assert config.output_dir == str(out)
    assert config.run_name == "run"
    assert config.max_steps == 3
    assert config.generation_kwargs == {"stop": list(spec.prompt.stop)}
    assert config.per_device_train_batch_size * config.gradient_accumulation_steps == 256
    assert config.beta == 0.0

    # run.json: only the trainable params count, adapter name lowercased, extra adapter kwargs included
    summary = json.loads((out / "run.json").read_text())
    assert summary["params"] == 5
    assert summary["adapter"] == "fakeadapter"
    assert summary["steps"] == 7
    assert summary["proj_dim"] == 3
    assert summary["gpu"] == "FakeGPU"
    assert summary["peak_vram_gb"] == 3.0


def test_run_keeps_large_rank(stubbed, tmp_path):
    grpo.run(parse(out=str(tmp_path / "run")), FakeAdapter, rank=32)
    assert stubbed["max_lora_rank"] == 32


def test_small_gpu_gives_vllm_half(stubbed, monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: types.SimpleNamespace(total_memory=48 * 2**30))
    grpo.run(parse(out=str(tmp_path / "run")), FakeAdapter, rank=2)
    assert stubbed["gpu_memory_utilization"] == 0.5


@pytest.mark.parametrize(
    "loss, level, eps, eps_high", [("grpo", "token", 0.2, None), ("gspo", "sequence", 3e-4, 4e-4)]
)
def test_loss_selects_importance_sampling(stubbed, tmp_path, loss, level, eps, eps_high):
    grpo.run(parse("--loss", loss, out=str(tmp_path / "run")), FakeAdapter, rank=2)
    config = FakeGRPOTrainer.instances[-1].args
    assert (config.importance_sampling_level, config.epsilon, config.epsilon_high) == (level, eps, eps_high)


def test_run_resumes_from_latest_checkpoint(stubbed, tmp_path):
    out = tmp_path / "run"
    for step in (25, 50):
        (out / f"checkpoint-{step}").mkdir(parents=True)
    grpo.run(parse(out=str(out)), FakeAdapter, rank=2)
    assert FakeGRPOTrainer.instances[-1].resume_from_checkpoint == str(out / "checkpoint-50")


@pytest.mark.parametrize(
    "script, argv, expected",
    [
        ("train_lora", [], dict(rank=32)),
        ("train_lora", ["--rank", "4"], dict(rank=4)),
        ("train_loraxs", [], dict(rank=2)),
        ("train_tinylora", [], dict(rank=2, proj_dim=4, tie=0)),
        ("train_tinylora", ["--rank", "3", "--proj-dim", "5", "--no-tie"], dict(rank=3, proj_dim=5, tie=1)),
    ],
)
def test_train_scripts_forward_adapter_args(monkeypatch, script, argv, expected):
    seen = {}
    monkeypatch.setattr(grpo, "run", lambda args, adapter, **kw: seen.update(adapter=adapter.__name__, **kw))
    monkeypatch.setattr(sys, "argv", [script, "--model", "qwen2.5-7b", "--task", "gsm8k", "--out", "o", *argv])
    runpy.run_module(f"turbolora.{script}", run_name="__main__")
    assert seen == dict(adapter=script.removeprefix("train_").replace("loraxs", "LoRAXS").replace("lora", "LoRA").replace("tinyLoRA", "TinyLoRA"), **expected)
