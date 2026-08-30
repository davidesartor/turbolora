"""CPU tests on a tiny Qwen2 with Unsloth's get_peft_model stubbed by plain PEFT."""

import sys
import types

import pytest
import safetensors.torch
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM


class FakeFastLanguageModel:
    @staticmethod
    def get_peft_model(model, r, lora_alpha, lora_dropout, target_modules, **_):
        return get_peft_model(
            model, LoraConfig(r=r, lora_alpha=lora_alpha, target_modules=target_modules)
        )


sys.modules["unsloth"] = types.SimpleNamespace(FastLanguageModel=FakeFastLanguageModel)  # type: ignore[assignment]

from turbolora.adapters import LoRAXS, TinyLoRA, TurboLoRA  # noqa: E402

CONFIG = Qwen2Config(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=100,
)
N_MODULES = 7 * CONFIG.num_hidden_layers


def n_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def attach(adapter, **kwargs):
    torch.manual_seed(0)
    return adapter.attach(Qwen2ForCausalLM(CONFIG), rank=2, seed=1, **kwargs)


def test_loraxs_trains_one_r_per_module():
    assert n_trainable(attach(LoRAXS)) == N_MODULES * 4


@pytest.mark.parametrize("proj_dim, tie", [(1, 1), (3, 7), (2, 100)])
def test_tinylora_param_count(proj_dim, tie):
    model = attach(TinyLoRA, proj_dim=proj_dim, tie=tie)
    assert n_trainable(model) == proj_dim * -(-N_MODULES // tie)


def test_tinylora_starts_at_zero_delta():
    model = attach(TinyLoRA, proj_dim=3, tie=1)
    layer = model.base_model.model.model.layers[0].self_attn.q_proj
    assert torch.all(layer.lora_B["default"].weight == 0)


def test_turbolora_splits_sigma_symmetrically():
    tiny = attach(TinyLoRA).base_model.model.model.layers[0].self_attn.q_proj
    turbo = attach(TurboLoRA).base_model.model.model.layers[0].self_attn.q_proj
    a_tiny, a_turbo = tiny.lora_A["default"].weight, turbo.lora_A["default"].weight
    us_tiny, us_turbo = tiny.lora_B["default"].B0, turbo.lora_B["default"].B0
    assert torch.allclose(a_turbo.norm(dim=1), us_turbo.norm(dim=0), atol=1e-5)
    assert torch.allclose(a_turbo.norm(dim=1) ** 2, us_tiny.norm(dim=0), atol=1e-4)
    assert torch.allclose(us_turbo.T @ us_turbo, a_turbo @ a_turbo.T, atol=1e-4)
    assert torch.allclose(us_tiny @ a_tiny, us_turbo @ a_turbo, atol=1e-4)


@pytest.mark.parametrize("adapter", [LoRAXS, TinyLoRA, TurboLoRA])
def test_gradients_reach_trainable_params(adapter):
    model = attach(adapter)
    x = torch.randint(0, 100, (1, 8))
    model(x, labels=x).loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("adapter", [LoRAXS, TinyLoRA, TurboLoRA])
def test_export_is_plain_lora_dir(adapter, tmp_path):
    model = attach(adapter, **({} if adapter is LoRAXS else {"proj_dim": 3, "tie": 7}))
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.1)
    adapter.export(model, str(tmp_path))
    sd = safetensors.torch.load_file(tmp_path / "adapter_model.safetensors")

    assert not any("lora_v" in k for k in sd)
    assert len(sd) == 2 * N_MODULES
    layer = model.base_model.model.model.layers[0].self_attn.q_proj
    key = next(k for k in sd if "layers.0.self_attn.q_proj.lora_B" in k)
    assert sd[key].abs().sum() > 0
    assert torch.allclose(sd[key], layer.lora_B["default"].weight.detach())
    assert isinstance(
        PeftModel.from_pretrained(Qwen2ForCausalLM(CONFIG), tmp_path), PeftModel
    )
