"""CPU tests on a tiny Qwen2 (unsloth stubbed by conftest.py)."""

import pytest
import safetensors.torch
import torch
from peft import PeftModel
from transformers import Qwen2Config, Qwen2ForCausalLM

from turbolora.adapters import LoRAXS, TinyLoRA

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


@pytest.mark.parametrize("adapter", [LoRAXS, TinyLoRA])
def test_gradients_reach_trainable_params(adapter):
    model = attach(adapter)
    x = torch.randint(0, 100, (1, 8))
    model(x, labels=x).loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("adapter", [LoRAXS, TinyLoRA])
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
