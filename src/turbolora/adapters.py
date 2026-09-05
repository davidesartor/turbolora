"""Adaptation methods on top of a base model: attach for training, export as a standard LoRA dir for vLLM eval."""

from typing import Protocol, cast, runtime_checkable
from jaxtyping import Float
from unsloth import FastLanguageModel  # must import before peft/transformers

import torch
from einops import einsum
from peft.tuners.lora import LoraLayer
from torch import Tensor, nn

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@runtime_checkable
class Adapter(Protocol):
    """`attach` wraps the base for training; `export` writes a PEFT LoRA dir so eval.py can load it as-is."""

    @staticmethod
    def attach(model, rank: int, seed: int, **kwargs) -> nn.Module: ...

    @staticmethod
    def export(model, out_dir: str) -> None: ...


class LoRA(Adapter):
    """Conventional LoRA on all linear projections, alpha = 2r (TinyLoRA baseline setup)."""

    @staticmethod
    def attach(model, rank: int, seed: int, **kwargs) -> nn.Module:
        return FastLanguageModel.get_peft_model(
            model,
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.0,
            target_modules=TARGET_MODULES,
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )

    @staticmethod
    def export(model, out_dir: str) -> None:
        model.save_pretrained(out_dir)


def svd_lora_layers(
    model, rank: int, seed: int, bases: dict[str, Tensor] | None = None
) -> tuple[nn.Module, list[tuple[LoraLayer, Tensor, Tensor, Tensor]]]:
    """PEFT-wrap `model` and return each adapted module with the top-`rank` SVD (U, S, Vᵀ) of its frozen weight.

    `bases` (lora_A tensors of an exported run, by state_dict key) fixes Vᵀ to that run's, so the search space is
    the same one it trained in: SVD signs differ between the cluster's solvers, and UΣ = W·V follows from Vᵀ alone.
    """
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=rank,  # scaling 1: the papers have no alpha
        lora_dropout=0.0,
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, LoraLayer)]
    svds = []
    for name, layer in layers:
        W = layer.weight.float()
        if bases is not None:
            if f"{name}.lora_A.weight" not in bases:
                raise KeyError(f"reference export has no lora_A for {name}: not an export of this adapter/model")
            Vh = bases[f"{name}.lora_A.weight"].to(W).float()
            US = W @ Vh.T
            S = US.norm(dim=0)
            svds.append((layer, US / S, S, Vh))
            continue
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        # clone: slices are views that would pin every module's full fp32 factors (~35 GB on a 7B)
        svds.append((layer, U[:, :rank].clone(), S[:rank].clone(), Vh[:rank].clone()))
    return model, svds


class LoRAXS_B(nn.Module):
    """PEFT lora_B with a trainable R ∈ ℝ^{r×r} absorbed into it: weight = B0·R for a frozen B0, recomputed on every access.

    Unsloth's fused kernels and vLLM weight sync only see lora_A/lora_B, so R cannot be a third module as in the
    official LoRA-XS code. R is not registered here: it lives in the LoraLayer's `lora_v`, which PEFT checkpoints but
    the vLLM sync (`.lora_A.`/`.lora_B.` keys only) never sees.
    """

    B0: Float[Tensor, "D r"]
    R: nn.Parameter

    def __init__(self, B0: Float[Tensor, "D r"], R: nn.Parameter, dtype: torch.dtype):
        super().__init__()
        self.B0, self.dtype = B0, dtype
        object.__setattr__(self, "R", R)
        self.out_features, self.in_features = B0.shape
        # materialize into state_dict so Unsloth's vLLM sync and PEFT's save see a plain LoRA B
        self.register_state_dict_post_hook(
            lambda module, sd, prefix, _: sd.__setitem__(
                prefix + "weight", module.weight.detach()
            )
        )

    @property
    def weight(self) -> Float[Tensor, "D r"]:
        return (self.B0 @ self.R).to(self.dtype)

    def forward(self, x: Float[Tensor, "... r"]) -> Float[Tensor, "... D"]:
        return nn.functional.linear(x, self.weight)


class LoRAXS(Adapter):
    """LoRA-XS (arXiv 2405.17604): W' = W + UΣRVᵀ with frozen truncated-SVD factors, only R ∈ ℝ^{r×r} per module trains.

    Reimplemented on PEFT LoRA via LoRAXS_B rather than reusing the official code (an old PEFT fork whose extra
    R module Unsloth's fused path would ignore).
    """

    @staticmethod
    def attach(model, rank: int, seed: int, bases: dict[str, Tensor] | None = None, **kwargs) -> nn.Module:
        model, svds = svd_lora_layers(model, rank, seed, bases)
        for layer, U, S, Vh in svds:
            lora_a = cast(nn.Linear, layer.lora_A["default"])
            lora_a.weight.data.copy_(Vh)
            lora_a.weight.requires_grad_(False)
            # trainable tensor goes under `lora_v` so PEFT checkpoints it for resume
            R = nn.Parameter(torch.zeros(rank, rank, device=U.device))
            layer.lora_B["default"] = LoRAXS_B(U * S, R, layer.weight.dtype)
            setattr(layer, "lora_v", nn.ParameterDict({"default": R}))
        print(
            f"{len(svds)} LoRA-XS modules -> {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
        )
        return model

    @staticmethod
    def export(model, out_dir: str) -> None:
        """PEFT save with the materialized lora_B, minus the `lora_v` entries vLLM's loader rejects."""
        tensors = {k: t for k, t in model.state_dict().items() if ".lora_v" not in k}
        model.save_pretrained(out_dir, state_dict=tensors)


class TinyLoRA_B(nn.Module):
    """Stands in for PEFT's lora_B: weight = B0·Σᵢ vᵢPᵢ with frozen B0 (as in LoRAXS_B), fixed random Pᵢ and trainable v ∈ ℝᵘ (possibly shared).

    Same plumbing as LoRAXS_B: v is unregistered here and lives in the LoraLayer's `lora_v`.
    """

    B0: Float[Tensor, "D r"]
    P: Float[Tensor, "u r r"]
    v: nn.Parameter

    def __init__(
        self,
        B0: Float[Tensor, "D r"],
        P: Float[Tensor, "u r r"],
        v: nn.Parameter,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.B0, self.P, self.dtype = B0, P, dtype
        object.__setattr__(self, "v", v)
        self.out_features, self.in_features = B0.shape
        # materialize into state_dict so Unsloth's vLLM sync and PEFT's save see a plain LoRA B
        self.register_state_dict_post_hook(
            lambda module, sd, prefix, _: sd.__setitem__(
                prefix + "weight", module.weight.detach()
            )
        )

    @property
    def weight(self) -> Float[Tensor, "D r"]:
        R = einsum(self.v, self.P, "u, u r s -> r s")
        return (self.B0 @ R).to(self.dtype)

    def forward(self, x: Float[Tensor, "... r"]) -> Float[Tensor, "... D"]:
        return nn.functional.linear(x, self.weight)


class TinyLoRA(Adapter):
    """TinyLoRA (arXiv 2602.04118): LoRA-XS with R = Σᵢ vᵢPᵢ, only v ∈ ℝᵘ trains, `tie` consecutive modules share a v.

    v starts at zero (ΔW = 0) and Pᵢ ~ N(0, 1) per module; the paper states neither.
    """

    @staticmethod
    def attach(
        model, rank: int, seed: int, proj_dim: int = 1, tie: int = 1, bases: dict[str, Tensor] | None = None, **kwargs
    ) -> nn.Module:
        model, svds = svd_lora_layers(model, rank, seed, bases)
        generator = torch.Generator().manual_seed(seed)
        device = svds[0][0].weight.device
        tie = tie or len(svds)  # 0 = one global v
        # one trainable v per `tie` consecutive modules
        vs = [
            nn.Parameter(torch.zeros(proj_dim, device=device))
            for _ in range(-(-len(svds) // tie))
        ]
        for i, (layer, U, S, Vh) in enumerate(svds):
            lora_a = cast(nn.Linear, layer.lora_A["default"])
            lora_a.weight.data.copy_(Vh)
            lora_a.weight.requires_grad_(False)
            P = torch.randn(proj_dim, rank, rank, generator=generator).to(device)
            layer.lora_B["default"] = TinyLoRA_B(
                U * S, P, vs[i // tie], layer.weight.dtype
            )
            # trainable tensor goes under `lora_v` so PEFT checkpoints it for resume
            setattr(layer, "lora_v", nn.ParameterDict({"default": vs[i // tie]}))
        print(
            f"{len(svds)} TinyLoRA modules, u={proj_dim}, tie={tie} -> {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
        )
        return model

    @staticmethod
    def export(model, out_dir: str) -> None:
        """PEFT save with the materialized lora_B, minus the `lora_v` entries vLLM's loader rejects."""
        tensors = {k: t for k, t in model.state_dict().items() if ".lora_v" not in k}
        model.save_pretrained(out_dir, state_dict=tensors)
