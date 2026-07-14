"""Lightweight LoRA adapters for TTT (no peft dependency).

Targets weight-matrix modules suitable for low-rank updates:
  - nn.Linear
  - nn.Conv1d
  - nn.MultiheadAttention (in_proj + out_proj)

Not wrapped (unsuitable / not matrix weights):
  - LayerNorm / BatchNorm
  - bare nn.Parameter (type embeds, mask tokens)
  - Dropout / DropPath / activations
  - custom natten NeighborhoodAttention1D
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        if rank > min(linear.in_features, linear.out_features):
            rank = min(linear.in_features, linear.out_features)

        self.linear = linear
        self.rank = rank
        self.scaling = alpha / rank

        for p in self.linear.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        # x: [..., in] -> [..., out]
        lora = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base + lora


class LoRAConv1d(nn.Module):
    def __init__(self, conv: nn.Conv1d, rank: int, alpha: float):
        super().__init__()
        if conv.groups != 1:
            raise ValueError("LoRAConv1d currently supports groups=1 only")
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")

        self.conv = conv
        out_c, in_c, k = conv.weight.shape
        max_rank = min(out_c, in_c * k)
        if rank > max_rank:
            rank = max_rank

        self.rank = rank
        self.scaling = alpha / rank

        for p in self.conv.parameters():
            p.requires_grad = False

        # Low-rank factorization of the flattened conv kernel.
        self.lora_A = nn.Parameter(torch.zeros(rank, in_c * k))
        self.lora_B = nn.Parameter(torch.zeros(out_c, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_w = (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
        return F.conv1d(
            x,
            self.conv.weight + delta_w,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )


class LoRAMultiheadAttention(nn.Module):
    """LoRA on packed in_proj_weight and out_proj.weight of nn.MultiheadAttention."""

    def __init__(self, mha: nn.MultiheadAttention, rank: int, alpha: float):
        super().__init__()
        if mha._qkv_same_embed_dim is False:
            raise ValueError("LoRAMultiheadAttention requires _qkv_same_embed_dim=True")
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")

        self.mha = mha
        embed_dim = mha.embed_dim
        max_rank = embed_dim
        if rank > max_rank:
            rank = max_rank

        self.rank = rank
        self.scaling = alpha / rank

        for p in self.mha.parameters():
            p.requires_grad = False

        # in_proj: (3E, E) ≈ B_in @ A_in
        self.lora_in_A = nn.Parameter(torch.zeros(rank, embed_dim))
        self.lora_in_B = nn.Parameter(torch.zeros(3 * embed_dim, rank))
        nn.init.kaiming_uniform_(self.lora_in_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_in_B)

        # out_proj: (E, E) ≈ B_out @ A_out
        self.lora_out_A = nn.Parameter(torch.zeros(rank, embed_dim))
        self.lora_out_B = nn.Parameter(torch.zeros(embed_dim, rank))
        nn.init.kaiming_uniform_(self.lora_out_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_out_B)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        mha = self.mha
        in_proj_weight = mha.in_proj_weight + (self.lora_in_B @ self.lora_in_A) * self.scaling
        out_proj_weight = mha.out_proj.weight + (self.lora_out_B @ self.lora_out_A) * self.scaling

        if mha.batch_first and query.dim() == 3:
            # Match nn.MultiheadAttention batch_first path.
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Older torch may not accept is_causal / batch_first on the functional API.
        kwargs = dict(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=mha.embed_dim,
            num_heads=mha.num_heads,
            in_proj_weight=in_proj_weight,
            in_proj_bias=mha.in_proj_bias,
            bias_k=mha.bias_k,
            bias_v=mha.bias_v,
            add_zero_attn=mha.add_zero_attn,
            dropout_p=mha.dropout,
            out_proj_weight=out_proj_weight,
            out_proj_bias=mha.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
        )
        try:
            attn_output, attn_weights = F.multi_head_attention_forward(
                **kwargs, is_causal=is_causal
            )
        except TypeError:
            attn_output, attn_weights = F.multi_head_attention_forward(**kwargs)

        if mha.batch_first and attn_output.dim() == 3:
            attn_output = attn_output.transpose(0, 1)
        return attn_output, attn_weights


def _set_by_name(root: nn.Module, name: str, module: nn.Module) -> None:
    atoms = name.split(".")
    parent = root
    for atom in atoms[:-1]:
        parent = getattr(parent, atom)
    setattr(parent, atoms[-1], module)


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_types: Sequence[str] = ("Linear", "Conv1d", "MultiheadAttention"),
) -> List[str]:
    """Replace eligible modules with LoRA wrappers and freeze non-LoRA params.

    Returns the list of replaced module names.
    """
    target_set = set(target_types)
    named = dict(model.named_modules())
    to_replace: List[str] = []

    for name, module in named.items():
        if not name:
            continue
        if "MultiheadAttention" in target_set and isinstance(module, nn.MultiheadAttention):
            to_replace.append(name)
        elif "Linear" in target_set and isinstance(module, nn.Linear):
            # out_proj is handled by LoRAMultiheadAttention when MHA is targeted.
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = named.get(parent_name)
            if (
                "MultiheadAttention" in target_set
                and isinstance(parent, nn.MultiheadAttention)
                and name.endswith("out_proj")
            ):
                continue
            to_replace.append(name)
        elif "Conv1d" in target_set and isinstance(module, nn.Conv1d):
            to_replace.append(name)

    # Replace deeper modules first so parent paths stay valid.
    to_replace.sort(key=lambda n: n.count("."), reverse=True)

    replaced: List[str] = []
    for name in to_replace:
        try:
            cur = model.get_submodule(name)
        except AttributeError:
            parent = model
            try:
                for atom in name.split(".")[:-1]:
                    parent = getattr(parent, atom)
                cur = getattr(parent, name.split(".")[-1])
            except AttributeError:
                continue

        if isinstance(cur, nn.MultiheadAttention):
            _set_by_name(model, name, LoRAMultiheadAttention(cur, rank=rank, alpha=alpha))
            replaced.append(name)
        elif isinstance(cur, nn.Linear):
            _set_by_name(model, name, LoRALinear(cur, rank=rank, alpha=alpha))
            replaced.append(name)
        elif isinstance(cur, nn.Conv1d):
            try:
                _set_by_name(model, name, LoRAConv1d(cur, rank=rank, alpha=alpha))
                replaced.append(name)
            except ValueError:
                # e.g. groups != 1
                continue

    # Freeze everything except LoRA parameters.
    for n, p in model.named_parameters():
        p.requires_grad = is_lora_param_name(n)

    return replaced


def is_lora_param_name(name: str) -> bool:
    return any(
        key in name
        for key in (
            "lora_A",
            "lora_B",
            "lora_in_A",
            "lora_in_B",
            "lora_out_A",
            "lora_out_B",
        )
    )


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters() if is_lora_param_name(n) and p.requires_grad]


def lora_param_stats(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for n, p in model.named_parameters() if is_lora_param_name(n))
    return {
        "total_params": total,
        "trainable_params": trainable,
        "lora_params": lora,
        "trainable_ratio": (trainable / total) if total else 0.0,
    }
