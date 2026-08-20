from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen Linear layer plus a trainable low-rank residual update.

    The base layer keeps the pretrained weight unchanged. ``lora_b`` starts at
    zero, so injecting LoRA does not alter the pretrained model's initial
    output. Only the low-rank matrices are enabled by the stage trainer.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.base_layer = base_layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.lora_dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    @property
    def weight(self) -> torch.nn.Parameter:
        """Expose the frozen base weight for compatible model introspection."""
        return self.base_layer.weight

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return self.base_layer.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        update = self.lora_b(self.lora_a(self.lora_dropout(inputs)))
        return base + update.to(base.dtype) * self.scaling


def inject_lora(
    module: nn.Module,
    *,
    target_modules: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> tuple[str, ...]:
    """Replace matching child Linear layers recursively and return their paths."""
    targets = {str(name) for name in target_modules}
    if not targets:
        raise ValueError("LoRA target_modules must not be empty")
    injected: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            path = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(
                    parent,
                    child_name,
                    LoRALinear(
                        child,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    ),
                )
                injected.append(path)
            else:
                visit(child, path)

    visit(module, "")
    if not injected:
        raise ValueError(
            f"LoRA injection found no Linear modules named {sorted(targets)}"
        )
    return tuple(injected)


def is_lora_parameter(name: str) -> bool:
    return ".lora_a." in name or ".lora_b." in name
