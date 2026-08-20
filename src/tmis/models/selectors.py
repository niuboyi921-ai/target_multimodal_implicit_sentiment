from __future__ import annotations

import torch
import torch.nn as nn


class TargetAwareTextSelector(nn.Module):
    """Learn a soft target-conditioned token selection without span labels."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_ta: torch.Tensor,
        h_f: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_expanded = h_f.unsqueeze(1).expand_as(h_ta)
        logits = self.scorer(torch.cat([h_ta, global_expanded], dim=-1)).squeeze(-1)
        mask = text_mask.bool()
        logits = logits.masked_fill(~mask, -20.0)
        weights = torch.sigmoid(logits) * mask.to(logits.dtype)
        denom = weights.sum(1, keepdim=True).clamp_min(1e-6)
        selected = (h_ta * weights.unsqueeze(-1)).sum(1) / denom
        return logits, weights, selected


class TargetAwareVisualSelector(nn.Module):
    """Learn a soft target-conditioned patch selection without evidence text."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self, h_va: torch.Tensor, h_f: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_expanded = h_f.unsqueeze(1).expand_as(h_va)
        logits = self.scorer(
            torch.cat([h_va, global_expanded], dim=-1)
        ).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        selected = (h_va * weights.unsqueeze(-1)).sum(1)
        return self.projection(selected), weights
