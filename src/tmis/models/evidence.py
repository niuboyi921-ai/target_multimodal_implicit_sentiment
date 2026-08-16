from __future__ import annotations

import torch
import torch.nn as nn


class TextEvidenceHead(nn.Module):
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
        logits = logits.masked_fill(~text_mask.bool(), -20.0)
        probs = torch.sigmoid(logits) * text_mask.to(logits.dtype)
        denom = probs.sum(1, keepdim=True).clamp_min(1e-6)
        pooled = (h_ta * probs.unsqueeze(-1)).sum(1) / denom
        return logits, probs, pooled


class VisualEvidenceHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.visual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.presence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, h_va: torch.Tensor, h_f: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = h_f.unsqueeze(1).expand_as(h_va)
        logits = self.scorer(torch.cat([h_va, g], dim=-1)).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        pooled = (h_va * weights.unsqueeze(-1)).sum(1)
        presence_logits = self.presence(h_f).squeeze(-1)
        # Suppress hallucinated visual evidence when the learned presence
        # probability is low, including samples with no annotated evidence.
        visual_repr = self.visual_proj(pooled) * torch.sigmoid(presence_logits).unsqueeze(-1)
        return visual_repr, weights, presence_logits

    def project_evidence_text(self, text_repr: torch.Tensor) -> torch.Tensor:
        return self.text_proj(text_repr)
