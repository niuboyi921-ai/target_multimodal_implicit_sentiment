from __future__ import annotations

import math
import torch
import torch.nn as nn


class TargetConditioner(nn.Module):
    """Soft target-conditioned attention with residual preservation."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.target_v = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q(target).unsqueeze(1)
        k = self.k(features)
        scores = (q * k).sum(dim=-1) / math.sqrt(features.size(-1))
        # A sigmoid gate avoids the 1 / sequence_length attenuation caused by
        # a softmax over 128 text or 257 vision tokens. Injecting a transformed
        # target also makes this a real target-conditioned residual rather than
        # merely rescaling the original features.
        weights = torch.sigmoid(scores)
        if mask is not None:
            weights = weights * mask.to(weights.dtype)
        target_value = self.target_v(target).unsqueeze(1)
        delta = weights.unsqueeze(-1) * (self.v(features) + target_value)
        out = self.norm(features + self.dropout(delta))
        return out, weights


class MultimodalFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.t2v = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.v2t = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.t_norm = nn.LayerNorm(hidden_dim)
        self.v_norm = nn.LayerNorm(hidden_dim)
        self.target_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.modality_gate = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        h_ta: torch.Tensor,
        h_va: torch.Tensor,
        h_a: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_ctx, _ = self.t2v(h_ta, h_va, h_va)
        v_ctx, _ = self.v2t(
            h_va,
            h_ta,
            h_ta,
            key_padding_mask=~text_mask.bool(),
        )
        t = self.t_norm(h_ta + t_ctx)
        v = self.v_norm(h_va + v_ctx)
        target_token = self.target_proj(h_a).unsqueeze(1)
        fused = self.out_norm(torch.cat([t, v, target_token], dim=1))
        vision_mask = torch.ones(h_va.shape[:2], dtype=torch.bool, device=h_va.device)
        target_mask = torch.ones((h_va.size(0), 1), dtype=torch.bool, device=h_va.device)
        fused_mask = torch.cat([text_mask.bool(), vision_mask, target_mask], dim=1)
        # Pool each modality first, then learn a three-way modality gate. A
        # direct mean over all tokens would give CLIP's 257 tokens roughly twice
        # the prior mass of a 128-token tweet solely because of token count.
        text_pool = (t * text_mask.unsqueeze(-1)).sum(1) / text_mask.sum(1, keepdim=True).clamp_min(1)
        vision_pool = v.mean(dim=1)
        summaries = torch.stack([text_pool, vision_pool, target_token.squeeze(1)], dim=1)
        modality_weights = torch.softmax(self.modality_gate(summaries).squeeze(-1), dim=-1)
        pooled = (summaries * modality_weights.unsqueeze(-1)).sum(dim=1)
        return fused, fused_mask, pooled
