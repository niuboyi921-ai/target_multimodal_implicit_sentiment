from __future__ import annotations

import torch
import torch.nn as nn


class ReasoningTagHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        h_text_selected: torch.Tensor,
        h_visual_selected: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Tag supervision must pass through both latent selectors. Excluding a
        # direct h_f bypass prevents Stage 1 from solving tag prediction while
        # ignoring selection. Product and difference features retain an
        # explicit target-aware cross-modal comparison signal.
        features = torch.cat(
            [
                h_text_selected,
                h_visual_selected,
                h_text_selected * h_visual_selected,
                torch.abs(h_text_selected - h_visual_selected),
            ],
            dim=-1,
        )
        logits = self.net(features)
        return logits, torch.sigmoid(logits)


class PathMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat(xs, dim=-1))


class MultiPathReasoner(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        # Keep the first two paths text-only. This makes their modality scope
        # match the routing-tag semantics instead of leaking the fused
        # multimodal representation (or selected visual features) into them.
        self.direct = PathMLP(hidden_dim * 2, hidden_dim, dropout)
        self.implicit = PathMLP(hidden_dim * 3, hidden_dim, dropout)
        # The Cross path receives deterministic comparison features directly.
        # This preserves an explicit cross-modal inductive bias without an
        # additional unsupervised relation representation.
        self.cross = PathMLP(hidden_dim * 5, hidden_dim, dropout)

    def forward(
        self,
        h_text_selected: torch.Tensor,
        h_visual_selected: torch.Tensor,
        h_text_global: torch.Tensor,
        h_a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_d = self.direct(h_text_selected, h_a)
        h_i = self.implicit(h_text_selected, h_text_global, h_a)
        h_c = self.cross(
            h_text_selected,
            h_visual_selected,
            torch.abs(h_text_selected - h_visual_selected),
            h_text_selected * h_visual_selected,
            h_a,
        )
        return h_d, h_i, h_c


class SoftRouter(nn.Module):
    def __init__(self, epsilon: float, alphas: tuple[float, float, float]) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float))

    def forward(
        self,
        paths: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        predicted_probs: torch.Tensor,
        gold_tags: torch.Tensor | None = None,
        gold_mix: float = 0.0,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        probs = predicted_probs
        if gold_tags is not None and gold_mix > 0:
            probs = gold_mix * gold_tags + (1.0 - gold_mix) * predicted_probs
        weights = self.epsilon + probs * self.alphas.to(probs.device)
        routed = tuple(paths[k] * weights[:, k : k + 1] for k in range(3))
        return routed, weights


class CrossPathInteraction(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        routed_paths: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        h_f: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.stack([*routed_paths, h_f], dim=1)
        z = self.encoder(x).mean(dim=1)
        g = self.gate(torch.cat([z, h_f], dim=-1))
        return self.norm(g * z + (1 - g) * h_f)
