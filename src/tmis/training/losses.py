from __future__ import annotations

import torch
import torch.nn.functional as F


def text_evidence_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: float | torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    mask = labels.ne(-100)
    if not mask.any():
        return logits.sum() * 0.0
    weight = None
    if pos_weight is not None:
        weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    raw = F.binary_cross_entropy_with_logits(
        logits[mask], labels[mask], pos_weight=weight, reduction="none"
    )
    if focal_gamma > 0:
        probs = torch.sigmoid(logits[mask])
        pt = torch.where(labels[mask].bool(), probs, 1.0 - probs)
        raw = raw * (1.0 - pt).pow(float(focal_gamma))
    return raw.mean()


def visual_evidence_contrastive_loss(
    visual_repr: torch.Tensor,
    text_repr: torch.Tensor | None,
    has_visual_evidence: torch.Tensor,
    temperature: float = 0.07,
    negative_bank: torch.Tensor | None = None,
    presence_logits: torch.Tensor | None = None,
    presence_weight: float = 1.0,
) -> torch.Tensor:
    presence = visual_repr.sum() * 0.0
    if presence_logits is not None:
        presence = F.binary_cross_entropy_with_logits(
            presence_logits, has_visual_evidence.to(presence_logits.dtype)
        )
    if text_repr is None:
        return float(presence_weight) * presence
    mask = has_visual_evidence.bool()
    if not mask.any():
        return float(presence_weight) * presence
    v = F.normalize(visual_repr[mask], dim=-1)
    t = F.normalize(text_repr[mask], dim=-1)
    candidates = t
    if negative_bank is not None and negative_bank.numel() > 0:
        candidates = torch.cat(
            [t, F.normalize(negative_bank.detach(), dim=-1)], dim=0
        )
    logits = v @ candidates.t() / temperature
    labels = torch.arange(v.size(0), device=v.device)
    alignment = F.cross_entropy(logits, labels)
    if v.size(0) >= 2:
        alignment = 0.5 * (
            alignment + F.cross_entropy((t @ v.t()) / temperature, labels)
        )
    elif negative_bank is None or negative_bank.numel() == 0:
        # Bootstrap the queue on the first positive example.
        alignment = (1.0 - (v * t).sum(-1)).mean()
    return alignment + float(presence_weight) * presence


def reasoning_tag_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def bridge_generation_loss(
    logits: torch.Tensor,
    full_bridge_ids: torch.Tensor,
    has_bridge: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    # T5 predicts the complete bridge target from a right-shifted decoder input:
    # <pad> -> BOS -> GROUND -> ... -> EOS.
    labels = full_bridge_ids
    mask = has_bridge.bool()
    if not mask.any():
        return logits.sum() * 0.0
    logits = logits[mask]
    labels = labels[mask]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_id,
    )


def sentiment_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.cross_entropy(logits, labels, weight=class_weight)
