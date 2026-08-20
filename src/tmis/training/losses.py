from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def selector_regularization_loss(
    text_weights: torch.Tensor,
    text_mask: torch.Tensor,
    visual_weights: torch.Tensor,
    *,
    text_min_ratio: float = 0.05,
    text_max_ratio: float = 0.50,
    visual_min_normalized_entropy: float = 0.20,
    visual_max_normalized_entropy: float = 0.90,
) -> torch.Tensor:
    """Prevent unlabeled latent selectors from selecting everything or collapsing.

    This regularizer does not pretend to identify a unique gold rationale. It
    only constrains the amount of selected text and the concentration of visual
    attention; semantic supervision still comes from reasoning tags and Bridge
    generation.
    """
    if not 0.0 <= text_min_ratio <= text_max_ratio <= 1.0:
        raise ValueError("text selector ratio bounds must satisfy 0 <= min <= max <= 1")
    if not (
        0.0
        <= visual_min_normalized_entropy
        <= visual_max_normalized_entropy
        <= 1.0
    ):
        raise ValueError(
            "visual selector entropy bounds must satisfy 0 <= min <= max <= 1"
        )

    mask = text_mask.to(dtype=torch.float32)
    text = text_weights.float() * mask
    text_ratio = text.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    text_penalty = (
        F.relu(float(text_min_ratio) - text_ratio).square()
        + F.relu(text_ratio - float(text_max_ratio)).square()
    ).mean()

    visual = visual_weights.float().clamp_min(1e-8)
    if visual.size(1) <= 1:
        normalized_entropy = torch.zeros(
            visual.size(0), device=visual.device, dtype=visual.dtype
        )
    else:
        normalized_entropy = -(visual * visual.log()).sum(dim=1) / math.log(
            visual.size(1)
        )
    visual_penalty = (
        F.relu(float(visual_min_normalized_entropy) - normalized_entropy).square()
        + F.relu(
            normalized_entropy - float(visual_max_normalized_entropy)
        ).square()
    ).mean()
    return text_penalty + visual_penalty


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


def sequence_log_probs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_id: int,
    *,
    length_normalize: bool = True,
) -> torch.Tensor:
    """Return one differentiable log-probability per generated Bridge.

    ``logits`` and ``target_ids`` follow the same right-shifted T5
    teacher-forcing contract as :func:`bridge_generation_loss`. Padding and
    tokens beyond the generated EOS mask do not contribute.
    """
    if logits.shape[:2] != target_ids.shape or target_ids.shape != attention_mask.shape:
        raise ValueError("Bridge logits, target IDs, and attention mask must align")
    token_log_probs = F.log_softmax(logits.float(), dim=-1).gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)
    mask = attention_mask.bool() & target_ids.ne(int(pad_id))
    summed = (token_log_probs * mask).sum(dim=-1)
    if length_normalize:
        return summed / mask.sum(dim=-1).clamp_min(1)
    return summed


def dpo_preference_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    chosen_reference_logp: torch.Tensor,
    rejected_reference_logp: torch.Tensor,
    *,
    beta: float = 0.1,
) -> torch.Tensor:
    """Direct Preference Optimization loss for Judge-selected Bridge pairs."""
    if beta <= 0:
        raise ValueError("DPO beta must be positive")
    policy_margin = chosen_logp - rejected_logp
    reference_margin = chosen_reference_logp - rejected_reference_logp
    return -F.logsigmoid(float(beta) * (policy_margin - reference_margin)).mean()


def sentiment_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if class_weight is None:
        return F.cross_entropy(logits, labels)

    # PyTorch's weighted mean divides by the sum of the observed target
    # weights. With the per-device batch size of one used by this project,
    # that would cancel the only sample's class weight exactly. The weights
    # are normalized to E_train[w_y] = 1 by the trainer, so an arithmetic mean
    # keeps the expected sentiment-loss scale unchanged while preserving the
    # intended per-sample reweighting through gradient accumulation.
    per_sample = F.cross_entropy(
        logits,
        labels,
        weight=class_weight,
        reduction="none",
    )
    return per_sample.mean()
