from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


TAG_NAMES = (
    "explicit_cue_present",
    "implicit_reasoning_required",
    "cross_modal_reasoning_required",
)


def _calc(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"n": 0, "accuracy": float("nan"), "macro_f1": float("nan")}
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),
    }


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    is_implicit: list[bool],
) -> dict[str, Any]:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    imp = np.asarray(is_implicit, dtype=bool)
    return {
        "full": _calc(yt, yp),
        "implicit": _calc(yt[imp], yp[imp]),
        "non_implicit": _calc(yt[~imp], yp[~imp]),
    }


def compute_reasoning_tag_metrics(
    gold: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    gold = np.asarray(gold, dtype=int)
    pred = (np.asarray(probs) >= threshold).astype(int)
    if gold.ndim != 2 or gold.shape[1] != 3 or pred.shape != gold.shape:
        raise ValueError("reasoning tag arrays must have shape [N, 3]")
    per_tag: dict[str, Any] = {}
    f1s = []
    for j, name in enumerate(TAG_NAMES):
        p, r, f, _ = precision_recall_fscore_support(
            gold[:, j], pred[:, j], average="binary", zero_division=0
        )
        per_tag[name] = {
            "accuracy": float(accuracy_score(gold[:, j], pred[:, j])),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
        }
        f1s.append(float(f))
    return {"threshold": threshold, "macro_tag_f1": float(np.mean(f1s)), "per_tag": per_tag}


def compute_text_evidence_token_metrics(
    gold_labels: list[np.ndarray],
    pred_probs: list[np.ndarray],
    threshold: float = 0.5,
) -> dict[str, Any]:
    gold_flat: list[int] = []
    pred_flat: list[int] = []
    for gold, probs in zip(gold_labels, pred_probs):
        gold = np.asarray(gold)
        probs = np.asarray(probs)
        mask = gold != -100
        gold_flat.extend(gold[mask].astype(int).tolist())
        pred_flat.extend((probs[mask] >= threshold).astype(int).tolist())
    if not gold_flat:
        return {"n_tokens": 0, "f1": float("nan")}
    p, r, f, _ = precision_recall_fscore_support(
        gold_flat, pred_flat, average="binary", zero_division=0
    )
    return {
        "n_tokens": len(gold_flat),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
    }
