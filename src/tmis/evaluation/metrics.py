from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from tmis.constants import ID_TO_SENTIMENT


TAG_NAMES = (
    "explicit_cue_present",
    "implicit_sentiment_present",
    "cross_modal_reasoning_required",
)


def _calc(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "prediction_counts": {name: 0 for name in ID_TO_SENTIMENT.values()},
            "per_class": {},
        }
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    class_precision, class_recall, class_f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=sorted(ID_TO_SENTIMENT),
            average=None,
            zero_division=0,
        )
    )
    prediction_counts = {
        ID_TO_SENTIMENT[class_id]: int(np.sum(y_pred == class_id))
        for class_id in sorted(ID_TO_SENTIMENT)
    }
    per_class = {
        ID_TO_SENTIMENT[class_id]: {
            "precision": float(class_precision[position]),
            "recall": float(class_recall[position]),
            "f1": float(class_f1[position]),
            "support": int(support[position]),
            "predicted": prediction_counts[ID_TO_SENTIMENT[class_id]],
        }
        for position, class_id in enumerate(sorted(ID_TO_SENTIMENT))
    }
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),
        "prediction_counts": prediction_counts,
        "per_class": per_class,
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
