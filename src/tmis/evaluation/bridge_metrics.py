from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from tmis.constants import (
    BRIDGE_BOS_TOKEN,
    BRIDGE_EOS_TOKEN,
    GROUND_TOKEN,
    IMPLICATION_TOKEN,
    TRANSITION_TOKEN,
)


BRIDGE_FIELDS = (
    "grounded_synthesis",
    "reasoning_transition",
    "evaluative_implication",
)


@dataclass
class ParsedBridge:
    fields: dict[str, str]
    structure_valid: bool
    error: str | None = None

    @property
    def flattened(self) -> str:
        return " ".join(self.fields[k] for k in BRIDGE_FIELDS if self.fields[k]).strip()


def parse_bridge_text(text: str) -> ParsedBridge:
    cleaned = str(text or "").replace(BRIDGE_BOS_TOKEN, "").replace(BRIDGE_EOS_TOKEN, "").strip()
    out = {k: "" for k in BRIDGE_FIELDS}
    try:
        g = cleaned.index(GROUND_TOKEN)
        t = cleaned.index(TRANSITION_TOKEN, g + len(GROUND_TOKEN))
        i = cleaned.index(IMPLICATION_TOKEN, t + len(TRANSITION_TOKEN))
        if not (g <= t <= i):
            raise ValueError("markers are out of order")
        out["grounded_synthesis"] = cleaned[g + len(GROUND_TOKEN) : t].strip()
        out["reasoning_transition"] = cleaned[t + len(TRANSITION_TOKEN) : i].strip()
        out["evaluative_implication"] = cleaned[i + len(IMPLICATION_TOKEN) :].strip()
        valid = all(out.values())
        return ParsedBridge(out, valid, None if valid else "one or more bridge fields are empty")
    except ValueError as exc:
        # Preserve the raw text for debugging, but mark structure invalid.
        out["grounded_synthesis"] = cleaned
        return ParsedBridge(out, False, str(exc))


def flatten_reference_bridge(bridge: dict[str, str] | None) -> str:
    if not bridge:
        return ""
    return " ".join(str(bridge.get(k, "")).strip() for k in BRIDGE_FIELDS).strip()


def _tokens(text: str) -> list[str]:
    # Whitespace tokens are deliberate: ROUGE-L is only a lexical auxiliary
    # metric here, not the semantic ground truth for reasoning quality.
    return str(text or "").lower().split()


def rouge_l_f1(reference: str, candidate: str) -> float:
    a = _tokens(reference)
    b = _tokens(candidate)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    precision = lcs / len(b)
    recall = lcs / len(a)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(np.mean(vals)) if vals else float("nan")


def compute_bridge_reference_metrics(
    generated: list[dict[str, str]],
    references: list[dict[str, str] | None],
    *,
    compute_bertscore: bool = False,
    bertscore_model: str | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    if len(generated) != len(references):
        raise ValueError("generated/references length mismatch")

    valid_pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for gen, ref in zip(generated, references):
        if ref is not None:
            valid_pairs.append((gen, ref))

    result: dict[str, Any] = {
        "n_total": len(generated),
        "n_with_reference": len(valid_pairs),
        "note": (
            "ROUGE-L/BERTScore compare against one reference wording only. "
            "They are auxiliary metrics because reasoning text has multiple valid paraphrases."
        ),
    }
    if not valid_pairs:
        return result

    field_rouge: dict[str, float] = {}
    for field in BRIDGE_FIELDS:
        field_rouge[field] = _mean(
            rouge_l_f1(ref[field], gen[field]) for gen, ref in valid_pairs
        )
    full_rouge = _mean(
        rouge_l_f1(flatten_reference_bridge(ref), flatten_reference_bridge(gen))
        for gen, ref in valid_pairs
    )
    result["rouge_l_f1"] = {"full": full_rouge, **field_rouge}

    if compute_bertscore:
        try:
            from bert_score import score as bert_score  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "BERTScore requested but bert-score is not installed. "
                "Install requirements-eval.txt or pip install bert-score."
            ) from exc

        def run_bs(cands: list[str], refs: list[str]) -> dict[str, float]:
            kwargs: dict[str, Any] = {"verbose": False, "rescale_with_baseline": False}
            if bertscore_model:
                kwargs["model_type"] = bertscore_model
            else:
                kwargs["lang"] = lang
            p, r, f = bert_score(cands, refs, **kwargs)
            return {
                "precision": float(p.mean().item()),
                "recall": float(r.mean().item()),
                "f1": float(f.mean().item()),
            }

        bs: dict[str, Any] = {}
        bs["full"] = run_bs(
            [flatten_reference_bridge(gen) for gen, _ in valid_pairs],
            [flatten_reference_bridge(ref) for _, ref in valid_pairs],
        )
        for field in BRIDGE_FIELDS:
            bs[field] = run_bs(
                [gen[field] for gen, _ in valid_pairs],
                [ref[field] for _, ref in valid_pairs],
            )
        result["bertscore"] = bs

    return result


def compute_structure_metrics(parsed: list[ParsedBridge]) -> dict[str, Any]:
    n = len(parsed)
    if n == 0:
        return {"n": 0, "valid_rate": float("nan")}
    errors = Counter(x.error or "valid" for x in parsed)
    return {
        "n": n,
        "valid_count": sum(x.structure_valid for x in parsed),
        "valid_rate": float(sum(x.structure_valid for x in parsed) / n),
        "error_counts": dict(errors),
    }
