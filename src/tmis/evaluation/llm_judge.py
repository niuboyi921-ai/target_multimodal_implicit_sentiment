from __future__ import annotations

import json
from typing import Any


JUDGE_SYSTEM_PROMPT = r"""
You are evaluating a generated target-level reasoning bridge for multimodal sentiment analysis.

You receive ONLY:
- restored_text
- target
- verified text_evidence
- verified visual_evidence (already represented as text)
- generated reasoning_bridge

Do not infer or use any hidden dataset sentiment label. Judge whether the generated bridge is a faithful, target-specific semantic transformation of the supplied information.

Score each dimension from 1 to 5:
1. evidence_faithfulness: substantive claims are supported by restored_text/evidence; no invented facts.
2. target_ownership: properties, consequences, and evaluation belong to the CURRENT target.
3. reasoning_coherence: grounded_synthesis -> reasoning_transition -> evaluative_implication is logically connected.
4. field_role_separation: synthesis states grounded semantic state; transition explains the transformation; implication states final target-level evaluation/presentation.
5. evaluative_clarity: evaluative_implication is clear enough for a classifier to infer favorable/unfavorable/descriptive meaning without raw class-label meta-language.

Also return hallucination as true/false and a concise rationale.
Do not reward lexical overlap with a reference answer. Multiple paraphrases can be equally valid.
Return valid JSON only with exactly:
{
  "evidence_faithfulness": 1,
  "target_ownership": 1,
  "reasoning_coherence": 1,
  "field_role_separation": 1,
  "evaluative_clarity": 1,
  "hallucination": false,
  "rationale": "..."
}
""".strip()


_SCORE_KEYS = (
    "evidence_faithfulness",
    "target_ownership",
    "reasoning_coherence",
    "field_role_separation",
    "evaluative_clarity",
)


def validate_judge_result(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("LLM judge output must be a JSON object")
    out: dict[str, Any] = {}
    for key in _SCORE_KEYS:
        score = value.get(key)
        if not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"{key} must be an integer in [1, 5]")
        out[key] = score
    hallucination = value.get("hallucination")
    if not isinstance(hallucination, bool):
        raise ValueError("hallucination must be boolean")
    rationale = str(value.get("rationale") or "").strip()
    if not rationale:
        raise ValueError("rationale is empty")
    out["hallucination"] = hallucination
    out["rationale"] = rationale
    return out


def judge_one_openai_compatible(
    *,
    client: Any,
    model: str,
    payload: dict[str, Any],
    temperature: float = 0.0,
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content
    if not isinstance(text, str):
        raise RuntimeError("LLM judge returned no text content")
    return validate_judge_result(json.loads(text))
