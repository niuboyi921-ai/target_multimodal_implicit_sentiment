from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tmis.constants import BRIDGE_KEYS, SENTIMENT_TO_ID


@dataclass
class NormalizedRecord:
    index: int
    text: str
    restored_text: str
    target: str
    image: str
    sentiment: str
    source_dataset: str
    text_evidence: list[str]
    visual_evidence: list[str]
    reasoning_tags: dict[str, bool]
    reasoning_bridge: dict[str, str] | None
    raw: dict[str, Any]

    @property
    def sentiment_id(self) -> int:
        return SENTIMENT_TO_ID[self.sentiment]

    @property
    def is_implicit(self) -> bool:
        return self.reasoning_tags["implicit_reasoning_required"]


def _strip_preserve_internal(value: Any) -> str:
    """Strip only outer whitespace; keep internal spacing for exact evidence spans."""
    return str(value or "").strip()


def _norm_scalar(value: Any) -> str:
    return " ".join(str(value or "").split())


def _norm_semantic_text(value: Any) -> str:
    # Bridge/visual evidence does not need source-character offsets.
    return " ".join(str(value or "").split())


def _exact_text_evidence_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise TypeError(f"text_evidence must be list[str], got {type(value).__name__}")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError("text_evidence items must be strings")
        # Preserve internal spaces because text_evidence is defined as an exact
        # contiguous substring of restored_text.
        text = item.strip()
        if text:
            out.append(text)
    return out


def _semantic_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be list[str], got {type(value).__name__}")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"{field_name} items must be strings")
        text = _norm_semantic_text(item)
        if text:
            out.append(text)
    return out


def _strict_bool(value: Any, *, field_name: str, record_index: int) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(
        f"record {record_index}: {field_name} must be JSON boolean true/false, "
        f"got {value!r} ({type(value).__name__})"
    )


def normalize_record(record: dict[str, Any], index: int) -> NormalizedRecord:
    if not isinstance(record, dict):
        raise TypeError(f"record {index}: record must be a JSON object")

    target = _norm_scalar(record.get("target") or record.get("targe"))
    restored_text = _strip_preserve_internal(record.get("restored_text") or record.get("text"))
    text = _strip_preserve_internal(record.get("text") or restored_text)
    image = _norm_scalar(record.get("image"))
    sentiment = _norm_scalar(record.get("sentiment")).lower()
    source_dataset = _norm_scalar(record.get("source_dataset"))

    if not target:
        raise ValueError(f"record {index}: missing target/targe")
    if not restored_text:
        raise ValueError(f"record {index}: missing restored_text/text")
    if not image:
        raise ValueError(f"record {index}: missing image")
    if sentiment not in SENTIMENT_TO_ID:
        raise ValueError(f"record {index}: invalid sentiment={sentiment!r}")

    tags_raw = record.get("reasoning_tags")
    if not isinstance(tags_raw, dict):
        raise TypeError(f"record {index}: reasoning_tags must be an object")

    required_tags = (
        "explicit_cue_present",
        "implicit_reasoning_required",
        "cross_modal_reasoning_required",
    )
    missing = [k for k in required_tags if k not in tags_raw]
    if missing:
        raise ValueError(f"record {index}: reasoning_tags missing keys: {missing}")

    tags = {
        key: _strict_bool(tags_raw[key], field_name=f"reasoning_tags.{key}", record_index=index)
        for key in required_tags
    }

    bridge_raw = record.get("reasoning_bridge")
    bridge: dict[str, str] | None = None
    if bridge_raw is not None:
        if not isinstance(bridge_raw, dict):
            raise TypeError(f"record {index}: reasoning_bridge must be an object")
        extra = set(bridge_raw) - set(BRIDGE_KEYS)
        if extra:
            # Extra keys are usually an annotation-pipeline mistake. Fail loudly
            # so training data stays stable and reproducible.
            raise ValueError(f"record {index}: unexpected reasoning_bridge keys: {sorted(extra)}")
        bridge = {k: _norm_semantic_text(bridge_raw.get(k)) for k in BRIDGE_KEYS}
        if any(not bridge[k] for k in BRIDGE_KEYS):
            raise ValueError(f"record {index}: reasoning_bridge missing/empty one of {BRIDGE_KEYS}")

    text_evidence = _exact_text_evidence_list(record.get("text_evidence"))
    for ev in text_evidence:
        if ev not in restored_text:
            raise ValueError(
                f"record {index}: text_evidence must be an exact contiguous substring "
                f"of restored_text: {ev!r}"
            )

    return NormalizedRecord(
        index=index,
        text=text,
        restored_text=restored_text,
        target=target,
        image=image,
        sentiment=sentiment,
        source_dataset=source_dataset,
        text_evidence=text_evidence,
        visual_evidence=_semantic_list(record.get("visual_evidence"), "visual_evidence"),
        reasoning_tags=tags,
        reasoning_bridge=bridge,
        raw=record,
    )
