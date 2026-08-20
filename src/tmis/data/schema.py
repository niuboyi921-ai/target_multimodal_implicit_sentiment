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
    reasoning_tags: dict[str, bool]
    reasoning_bridge: dict[str, str] | None
    raw: dict[str, Any]

    @property
    def sentiment_id(self) -> int:
        return SENTIMENT_TO_ID[self.sentiment]

    @property
    def is_implicit(self) -> bool:
        return self.reasoning_tags["implicit_sentiment_present"]


def _strip_preserve_internal(value: Any) -> str:
    """Strip only outer whitespace while preserving the original tweet text."""
    return str(value or "").strip()


def _norm_scalar(value: Any) -> str:
    return " ".join(str(value or "").split())


def _norm_semantic_text(value: Any) -> str:
    # Bridge/visual evidence does not need source-character offsets.
    return " ".join(str(value or "").split())


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

    legacy_evidence = [
        name for name in ("text_evidence", "visual_evidence") if name in record
    ]
    if legacy_evidence:
        raise ValueError(
            f"record {index}: legacy artificial evidence fields are not supported: "
            f"{legacy_evidence}; run scripts/remove_evidence_supervision.py"
        )

    tags_raw = record.get("reasoning_tags")
    if not isinstance(tags_raw, dict):
        raise TypeError(f"record {index}: reasoning_tags must be an object")

    required_tags = (
        "explicit_cue_present",
        "implicit_sentiment_present",
        "cross_modal_reasoning_required",
    )
    missing = [k for k in required_tags if k not in tags_raw]
    if missing:
        raise ValueError(f"record {index}: reasoning_tags missing keys: {missing}")

    tags = {
        key: _strict_bool(tags_raw[key], field_name=f"reasoning_tags.{key}", record_index=index)
        for key in required_tags
    }
    if tags["implicit_sentiment_present"] and sentiment == "neutral":
        raise ValueError(
            f"record {index}: implicit_sentiment_present=true requires positive or negative sentiment"
        )
    if sentiment != "neutral" and not (
        tags["explicit_cue_present"] or tags["implicit_sentiment_present"]
    ):
        raise ValueError(
            f"record {index}: positive/negative sentiment requires at least one of "
            "explicit_cue_present or implicit_sentiment_present to be true; both may be true"
        )

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

    return NormalizedRecord(
        index=index,
        text=text,
        restored_text=restored_text,
        target=target,
        image=image,
        sentiment=sentiment,
        source_dataset=source_dataset,
        reasoning_tags=tags,
        reasoning_bridge=bridge,
        raw=record,
    )
