from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize_training_judge(
    output_dir: str | Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Report actual Judge usage without exposing credentials or prompt payloads."""
    output_dir = Path(output_dir)
    feedback = (cfg.get("training", {}).get("stage3_bridge", {}).get("ai_feedback") or {})
    totals = {
        "api_calls": 0,
        "cache_hits": 0,
        "cache_writes": 0,
        "failed_attempts": 0,
        "preference_pairs": 0,
        "quality_rejected_pairs": 0,
        "absolute_judged_records": 0,
        "reviewed_records_or_pairs": 0,
    }
    artifacts: list[str] = []
    for path in sorted(output_dir.glob("stage3_ai_feedback_epoch_*_preferences.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        summary = payload.get("summary") or {}
        usage = summary.get("judge_usage") or {}
        for name in ("api_calls", "cache_hits", "cache_writes", "failed_attempts"):
            totals[name] += int(usage.get(name, 0))
        totals["preference_pairs"] += int(summary.get("preference_pairs", 0))
        totals["quality_rejected_pairs"] += int(
            summary.get("quality_rejected_pairs", 0)
        )
        totals["reviewed_records_or_pairs"] += int(summary.get("reviewed_pairs", 0))
        artifacts.append(path.name)
    for path in sorted(output_dir.glob("stage3_dev_bridge_metrics_epoch_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        absolute = payload.get("absolute_judge") or {}
        if not absolute:
            continue
        usage = absolute.get("judge_usage") or {}
        for name in ("api_calls", "cache_hits", "cache_writes", "failed_attempts"):
            totals[name] += int(usage.get(name, 0))
        totals["absolute_judged_records"] += int(absolute.get("sample_size", 0))
        totals["reviewed_records_or_pairs"] += int(
            absolute.get("reviewed_records", 0)
        )
        artifacts.append(path.name)
    primary = feedback.get("primary_judge") or {}
    review = feedback.get("review_judge") or {}
    used = bool(
        totals["preference_pairs"]
        or totals["absolute_judged_records"]
        or totals["cache_hits"]
        or totals["api_calls"]
    )
    return {
        "configured": bool(feedback.get("enabled", False)),
        "used_during_training": used,
        "remote_api_called_during_training": totals["api_calls"] > 0,
        "primary_model": primary.get("model"),
        "review_model": review.get("model"),
        "pairwise_prompt_version": "bridge-pair-v1",
        "absolute_prompt_version": "bridge-absolute-v1",
        "totals": totals,
        "source_artifacts": artifacts,
        "credentials_included": False,
    }
