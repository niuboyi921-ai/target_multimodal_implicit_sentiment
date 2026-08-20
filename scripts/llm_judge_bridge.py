#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmis.config import load_config, resolve_project_path
from tmis.data.dataset import load_json_records
from tmis.data.schema import normalize_record
from tmis.evaluation.llm_judge import judge_one_openai_compatible
from tmis.training.ai_feedback import load_bailian_credentials
from tmis.utils import write_json


def main() -> None:
    ap = argparse.ArgumentParser(
        description="OFFLINE post-training LLM judge for generated reasoning bridges. Never used as training loss."
    )
    ap.add_argument("--config", default="configs/twitter2015.yaml")
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install optional evaluator dependencies: pip install -r requirements-eval.txt") from exc

    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)
    out_dir = resolve_project_path(cfg, cfg["output_dir"])
    pred_path = Path(args.predictions) if args.predictions else out_dir / "test_predictions.json"
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))

    test_path = resolve_project_path(cfg, cfg["data"]["test_file"])
    records = [normalize_record(r, i) for i, r in enumerate(load_json_records(test_path))]
    by_index = {r.index: r for r in records}

    api_key, base_url = load_bailian_credentials()
    client = OpenAI(api_key=api_key, base_url=base_url)

    judged = []
    rows = predictions[: args.limit] if args.limit else predictions
    for n, pred in enumerate(rows, start=1):
        idx = int(pred["index"])
        record = by_index[idx]
        generated = pred.get("reasoning_bridge_structured") or pred.get("reasoning_bridge")
        payload = {
            "restored_text": record.restored_text,
            "target": record.target,
            "generated_reasoning_bridge": generated,
        }
        try:
            result = judge_one_openai_compatible(
                client=client,
                model=args.model,
                payload=payload,
            )
            judged.append({"index": idx, "judge": result})
            print(f"[{n}/{len(rows)}] index={idx} ok")
        except Exception as exc:
            judged.append({"index": idx, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{n}/{len(rows)}] index={idx} ERROR: {exc}")

    score_keys = [
        "evidence_faithfulness",
        "target_ownership",
        "reasoning_coherence",
        "field_role_separation",
        "evaluative_clarity",
    ]
    successful = [x["judge"] for x in judged if "judge" in x]
    summary = {
        "n_requested": len(rows),
        "n_success": len(successful),
        "n_failed": len(rows) - len(successful),
        "model": args.model,
        "gold_sentiment_was_not_sent_to_judge": True,
        "mean_scores": {
            k: mean([x[k] for x in successful]) if successful else None for k in score_keys
        },
        "hallucination_rate": (
            mean([1.0 if x["hallucination"] else 0.0 for x in successful]) if successful else None
        ),
    }

    output = Path(args.output) if args.output else out_dir / "bridge_llm_judge.json"
    write_json(output, {"summary": summary, "items": judged})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
