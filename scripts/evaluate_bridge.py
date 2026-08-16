#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmis.config import load_config, resolve_project_path
from tmis.evaluation import compute_bridge_reference_metrics, compute_structure_metrics, parse_bridge_text
from tmis.utils import write_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline automatic evaluation of generated reasoning_bridge text.")
    ap.add_argument("--config", default="configs/twitter2015.yaml")
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--bertscore", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)
    out_dir = resolve_project_path(cfg, cfg["output_dir"])
    pred_path = Path(args.predictions) if args.predictions else out_dir / "test_predictions.json"
    rows = json.loads(pred_path.read_text(encoding="utf-8"))

    parsed = []
    generated = []
    refs = []
    for row in rows:
        struct = row.get("reasoning_bridge_structured")
        if isinstance(struct, dict):
            generated.append(struct)
            # Recreate ParsedBridge from saved validity metadata for structure summary.
            raw = (
                f"[GROUND] {struct.get('grounded_synthesis','')} "
                f"[TRANSITION] {struct.get('reasoning_transition','')} "
                f"[IMPLICATION] {struct.get('evaluative_implication','')}"
            )
            parsed.append(parse_bridge_text(raw))
        else:
            p = parse_bridge_text(str(row.get("reasoning_bridge", "")))
            parsed.append(p)
            generated.append(p.fields)
        ref = row.get("reference_reasoning_bridge")
        refs.append(ref if isinstance(ref, dict) else None)

    bridge_cfg = cfg.get("evaluation", {}).get("bridge_metrics", {})
    result = {
        "structure": compute_structure_metrics(parsed),
        "reference": compute_bridge_reference_metrics(
            generated,
            refs,
            compute_bertscore=args.bertscore,
            bertscore_model=bridge_cfg.get("bertscore_model"),
        ),
    }
    output = Path(args.output) if args.output else out_dir / "bridge_metrics.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
