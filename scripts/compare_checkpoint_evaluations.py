#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmis.utils import write_json


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint evaluation: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare best-joint and final generated-only test evaluations."
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    paths = {
        "best_joint": output_dir / "test_metrics_best_joint.json",
        "generated_only": output_dir / "test_metrics_generated_only.json",
    }
    reports = {name: load(path) for name, path in paths.items()}
    comparison: dict[str, Any] = {
        "checkpoints": {
            name: report.get("checkpoint") for name, report in reports.items()
        },
        "sentiment": {},
        "bridge_structure": {},
        "interpretation": (
            "best_joint is selected on gated development metrics; generated_only is "
            "the final 100% Generated-Bridge adaptation checkpoint. Test results are "
            "reported side by side and must not be used to re-select a checkpoint."
        ),
    }
    for subset in ("full", "implicit", "non_implicit"):
        comparison["sentiment"][subset] = {
            name: {
                metric: report["sentiment"][subset].get(metric)
                for metric in ("n", "accuracy", "macro_f1", "macro_recall")
            }
            for name, report in reports.items()
        }
    comparison["bridge_structure"] = {
        name: report.get("bridge_structure") for name, report in reports.items()
    }
    write_json(output_dir / "test_checkpoint_comparison.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
