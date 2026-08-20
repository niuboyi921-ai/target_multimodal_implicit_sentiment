#!/usr/bin/env python3
"""Remove legacy artificial evidence fields from local dataset JSON files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REMOVED_FIELDS = ("text_evidence", "visual_evidence")


def _records(root: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(root, list):
        records = root
    elif isinstance(root, dict) and isinstance(root.get("data"), list):
        records = root["data"]
    elif isinstance(root, dict) and isinstance(root.get("records"), list):
        records = root["records"]
    elif isinstance(root, dict) and "reasoning_tags" in root:
        records = [root]
    else:
        raise ValueError(f"{path}: expected a record, list, or data/records list")
    if not all(isinstance(record, dict) for record in records):
        raise TypeError(f"{path}: every dataset record must be a JSON object")
    return records


def migrate(path: Path) -> tuple[int, dict[str, int]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        root = json.load(stream)
    records = _records(root, path)
    removed = {field: 0 for field in REMOVED_FIELDS}
    changed = False
    for record in records:
        for field in REMOVED_FIELDS:
            if field in record:
                del record[field]
                removed[field] += 1
                changed = True
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(root, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
    return len(records), removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="JSON files to migrate; defaults to both configured datasets and data/sample_record.json.",
    )
    args = parser.parse_args()
    paths = args.paths or [
        ROOT / "data" / dataset / f"{split}.json"
        for dataset in ("twitter2015", "twitter2017")
        for split in ("train", "dev", "test")
    ] + [ROOT / "data" / "sample_record.json"]
    existing = [path.resolve() for path in paths if path.is_file()]
    if not existing:
        raise SystemExit("No dataset JSON files were found.")
    for path in existing:
        count, removed = migrate(path)
        print(
            f"{path}: records={count} "
            f"removed_text_evidence={removed['text_evidence']} "
            f"removed_visual_evidence={removed['visual_evidence']}"
        )


if __name__ == "__main__":
    main()
