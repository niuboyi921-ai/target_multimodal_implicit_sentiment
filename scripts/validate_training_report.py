#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "run_id",
    "experiment_name",
    "dataset_name",
    "status",
    "training_git",
    "environment",
    "artifacts",
}
REQUIRED_FILES = {
    "config.yaml",
    "environment.json",
    "dataset_summary.json",
    "learning_curves.csv",
    "checkpoint_manifest.json",
    "RUN_SUMMARY.md",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_finite(value: Any, location: str, errors: list[str]) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{location}: non-finite number {value}")
    elif isinstance(value, dict):
        for key, item in value.items():
            check_finite(item, f"{location}.{key}", errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            check_finite(item, f"{location}[{index}]", errors)


def validate(report_dir: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = report_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return [f"{report_dir}: missing run_manifest.json"]
    try:
        manifest = load_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{manifest_path}: invalid JSON: {exc}"]
    if not isinstance(manifest, dict):
        return [f"{manifest_path}: root must be an object"]
    missing = REQUIRED_MANIFEST_KEYS - set(manifest)
    if missing:
        errors.append(f"{manifest_path}: missing keys {sorted(missing)}")
    if manifest.get("schema_version") != 1:
        errors.append(f"{manifest_path}: unsupported schema_version")
    if manifest.get("status") not in {"running", "completed", "failed", "interrupted"}:
        errors.append(f"{manifest_path}: invalid status")
    check_finite(manifest, str(manifest_path), errors)

    for relative in REQUIRED_FILES:
        if not (report_dir / relative).is_file():
            errors.append(f"{report_dir}: missing required file {relative}")

    seen: set[str] = set()
    for record in manifest.get("artifacts", []):
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            errors.append(f"{manifest_path}: malformed artifact record")
            continue
        relative = record["path"]
        if relative in seen:
            errors.append(f"{manifest_path}: duplicate artifact {relative}")
        seen.add(relative)
        path = (report_dir / relative).resolve()
        try:
            path.relative_to(report_dir.resolve())
        except ValueError:
            errors.append(f"{manifest_path}: artifact escapes report directory: {relative}")
            continue
        if not path.is_file():
            errors.append(f"{manifest_path}: missing artifact {relative}")
            continue
        if int(record.get("size_bytes", -1)) != path.stat().st_size:
            errors.append(f"{relative}: size does not match manifest")
        if record.get("sha256") != sha256(path):
            errors.append(f"{relative}: SHA256 does not match manifest")
    missing_from_index = REQUIRED_FILES - seen
    if missing_from_index:
        errors.append(
            f"{manifest_path}: required files missing from artifact index: "
            f"{sorted(missing_from_index)}"
        )

    forbidden_checkpoints = [
        path for suffix in ("*.pt", "*.pth", "*.ckpt", "*.safetensors")
        for path in report_dir.rglob(suffix)
    ]
    if forbidden_checkpoints:
        errors.append(
            f"{report_dir}: checkpoint binaries are forbidden in reports: "
            f"{[path.name for path in forbidden_checkpoints]}"
        )

    for path in report_dir.rglob("*.json"):
        try:
            payload = load_json(path)
            check_finite(payload, str(path), errors)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: invalid JSON: {exc}")

    curves = report_dir / "learning_curves.csv"
    if curves.is_file():
        try:
            with curves.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or not {"stage", "epoch"}.issubset(reader.fieldnames):
                    errors.append(f"{curves}: missing stage/epoch columns")
                list(reader)
        except (OSError, csv.Error) as exc:
            errors.append(f"{curves}: invalid CSV: {exc}")

    checkpoints = report_dir / "checkpoint_manifest.json"
    if checkpoints.is_file():
        payload = load_json(checkpoints)
        if not isinstance(payload, list):
            errors.append(f"{checkpoints}: root must be a list")
        elif any(item.get("stored_in_git") is not False for item in payload if isinstance(item, dict)):
            errors.append(f"{checkpoints}: checkpoint files must remain external to Git")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one or all exported training reports.")
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if bool(args.report_dir) == bool(args.all):
        parser.error("choose exactly one of --report-dir or --all")

    if args.all:
        root = Path(__file__).resolve().parents[1] / "reports"
        report_dirs = sorted(path.parent for path in root.rglob("run_manifest.json"))
    else:
        report_dirs = [Path(args.report_dir).resolve()]
    if not report_dirs:
        print("No exported training reports found; nothing to validate.")
        return

    all_errors: list[str] = []
    for report_dir in report_dirs:
        errors = validate(report_dir)
        if errors:
            all_errors.extend(errors)
        else:
            print(f"REPORT_OK {report_dir}")
    if all_errors:
        for error in all_errors:
            print(f"REPORT_ERROR {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
