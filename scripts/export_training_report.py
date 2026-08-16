#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmis.config import load_config, resolve_project_path
from tmis.utils import collect_environment, collect_git_state, write_json


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def records_from_json(path: Path) -> list[dict[str, Any]]:
    root = load_json(path)
    if isinstance(root, list):
        return root
    if isinstance(root, dict):
        for key in ("data", "records"):
            if isinstance(root.get(key), list):
                return root[key]
    raise ValueError(f"{path}: expected a list or data/records list")


def assert_finite_json(value: Any, location: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number at {location}: {value}")
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_json(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_json(item, f"{location}[{index}]")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def summarize_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg["data"]
    summary: dict[str, Any] = {
        "dataset_name": data_cfg["dataset_name"],
        "splits": {},
    }
    for split in ("train", "dev", "test"):
        path = resolve_project_path(cfg, data_cfg[f"{split}_file"])
        records = records_from_json(path)
        sentiment = Counter(str(record.get("sentiment", "")).lower() for record in records)
        implicit = sum(
            bool((record.get("reasoning_tags") or {}).get("implicit_reasoning_required"))
            for record in records
        )
        summary["splits"][split] = {
            "records": len(records),
            "sentiment": dict(sorted(sentiment.items())),
            "implicit": implicit,
            "non_implicit": len(records) - implicit,
            "has_text_evidence": sum(bool(record.get("text_evidence")) for record in records),
            "has_visual_evidence": sum(bool(record.get("visual_evidence")) for record in records),
            "has_reasoning_bridge": sum(record.get("reasoning_bridge") is not None for record in records),
        }

    image_dir = resolve_project_path(cfg, data_cfg["image_dir"])
    extensions = {str(item).lower() for item in data_cfg["image_extensions"]}
    images = [
        path for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    summary["images"] = {
        "count": len(images),
        "total_bytes": sum(path.stat().st_size for path in images),
    }
    return summary


def copy_json_artifacts(
    output_dir: Path, artifact_dir: Path, include_console_tail: bool
) -> list[Path]:
    copied: list[Path] = []
    for source in sorted(output_dir.glob("*.json")):
        value = load_json(source)
        assert_finite_json(value, source.name)
        destination = artifact_dir / source.name
        shutil.copy2(source, destination)
        copied.append(destination)
    console_tail = output_dir / "console_tail.txt"
    if include_console_tail and console_tail.is_file():
        destination = artifact_dir / console_tail.name
        shutil.copy2(console_tail, destination)
        copied.append(destination)
    return copied


def create_learning_curves(artifact_dir: Path, destination: Path) -> int:
    rows: list[dict[str, Any]] = []
    for path in sorted(artifact_dir.glob("*_epoch_*_train.json")):
        payload = load_json(path)
        row: dict[str, Any] = {
            "stage": payload.get("stage"),
            "epoch": payload.get("epoch"),
            "routing_gold_mix": payload.get("routing_gold_mix"),
            "generated_bridge_ratio": payload.get("generated_bridge_ratio"),
            "trainable_parameters": payload.get("trainable_parameters"),
            "mixed_precision": payload.get("mixed_precision"),
            "optimizer": payload.get("optimizer"),
        }
        for name, value in (payload.get("mean_losses") or {}).items():
            row[f"loss_{name}"] = value
        rows.append(row)
    fields = [
        "stage",
        "epoch",
        "routing_gold_mix",
        "generated_bridge_ratio",
        "trainable_parameters",
        "mixed_precision",
        "optimizer",
        *sorted({key for row in rows for key in row if key.startswith("loss_")}),
    ]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def checkpoint_manifest(output_dir: Path, include_hashes: bool) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    for pattern in ("*.pt", "*.pth", "*.ckpt"):
        for path in sorted(output_dir.glob(pattern)):
            row = {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "modified_at_utc": utc_mtime(path),
                "sha256": sha256(path) if include_hashes else None,
                "stored_in_git": False,
            }
            checkpoints.append(row)
    return checkpoints


def artifact_record(path: Path, report_dir: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(report_dir).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def markdown_summary(manifest: dict[str, Any], curves: list[dict[str, Any]]) -> str:
    git = manifest["training_git"]
    environment = manifest["environment"]
    lines = [
        f"# Training run: {manifest['run_id']}",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Experiment: `{manifest['experiment_name']}`",
        f"- Dataset: `{manifest['dataset_name']}`",
        f"- Training commit: `{git.get('commit')}`",
        f"- Training branch: `{git.get('branch')}`",
        f"- Tracked worktree dirty: `{git.get('tracked_worktree_dirty')}`",
        f"- Started: `{manifest.get('started_at_utc')}`",
        f"- Finished: `{manifest.get('finished_at_utc')}`",
        "",
        "## Hardware",
        "",
        f"- CUDA available: `{environment.get('cuda_available')}`",
        f"- CUDA version: `{environment.get('torch_cuda_version')}`",
        f"- GPU count: `{environment.get('gpu_count')}`",
    ]
    for gpu in environment.get("gpus", []):
        gib = gpu["total_memory_bytes"] / (1024 ** 3)
        lines.append(f"- GPU {gpu['index']}: `{gpu['name']}` ({gib:.1f} GiB)")
    lines.extend(["", "## Epoch summaries", ""])
    if curves:
        loss_names = sorted({key for row in curves for key in row if key.startswith("loss_")})
        headers = ["stage", "epoch", *loss_names]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in curves:
            values = [str(row.get("stage")), str(row.get("epoch"))]
            for name in loss_names:
                value = row.get(name)
                values.append("" if value is None else f"{float(value):.6g}")
            lines.append("| " + " | ".join(values) + " |")
    else:
        lines.append("No completed epoch summaries were found.")
    lines.extend(
        [
            "",
            "## Analysis contract",
            "",
            "Treat JSON metrics as observed results and architectural explanations as inferences from the referenced code commit. Do not infer checkpoint quality from file size.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export lightweight, Git-trackable training evidence.")
    parser.add_argument("--config", default="configs/twitter2015.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument(
        "--reports-root",
        default=None,
        help="Override the reports root (primarily for isolated validation tests).",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--status", choices=("running", "completed", "failed", "interrupted"))
    parser.add_argument(
        "--hash-checkpoints",
        action="store_true",
        help="Compute checkpoint SHA256 values without copying checkpoint files.",
    )
    parser.add_argument(
        "--include-console-tail",
        action="store_true",
        help="Include console_tail.txt after manually checking it for sensitive content.",
    )
    args = parser.parse_args()

    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(config_path)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else resolve_project_path(cfg, cfg["output_dir"]).resolve()
    )
    if not output_dir.is_dir():
        raise FileNotFoundError(f"training output directory does not exist: {output_dir}")

    state_path = output_dir / "run_state.json"
    run_state = load_json(state_path) if state_path.is_file() else {}
    run_id = args.run_id or run_state.get("run_id")
    if not run_id:
        run_id = f"{cfg['experiment_name']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in run_id):
        raise ValueError("run ID contains unsupported characters")

    reports_root = (
        Path(args.reports_root).resolve()
        if args.reports_root
        else (ROOT / "reports").resolve()
    )
    report_dir = (
        Path(args.report_dir).resolve()
        if args.report_dir
        else (reports_root / cfg["data"]["dataset_name"] / run_id).resolve()
    )
    try:
        report_dir.relative_to(reports_root)
    except ValueError as exc:
        raise ValueError(f"report directory must stay inside {reports_root}") from exc
    if (report_dir / "run_manifest.json").exists():
        raise FileExistsError(f"report already exists and will not be overwritten: {report_dir}")
    artifact_dir = report_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_snapshot = report_dir / "config.yaml"
    config_snapshot.write_text(
        yaml.safe_dump(raw_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    environment = run_state.get("environment") or collect_environment()
    training_git = run_state.get("git") or collect_git_state(ROOT)
    write_json(report_dir / "environment.json", environment)
    write_json(report_dir / "dataset_summary.json", summarize_dataset(cfg))
    copied = copy_json_artifacts(output_dir, artifact_dir, args.include_console_tail)
    curves_path = report_dir / "learning_curves.csv"
    curve_count = create_learning_curves(artifact_dir, curves_path)
    checkpoint_rows = checkpoint_manifest(output_dir, args.hash_checkpoints)
    write_json(report_dir / "checkpoint_manifest.json", checkpoint_rows)

    curve_rows: list[dict[str, Any]] = []
    if curve_count:
        with curves_path.open("r", encoding="utf-8", newline="") as stream:
            curve_rows = list(csv.DictReader(stream))
    status = args.status or run_state.get("status") or "interrupted"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "source_run_id": run_state.get("run_id"),
        "experiment_name": cfg["experiment_name"],
        "dataset_name": cfg["data"]["dataset_name"],
        "status": status,
        "started_at_utc": run_state.get("started_at_utc"),
        "finished_at_utc": run_state.get("finished_at_utc"),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_git": training_git,
        "export_git": collect_git_state(ROOT),
        "environment": environment,
        "source_output_dir": f"outputs/{cfg['data']['dataset_name']}/runs/{run_id}",
        "checkpoint_files_are_external": True,
        "checkpoint_hashes_computed": bool(args.hash_checkpoints),
        "console_tail_included": bool(args.include_console_tail),
        "copied_output_artifact_count": len(copied),
        "epoch_summary_count": curve_count,
        "error": run_state.get("error"),
    }
    summary_path = report_dir / "RUN_SUMMARY.md"
    summary_path.write_text(markdown_summary(manifest, curve_rows), encoding="utf-8")

    report_files = [
        config_snapshot,
        report_dir / "environment.json",
        report_dir / "dataset_summary.json",
        curves_path,
        report_dir / "checkpoint_manifest.json",
        summary_path,
        *copied,
    ]
    manifest["artifacts"] = [artifact_record(path, report_dir) for path in report_files]
    write_json(report_dir / "run_manifest.json", manifest)
    print(report_dir)


if __name__ == "__main__":
    main()
