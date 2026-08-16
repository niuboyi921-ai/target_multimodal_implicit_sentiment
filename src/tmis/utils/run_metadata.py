from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch

from .io import write_json


TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "sentencepiece",
    "numpy",
    "scikit-learn",
    "Pillow",
    "PyYAML",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(args: list[str], project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def collect_git_state(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    commit = _git(["rev-parse", "HEAD"], root)
    branch = _git(["branch", "--show-current"], root)
    status = _git(["status", "--porcelain", "--untracked-files=no"], root)
    return {
        "commit": commit,
        "branch": branch,
        "tracked_worktree_dirty": bool(status) if status is not None else None,
        "repository_available": commit is not None,
    }


def collect_environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    gpus: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": len(gpus),
        "gpus": gpus,
    }


def initialize_run_state(
    output_dir: str | Path,
    *,
    run_id: str,
    experiment_name: str,
    config_path: str,
    project_root: str | Path,
    command: list[str],
) -> tuple[Path, dict[str, Any]]:
    path = Path(output_dir) / "run_state.json"
    state = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_name": experiment_name,
        "status": "running",
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "config_path": config_path,
        "command": command,
        "git": collect_git_state(project_root),
        "environment": collect_environment(),
        "error": None,
    }
    write_json(path, state)
    return path, state


def finish_run_state(
    path: str | Path,
    state: dict[str, Any],
    *,
    status: str,
    error: dict[str, str] | None = None,
) -> None:
    if status not in {"completed", "failed", "interrupted"}:
        raise ValueError(f"invalid terminal run status: {status}")
    updated = dict(state)
    updated["status"] = status
    updated["finished_at_utc"] = utc_now()
    updated["error"] = error
    write_json(path, updated)
