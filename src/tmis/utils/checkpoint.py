from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    meta=None,
) -> None:
    """Atomically write a restartable checkpoint on the same filesystem."""
    state = {"model": model.state_dict(), "meta": meta or {}}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
) -> dict:
    # weights_only prevents arbitrary Python object construction when loading
    # a checkpoint. All objects saved above are tensors and primitive metadata.
    state = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(f"invalid checkpoint format: {path}")
    model.load_state_dict(state["model"], strict=True)
    for name, target in (
        ("optimizer", optimizer),
        ("scheduler", scheduler),
        ("scaler", scaler),
    ):
        if target is None:
            continue
        if name not in state:
            raise ValueError(
                f"checkpoint {path} has no {name} state; resume from latest.pt, "
                "not a model-selection checkpoint"
            )
        target.load_state_dict(state[name])
    return state.get("meta", {})
