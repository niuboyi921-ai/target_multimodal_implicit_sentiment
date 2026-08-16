from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def _validate_config(cfg: dict[str, Any], path: Path) -> None:
    try:
        data = cfg["data"]
        model = cfg["model"]
        training = cfg["training"]
        evaluation = cfg["evaluation"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: config is missing a required top-level section") from exc

    heads = int(model["num_heads"])
    if heads <= 0:
        raise ValueError(f"{path}: model.num_heads must be positive")
    for name in ("hidden_dim", "bridge_encoder_dim"):
        width = int(model[name])
        if width <= 0 or width % heads != 0:
            raise ValueError(f"{path}: model.{name} must be positive and divisible by num_heads")

    bridge_length = int(data["max_bridge_length"])
    min_field = int(model.get("bridge_min_tokens_per_field", 2))
    min_bridge_length = 3 * min_field + 5
    if bridge_length < min_bridge_length:
        raise ValueError(
            f"{path}: data.max_bridge_length must be at least {min_bridge_length}"
        )
    generation_length = int(evaluation["max_generation_length"])
    if generation_length < min_bridge_length or generation_length > bridge_length:
        raise ValueError(
            f"{path}: evaluation.max_generation_length must be in "
            f"[{min_bridge_length}, data.max_bridge_length]"
        )

    for name in ("batch_size", "eval_batch_size", "grad_accum_steps"):
        if int(training[name]) <= 0:
            raise ValueError(f"{path}: training.{name} must be positive")
    if not 0.0 <= float(training.get("warmup_ratio", 0.1)) <= 1.0:
        raise ValueError(f"{path}: training.warmup_ratio must be between 0 and 1")
    if str(training.get("optimizer", "adafactor")).lower() not in {"adafactor", "adamw"}:
        raise ValueError(f"{path}: training.optimizer must be adafactor or adamw")
    if str(training.get("mixed_precision", "auto")).lower() not in {
        "auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32", "none"
    }:
        raise ValueError(f"{path}: invalid training.mixed_precision")

    for stage in (
        "stage1_aux",
        "stage2_reasoning_warmup",
        "stage3_bridge",
        "stage4_classifier",
        "stage5_joint",
    ):
        stage_cfg = training[stage]
        if stage_cfg.get("enabled", True) and int(stage_cfg["epochs"]) <= 0:
            raise ValueError(f"{path}: enabled {stage}.epochs must be positive")
        for ratio_name in (
            "routing_gold_mix_start",
            "routing_gold_mix_end",
            "generated_bridge_ratio_start",
            "generated_bridge_ratio_end",
        ):
            if ratio_name in stage_cfg and not 0.0 <= float(stage_cfg[ratio_name]) <= 1.0:
                raise ValueError(f"{path}: {stage}.{ratio_name} must be between 0 and 1")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: YAML root must be an object")
    _validate_config(cfg, path)
    cfg["_config_path"] = str(path.resolve())
    cfg["_project_root"] = str(path.resolve().parent.parent)
    return cfg


def resolve_project_path(cfg: dict[str, Any], value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return Path(cfg["_project_root"]) / p
