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

    if not bool(model.get("freeze_text_backbone", False)):
        raise ValueError(f"{path}: model.freeze_text_backbone must be true for this dataset size")
    if not bool(model.get("freeze_vision_backbone", False)):
        raise ValueError(f"{path}: model.freeze_vision_backbone must be true for this dataset size")
    if not bool(model.get("freeze_bridge_token_embeddings", False)):
        raise ValueError(
            f"{path}: model.freeze_bridge_token_embeddings must be true for this dataset size"
        )
    lora = model.get("lora") or {}
    if not bool(lora.get("enabled", False)):
        raise ValueError(f"{path}: model.lora.enabled must be true")
    if int(lora.get("rank", 0)) <= 0 or float(lora.get("alpha", 0.0)) <= 0:
        raise ValueError(f"{path}: model.lora rank and alpha must be positive")
    if not 0.0 <= float(lora.get("dropout", -1.0)) < 1.0:
        raise ValueError(f"{path}: model.lora.dropout must be in [0, 1)")
    targets = lora.get("target_modules")
    if not isinstance(targets, list) or not targets or not all(
        isinstance(name, str) and name.strip() for name in targets
    ):
        raise ValueError(f"{path}: model.lora.target_modules must be a non-empty list")

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

    text_min = float(training.get("selector_text_min_ratio", 0.05))
    text_max = float(training.get("selector_text_max_ratio", 0.50))
    visual_min = float(
        training.get("selector_visual_min_normalized_entropy", 0.20)
    )
    visual_max = float(
        training.get("selector_visual_max_normalized_entropy", 0.90)
    )
    if not 0.0 <= text_min <= text_max <= 1.0:
        raise ValueError(
            f"{path}: selector text ratio bounds must satisfy 0 <= min <= max <= 1"
        )
    if not 0.0 <= visual_min <= visual_max <= 1.0:
        raise ValueError(
            f"{path}: selector visual entropy bounds must satisfy 0 <= min <= max <= 1"
        )

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
        loss_weights = stage_cfg.get("loss_weights") or {}
        legacy_losses = {
            "text_evidence",
            "visual_evidence",
        }.intersection(loss_weights)
        if legacy_losses:
            raise ValueError(
                f"{path}: {stage} uses removed artificial-evidence losses: "
                f"{sorted(legacy_losses)}"
            )
        if bool(stage_cfg.get("train_text_backbone", False)):
            raise ValueError(f"{path}: {stage}.train_text_backbone cannot be true")
        if bool(stage_cfg.get("train_vision_backbone", False)):
            raise ValueError(f"{path}: {stage}.train_vision_backbone cannot be true")
        if float(stage_cfg.get("lora_lr_scale", 1.0)) <= 0:
            raise ValueError(f"{path}: {stage}.lora_lr_scale must be positive")

    feedback = training["stage3_bridge"].get("ai_feedback") or {}
    if feedback.get("enabled", False):
        checkpoint_selection = (
            training["stage3_bridge"].get("checkpoint_selection") or {}
        )
        if int(checkpoint_selection.get("sample_size", 0)) <= 0:
            raise ValueError(
                f"{path}: stage3_bridge.checkpoint_selection.sample_size must be positive"
            )
        for name in ("min_structure_valid_rate", "max_critical_error_rate"):
            if not 0.0 <= float(checkpoint_selection.get(name, -1.0)) <= 1.0:
                raise ValueError(
                    f"{path}: stage3_bridge.checkpoint_selection.{name} must be in [0, 1]"
                )
        if not 1.0 <= float(
            checkpoint_selection.get("min_mean_dimension_score", 0.0)
        ) <= 5.0:
            raise ValueError(
                f"{path}: checkpoint_selection.min_mean_dimension_score must be in [1, 5]"
            )
        if int(feedback.get("sample_size_per_epoch", 0)) <= 0:
            raise ValueError(
                f"{path}: stage3_bridge.ai_feedback.sample_size_per_epoch must be positive"
            )
        if int(feedback.get("candidate_count", 0)) < 2:
            raise ValueError(
                f"{path}: stage3_bridge.ai_feedback.candidate_count must be at least 2"
            )
        if float(feedback.get("sampling_temperature", 0.0)) <= 0:
            raise ValueError(
                f"{path}: stage3_bridge.ai_feedback.sampling_temperature must be positive"
            )
        if not 0.0 < float(feedback.get("top_p", 0.0)) <= 1.0:
            raise ValueError(
                f"{path}: stage3_bridge.ai_feedback.top_p must be in (0, 1]"
            )
        for judge_name in ("primary_judge", "review_judge"):
            judge = feedback.get(judge_name)
            if not isinstance(judge, dict) or not str(judge.get("model", "")).strip():
                raise ValueError(
                    f"{path}: stage3_bridge.ai_feedback.{judge_name}.model is required"
                )
            temperature = float(judge.get("temperature", 0.0))
            if not 0.0 <= temperature <= 2.0:
                raise ValueError(
                    f"{path}: stage3_bridge.ai_feedback.{judge_name}.temperature "
                    "must be between 0 and 2"
                )
        rules = feedback.get("review_rules")
        if not isinstance(rules, dict):
            raise ValueError(
                f"{path}: stage3_bridge.ai_feedback.review_rules is required"
            )
        audit_ratio = float(rules.get("random_audit_ratio", -1.0))
        if not 0.0 <= audit_ratio <= 1.0:
            raise ValueError(
                f"{path}: ai_feedback.review_rules.random_audit_ratio must be in [0, 1]"
            )
        dpo = feedback.get("dpo") or {}
        quality_gate = feedback.get("quality_gate") or {}
        for name in (
            "min_faithfulness",
            "min_reasoning_coherence",
            "min_target_consistency",
        ):
            if not 1 <= int(quality_gate.get(name, 0)) <= 5:
                raise ValueError(
                    f"{path}: stage3_bridge.ai_feedback.quality_gate.{name} "
                    "must be in [1, 5]"
                )
        if not 3 <= int(quality_gate.get("min_total_score", 0)) <= 15:
            raise ValueError(
                f"{path}: quality_gate.min_total_score must be in [3, 15]"
            )
        if int(quality_gate.get("max_generation_attempts", 0)) <= 0:
            raise ValueError(
                f"{path}: quality_gate.max_generation_attempts must be positive"
            )
        if dpo.get("enabled", True):
            for name in ("epochs", "batch_size", "grad_accum_steps"):
                if int(dpo.get(name, 0)) <= 0:
                    raise ValueError(
                        f"{path}: stage3_bridge.ai_feedback.dpo.{name} must be positive"
                    )
            if float(dpo.get("lr", 0.0)) <= 0 or float(dpo.get("beta", 0.0)) <= 0:
                raise ValueError(
                    f"{path}: stage3_bridge.ai_feedback DPO lr and beta must be positive"
                )

    joint_selection = training["stage5_joint"].get("selection") or {}
    full_weight = float(joint_selection.get("full_macro_f1_weight", 0.0))
    implicit_weight = float(joint_selection.get("implicit_macro_f1_weight", 0.0))
    if full_weight < 0 or implicit_weight < 0 or full_weight + implicit_weight <= 0:
        raise ValueError(
            f"{path}: Stage-5 Full/Implicit checkpoint weights must be non-negative "
            "and sum to a positive value"
        )
    for name in (
        "min_negative_recall",
        "max_implicit_macro_f1_drop",
        "min_bridge_structure_valid_rate",
    ):
        if not 0.0 <= float(joint_selection.get(name, -1.0)) <= 1.0:
            raise ValueError(f"{path}: stage5_joint.selection.{name} must be in [0, 1]")
    if int(joint_selection.get("min_predictions_per_class", 0)) <= 0:
        raise ValueError(
            f"{path}: stage5_joint.selection.min_predictions_per_class must be positive"
        )


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
