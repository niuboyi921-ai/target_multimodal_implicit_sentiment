from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

from tmis.evaluation import (
    compute_bridge_reference_metrics,
    compute_metrics,
    compute_structure_metrics,
    parse_bridge_text,
)
from tmis.models.lora import is_lora_parameter
from tmis.training.losses import (
    bridge_generation_loss,
    dpo_preference_loss,
    reasoning_tag_loss,
    sequence_log_probs,
    selector_regularization_loss,
    sentiment_loss,
)
from tmis.training.ai_feedback import (
    AbsoluteBailianJudge,
    BridgePreferenceCollator,
    BridgePreferenceDataset,
    PairwiseBailianJudge,
    passes_quality_gate,
)
from tmis.utils import load_checkpoint, save_checkpoint, set_seed, write_json


STAGE_ORDER = [
    "stage1_aux",
    "stage2_reasoning_warmup",
    "stage3_bridge",
    "stage4_classifier",
    "stage5_joint",
]
TAG_NAMES = (
    "explicit_cue_present",
    "implicit_sentiment_present",
    "cross_modal_reasoning_required",
)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def linear_schedule(start: float, end: float, epoch: int, epochs: int) -> float:
    if epochs <= 1:
        return float(end)
    t = epoch / (epochs - 1)
    return float(start + (end - start) * t)


def restore_optional_metric(value: Any, default: float = -math.inf) -> float:
    """Restore a checkpoint metric that is serialized as null before it exists."""
    return float(default) if value is None else float(value)


def effective_number_class_weights(
    counts: torch.Tensor,
    beta: float = 0.999,
) -> torch.Tensor:
    """Return effective-number weights normalized to E_train[w_y] = 1."""
    if counts.ndim != 1:
        raise ValueError("class counts must be a one-dimensional tensor")
    if not 0.0 <= beta < 1.0:
        raise ValueError("class_balance_beta must satisfy 0 <= beta < 1")

    counts = counts.to(dtype=torch.float32)
    present = counts.gt(0)
    total = counts.sum()
    if not present.any() or not torch.isfinite(total) or total <= 0:
        raise ValueError("class counts must contain at least one positive finite value")

    weights = torch.zeros_like(counts)
    effective_number = 1.0 - torch.pow(
        torch.as_tensor(beta, dtype=counts.dtype, device=counts.device),
        counts[present],
    )
    weights[present] = (1.0 - beta) / effective_number.clamp_min(
        torch.finfo(counts.dtype).eps
    )

    class_probability = counts / total
    expected_weight = (class_probability * weights).sum()
    if not torch.isfinite(expected_weight) or expected_weight <= 0:
        raise ValueError("effective-number class weights are not finite and positive")
    return weights / expected_weight


def bridge_selection_score(metrics: dict[str, Any], metric_name: str) -> float:
    """Extract a finite Stage-3 checkpoint score from autoregressive metrics."""
    if metric_name == "rouge_l_f1_full":
        try:
            score = float(metrics["reference"]["rouge_l_f1"]["full"])
        except KeyError as exc:
            raise ValueError(
                "Stage 3 selection_metric=rouge_l_f1_full requires at least one "
                "reference reasoning_bridge in the dev split"
            ) from exc
    elif metric_name == "structure_valid_rate":
        score = float(metrics["structure"]["valid_rate"])
    else:
        raise ValueError(
            f"unsupported Stage 3 selection_metric={metric_name!r}; expected "
            "'rouge_l_f1_full' or 'structure_valid_rate'"
        )
    if not math.isfinite(score):
        raise ValueError(
            f"Stage 3 selection metric {metric_name!r} is not finite: {score}. "
            "Check that the dev split contains reference reasoning bridges."
        )
    return score


def stage3_checkpoint_decision(
    metrics: dict[str, Any], selection_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Apply absolute-quality checkpoint gates before ranking Stage-3 epochs."""
    judged = metrics.get("absolute_judge") or {}
    dimension_means = judged.get("dimension_means") or {}
    sample_size = int(judged.get("sample_size", 0))
    score = (
        sum(float(dimension_means.get(name, 0.0)) for name in (
            "faithfulness", "reasoning_coherence", "target_consistency"
        ))
        / 15.0
    )
    reasons: list[str] = []
    structure_rate = float((metrics.get("structure") or {}).get("valid_rate", 0.0))
    if sample_size <= 0:
        reasons.append("no_absolute_judge_samples")
    if structure_rate < float(selection_cfg.get("min_structure_valid_rate", 0.90)):
        reasons.append("bridge_structure_valid_rate_below_minimum")
    minimum_dimension = float(selection_cfg.get("min_mean_dimension_score", 3.0))
    for name in ("faithfulness", "reasoning_coherence", "target_consistency"):
        if float(dimension_means.get(name, 0.0)) < minimum_dimension:
            reasons.append(f"mean_{name}_below_minimum")
    if float(judged.get("critical_error_rate", 1.0)) > float(
        selection_cfg.get("max_critical_error_rate", 0.20)
    ):
        reasons.append("critical_error_rate_above_maximum")
    return {
        "metric": "absolute_judge_mean_normalized",
        "score": float(score),
        "eligible": not reasons,
        "gate_failures": reasons,
        "rouge_l_f1_tiebreak": float(
            (((metrics.get("reference") or {}).get("rouge_l_f1") or {}).get("full", 0.0))
        ),
    }


def stage5_checkpoint_decision(
    metrics: dict[str, Any],
    selection_cfg: dict[str, Any],
    best_implicit_macro_f1: float,
) -> dict[str, Any]:
    """Rank joint checkpoints while rejecting collapse and implicit degradation."""
    full = metrics["full"]
    implicit = metrics["implicit"]
    full_weight = float(selection_cfg.get("full_macro_f1_weight", 0.40))
    implicit_weight = float(selection_cfg.get("implicit_macro_f1_weight", 0.60))
    weight_sum = full_weight + implicit_weight
    if weight_sum <= 0:
        raise ValueError("Stage-5 checkpoint weights must sum to a positive value")
    score = (
        full_weight * float(full["macro_f1"])
        + implicit_weight * float(implicit["macro_f1"])
    ) / weight_sum
    reasons: list[str] = []
    prediction_counts = full.get("prediction_counts") or {}
    missing = [name for name in ("positive", "neutral", "negative") if int(
        prediction_counts.get(name, 0)
    ) < int(selection_cfg.get("min_predictions_per_class", 1))]
    if missing:
        reasons.append("unpredicted_classes:" + ",".join(missing))
    negative_recall = float(
        ((full.get("per_class") or {}).get("negative") or {}).get("recall", 0.0)
    )
    if negative_recall < float(selection_cfg.get("min_negative_recall", 0.01)):
        reasons.append("negative_recall_below_minimum")
    if best_implicit_macro_f1 >= 0 and float(implicit["macro_f1"]) < (
        best_implicit_macro_f1
        - float(selection_cfg.get("max_implicit_macro_f1_drop", 0.02))
    ):
        reasons.append("implicit_macro_f1_degraded")
    structure_rate = float(
        (metrics.get("bridge_structure") or {}).get("valid_rate", 0.0)
    )
    if structure_rate < float(selection_cfg.get("min_bridge_structure_valid_rate", 0.90)):
        reasons.append("bridge_structure_valid_rate_below_minimum")
    return {
        "metric": "weighted_full_implicit_macro_f1",
        "score": float(score),
        "eligible": not reasons,
        "gate_failures": reasons,
        "components": {
            "full_macro_f1": float(full["macro_f1"]),
            "implicit_macro_f1": float(implicit["macro_f1"]),
            "full_weight": full_weight / weight_sum,
            "implicit_weight": implicit_weight / weight_sum,
            "negative_recall": negative_recall,
            "bridge_structure_valid_rate": structure_rate,
            "best_implicit_macro_f1_before_epoch": (
                best_implicit_macro_f1 if best_implicit_macro_f1 >= 0 else None
            ),
        },
    }


def generated_attention_mask(ids: torch.Tensor, eos_id: int, pad_id: int) -> torch.Tensor:
    positions = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
    eos_positions = torch.where(
        ids.eq(eos_id), positions, torch.full_like(positions, ids.size(1))
    )
    first_eos = eos_positions.min(dim=1).values.unsqueeze(1)
    return positions.le(first_eos).logical_and(ids.ne(pad_id)).long()


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def resolve_amp_dtype(training_cfg: dict[str, Any], device: torch.device) -> torch.dtype | None:
    requested = str(
        training_cfg.get(
            "mixed_precision", "fp16" if training_cfg.get("fp16", False) else "fp32"
        )
    ).lower()
    if requested == "auto":
        if device.type != "cuda":
            return None
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if requested in {"none", "fp32", "float32"}:
        return None
    if requested in {"bf16", "bfloat16"}:
        if device.type not in {"cuda", "cpu"}:
            raise ValueError(f"bf16 autocast is not supported on device type {device.type!r}")
        return torch.bfloat16
    if requested in {"fp16", "float16"}:
        return torch.float16 if device.type == "cuda" else None
    raise ValueError(
        f"unsupported training.mixed_precision={requested!r}; "
        "expected auto, bf16, fp16, or fp32"
    )


class StageTrainer:
    def __init__(
        self,
        cfg: dict[str, Any],
        model,
        tokenizer,
        train_dataset,
        dev_dataset,
        collator,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.collator = collator
        self.device = device
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.is_main = self.rank == 0
        self.out_dir = Path(cfg["output_dir"])
        if self.is_main:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self._barrier()
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = model.eos_id
        self.best_macro_f1 = -1.0
        self.best_implicit_macro_f1 = -1.0
        self.best_bridge_score = -math.inf
        self.best_bridge_tiebreak = -math.inf
        self.amp_dtype = resolve_amp_dtype(cfg["training"], device)
        self.sentiment_class_weight, self.tag_pos_weight = self._class_weights()

        feedback = cfg["training"]["stage3_bridge"].get("ai_feedback") or {}
        if feedback.get("enabled", False) and self.distributed:
            raise ValueError(
                "Stage-3 Bailian Judge + DPO v1 currently requires a single training "
                "process. Set NPROC_PER_NODE=1; multi-GPU Judge synchronization is not "
                "implemented."
            )

        save_best_by = str(
            cfg["training"].get(
                "save_best_by", "gated_full_implicit_macro_f1"
            )
        )
        if save_best_by != "gated_full_implicit_macro_f1":
            raise ValueError(
                f"unsupported training.save_best_by={save_best_by!r}; only "
                "'gated_full_implicit_macro_f1' is implemented"
            )

    def _barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def _class_weights(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        tcfg = self.cfg["training"]
        if not bool(tcfg.get("use_class_balancing", True)):
            return None, None

        sentiment_counts = torch.zeros(3, dtype=torch.float32)
        tag_positive = torch.zeros(len(TAG_NAMES), dtype=torch.float32)
        for record in self.train_dataset.records:
            sentiment_counts[record.sentiment_id] += 1
            for i, name in enumerate(TAG_NAMES):
                tag_positive[i] += float(record.reasoning_tags[name])

        sentiment_weight = effective_number_class_weights(
            sentiment_counts,
            beta=float(tcfg.get("class_balance_beta", 0.999)),
        )

        total = float(len(self.train_dataset.records))
        max_pos_weight = float(tcfg.get("max_tag_pos_weight", 20.0))
        tag_weight = torch.ones_like(tag_positive)
        has_positive = tag_positive.gt(0)
        tag_weight[has_positive] = (
            (total - tag_positive[has_positive]) / tag_positive[has_positive]
        ).clamp(min=0.1, max=max_pos_weight)
        return sentiment_weight.to(self.device), tag_weight.to(self.device)

    def _loader(self, dataset, train: bool, require_bridge: bool = False) -> DataLoader:
        selected = dataset
        if require_bridge:
            indices = [
                i for i, record in enumerate(dataset.records)
                if record.reasoning_bridge is not None
            ]
            if not indices:
                raise ValueError("the selected split contains no reference reasoning bridges")
            selected = Subset(dataset, indices)
        if len(selected) == 0:
            raise ValueError("the selected dataset split is empty")
        sampler = None
        if train and self.distributed:
            sampler = DistributedSampler(
                selected,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=int(self.cfg["seed"]),
            )
        workers = int(self.cfg["training"]["num_workers"])
        return DataLoader(
            selected,
            batch_size=self.cfg["training"]["batch_size" if train else "eval_batch_size"],
            shuffle=train and sampler is None,
            sampler=sampler,
            num_workers=workers,
            persistent_workers=workers > 0 and bool(
                self.cfg["training"].get("persistent_workers", True)
            ),
            pin_memory=self.device.type == "cuda",
            collate_fn=self.collator,
        )

    def _set_trainable(self, stage: str, scfg: dict[str, Any]) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        def enable(module) -> None:
            for parameter in module.parameters():
                parameter.requires_grad = True

        def enable_t5_lora() -> None:
            enabled = 0
            for name, parameter in self.model.text_encoder.backbone.named_parameters():
                if is_lora_parameter(name):
                    parameter.requires_grad = True
                    enabled += parameter.numel()
            if bool(self.cfg["model"].get("lora", {}).get("enabled", False)) and not enabled:
                raise RuntimeError("T5 LoRA is enabled but no LoRA parameters were found")

        def enable_bridge_classifier() -> None:
            enable(self.model.bridge_classifier)
            # This large token table is copied from frozen T5 embeddings. The
            # Bridge encoder and classifier learn the task while the table stays
            # fixed, avoiding a high-capacity memorization path on ~3k samples.
            self.model.bridge_classifier.token_embedding.weight.requires_grad = False

        frontend_modules = [
            self.model.text_proj,
            self.model.vision_proj,
            self.model.target_proj,
            self.model.text_conditioner,
            self.model.visual_conditioner,
            self.model.fusion,
            self.model.text_selector,
            self.model.visual_selector,
            self.model.tag_head,
        ]
        reasoning_modules = [
            self.model.reasoner,
            self.model.path_interaction,
            self.model.bridge_generator,
        ]

        if stage == "stage1_aux":
            for module in frontend_modules:
                enable(module)
        elif stage == "stage2_reasoning_warmup":
            enable_t5_lora()
            for module in [*frontend_modules, *reasoning_modules]:
                enable(module)
        elif stage == "stage3_bridge":
            enable_t5_lora()
            for module in [*frontend_modules, *reasoning_modules]:
                enable(module)
        elif stage == "stage4_classifier":
            enable_bridge_classifier()
        elif stage == "stage5_joint":
            enable_t5_lora()
            for module in [*frontend_modules, *reasoning_modules]:
                enable(module)
            enable_bridge_classifier()
        else:
            raise ValueError(stage)

        self.model.text_encoder.set_lora_input_gradients(
            stage in {"stage2_reasoning_warmup", "stage3_bridge", "stage5_joint"}
        )
        self._assert_parameter_efficient_contract(stage)

    def _set_dpo_trainable(self) -> None:
        """Restrict the small Judge preference set to LoRA + Bridge adapter."""
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for name, parameter in self.model.text_encoder.backbone.named_parameters():
            if is_lora_parameter(name):
                parameter.requires_grad = True
        for parameter in self.model.bridge_generator.parameters():
            parameter.requires_grad = True
        self.model.text_encoder.set_lora_input_gradients(True)
        self._assert_parameter_efficient_contract("stage3_dpo")

    def _assert_parameter_efficient_contract(self, stage: str) -> None:
        text_base = [
            name
            for name, parameter in self.model.text_encoder.backbone.named_parameters()
            if parameter.requires_grad and not is_lora_parameter(name)
        ]
        vision = [
            name
            for name, parameter in self.model.vision_encoder.backbone.named_parameters()
            if parameter.requires_grad
        ]
        if text_base:
            raise RuntimeError(
                f"{stage}: frozen T5 base parameters became trainable: {text_base[:3]}"
            )
        if vision:
            raise RuntimeError(
                f"{stage}: frozen CLIP parameters became trainable: {vision[:3]}"
            )
        if self.model.bridge_classifier.token_embedding.weight.requires_grad:
            raise RuntimeError(
                f"{stage}: frozen Bridge token embeddings became trainable"
            )

    def _parameter_summary(self) -> dict[str, int | float]:
        total = 0
        trainable = 0
        lora = 0
        task = 0
        text_base = 0
        vision = 0
        bridge_token_embeddings = 0
        for name, parameter in self.model.named_parameters():
            count = parameter.numel()
            total += count
            if not parameter.requires_grad:
                continue
            trainable += count
            if name.startswith("text_encoder.backbone."):
                if is_lora_parameter(name):
                    lora += count
                else:
                    text_base += count
            elif name.startswith("vision_encoder.backbone."):
                vision += count
            elif name == "bridge_classifier.token_embedding.weight":
                bridge_token_embeddings += count
            else:
                task += count
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "trainable_ratio": float(trainable / max(1, total)),
            "trainable_lora_parameters": int(lora),
            "trainable_task_parameters": int(task),
            "trainable_t5_base_parameters": int(text_base),
            "trainable_clip_parameters": int(vision),
            "trainable_bridge_token_embedding_parameters": int(
                bridge_token_embeddings
            ),
        }

    def _parameter_groups(self, scfg: dict[str, Any]) -> list[dict[str, Any]]:
        base_lr = float(scfg["lr"])
        weight_decay = float(self.cfg["training"]["weight_decay"])
        groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
        seen: set[int] = set()
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            is_lora = is_lora_parameter(name)
            lora_scale = float(scfg.get("lora_lr_scale", 1.0))
            lr = base_lr * lora_scale if is_lora else base_lr
            lower = name.lower()
            no_decay = parameter.ndim <= 1 or lower.endswith(".bias") or "norm" in lower
            decay = 0.0 if no_decay else weight_decay
            groups.setdefault((lr, decay), []).append(parameter)
        if not groups:
            raise RuntimeError("stage has no trainable parameters")
        return [
            {"params": parameters, "lr": lr, "weight_decay": decay}
            for (lr, decay), parameters in groups.items()
        ]

    def _optimizer(self, scfg: dict[str, Any]):
        groups = self._parameter_groups(scfg)
        name = str(self.cfg["training"].get("optimizer", "adafactor")).lower()
        if name == "adafactor":
            optimizer_cls = getattr(torch.optim, "Adafactor", None)
            if optimizer_cls is None:
                raise RuntimeError("optimizer=adafactor requires PyTorch 2.5 or newer")
            return optimizer_cls(groups, lr=float(scfg["lr"]))
        if name == "adamw":
            return AdamW(groups, lr=float(scfg["lr"]))
        raise ValueError(f"unsupported training.optimizer={name!r}")

    def _scheduler(self, optimizer, total_steps: int) -> LambdaLR:
        warmup_ratio = float(self.cfg["training"].get("warmup_ratio", 0.1))
        warmup_steps = min(total_steps, int(round(total_steps * warmup_ratio)))

        def multiplier(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            remaining = max(1, total_steps - warmup_steps)
            return max(0.0, float(total_steps - step) / float(remaining))

        return LambdaLR(optimizer, multiplier)

    def _losses(self, batch, outputs, weights):
        tcfg = self.cfg["training"]
        losses: dict[str, torch.Tensor] = {}
        if weights.get("selector_regularization", 0) > 0:
            losses["selector_regularization"] = selector_regularization_loss(
                outputs["text_selector_weights"],
                batch["text_token_mask"],
                outputs["visual_selector_weights"],
                text_min_ratio=float(tcfg.get("selector_text_min_ratio", 0.05)),
                text_max_ratio=float(tcfg.get("selector_text_max_ratio", 0.50)),
                visual_min_normalized_entropy=float(
                    tcfg.get("selector_visual_min_normalized_entropy", 0.20)
                ),
                visual_max_normalized_entropy=float(
                    tcfg.get("selector_visual_max_normalized_entropy", 0.90)
                ),
            )
        if weights.get("reasoning_tags", 0) > 0:
            losses["reasoning_tags"] = reasoning_tag_loss(
                outputs["tag_logits"],
                batch["reasoning_tag_labels"],
                pos_weight=self.tag_pos_weight,
            )
        if weights.get("bridge", 0) > 0:
            losses["bridge"] = bridge_generation_loss(
                outputs["bridge_logits"],
                batch["bridge_input_ids"],
                batch["has_bridge"],
                self.pad_id,
            )
        if weights.get("sentiment", 0) > 0:
            losses["sentiment"] = sentiment_loss(
                outputs["sentiment_logits"],
                batch["sentiment_labels"],
                class_weight=self.sentiment_class_weight,
            )
        if not losses:
            raise ValueError("each enabled stage must configure at least one non-zero loss")
        total = sum(float(weights[name]) * loss for name, loss in losses.items())
        return total, losses

    def _reduce_losses(self, running: dict[str, float], batches: int) -> dict[str, float]:
        if not self.distributed:
            return {name: value / max(1, batches) for name, value in running.items()}
        names = sorted(running)
        values = torch.tensor(
            [running[name] for name in names] + [float(batches)],
            device=self.device,
            dtype=torch.float64,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        denominator = max(1.0, float(values[-1].item()))
        return {name: float(values[i].item()) / denominator for i, name in enumerate(names)}

    @staticmethod
    def _trim_generated_ids(
        ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[list[int]]:
        return [
            row[row_mask.bool()].detach().cpu().tolist()
            for row, row_mask in zip(ids, mask)
        ]

    @torch.no_grad()
    def _reference_candidate_logps(
        self,
        core,
        ids: torch.Tensor,
        mask: torch.Tensor,
        *,
        length_normalize: bool,
    ) -> torch.Tensor:
        expanded_core = SimpleNamespace(
            bridge_memory=core.bridge_memory.expand(ids.size(0), -1, -1)
        )
        with autocast_context(self.device, self.amp_dtype):
            logits = self.model.bridge_logits(expanded_core, ids, mask)
        return sequence_log_probs(
            logits,
            ids,
            mask,
            self.pad_id,
            length_normalize=length_normalize,
        ).float()

    def _collect_stage3_preferences(
        self,
        feedback_cfg: dict[str, Any],
        *,
        epoch: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Generate candidates and obtain cached, leakage-free Judge preferences."""
        cache_path = self.out_dir / str(
            feedback_cfg.get("cache_filename", "stage3_bailian_judge_cache.jsonl")
        )
        # Candidate sampling must be reproducible across an interrupted retry,
        # independently of how many random draws the teacher-forced epoch used.
        set_seed(int(self.cfg["seed"]) + 300_007 * (epoch + 1))
        judge = PairwiseBailianJudge(feedback_cfg, cache_path)
        eligible = [
            position
            for position, record in enumerate(self.train_dataset.records)
            if record.reasoning_bridge is not None
        ]
        sample_size = min(int(feedback_cfg["sample_size_per_epoch"]), len(eligible))
        rng = random.Random(int(self.cfg["seed"]) + 100_003 * (epoch + 1))
        positions = rng.sample(eligible, sample_size)
        candidate_count = int(feedback_cfg["candidate_count"])
        length_normalize = bool(
            (feedback_cfg.get("dpo") or {}).get("length_normalize_logps", True)
        )
        max_length = int(self.cfg["evaluation"]["max_generation_length"])
        entries: list[dict[str, Any]] = []
        samples_with_pairs = 0
        reviewed_pairs = 0
        tied_pairs = 0
        duplicate_or_invalid = 0
        quality_rejected_pairs = 0
        resampled_records = 0
        resampling_attempts = 0
        exhausted_records = 0
        generation_attempts = 0
        quality_gate = feedback_cfg.get("quality_gate") or {}
        max_generation_attempts = max(
            1, int(quality_gate.get("max_generation_attempts", 3))
        )
        previous_mode = self.model.training
        self.model.eval()
        progress = tqdm(
            positions,
            desc=f"stage3-ai-feedback {epoch + 1}",
            leave=False,
            disable=not self.is_main,
        )
        try:
            for position in progress:
                item = self.train_dataset[position]
                raw_batch = self.collator([item])
                batch = move_batch(raw_batch, self.device)
                with torch.no_grad(), autocast_context(self.device, self.amp_dtype):
                    core = self.model.encode_and_reason(batch, routing_gold_mix=0.0)
                    predicted_cross_modal = bool(
                        core.tag_probs[0, 2].detach().float().cpu().item() >= 0.5
                    )
                accepted_for_record = False
                for attempt in range(max_generation_attempts):
                    generation_attempts += 1
                    if attempt > 0:
                        resampling_attempts += 1
                        if attempt == 1:
                            resampled_records += 1
                    with torch.no_grad(), autocast_context(
                        self.device, self.amp_dtype
                    ):
                        generated = self.model.generate_bridge(
                            core,
                            max_length,
                            do_sample=True,
                            temperature=float(feedback_cfg["sampling_temperature"]),
                            top_p=float(feedback_cfg["top_p"]),
                            num_return_sequences=candidate_count,
                        )
                        generated_mask = generated_attention_mask(
                            generated, self.eos_id, self.pad_id
                        )
                        reference_logps = self._reference_candidate_logps(
                            core,
                            generated,
                            generated_mask,
                            length_normalize=length_normalize,
                        )

                    texts = self.tokenizer.batch_decode(
                        generated.detach().cpu(), skip_special_tokens=False
                    )
                    trimmed_ids = self._trim_generated_ids(
                        generated, generated_mask
                    )
                    candidates: list[dict[str, Any]] = []
                    seen: set[tuple[int, ...]] = set()
                    for ids_row, text, ref_logp in zip(
                        trimmed_ids, texts, reference_logps.detach().cpu().tolist()
                    ):
                        identity = tuple(ids_row)
                        parsed = parse_bridge_text(text)
                        if identity in seen or not parsed.structure_valid:
                            duplicate_or_invalid += 1
                            continue
                        seen.add(identity)
                        candidates.append(
                            {
                                "ids": ids_row,
                                "text": text,
                                "reference_logp": float(ref_logp),
                            }
                        )
                    if len(candidates) < 2:
                        continue

                    incumbent = 0
                    for challenger in range(1, len(candidates)):
                        material = (
                            f"{self.cfg.get('_run_id')}:{epoch + 1}:"
                            f"{item['index']}:{attempt}:{incumbent}:{challenger}"
                        )
                        decision = judge.compare(
                            text=str(item["restored_text"]),
                            target=str(item["target"]),
                            image=item["image"],
                            candidate_a=candidates[incumbent]["text"],
                            candidate_b=candidates[challenger]["text"],
                            cross_modal=predicted_cross_modal,
                            audit_material=material,
                        )
                        if decision.review is not None:
                            reviewed_pairs += 1
                        if decision.winner is None:
                            tied_pairs += 1
                            continue
                        chosen_index = (
                            incumbent if decision.winner == "A" else challenger
                        )
                        rejected_index = (
                            challenger if decision.winner == "A" else incumbent
                        )
                        if not passes_quality_gate(
                            decision.selected_scores, quality_gate
                        ):
                            quality_rejected_pairs += 1
                            incumbent = chosen_index
                            continue
                        chosen = candidates[chosen_index]
                        rejected = candidates[rejected_index]
                        entries.append(
                            {
                                "record_index": int(item["index"]),
                                "chosen_ids": chosen["ids"],
                                "rejected_ids": rejected["ids"],
                                "chosen_ref_logp": chosen["reference_logp"],
                                "rejected_ref_logp": rejected["reference_logp"],
                                "chosen_bridge": chosen["text"],
                                "rejected_bridge": rejected["text"],
                                "generation_attempt": attempt + 1,
                                "quality_gate_passed": True,
                                "review_reasons": list(decision.review_reasons),
                                "decision_source": (
                                    "review_judge"
                                    if decision.review is not None
                                    else "primary_judge"
                                ),
                                "selected_scores": decision.selected_scores,
                                "primary_decision": decision.primary,
                                "primary_reversed_decision": decision.primary_reversed,
                                "review_decision": decision.review,
                            }
                        )
                        accepted_for_record = True
                        break
                    if accepted_for_record:
                        samples_with_pairs += 1
                        break
                if not accepted_for_record:
                    exhausted_records += 1
        finally:
            self.model.train(previous_mode)

        summary = {
            "sampled_records": sample_size,
            "records_with_preference_pairs": samples_with_pairs,
            "preference_pairs": len(entries),
            "reviewed_pairs": reviewed_pairs,
            "tied_pairs": tied_pairs,
            "duplicate_or_invalid_candidates": duplicate_or_invalid,
            "quality_rejected_pairs": quality_rejected_pairs,
            "resampled_records": resampled_records,
            "resampling_attempts": resampling_attempts,
            "exhausted_records": exhausted_records,
            "generation_attempts": generation_attempts,
            "quality_gate": quality_gate,
            "judge_usage": judge.usage_summary(),
            "cache_path": str(cache_path),
            "judge_inputs": {
                "uses_original_text": True,
                "uses_image": True,
                "uses_current_target": True,
                "uses_gold_sentiment": False,
                "uses_gold_reasoning_tags": False,
                "uses_reference_bridge": False,
                "cross_modal_review_uses_predicted_tag": True,
            },
        }
        write_json(
            self.out_dir / f"stage3_ai_feedback_epoch_{epoch + 1}_preferences.json",
            {"summary": summary, "preferences": entries},
        )
        if not entries:
            raise RuntimeError(
                "Stage-3 AI feedback produced no decisive Bridge preference pairs. "
                "Increase candidate_count/sample_size or inspect the Judge cache."
            )
        return entries, summary

    def _train_stage3_dpo(
        self,
        scfg: dict[str, Any],
        feedback_cfg: dict[str, Any],
        entries: list[dict[str, Any]],
        *,
        epoch: int,
    ) -> dict[str, Any]:
        dpo_cfg = feedback_cfg.get("dpo") or {}
        if not dpo_cfg.get("enabled", True):
            return {"enabled": False, "preference_pairs": len(entries)}
        self._set_dpo_trainable()
        dpo_parameter_summary = self._parameter_summary()
        dataset = BridgePreferenceDataset(self.train_dataset, entries)
        collator = BridgePreferenceCollator(self.collator, self.pad_id)
        generator = torch.Generator()
        generator.manual_seed(int(self.cfg["seed"]) + 200_003 * (epoch + 1))
        loader = DataLoader(
            dataset,
            batch_size=int(dpo_cfg["batch_size"]),
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            collate_fn=collator,
        )
        dpo_scfg = dict(scfg)
        dpo_scfg["lr"] = float(dpo_cfg["lr"])
        optimizer = self._optimizer(dpo_scfg)
        use_scaler = self.amp_dtype == torch.float16 and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        grad_accum = int(dpo_cfg["grad_accum_steps"])
        dpo_epochs = int(dpo_cfg["epochs"])
        beta = float(dpo_cfg["beta"])
        reference_ce_weight = float(dpo_cfg.get("reference_ce_weight", 0.0))
        length_normalize = bool(dpo_cfg.get("length_normalize_logps", True))
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        running = {"dpo": 0.0, "reference_ce": 0.0, "total": 0.0}
        batches = 0
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        for dpo_epoch in range(dpo_epochs):
            progress = tqdm(
                loader,
                desc=f"stage3-dpo {epoch + 1}.{dpo_epoch + 1}",
                leave=False,
                disable=not self.is_main,
            )
            for step, raw_batch in enumerate(progress, start=1):
                batch = move_batch(raw_batch, self.device)
                group_start = ((step - 1) // grad_accum) * grad_accum + 1
                group_end = min(group_start + grad_accum - 1, len(loader))
                group_size = group_end - group_start + 1
                with autocast_context(self.device, self.amp_dtype):
                    core = self.model.encode_and_reason(batch, routing_gold_mix=0.0)
                    chosen_logits = self.model.bridge_logits(
                        core, batch["chosen_ids"], batch["chosen_mask"]
                    )
                    rejected_logits = self.model.bridge_logits(
                        core, batch["rejected_ids"], batch["rejected_mask"]
                    )
                    chosen_logp = sequence_log_probs(
                        chosen_logits,
                        batch["chosen_ids"],
                        batch["chosen_mask"],
                        self.pad_id,
                        length_normalize=length_normalize,
                    )
                    rejected_logp = sequence_log_probs(
                        rejected_logits,
                        batch["rejected_ids"],
                        batch["rejected_mask"],
                        self.pad_id,
                        length_normalize=length_normalize,
                    )
                    dpo_loss = dpo_preference_loss(
                        chosen_logp,
                        rejected_logp,
                        batch["chosen_ref_logp"],
                        batch["rejected_ref_logp"],
                        beta=beta,
                    )
                    reference_logits = self.model.bridge_logits(
                        core,
                        batch["bridge_input_ids"],
                        batch["bridge_attention_mask"],
                    )
                    reference_ce = bridge_generation_loss(
                        reference_logits,
                        batch["bridge_input_ids"],
                        batch["has_bridge"],
                        self.pad_id,
                    )
                    total = dpo_loss + reference_ce_weight * reference_ce
                scaler.scale(total / group_size).backward()
                if step == group_end:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        trainable, float(self.cfg["training"]["max_grad_norm"])
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                running["dpo"] += float(dpo_loss.detach().cpu())
                running["reference_ce"] += float(reference_ce.detach().cpu())
                running["total"] += float(total.detach().cpu())
                batches += 1
                progress.set_postfix(
                    dpo=f"{running['dpo'] / batches:.3f}",
                    ref_ce=f"{running['reference_ce'] / batches:.3f}",
                )
        summary = {
            "enabled": True,
            "preference_pairs": len(entries),
            "epochs": dpo_epochs,
            "beta": beta,
            "reference_ce_weight": reference_ce_weight,
            "length_normalize_logps": length_normalize,
            "mean_losses": {
                name: value / max(1, batches) for name, value in running.items()
            },
            "parameter_summary": dpo_parameter_summary,
        }
        self._set_trainable("stage3_bridge", scfg)
        return summary

    def _run_stage3_ai_feedback(
        self,
        scfg: dict[str, Any],
        *,
        epoch: int,
    ) -> dict[str, Any] | None:
        feedback_cfg = scfg.get("ai_feedback") or {}
        if not feedback_cfg.get("enabled", False):
            return None
        entries, collection = self._collect_stage3_preferences(
            feedback_cfg, epoch=epoch
        )
        dpo = self._train_stage3_dpo(
            scfg, feedback_cfg, entries, epoch=epoch
        )
        return {"collection": collection, "dpo": dpo}

    def train_all(self, resume_path: str | Path | None = None) -> None:
        resume_meta: dict[str, Any] | None = None
        resume_stage_index = -1
        if resume_path is not None:
            resume_path = Path(resume_path).resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
            resume_meta = load_checkpoint(resume_path, self.model, map_location=self.device)
            resume_stage = resume_meta.get("stage")
            if resume_stage not in STAGE_ORDER:
                raise ValueError("resume checkpoint metadata is missing a valid stage")
            resume_stage_index = STAGE_ORDER.index(resume_stage)
            self.best_macro_f1 = float(resume_meta.get("best_macro_f1", -1.0))
            self.best_implicit_macro_f1 = float(
                resume_meta.get("best_implicit_macro_f1", -1.0)
            )
            self.best_bridge_score = restore_optional_metric(
                resume_meta.get("best_bridge_score")
            )
            self.best_bridge_tiebreak = restore_optional_metric(
                resume_meta.get("best_bridge_tiebreak")
            )
            saved_world_size = resume_meta.get("world_size")
            if (
                not bool(resume_meta.get("stage_complete", False))
                and saved_world_size is not None
                and int(saved_world_size) != self.world_size
            ):
                raise ValueError(
                    "cannot resume an incomplete stage with a different DDP world size; "
                    "optimizer steps per epoch would change"
                )

        for stage_index, stage in enumerate(STAGE_ORDER):
            scfg = self.cfg["training"][stage]
            if not scfg.get("enabled", True):
                if stage_index == resume_stage_index:
                    raise ValueError(f"cannot resume disabled stage {stage!r}")
                continue
            if stage_index < resume_stage_index:
                continue
            if (
                stage_index == resume_stage_index
                and resume_meta is not None
                and bool(resume_meta.get("stage_complete", False))
            ):
                continue

            start_epoch = 0
            stage_resume_path = None
            if stage_index == resume_stage_index and resume_meta is not None:
                start_epoch = int(resume_meta.get("epoch", 0))
                stage_resume_path = resume_path
            self.train_stage(
                stage,
                scfg,
                start_epoch=start_epoch,
                resume_path=stage_resume_path,
            )
            resume_meta = None

    def train_stage(
        self,
        stage: str,
        scfg: dict[str, Any],
        *,
        start_epoch: int = 0,
        resume_path: str | Path | None = None,
    ) -> None:
        self._set_trainable(stage, scfg)
        if stage == "stage4_classifier":
            # Reconstruct the frozen Bridge token table from the pinned T5
            # base before classifier training. T5/LoRA training does not alter
            # this table, but refreshing makes the checkpoint contract explicit.
            self.model.bridge_classifier.initialize_token_embeddings(
                self.model.text_encoder.backbone.shared.weight
            )
        parameter_summary = self._parameter_summary()
        if self.is_main:
            print(
                f"[{stage}] trainable={parameter_summary['trainable_parameters']:,} "
                f"/ total={parameter_summary['total_parameters']:,} "
                f"({100.0 * parameter_summary['trainable_ratio']:.3f}%), "
                f"LoRA={parameter_summary['trainable_lora_parameters']:,}, "
                f"T5-base={parameter_summary['trainable_t5_base_parameters']:,}, "
                f"CLIP={parameter_summary['trainable_clip_parameters']:,}"
            )
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = self._optimizer(scfg)
        require_bridge = stage in {
            "stage2_reasoning_warmup",
            "stage3_bridge",
            "stage4_classifier",
        }
        loader = self._loader(self.train_dataset, train=True, require_bridge=require_bridge)
        epochs = int(scfg["epochs"])
        if start_epoch < 0 or start_epoch > epochs:
            raise ValueError(f"invalid resume epoch {start_epoch} for {stage} with {epochs} epochs")
        grad_accum = max(1, int(self.cfg["training"]["grad_accum_steps"]))
        optimizer_steps_per_epoch = math.ceil(len(loader) / grad_accum)
        scheduler = self._scheduler(optimizer, max(1, optimizer_steps_per_epoch * epochs))
        use_scaler = self.amp_dtype == torch.float16 and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        if resume_path is not None:
            loaded_meta = load_checkpoint(
                resume_path,
                self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                map_location=self.device,
            )
            saved_steps = loaded_meta.get("optimizer_steps_per_epoch")
            if saved_steps is not None and int(saved_steps) != optimizer_steps_per_epoch:
                raise ValueError(
                    "resume checkpoint was created with a different number of optimizer "
                    "steps per epoch; keep batch size, gradient accumulation, dataset, "
                    "and DDP world size unchanged"
                )

        if stage == "stage3_bridge" and start_epoch == 0:
            self.best_bridge_score = -math.inf
            self.best_bridge_tiebreak = -math.inf
        if stage == "stage5_joint" and start_epoch == 0:
            self.best_macro_f1 = -1.0
            self.best_implicit_macro_f1 = -1.0

        train_model = self.model
        if self.distributed:
            ddp_kwargs: dict[str, Any] = {
                "find_unused_parameters": True,
                "broadcast_buffers": False,
            }
            if self.device.type == "cuda":
                ddp_kwargs.update(
                    device_ids=[self.device.index], output_device=self.device.index
                )
            train_model = DistributedDataParallel(self.model, **ddp_kwargs)

        best_bridge_path = self.out_dir / "best_bridge.pt"
        best_joint_path = self.out_dir / "best_joint.pt"
        if (
            stage == "stage3_bridge"
            and start_epoch > 0
            and self.is_main
            and not best_bridge_path.is_file()
        ):
            # A copied latest.pt may not be accompanied by best_bridge.pt.
            # Fall back to selecting the first remaining epoch in that case.
            self.best_bridge_score = -math.inf
            self.best_bridge_tiebreak = -math.inf
        if (
            stage == "stage5_joint"
            and start_epoch > 0
            and self.is_main
            and not best_joint_path.is_file()
        ):
            self.best_macro_f1 = -1.0
        for epoch in range(start_epoch, epochs):
            # Checkpoints are written at epoch boundaries. Epoch-scoped random
            # seeds make a resumed next epoch reproducible.
            set_seed(int(self.cfg["seed"]) + epoch * self.world_size + self.rank)
            if isinstance(loader.sampler, DistributedSampler):
                loader.sampler.set_epoch(epoch)
            train_model.train()
            routing_mix = linear_schedule(
                scfg.get("routing_gold_mix_start", 0.0),
                scfg.get("routing_gold_mix_end", 0.0),
                epoch,
                epochs,
            )
            generated_ratio = linear_schedule(
                scfg.get("generated_bridge_ratio_start", 0.0),
                scfg.get("generated_bridge_ratio_end", 0.0),
                epoch,
                epochs,
            )
            if stage == "stage5_joint" and epoch == epochs - 1:
                generated_ratio = 1.0

            optimizer.zero_grad(set_to_none=True)
            running: dict[str, float] = {}
            progress = tqdm(
                loader,
                desc=f"{stage} {epoch + 1}/{epochs}",
                disable=not self.is_main,
            )
            for step, raw_batch in enumerate(progress, start=1):
                batch = move_batch(raw_batch, self.device)
                group_start = ((step - 1) // grad_accum) * grad_accum + 1
                group_end = min(group_start + grad_accum - 1, len(loader))
                group_size = group_end - group_start + 1
                should_step = step == group_end
                sync_context = nullcontext()
                if self.distributed and not should_step:
                    sync_context = train_model.no_sync()

                with sync_context:
                    with autocast_context(self.device, self.amp_dtype):
                        weights = scfg["loss_weights"]
                        outputs = train_model(
                            batch,
                            stage=stage,
                            routing_gold_mix=routing_mix,
                            compute_bridge=weights.get("bridge", 0) > 0,
                            compute_sentiment=weights.get("sentiment", 0) > 0,
                            generated_bridge_ratio=generated_ratio,
                            max_generation_length=int(
                                self.cfg["evaluation"]["max_generation_length"]
                            ),
                        )
                        total, losses = self._losses(batch, outputs, weights)
                        scaled_loss = total / group_size
                    scaler.scale(scaled_loss).backward()

                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        trainable, float(self.cfg["training"]["max_grad_norm"])
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                for name, loss in losses.items():
                    running[name] = running.get(name, 0.0) + float(loss.detach().cpu())
                if self.is_main:
                    progress.set_postfix(
                        {name: f"{value / step:.3f}" for name, value in running.items()}
                    )

            mean_losses = self._reduce_losses(running, len(loader))
            epoch_summary: dict[str, Any] = {
                "run_id": self.cfg.get("_run_id"),
                "stage": stage,
                "epoch": epoch + 1,
                "stage_complete": False,
                "routing_gold_mix": routing_mix,
                "generated_bridge_ratio": generated_ratio,
                "trainable_parameters": int(sum(p.numel() for p in trainable)),
                "parameter_summary": parameter_summary,
                "world_size": self.world_size,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "mean_losses": mean_losses,
                "optimizer": str(self.cfg["training"].get("optimizer", "adafactor")),
                "mixed_precision": str(self.amp_dtype).replace("torch.", "")
                if self.amp_dtype is not None
                else "fp32",
            }

            if stage == "stage3_bridge":
                ai_feedback_summary = self._run_stage3_ai_feedback(
                    scfg,
                    epoch=epoch,
                )
                if ai_feedback_summary is not None:
                    epoch_summary["ai_feedback"] = ai_feedback_summary
                if self.is_main:
                    bridge_metrics = self.evaluate_bridge_generation(self.dev_dataset)
                    feedback_cfg = scfg.get("ai_feedback") or {}
                    checkpoint_selection_cfg = (
                        scfg.get("checkpoint_selection") or {}
                    )
                    bridge_metrics["absolute_judge"] = (
                        self.evaluate_stage3_absolute_judge(
                            self.dev_dataset,
                            feedback_cfg,
                            checkpoint_selection_cfg,
                            epoch=epoch,
                        )
                    )
                    selection = stage3_checkpoint_decision(
                        bridge_metrics, checkpoint_selection_cfg
                    )
                    bridge_metrics["selection"] = selection
                    write_json(
                        self.out_dir
                        / f"stage3_dev_bridge_metrics_epoch_{epoch + 1}.json",
                        bridge_metrics,
                    )
                    selection_score = float(selection["score"])
                    selection_tiebreak = float(selection["rouge_l_f1_tiebreak"])
                    is_better = selection_score > self.best_bridge_score or (
                        math.isclose(selection_score, self.best_bridge_score)
                        and selection_tiebreak > self.best_bridge_tiebreak
                    )
                    if selection["eligible"] and is_better:
                        self.best_bridge_score = selection_score
                        self.best_bridge_tiebreak = selection_tiebreak
                        save_checkpoint(
                            best_bridge_path,
                            self.model,
                            meta={
                                "stage": stage,
                                "epoch": epoch + 1,
                                "stage_complete": False,
                                "selection_metric": selection["metric"],
                                "selection_score": selection_score,
                                "selection_tiebreak": selection_tiebreak,
                                "metrics": bridge_metrics,
                            },
                        )
                self._barrier()

            elif stage == "stage5_joint":
                if self.is_main:
                    metrics = self.evaluate(self.dev_dataset)
                    selection = stage5_checkpoint_decision(
                        metrics,
                        scfg.get("selection") or {},
                        self.best_implicit_macro_f1,
                    )
                    metrics["selection"] = selection
                    write_json(
                        self.out_dir / f"dev_metrics_epoch_{epoch + 1}.json", metrics
                    )
                    implicit_macro_f1 = float(metrics["implicit"]["macro_f1"])
                    selection_score = float(selection["score"])
                    if selection["eligible"]:
                        self.best_implicit_macro_f1 = max(
                            self.best_implicit_macro_f1, implicit_macro_f1
                        )
                    if selection["eligible"] and selection_score > self.best_macro_f1:
                        self.best_macro_f1 = selection_score
                        save_checkpoint(
                            best_joint_path,
                            self.model,
                            meta={
                                "stage": stage,
                                "epoch": epoch + 1,
                                "stage_complete": False,
                                "selection_metric": selection["metric"],
                                "selection_score": selection_score,
                                "best_implicit_macro_f1": self.best_implicit_macro_f1,
                                "metrics": metrics,
                            },
                        )
                    if epoch == epochs - 1:
                        save_checkpoint(
                            self.out_dir / "stage5_generated_only.pt",
                            self.model,
                            meta={
                                "stage": stage,
                                "epoch": epoch + 1,
                                "stage_complete": True,
                                "generated_bridge_ratio": 1.0,
                                "selection": selection,
                                "metrics": metrics,
                            },
                        )
                self._barrier()

            epoch_summary["best_macro_f1"] = self.best_macro_f1
            epoch_summary["best_implicit_macro_f1"] = self.best_implicit_macro_f1
            epoch_summary["best_bridge_score"] = (
                self.best_bridge_score
                if math.isfinite(self.best_bridge_score)
                else None
            )
            epoch_summary["best_bridge_tiebreak"] = (
                self.best_bridge_tiebreak
                if math.isfinite(self.best_bridge_tiebreak)
                else None
            )
            if self.is_main:
                write_json(
                    self.out_dir / f"{stage}_epoch_{epoch + 1}_train.json",
                    epoch_summary,
                )
                save_checkpoint(
                    self.out_dir / "latest.pt",
                    self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    meta=epoch_summary,
                )
                if bool(self.cfg["training"].get("keep_stage_checkpoints", False)):
                    save_checkpoint(
                        self.out_dir / f"{stage}_last.pt",
                        self.model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        meta=epoch_summary,
                    )
            self._barrier()

        if stage == "stage3_bridge":
            if self.is_main and (
                not math.isfinite(self.best_bridge_score)
                or not best_bridge_path.is_file()
            ):
                raise RuntimeError(
                    "Stage 3 finished without producing a valid best Bridge checkpoint"
                )
            self._barrier()
            best_meta = load_checkpoint(
                best_bridge_path, self.model, map_location=self.device
            )
            if self.is_main:
                write_json(
                    self.out_dir / "stage3_best_bridge_selected.json",
                    {
                        "checkpoint": str(best_bridge_path),
                        "selection_metric": best_meta["selection_metric"],
                        "selection_score": best_meta["selection_score"],
                        "selection_tiebreak": best_meta.get("selection_tiebreak"),
                        "epoch": best_meta["epoch"],
                        "restored_before_stage4": True,
                    },
                )

        if stage == "stage5_joint":
            if self.is_main and (
                self.best_macro_f1 < 0 or not best_joint_path.is_file()
            ):
                raise RuntimeError(
                    "Stage 5 finished without any checkpoint passing the class, "
                    "implicit-subset, and Bridge-structure quality gates"
                )
            self._barrier()

        completion_meta = {
            "run_id": self.cfg.get("_run_id"),
            "stage": stage,
            "epoch": epochs,
            "stage_complete": True,
            "best_macro_f1": self.best_macro_f1,
            "best_implicit_macro_f1": self.best_implicit_macro_f1,
            "best_bridge_score": self.best_bridge_score
            if math.isfinite(self.best_bridge_score)
            else None,
            "best_bridge_tiebreak": self.best_bridge_tiebreak
            if math.isfinite(self.best_bridge_tiebreak)
            else None,
            "world_size": self.world_size,
        }
        if self.is_main:
            save_checkpoint(self.out_dir / "latest.pt", self.model, meta=completion_meta)
        self._barrier()

    @torch.no_grad()
    def evaluate_stage3_absolute_judge(
        self,
        dataset,
        feedback_cfg: dict[str, Any],
        selection_cfg: dict[str, Any],
        *,
        epoch: int,
    ) -> dict[str, Any]:
        """Score the same deterministic dev records every epoch without gold inputs."""
        sample_size = min(int(selection_cfg["sample_size"]), len(dataset.records))
        rng = random.Random(
            int(self.cfg["seed"]) + int(selection_cfg.get("sample_seed_offset", 53_021))
        )
        positions = rng.sample(range(len(dataset.records)), sample_size)
        cache_path = self.out_dir / str(
            selection_cfg.get(
                "cache_filename", "stage3_dev_absolute_judge_cache.jsonl"
            )
        )
        judge = AbsoluteBailianJudge(feedback_cfg, cache_path)
        rows: list[dict[str, Any]] = []
        review_count = 0
        previous_mode = self.model.training
        self.model.eval()
        try:
            for position in tqdm(
                positions,
                desc=f"stage3-absolute-judge {epoch + 1}",
                leave=False,
                disable=not self.is_main,
            ):
                item = dataset[position]
                batch = move_batch(self.collator([item]), self.device)
                with autocast_context(self.device, self.amp_dtype):
                    core = self.model.encode_and_reason(batch, routing_gold_mix=0.0)
                    ids = self.model.generate_bridge(
                        core, self.cfg["evaluation"]["max_generation_length"]
                    )
                text = self.tokenizer.batch_decode(
                    ids.detach().cpu(), skip_special_tokens=False
                )[0]
                parsed = parse_bridge_text(text)
                row: dict[str, Any] = {
                    "record_index": int(item["index"]),
                    "structure_valid": bool(parsed.structure_valid),
                    "structure_error": parsed.error,
                }
                if not parsed.structure_valid:
                    row.update(
                        {
                            "scores": {
                                "faithfulness": 0,
                                "reasoning_coherence": 0,
                                "target_consistency": 0,
                            },
                            "critical_error": True,
                            "judge_skipped": "invalid_bridge_structure",
                        }
                    )
                    rows.append(row)
                    continue
                predicted_cross_modal = bool(
                    core.tag_probs[0, 2].detach().float().cpu().item() >= 0.5
                )
                decision = judge.score(
                    text=str(item["restored_text"]),
                    target=str(item["target"]),
                    image=item["image"],
                    candidate=text,
                    cross_modal=predicted_cross_modal,
                    audit_material=(
                        f"{self.cfg.get('_run_id')}:stage3-checkpoint:"
                        f"{item['index']}"
                    ),
                )
                if decision.review is not None:
                    review_count += 1
                row.update(
                    {
                        "scores": decision.result["scores"],
                        "critical_error": bool(decision.result["critical_error"]),
                        "decision_source": (
                            "review_judge"
                            if decision.review is not None
                            else "primary_judge"
                        ),
                        "review_reasons": list(decision.review_reasons),
                    }
                )
                rows.append(row)
        finally:
            self.model.train(previous_mode)

        dimensions = ("faithfulness", "reasoning_coherence", "target_consistency")
        denominator = max(1, len(rows))
        dimension_means = {
            name: sum(float(row["scores"][name]) for row in rows) / denominator
            for name in dimensions
        }
        return {
            "sample_size": len(rows),
            "fixed_record_indices": [int(dataset.records[p].index) for p in positions],
            "dimension_means": dimension_means,
            "mean_score_normalized": sum(dimension_means.values()) / 15.0,
            "structure_valid_rate": sum(
                bool(row["structure_valid"]) for row in rows
            ) / denominator,
            "critical_error_rate": sum(
                bool(row["critical_error"]) for row in rows
            ) / denominator,
            "reviewed_records": review_count,
            "judge_usage": judge.usage_summary(),
            "records": rows,
            "judge_inputs": {
                "uses_original_text": True,
                "uses_image": True,
                "uses_current_target": True,
                "uses_gold_sentiment": False,
                "uses_gold_reasoning_tags": False,
                "uses_reference_bridge": False,
            },
        }

    @torch.no_grad()
    def evaluate_bridge_generation(self, dataset) -> dict[str, Any]:
        """Evaluate Stage 3 with gold-free routing and autoregressive decoding."""
        self.model.eval()
        loader = self._loader(dataset, train=False, require_bridge=False)
        parsed = []
        generated = []
        references = []
        for raw_batch in tqdm(loader, desc="stage3-bridge-eval", leave=False):
            batch = move_batch(raw_batch, self.device)
            with autocast_context(self.device, self.amp_dtype):
                core = self.model.encode_and_reason(
                    batch,
                    routing_gold_mix=0.0,
                )
                ids = self.model.generate_bridge(
                    core, self.cfg["evaluation"]["max_generation_length"]
                )
            texts = self.tokenizer.batch_decode(
                ids.detach().cpu(), skip_special_tokens=False
            )
            for text, reference in zip(texts, batch["reference_bridges"]):
                candidate = parse_bridge_text(text)
                parsed.append(candidate)
                generated.append(candidate.fields)
                references.append(reference)

        return {
            "structure": compute_structure_metrics(parsed),
            "reference": compute_bridge_reference_metrics(
                generated, references, compute_bertscore=False
            ),
            "inference_contract": {
                "autoregressive_generation": True,
                "routing_gold_mix": 0.0,
                "uses_gold_reasoning_tags": False,
                "uses_reference_bridge_as_decoder_input": False,
                "reference_bridge_used_only_as_evaluation_target": True,
            },
        }

    @torch.no_grad()
    def evaluate(self, dataset) -> dict[str, Any]:
        self.model.eval()
        loader = self._loader(dataset, train=False, require_bridge=False)
        y_true, y_pred, implicit = [], [], []
        parsed_bridges = []
        for raw_batch in tqdm(loader, desc="evaluate", leave=False):
            batch = move_batch(raw_batch, self.device)
            with autocast_context(self.device, self.amp_dtype):
                core = self.model.encode_and_reason(
                    batch,
                    routing_gold_mix=0.0,
                )
                ids = self.model.generate_bridge(
                    core, self.cfg["evaluation"]["max_generation_length"]
                )
                mask = generated_attention_mask(ids, self.eos_id, self.pad_id)
                logits, _ = self.model.classify_bridge(ids, mask)
            pred = logits.argmax(dim=-1)
            texts = self.tokenizer.batch_decode(
                ids.detach().cpu(), skip_special_tokens=False
            )
            parsed_bridges.extend(parse_bridge_text(text) for text in texts)
            y_true.extend(batch["sentiment_labels"].cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            implicit.extend(batch["is_implicit"].cpu().tolist())
        metrics = compute_metrics(y_true, y_pred, implicit)
        metrics["bridge_structure"] = compute_structure_metrics(parsed_bridges)
        return metrics
