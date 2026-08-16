from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
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
from tmis.training.losses import (
    bridge_generation_loss,
    reasoning_tag_loss,
    sentiment_loss,
    text_evidence_loss,
    visual_evidence_contrastive_loss,
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
    "implicit_reasoning_required",
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
        self.best_bridge_score = -math.inf
        self.amp_dtype = resolve_amp_dtype(cfg["training"], device)
        self.sentiment_class_weight, self.tag_pos_weight = self._class_weights()
        self.visual_negative_bank: torch.Tensor | None = None

        save_best_by = str(cfg["training"].get("save_best_by", "macro_f1"))
        if save_best_by != "macro_f1":
            raise ValueError(
                f"unsupported training.save_best_by={save_best_by!r}; only 'macro_f1' is implemented"
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

        present = sentiment_counts.gt(0)
        sentiment_weight = torch.zeros_like(sentiment_counts)
        sentiment_weight[present] = sentiment_counts.sum() / (
            present.sum().clamp_min(1) * sentiment_counts[present]
        )
        if present.any():
            sentiment_weight[present] /= sentiment_weight[present].mean()

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

        def enable_t5_encoder() -> None:
            enable(self.model.text_encoder.backbone.shared)
            enable(self.model.text_encoder.backbone.encoder)

        def enable_t5_decoder() -> None:
            enable(self.model.text_encoder.backbone.shared)
            enable(self.model.text_encoder.backbone.decoder)
            enable(self.model.text_encoder.backbone.lm_head)

        if stage == "stage1_aux":
            enable_t5_encoder()
            for module in [
                self.model.vision_encoder,
                self.model.text_proj,
                self.model.vision_proj,
                self.model.target_proj,
                self.model.aux_text_proj,
                self.model.text_conditioner,
                self.model.visual_conditioner,
                self.model.fusion,
                self.model.text_evidence_head,
                self.model.visual_evidence_head,
                self.model.tag_head,
            ]:
                enable(module)
        elif stage == "stage2_reasoning_warmup":
            enable_t5_decoder()
            for module in [
                self.model.tag_head,
                self.model.reasoner,
                self.model.path_interaction,
                self.model.relation,
                self.model.bridge_generator,
            ]:
                enable(module)
        elif stage == "stage3_bridge":
            enable_t5_decoder()
            for module in [
                self.model.text_evidence_head,
                self.model.visual_evidence_head,
                self.model.aux_text_proj,
                self.model.tag_head,
                self.model.reasoner,
                self.model.path_interaction,
                self.model.relation,
                self.model.bridge_generator,
            ]:
                enable(module)
        elif stage == "stage4_classifier":
            enable(self.model.bridge_classifier)
        elif stage == "stage5_joint":
            for parameter in self.model.parameters():
                parameter.requires_grad = True
        else:
            raise ValueError(stage)

        if not bool(scfg.get("train_text_backbone", True)):
            for parameter in self.model.text_encoder.backbone.parameters():
                parameter.requires_grad = False
        if not bool(scfg.get("train_vision_backbone", True)):
            for parameter in self.model.vision_encoder.backbone.parameters():
                parameter.requires_grad = False

    def _parameter_groups(self, scfg: dict[str, Any]) -> list[dict[str, Any]]:
        base_lr = float(scfg["lr"])
        backbone_scale = float(scfg.get("backbone_lr_scale", 0.1))
        weight_decay = float(self.cfg["training"]["weight_decay"])
        groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
        seen: set[int] = set()
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            is_backbone = name.startswith("text_encoder.backbone.") or name.startswith(
                "vision_encoder.backbone."
            )
            lr = base_lr * backbone_scale if is_backbone else base_lr
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
        if weights.get("text_evidence", 0) > 0:
            losses["text_evidence"] = text_evidence_loss(
                outputs["text_evidence_logits"],
                batch["text_evidence_labels"],
                pos_weight=float(tcfg.get("text_evidence_pos_weight", 3.0)),
                focal_gamma=float(tcfg.get("text_evidence_focal_gamma", 2.0)),
            )
        if weights.get("visual_evidence", 0) > 0:
            losses["visual_evidence"] = visual_evidence_contrastive_loss(
                outputs["visual_evidence_repr"],
                outputs["visual_evidence_text_repr"],
                batch["has_visual_evidence"],
                temperature=float(tcfg.get("visual_evidence_temperature", 0.07)),
                negative_bank=self.visual_negative_bank,
                presence_logits=outputs["visual_presence_logits"],
                presence_weight=float(tcfg.get("visual_presence_weight", 0.5)),
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

    @torch.no_grad()
    def _update_visual_negative_bank(self, outputs, batch) -> None:
        text_repr = outputs.get("visual_evidence_text_repr")
        if text_repr is None:
            return
        positive = text_repr[batch["has_visual_evidence"].bool()].detach()
        if positive.numel() == 0:
            return
        bank = positive if self.visual_negative_bank is None else torch.cat(
            [self.visual_negative_bank, positive], dim=0
        )
        capacity = int(self.cfg["training"].get("visual_evidence_queue_size", 256))
        self.visual_negative_bank = bank[-capacity:]

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
            saved_bridge_score = resume_meta.get("best_bridge_score")
            self.best_bridge_score = (
                float(saved_bridge_score)
                if saved_bridge_score is not None
                else -math.inf
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
            # Refresh from the Stage-3-selected T5 table so the custom marker
            # embeddings include Bridge-generation training, not only their
            # random values from tokenizer resizing at model construction.
            self.model.bridge_classifier.initialize_token_embeddings(
                self.model.text_encoder.backbone.shared.weight
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
        if stage == "stage5_joint" and start_epoch == 0:
            self.best_macro_f1 = -1.0

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
        if (
            stage == "stage3_bridge"
            and start_epoch > 0
            and self.is_main
            and not best_bridge_path.is_file()
        ):
            # A copied latest.pt may not be accompanied by best_bridge.pt.
            # Fall back to selecting the first remaining epoch in that case.
            self.best_bridge_score = -math.inf
        for epoch in range(start_epoch, epochs):
            # Checkpoints are written at epoch boundaries. Epoch-scoped random
            # seeds and queue state make a resumed next epoch reproducible.
            set_seed(int(self.cfg["seed"]) + epoch * self.world_size + self.rank)
            self.visual_negative_bank = None
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
                            compute_visual_evidence_target=weights.get("visual_evidence", 0) > 0,
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

                self._update_visual_negative_bank(outputs, batch)
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
                "world_size": self.world_size,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "mean_losses": mean_losses,
                "optimizer": str(self.cfg["training"].get("optimizer", "adafactor")),
                "mixed_precision": str(self.amp_dtype).replace("torch.", "")
                if self.amp_dtype is not None
                else "fp32",
            }

            if stage == "stage3_bridge":
                if self.is_main:
                    bridge_metrics = self.evaluate_bridge_generation(self.dev_dataset)
                    selection_metric = str(
                        scfg.get("selection_metric", "rouge_l_f1_full")
                    )
                    selection_score = bridge_selection_score(
                        bridge_metrics, selection_metric
                    )
                    bridge_metrics["selection"] = {
                        "metric": selection_metric,
                        "score": selection_score,
                    }
                    write_json(
                        self.out_dir
                        / f"stage3_dev_bridge_metrics_epoch_{epoch + 1}.json",
                        bridge_metrics,
                    )
                    if selection_score > self.best_bridge_score:
                        self.best_bridge_score = selection_score
                        save_checkpoint(
                            best_bridge_path,
                            self.model,
                            meta={
                                "stage": stage,
                                "epoch": epoch + 1,
                                "stage_complete": False,
                                "selection_metric": selection_metric,
                                "selection_score": selection_score,
                                "metrics": bridge_metrics,
                            },
                        )
                self._barrier()

            elif stage == "stage5_joint":
                if self.is_main:
                    metrics = self.evaluate(self.dev_dataset)
                    write_json(
                        self.out_dir / f"dev_metrics_epoch_{epoch + 1}.json", metrics
                    )
                    macro_f1 = float(metrics["full"]["macro_f1"])
                    if macro_f1 > self.best_macro_f1:
                        self.best_macro_f1 = macro_f1
                        save_checkpoint(
                            self.out_dir / "best_joint.pt",
                            self.model,
                            meta={
                                "stage": stage,
                                "epoch": epoch + 1,
                                "stage_complete": False,
                                "selection_metric": "macro_f1",
                                "selection_score": macro_f1,
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
                                "metrics": metrics,
                            },
                        )
                self._barrier()

            epoch_summary["best_macro_f1"] = self.best_macro_f1
            epoch_summary["best_bridge_score"] = (
                self.best_bridge_score
                if math.isfinite(self.best_bridge_score)
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
            if self.is_main and not best_bridge_path.is_file():
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
                        "epoch": best_meta["epoch"],
                        "restored_before_stage4": True,
                    },
                )

        completion_meta = {
            "run_id": self.cfg.get("_run_id"),
            "stage": stage,
            "epoch": epochs,
            "stage_complete": True,
            "best_macro_f1": self.best_macro_f1,
            "best_bridge_score": self.best_bridge_score
            if math.isfinite(self.best_bridge_score)
            else None,
            "world_size": self.world_size,
        }
        if self.is_main:
            save_checkpoint(self.out_dir / "latest.pt", self.model, meta=completion_meta)
        self._barrier()

    @torch.no_grad()
    def evaluate_bridge_generation(self, dataset) -> dict[str, Any]:
        """Evaluate Stage 3 with gold-free routing and autoregressive decoding."""
        self.model.eval()
        loader = self._loader(dataset, train=False, require_bridge=True)
        parsed = []
        generated = []
        references = []
        for raw_batch in tqdm(loader, desc="stage3-bridge-eval", leave=False):
            batch = move_batch(raw_batch, self.device)
            with autocast_context(self.device, self.amp_dtype):
                core = self.model.encode_and_reason(
                    batch,
                    routing_gold_mix=0.0,
                    compute_visual_evidence_target=False,
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
        for raw_batch in tqdm(loader, desc="evaluate", leave=False):
            batch = move_batch(raw_batch, self.device)
            with autocast_context(self.device, self.amp_dtype):
                core = self.model.encode_and_reason(
                    batch,
                    routing_gold_mix=0.0,
                    compute_visual_evidence_target=False,
                )
                ids = self.model.generate_bridge(
                    core, self.cfg["evaluation"]["max_generation_length"]
                )
                mask = generated_attention_mask(ids, self.eos_id, self.pad_id)
                logits, _ = self.model.classify_bridge(ids, mask)
            pred = logits.argmax(dim=-1)
            y_true.extend(batch["sentiment_labels"].cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            implicit.extend(batch["is_implicit"].cpu().tolist())
        return compute_metrics(y_true, y_pred, implicit)
