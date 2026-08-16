from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from tmis.constants import (
    BRIDGE_BOS_TOKEN,
    BRIDGE_EOS_TOKEN,
    GROUND_TOKEN,
    IMPLICATION_TOKEN,
    TRANSITION_TOKEN,
)
from tmis.models.bridge import (
    BridgeEncoderClassifier,
    ReasoningBridgeGenerator,
    shift_tokens_right,
)
from tmis.models.conditioning import MultimodalFusion, TargetConditioner
from tmis.models.encoders import TextTargetEncoder, VisionEncoder
from tmis.models.evidence import TextEvidenceHead, VisualEvidenceHead
from tmis.models.reasoning import (
    CrossModalRelationModule,
    CrossPathInteraction,
    MultiPathReasoner,
    ReasoningTagHead,
    SoftRouter,
)


@dataclass
class CoreOutput:
    h_ta: torch.Tensor
    h_va: torch.Tensor
    h_a: torch.Tensor
    h_f: torch.Tensor
    text_evidence_logits: torch.Tensor
    text_evidence_probs: torch.Tensor
    h_te: torch.Tensor
    visual_evidence_weights: torch.Tensor
    visual_presence_logits: torch.Tensor
    h_ve: torch.Tensor
    tag_logits: torch.Tensor
    tag_probs: torch.Tensor
    route_weights: torch.Tensor
    h_reasoning: torch.Tensor
    h_relation: torch.Tensor
    bridge_memory: torch.Tensor
    visual_evidence_text_repr: torch.Tensor | None


class EvidenceAwareMultiPathModel(nn.Module):
    def __init__(self, cfg: dict[str, Any], tokenizer: Any) -> None:
        super().__init__()
        m = cfg["model"]
        d = cfg["data"]
        self.cfg_model = m
        hidden = int(m["hidden_dim"])
        bridge_dim = int(m["bridge_encoder_dim"])
        heads = int(m["num_heads"])
        dropout = float(m["dropout"])

        self.text_encoder = TextTargetEncoder(
            m["text_backbone"],
            len(tokenizer),
            gradient_checkpointing=bool(m.get("text_gradient_checkpointing", False)),
            revision=m.get("text_backbone_revision"),
            local_files_only=bool(m.get("local_files_only", False)),
        )
        self.vision_encoder = VisionEncoder(
            m["vision_backbone"],
            gradient_checkpointing=bool(m.get("vision_gradient_checkpointing", False)),
            revision=m.get("vision_backbone_revision"),
            local_files_only=bool(m.get("local_files_only", False)),
        )
        self.text_proj = nn.Linear(self.text_encoder.hidden_size, hidden)
        self.vision_proj = nn.Linear(self.vision_encoder.hidden_size, hidden)
        self.target_proj = nn.Linear(self.text_encoder.hidden_size, hidden)
        self.aux_text_proj = nn.Linear(self.text_encoder.hidden_size, hidden)

        self.text_conditioner = TargetConditioner(hidden, dropout)
        self.visual_conditioner = TargetConditioner(hidden, dropout)
        self.fusion = MultimodalFusion(hidden, heads, dropout)
        self.text_evidence_head = TextEvidenceHead(hidden, dropout)
        self.visual_evidence_head = VisualEvidenceHead(hidden, dropout)
        self.tag_head = ReasoningTagHead(hidden, dropout)
        self.reasoner = MultiPathReasoner(hidden, dropout)
        self.router = SoftRouter(
            epsilon=m["router_epsilon"],
            alphas=(
                m["router_alpha_explicit"],
                m["router_alpha_implicit"],
                m["router_alpha_cross"],
            ),
        )
        self.path_interaction = CrossPathInteraction(
            hidden, heads, int(m["interaction_layers"]), dropout
        )
        self.relation = CrossModalRelationModule(hidden, dropout)

        self.bridge_generator = ReasoningBridgeGenerator(
            hidden_dim=hidden,
            t5_hidden_dim=self.text_encoder.hidden_size,
            pad_id=tokenizer.pad_token_id,
            max_length=int(d["max_bridge_length"]),
        )
        self.bridge_classifier = BridgeEncoderClassifier(
            vocab_size=len(tokenizer),
            pad_id=tokenizer.pad_token_id,
            bridge_dim=bridge_dim,
            num_heads=heads,
            layers=int(m["bridge_encoder_layers"]),
            dropout=dropout,
            max_length=int(d["max_bridge_length"]),
            ground_id=tokenizer.convert_tokens_to_ids(GROUND_TOKEN),
            transition_id=tokenizer.convert_tokens_to_ids(TRANSITION_TOKEN),
            implication_id=tokenizer.convert_tokens_to_ids(IMPLICATION_TOKEN),
            pretrained_token_embeddings=self.text_encoder.backbone.shared.weight,
        )

        self.bos_id = tokenizer.convert_tokens_to_ids(BRIDGE_BOS_TOKEN)
        self.eos_id = tokenizer.convert_tokens_to_ids(BRIDGE_EOS_TOKEN)
        self.ground_id = tokenizer.convert_tokens_to_ids(GROUND_TOKEN)

    def encode_and_reason(
        self,
        batch: dict[str, Any],
        routing_gold_mix: float = 0.0,
        compute_visual_evidence_target: bool = False,
    ) -> CoreOutput:
        h_t_raw, h_a_raw = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            target_input_ids=batch["target_input_ids"],
            target_attention_mask=batch["target_attention_mask"],
            target_token_mask=batch["target_token_mask"],
        )
        h_v_raw = self.vision_encoder(batch["pixel_values"])
        h_t = self.text_proj(h_t_raw)
        h_v = self.vision_proj(h_v_raw)
        h_a = self.target_proj(h_a_raw)

        text_mask = batch["text_token_mask"].bool()
        h_ta, _ = self.text_conditioner(h_t, h_a, text_mask)
        h_va, _ = self.visual_conditioner(h_v, h_a, None)
        _, _, h_f = self.fusion(h_ta, h_va, h_a, text_mask)

        te_logits, te_probs, h_te = self.text_evidence_head(h_ta, h_f, text_mask)
        h_ve, ve_weights, visual_presence_logits = self.visual_evidence_head(h_va, h_f)

        # Gold visual_evidence text is a TRAINING TARGET only. It is encoded
        # exclusively when the visual-evidence loss is requested; normal
        # dev/test inference leaves this branch off, eliminating auxiliary-label
        # computation from the prediction path.
        visual_text_repr = None
        if compute_visual_evidence_target and batch.get("visual_evidence_input_ids") is not None:
            raw = self.text_encoder.encode_text_only(
                batch["visual_evidence_input_ids"],
                batch["visual_evidence_attention_mask"],
            )
            visual_text_repr = self.visual_evidence_head.project_evidence_text(self.aux_text_proj(raw))

        tag_logits, tag_probs = self.tag_head(h_f)
        # Model-structure contract from 模型结构.docx: Evidence Head,
        # Reasoning Router, and Cross-modal Head are computed before the three
        # reasoning paths. The cross-modal path consumes h_rel explicitly.
        h_rel = self.relation(h_te, h_ve, h_a, h_f)
        paths = self.reasoner(h_f, h_te, h_ve, h_a, h_rel)
        # Gold tags are only passed when the curriculum mix is strictly > 0.
        # This makes the no-gold-tag test path explicit rather than relying on a
        # multiplication by zero inside the router.
        gold_tags = batch.get("reasoning_tag_labels") if routing_gold_mix > 0 else None
        routed, route_weights = self.router(
            paths,
            tag_probs,
            gold_tags=gold_tags,
            gold_mix=routing_gold_mix,
        )
        h_r = self.path_interaction(routed, h_f)
        memory = torch.stack([h_r, h_rel, h_te, h_ve, h_a], dim=1)

        return CoreOutput(
            h_ta=h_ta,
            h_va=h_va,
            h_a=h_a,
            h_f=h_f,
            text_evidence_logits=te_logits,
            text_evidence_probs=te_probs,
            h_te=h_te,
            visual_evidence_weights=ve_weights,
            visual_presence_logits=visual_presence_logits,
            h_ve=h_ve,
            tag_logits=tag_logits,
            tag_probs=tag_probs,
            route_weights=route_weights,
            h_reasoning=h_r,
            h_relation=h_rel,
            bridge_memory=memory,
            visual_evidence_text_repr=visual_text_repr,
        )

    def bridge_logits(self, core: CoreOutput, bridge_ids: torch.Tensor, bridge_mask: torch.Tensor) -> torch.Tensor:
        decoder_ids = shift_tokens_right(bridge_ids, self.bridge_generator.pad_id)
        return self.bridge_generator(
            self.text_encoder.backbone,
            decoder_input_ids=decoder_ids,
            memory=core.bridge_memory,
            decoder_attention_mask=bridge_mask,
        )

    @torch.no_grad()
    def generate_bridge(self, core: CoreOutput, max_length: int) -> torch.Tensor:
        # Generation must use inference-mode dropout semantics even when called
        # from a Stage-5 training forward. Restore the previous module modes so
        # differentiable losses continue in the requested training state.
        t5_was_training = self.text_encoder.backbone.training
        bridge_was_training = self.bridge_generator.training
        self.text_encoder.backbone.eval()
        self.bridge_generator.eval()
        try:
            return self.bridge_generator.generate(
                t5_model=self.text_encoder.backbone,
                memory=core.bridge_memory,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                ground_id=self.ground_id,
                transition_id=self.bridge_classifier.transition_id,
                implication_id=self.bridge_classifier.implication_id,
                max_length=max_length,
                min_tokens_per_field=int(self.cfg_model.get("bridge_min_tokens_per_field", 2)),
            )
        finally:
            self.text_encoder.backbone.train(t5_was_training)
            self.bridge_generator.train(bridge_was_training)

    def classify_bridge(self, bridge_ids: torch.Tensor, bridge_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.bridge_classifier(bridge_ids, bridge_mask)

    @staticmethod
    def _generated_attention_mask(
        ids: torch.Tensor, eos_id: int, pad_id: int
    ) -> torch.Tensor:
        positions = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        eos_positions = torch.where(
            ids.eq(eos_id),
            positions,
            torch.full_like(positions, ids.size(1)),
        )
        first_eos = eos_positions.min(dim=1).values.unsqueeze(1)
        return positions.le(first_eos).logical_and(ids.ne(pad_id)).long()

    @staticmethod
    def _pad_ids(ids: torch.Tensor, length: int, pad_id: int) -> torch.Tensor:
        if ids.size(1) >= length:
            return ids[:, :length]
        out = torch.full(
            (ids.size(0), length), pad_id, dtype=ids.dtype, device=ids.device
        )
        out[:, : ids.size(1)] = ids
        return out

    def forward(
        self,
        batch: dict[str, Any],
        *,
        stage: str,
        routing_gold_mix: float = 0.0,
        compute_visual_evidence_target: bool = False,
        compute_bridge: bool = False,
        compute_sentiment: bool = False,
        generated_bridge_ratio: float = 0.0,
        max_generation_length: int = 160,
    ) -> dict[str, torch.Tensor | None]:
        """One DDP-safe training forward covering every stage-specific branch."""
        if stage == "stage4_classifier":
            sentiment_logits, _ = self.classify_bridge(
                batch["bridge_input_ids"], batch["bridge_attention_mask"]
            )
            return {"sentiment_logits": sentiment_logits}

        core = self.encode_and_reason(
            batch,
            routing_gold_mix=routing_gold_mix,
            compute_visual_evidence_target=compute_visual_evidence_target,
        )
        outputs: dict[str, torch.Tensor | None] = {
            "text_evidence_logits": core.text_evidence_logits,
            "visual_evidence_repr": core.h_ve,
            "visual_evidence_text_repr": core.visual_evidence_text_repr,
            "visual_presence_logits": core.visual_presence_logits,
            "tag_logits": core.tag_logits,
        }
        if compute_bridge:
            outputs["bridge_logits"] = self.bridge_logits(
                core, batch["bridge_input_ids"], batch["bridge_attention_mask"]
            )

        if compute_sentiment:
            reference = batch["bridge_input_ids"]
            reference_mask = batch["bridge_attention_mask"]
            use_generated = (
                torch.rand(reference.size(0), device=reference.device)
                < generated_bridge_ratio
            ) | (~batch["has_bridge"])
            if not use_generated.any():
                sentiment_logits, _ = self.classify_bridge(
                    reference, reference_mask
                )
                outputs["sentiment_logits"] = sentiment_logits
                return outputs

            # Recompute the generation path under eval semantics. This makes
            # the generated classifier inputs match dev/test behavior instead
            # of retaining Dropout noise from the differentiable training core.
            # With batch size 1, reference-only draws skip this expensive pass.
            was_training = self.training
            try:
                self.eval()
                with torch.no_grad():
                    generation_core = self.encode_and_reason(
                        batch,
                        routing_gold_mix=0.0,
                        compute_visual_evidence_target=False,
                    )
                    generated = self.generate_bridge(
                        generation_core, max_generation_length
                    )
            finally:
                self.train(was_training)

            generated_mask = self._generated_attention_mask(
                generated, self.eos_id, self.bridge_generator.pad_id
            )
            length = max(generated.size(1), reference.size(1))
            generated = self._pad_ids(
                generated, length, self.bridge_generator.pad_id
            )
            generated_mask = self._pad_ids(generated_mask, length, 0)
            reference = self._pad_ids(
                reference, length, self.bridge_generator.pad_id
            )
            reference_mask = self._pad_ids(reference_mask, length, 0)
            classifier_ids = torch.where(
                use_generated.unsqueeze(-1), generated, reference
            )
            classifier_mask = torch.where(
                use_generated.unsqueeze(-1), generated_mask, reference_mask
            )
            sentiment_logits, _ = self.classify_bridge(
                classifier_ids, classifier_mask
            )
            outputs["sentiment_logits"] = sentiment_logits
        return outputs
