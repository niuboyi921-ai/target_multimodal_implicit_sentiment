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
from tmis.models.lora import is_lora_parameter
from tmis.models.reasoning import (
    CrossPathInteraction,
    MultiPathReasoner,
    ReasoningTagHead,
    SoftRouter,
)
from tmis.models.selectors import TargetAwareTextSelector, TargetAwareVisualSelector


@dataclass
class CoreOutput:
    h_ta: torch.Tensor
    h_va: torch.Tensor
    h_a: torch.Tensor
    h_f: torch.Tensor
    text_selector_logits: torch.Tensor
    text_selector_weights: torch.Tensor
    h_text_selected: torch.Tensor
    visual_selector_weights: torch.Tensor
    h_visual_selected: torch.Tensor
    h_text_global: torch.Tensor
    tag_logits: torch.Tensor
    tag_probs: torch.Tensor
    route_weights: torch.Tensor
    h_reasoning: torch.Tensor
    bridge_memory: torch.Tensor


class SelectorGuidedMultiPathModel(nn.Module):
    checkpoint_state_format = "parameter_efficient_v1"

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
            lora_config=m.get("lora"),
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
        self.text_conditioner = TargetConditioner(hidden, dropout)
        self.visual_conditioner = TargetConditioner(hidden, dropout)
        self.fusion = MultimodalFusion(hidden, heads, dropout)
        self.text_selector = TargetAwareTextSelector(hidden, dropout)
        self.visual_selector = TargetAwareVisualSelector(hidden, dropout)
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
        self.bridge_classifier.token_embedding.weight.requires_grad_(False)

        self.bos_id = tokenizer.convert_tokens_to_ids(BRIDGE_BOS_TOKEN)
        self.eos_id = tokenizer.convert_tokens_to_ids(BRIDGE_EOS_TOKEN)
        self.ground_id = tokenizer.convert_tokens_to_ids(GROUND_TOKEN)

    @staticmethod
    def _is_frozen_pretrained_state(name: str) -> bool:
        if name.startswith("vision_encoder.backbone."):
            return True
        if name.startswith("text_encoder.backbone.") and not is_lora_parameter(name):
            return True
        # This large table is copied from the frozen T5 embedding and remains
        # fixed under the small-data protocol, so it can be reconstructed.
        return name == "bridge_classifier.token_embedding.weight"

    def checkpoint_state_dict(self) -> dict[str, torch.Tensor]:
        """Save LoRA and task modules without duplicating pretrained weights."""
        return {
            name: value
            for name, value in self.state_dict().items()
            if not self._is_frozen_pretrained_state(name)
        }

    def load_checkpoint_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(state, strict=False)
        illegal_missing = [
            name
            for name in incompatible.missing_keys
            if not self._is_frozen_pretrained_state(name)
        ]
        if illegal_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "invalid parameter-efficient checkpoint; "
                f"missing={illegal_missing[:5]}, "
                f"unexpected={incompatible.unexpected_keys[:5]}"
            )

    def encode_and_reason(
        self,
        batch: dict[str, Any],
        routing_gold_mix: float = 0.0,
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

        text_selector_logits, text_selector_weights, h_text_selected = (
            self.text_selector(h_ta, h_f, text_mask)
        )
        # CLIP position 0 is the global CLS token. Excluding it forces the
        # visual selector to localize over patch tokens instead of taking a
        # shortcut through an already pooled image representation.
        h_visual_selected, visual_selector_weights = self.visual_selector(
            h_va[:, 1:, :], h_f
        )

        # Reasoning tags supervise routing and pass gradients through both
        # latent modality selectors. No manually annotated evidence enters
        # this computation.
        tag_logits, tag_probs = self.tag_head(
            h_text_selected,
            h_visual_selected,
        )
        # A masked global summary of target-conditioned text gives the
        # implicit path broad textual context without exposing any visual or
        # fused multimodal representation to that path.
        text_mask_float = text_mask.to(h_ta.dtype)
        h_text_global = (
            (h_ta * text_mask_float.unsqueeze(-1)).sum(dim=1)
            / text_mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        # Direct and Implicit remain text-only. The Cross path itself receives
        # the selected modalities plus deterministic product/difference
        # features, so no separately learned relation vector is required.
        paths = self.reasoner(
            h_text_selected,
            h_visual_selected,
            h_text_global,
            h_a,
        )
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
        memory = torch.stack(
            [h_r, h_text_selected, h_visual_selected, h_a],
            dim=1,
        )

        return CoreOutput(
            h_ta=h_ta,
            h_va=h_va,
            h_a=h_a,
            h_f=h_f,
            text_selector_logits=text_selector_logits,
            text_selector_weights=text_selector_weights,
            h_text_selected=h_text_selected,
            visual_selector_weights=visual_selector_weights,
            h_visual_selected=h_visual_selected,
            h_text_global=h_text_global,
            tag_logits=tag_logits,
            tag_probs=tag_probs,
            route_weights=route_weights,
            h_reasoning=h_r,
            bridge_memory=memory,
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
    def generate_bridge(
        self,
        core: CoreOutput,
        max_length: int,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
    ) -> torch.Tensor:
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
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
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
        )
        outputs: dict[str, torch.Tensor | None] = {
            "text_selector_logits": core.text_selector_logits,
            "text_selector_weights": core.text_selector_weights,
            "visual_selector_weights": core.visual_selector_weights,
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
