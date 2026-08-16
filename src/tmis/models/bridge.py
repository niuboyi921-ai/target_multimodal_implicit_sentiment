from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput


def shift_tokens_right(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Create T5 teacher-forcing inputs for a full bridge target sequence."""
    shifted = torch.full_like(input_ids, pad_id)
    shifted[:, 1:] = input_ids[:, :-1]
    shifted.masked_fill_(shifted.eq(-100), pad_id)
    return shifted


class ReasoningBridgeGenerator(nn.Module):
    """Multimodal adapter and structural decoding guard for the T5 decoder.

    The language decoder, shared token embeddings, and LM head come from the
    same pretrained T5-large used by the text encoder. This module only projects
    the five multimodal bridge-memory tokens into T5's hidden space and applies
    the required [GROUND] -> [TRANSITION] -> [IMPLICATION] generation contract.
    """

    def __init__(
        self,
        hidden_dim: int,
        t5_hidden_dim: int,
        pad_id: int,
        max_length: int,
    ) -> None:
        super().__init__()
        self.pad_id = int(pad_id)
        self.max_length = int(max_length)
        self.memory_proj = nn.Sequential(
            nn.Linear(hidden_dim, t5_hidden_dim),
            nn.LayerNorm(t5_hidden_dim),
        )
        self.memory_type_embeddings = nn.Embedding(5, t5_hidden_dim)

    def _encoder_outputs(self, memory: torch.Tensor) -> tuple[BaseModelOutput, torch.Tensor]:
        projected = self.memory_proj(memory)
        if projected.size(1) != self.memory_type_embeddings.num_embeddings:
            raise ValueError(
                f"expected {self.memory_type_embeddings.num_embeddings} Bridge memory tokens, "
                f"got {projected.size(1)}"
            )
        memory_types = torch.arange(projected.size(1), device=projected.device)
        projected = projected + self.memory_type_embeddings(memory_types).unsqueeze(0)
        memory_mask = torch.ones(
            projected.shape[:2], dtype=torch.long, device=projected.device
        )
        return BaseModelOutput(last_hidden_state=projected), memory_mask

    def forward(
        self,
        t5_model: Any,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_outputs, memory_mask = self._encoder_outputs(memory)
        out = t5_model(
            encoder_outputs=encoder_outputs,
            attention_mask=memory_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.logits

    @torch.no_grad()
    def generate(
        self,
        t5_model: Any,
        memory: torch.Tensor,
        bos_id: int,
        eos_id: int,
        ground_id: int,
        transition_id: int,
        implication_id: int,
        max_length: int | None = None,
        min_tokens_per_field: int = 2,
    ) -> torch.Tensor:
        max_length = min(int(max_length or self.max_length), self.max_length)
        min_required = 3 * int(min_tokens_per_field) + 5
        if max_length < min_required:
            raise ValueError(
                f"max bridge generation length must be at least {min_required} "
                f"for min_tokens_per_field={min_tokens_per_field}"
            )

        encoder_outputs, memory_mask = self._encoder_outputs(memory)
        vocab_size = int(t5_model.get_output_embeddings().weight.size(0))
        structural = {self.pad_id, bos_id, ground_id, transition_id, implication_id, eos_id}
        content_tokens = tuple(i for i in range(vocab_size) if i not in structural)

        def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
            del batch_id
            sequence = input_ids.tolist()
            if sequence and sequence[0] == self.pad_id:
                sequence = sequence[1:]
            remaining_slots = max_length - len(sequence)
            has_transition = transition_id in sequence
            has_implication = implication_id in sequence

            if not has_transition:
                marker_pos = len(sequence) - 1 - sequence[::-1].index(ground_id)
                field_len = len(sequence) - marker_pos - 1
                if field_len < min_tokens_per_field:
                    return list(content_tokens)
                if remaining_slots <= 2 * min_tokens_per_field + 3:
                    return [transition_id]
                return [*content_tokens, transition_id]

            if not has_implication:
                marker_pos = len(sequence) - 1 - sequence[::-1].index(transition_id)
                field_len = len(sequence) - marker_pos - 1
                if field_len < min_tokens_per_field:
                    return list(content_tokens)
                if remaining_slots <= min_tokens_per_field + 2:
                    return [implication_id]
                return [*content_tokens, implication_id]

            marker_pos = len(sequence) - 1 - sequence[::-1].index(implication_id)
            field_len = len(sequence) - marker_pos - 1
            if field_len < min_tokens_per_field:
                return list(content_tokens)
            if remaining_slots <= 1:
                return [eos_id]
            return [*content_tokens, eos_id]

        decoder_input_ids = torch.full(
            (memory.size(0), 3), self.pad_id, dtype=torch.long, device=memory.device
        )
        decoder_input_ids[:, 1] = bos_id
        decoder_input_ids[:, 2] = ground_id
        generated = t5_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=memory_mask,
            decoder_input_ids=decoder_input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            max_length=max_length + 1,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=self.pad_id,
            eos_token_id=eos_id,
            return_dict_in_generate=False,
        )
        # T5 prepends its internal decoder-start PAD. Exclude it from the
        # externally visible structured Bridge sequence.
        return generated[:, 1 : max_length + 1]


class BridgeEncoderClassifier(nn.Module):
    """Final sentiment classifier. It consumes bridge tokens only."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        bridge_dim: int,
        num_heads: int,
        layers: int,
        dropout: float,
        max_length: int,
        ground_id: int,
        transition_id: int,
        implication_id: int,
        pretrained_token_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.ground_id = ground_id
        self.transition_id = transition_id
        self.implication_id = implication_id
        self.token_embedding = nn.Embedding(vocab_size, bridge_dim, padding_idx=pad_id)
        if pretrained_token_embeddings is not None:
            self.initialize_token_embeddings(pretrained_token_embeddings)
        self.pos_embedding = nn.Embedding(max_length + 8, bridge_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=bridge_dim,
            nhead=num_heads,
            dim_feedforward=bridge_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.hier_gru = nn.GRU(bridge_dim, bridge_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(bridge_dim),
            nn.Dropout(dropout),
            nn.Linear(bridge_dim, 3),
        )

    @torch.no_grad()
    def initialize_token_embeddings(self, pretrained: torch.Tensor) -> None:
        source = pretrained.detach()
        target = self.token_embedding.weight
        if source.size(0) != target.size(0) or source.size(1) < target.size(1):
            raise ValueError(
                "pretrained Bridge token embeddings must match vocab size and "
                "have at least bridge_dim features"
            )
        target.copy_(source[:, : target.size(1)])
        target[self.pad_id].zero_()

    def _marker_state(
        self,
        encoded: torch.Tensor,
        ids: torch.Tensor,
        marker_id: int,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        mask = ids.eq(marker_id)
        first = mask.float().argmax(dim=1)
        has = mask.any(dim=1)
        gathered = encoded[torch.arange(ids.size(0), device=ids.device), first]
        return torch.where(has.unsqueeze(-1), gathered, fallback)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, length = input_ids.shape
        pos = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_embedding(input_ids) + self.pos_embedding(pos)
        encoded = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        mask = attention_mask.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        h_g = self._marker_state(encoded, input_ids, self.ground_id, pooled)
        h_t = self._marker_state(encoded, input_ids, self.transition_id, pooled)
        h_i = self._marker_state(encoded, input_ids, self.implication_id, pooled)
        seq = torch.stack([h_g, h_t, h_i], dim=1)
        _, h = self.hier_gru(seq)
        bridge_repr = h[-1]
        return self.classifier(bridge_repr), bridge_repr
