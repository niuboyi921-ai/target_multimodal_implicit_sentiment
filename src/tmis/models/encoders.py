from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, T5ForConditionalGeneration


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask = mask.to(x.dtype).unsqueeze(-1)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(eps)


class TextTargetEncoder(nn.Module):
    """Shared T5-large encoder/decoder backbone.

    The encoder is called separately for the tweet, target, and optional
    auxiliary evidence text. The same pretrained T5 decoder is later reused by
    the reasoning-bridge generator, so the project loads only one T5-large.
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_size: int,
        gradient_checkpointing: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = T5ForConditionalGeneration.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
            use_safetensors=True,
        )
        self.backbone.resize_token_embeddings(tokenizer_size)
        self.hidden_size = int(self.backbone.config.d_model)
        self.backbone.config.use_cache = False
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.backbone.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        target_token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_text = self._encode(input_ids, attention_mask)
        h_target_tokens = self._encode(target_input_ids, target_attention_mask)
        h_target = masked_mean(h_target_tokens, target_token_mask)
        return h_text, h_target

    def encode_text_only(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._encode(input_ids, attention_mask)
        return masked_mean(hidden, attention_mask.bool())


class VisionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        gradient_checkpointing: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
            use_safetensors=True,
        )
        self.hidden_size = self.backbone.config.hidden_size
        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values=pixel_values).last_hidden_state
