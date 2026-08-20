from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    PreTrainedTokenizerBase = Any

from tmis.constants import (
    BRIDGE_BOS_TOKEN,
    BRIDGE_EOS_TOKEN,
    GROUND_TOKEN,
    IMPLICATION_TOKEN,
    TRANSITION_TOKEN,
)


def serialize_bridge(bridge: dict[str, str]) -> str:
    return (
        f"{GROUND_TOKEN} {bridge['grounded_synthesis']} "
        f"{TRANSITION_TOKEN} {bridge['reasoning_transition']} "
        f"{IMPLICATION_TOKEN} {bridge['evaluative_implication']}"
    )


def _trim_field_token_ids(field_ids: list[list[int]], budget: int) -> list[list[int]]:
    """Trim longest field first while preserving every structural marker.

    The bridge layout always contains BOS + three markers + EOS. This helper only
    allocates the remaining token budget across S/R/E. It avoids the old failure
    mode where plain string truncation could remove [TRANSITION] or [IMPLICATION].
    """
    out = [list(x) for x in field_ids]
    if budget <= 0:
        return [[], [], []]
    while sum(len(x) for x in out) > budget:
        longest = max(range(3), key=lambda i: len(out[i]))
        if out[longest]:
            out[longest].pop()
        else:
            break
    return out


class MultimodalCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        image_processor: Any,
        max_text_length: int,
        max_target_length: int,
        max_bridge_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_text_length = max_text_length
        self.max_target_length = max_target_length
        self.max_bridge_length = max_bridge_length
        self.bos_id = tokenizer.convert_tokens_to_ids(BRIDGE_BOS_TOKEN)
        self.eos_id = tokenizer.convert_tokens_to_ids(BRIDGE_EOS_TOKEN)
        self.ground_id = tokenizer.convert_tokens_to_ids(GROUND_TOKEN)
        self.transition_id = tokenizer.convert_tokens_to_ids(TRANSITION_TOKEN)
        self.implication_id = tokenizer.convert_tokens_to_ids(IMPLICATION_TOKEN)

    def _tokenize_text_target(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [x["restored_text"] for x in batch]
        targets = [x["target"] for x in batch]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        text_mask = enc["attention_mask"].bool()
        for special_id in (self.tokenizer.pad_token_id, self.tokenizer.eos_token_id):
            if special_id is not None:
                text_mask &= enc["input_ids"].ne(special_id)
        if not text_mask.any(dim=1).all():
            bad = [batch[i]["index"] for i in range(len(batch)) if not text_mask[i].any()]
            raise ValueError(f"text tokenization produced no content tokens for records: {bad}")

        target_enc = self.tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        target_token_mask = target_enc["attention_mask"].bool()
        for special_id in (self.tokenizer.pad_token_id, self.tokenizer.eos_token_id):
            if special_id is not None:
                target_token_mask &= target_enc["input_ids"].ne(special_id)
        if not target_token_mask.any(dim=1).all():
            bad = [batch[i]["index"] for i in range(bsz) if not target_token_mask[i].any()]
            raise ValueError(f"target tokenization produced no content tokens for records: {bad}")

        enc["target_input_ids"] = target_enc["input_ids"]
        enc["target_attention_mask"] = target_enc["attention_mask"]
        enc["target_token_mask"] = target_token_mask
        enc["text_token_mask"] = text_mask
        return enc

    def _encode_bridge(self, bridge: dict[str, str] | None) -> tuple[list[int], int]:
        if bridge is None:
            return [self.bos_id, self.ground_id, self.transition_id, self.implication_id, self.eos_id], 0

        fields = [
            self.tokenizer.encode(bridge["grounded_synthesis"], add_special_tokens=False),
            self.tokenizer.encode(bridge["reasoning_transition"], add_special_tokens=False),
            self.tokenizer.encode(bridge["evaluative_implication"], add_special_tokens=False),
        ]
        structural_tokens = 5  # BOS + GROUND + TRANSITION + IMPLICATION + EOS
        budget = max(0, self.max_bridge_length - structural_tokens)
        fields = _trim_field_token_ids(fields, budget)
        ids = [self.bos_id, self.ground_id]
        ids.extend(fields[0])
        ids.append(self.transition_id)
        ids.extend(fields[1])
        ids.append(self.implication_id)
        ids.extend(fields[2])
        ids.append(self.eos_id)
        return ids[: self.max_bridge_length], 1

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        text_enc = self._tokenize_text_target(batch)
        image_batch = self.image_processor(images=[x["image"] for x in batch], return_tensors="pt")

        bridge_items = [self._encode_bridge(x["reasoning_bridge"]) for x in batch]
        bridge_ids = [x[0] for x in bridge_items]
        has_bridge = torch.tensor([x[1] for x in bridge_items], dtype=torch.bool)
        max_len = min(max(len(x) for x in bridge_ids), self.max_bridge_length)
        pad_id = self.tokenizer.pad_token_id
        bridge_input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        bridge_attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, ids in enumerate(bridge_ids):
            ids = ids[:max_len]
            bridge_input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            bridge_attention_mask[i, : len(ids)] = 1

        tag_labels = torch.tensor(
            [
                [
                    float(x["reasoning_tags"]["explicit_cue_present"]),
                    float(x["reasoning_tags"]["implicit_sentiment_present"]),
                    float(x["reasoning_tags"]["cross_modal_reasoning_required"]),
                ]
                for x in batch
            ],
            dtype=torch.float,
        )

        return {
            "indices": torch.tensor([x["index"] for x in batch], dtype=torch.long),
            "input_ids": text_enc["input_ids"],
            "attention_mask": text_enc["attention_mask"],
            "target_input_ids": text_enc["target_input_ids"],
            "target_attention_mask": text_enc["target_attention_mask"],
            "target_token_mask": text_enc["target_token_mask"],
            "text_token_mask": text_enc["text_token_mask"],
            "pixel_values": image_batch["pixel_values"],
            "reasoning_tag_labels": tag_labels,
            "bridge_input_ids": bridge_input_ids,
            "bridge_attention_mask": bridge_attention_mask,
            "has_bridge": has_bridge,
            "sentiment_labels": torch.tensor([x["sentiment_id"] for x in batch], dtype=torch.long),
            "is_implicit": torch.tensor([x["is_implicit"] for x in batch], dtype=torch.bool),
            "targets": [x["target"] for x in batch],
            "restored_texts": [x["restored_text"] for x in batch],
            "reference_bridges": [x["reasoning_bridge"] for x in batch],
            "image_names": [x["image_name"] for x in batch],
            "gold_sentiments": [x["sentiment"] for x in batch],
        }
