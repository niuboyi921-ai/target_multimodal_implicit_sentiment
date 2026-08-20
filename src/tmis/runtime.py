from __future__ import annotations

import torch
from transformers import AutoImageProcessor, AutoTokenizer

from tmis.config import resolve_project_path
from tmis.constants import BRIDGE_EOS_TOKEN, BRIDGE_SPECIAL_TOKENS
from tmis.data.dataset import TwitterMultimodalDataset
from tmis.data.collator import MultimodalCollator
from tmis.models import SelectorGuidedMultiPathModel


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_tokenizer_and_processor(cfg):
    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["text_backbone"],
        revision=model_cfg.get("text_backbone_revision"),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise RuntimeError("T5 tokenizer must define pad_token_id and eos_token_id")
    tokenizer.add_special_tokens(BRIDGE_SPECIAL_TOKENS)
    if tokenizer.convert_tokens_to_ids(BRIDGE_EOS_TOKEN) != tokenizer.eos_token_id:
        raise RuntimeError("Bridge EOS must resolve to T5's native </s> token")
    image_processor = AutoImageProcessor.from_pretrained(
        model_cfg["vision_backbone"],
        revision=model_cfg.get("vision_backbone_revision"),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    return tokenizer, image_processor


def build_datasets(cfg):
    d = cfg["data"]
    image_dir = resolve_project_path(cfg, d["image_dir"])
    kwargs = dict(image_dir=image_dir, image_extensions=d["image_extensions"], require_bridge=False)
    train = TwitterMultimodalDataset(resolve_project_path(cfg, d["train_file"]), **kwargs)
    dev = TwitterMultimodalDataset(resolve_project_path(cfg, d["dev_file"]), **kwargs)
    test = TwitterMultimodalDataset(resolve_project_path(cfg, d["test_file"]), **kwargs)
    return train, dev, test


def build_collator(cfg, tokenizer, image_processor):
    d = cfg["data"]
    return MultimodalCollator(
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_text_length=d["max_text_length"],
        max_target_length=d["max_target_length"],
        max_bridge_length=d["max_bridge_length"],
    )


def build_model(cfg, tokenizer):
    return SelectorGuidedMultiPathModel(cfg, tokenizer)
