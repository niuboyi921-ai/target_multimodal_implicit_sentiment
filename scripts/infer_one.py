#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmis.config import load_config, resolve_project_path
from tmis.constants import ID_TO_SENTIMENT
from tmis.evaluation import parse_bridge_text
from tmis.runtime import build_collator, build_model, build_tokenizer_and_processor, choose_device
from tmis.training import (
    autocast_context,
    generated_attention_mask,
    move_batch,
    resolve_amp_dtype,
)
from tmis.utils import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/twitter2015.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--text", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)
    cfg["output_dir"] = str(resolve_project_path(cfg, cfg["output_dir"]))
    device = choose_device(cfg["device"])
    tokenizer, image_processor = build_tokenizer_and_processor(cfg)
    collator = build_collator(cfg, tokenizer, image_processor)
    model = build_model(cfg, tokenizer).to(device)
    load_checkpoint(
        args.checkpoint or str(Path(cfg["output_dir"]) / "best_joint.pt"),
        model,
        map_location=device,
    )
    model.eval()
    amp_dtype = resolve_amp_dtype(cfg["training"], device)

    with Image.open(args.image) as im:
        image = im.convert("RGB")

    # Reasoning tags are training labels only. Placeholder values are required
    # by the shared collator but are not mixed into strict inference routing.
    sample = {
        "index": 0,
        "restored_text": args.text,
        "target": args.target,
        "image": image,
        "image_name": Path(args.image).name,
        "sentiment": "neutral",
        "sentiment_id": 1,
        "reasoning_tags": {
            "explicit_cue_present": False,
            "implicit_sentiment_present": False,
            "cross_modal_reasoning_required": False,
        },
        "reasoning_bridge": None,
        "is_implicit": False,
    }
    batch = move_batch(collator([sample]), device)
    with torch.no_grad():
        with autocast_context(device, amp_dtype):
            core = model.encode_and_reason(
                batch,
                routing_gold_mix=0.0,
            )
            ids = model.generate_bridge(core, cfg["evaluation"]["max_generation_length"])
            mask = generated_attention_mask(ids, model.eos_id, tokenizer.pad_token_id)
            logits, _ = model.classify_bridge(ids, mask)

    parsed = parse_bridge_text(tokenizer.decode(ids[0], skip_special_tokens=False))
    result = {
        "reasoning_bridge": parsed.flattened,
        "sentiment": ID_TO_SENTIMENT[int(logits.argmax(-1)[0].item())],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
