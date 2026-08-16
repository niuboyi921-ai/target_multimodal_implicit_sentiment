#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmis.config import load_config, resolve_project_path
from tmis.constants import ID_TO_SENTIMENT
from tmis.evaluation import (
    compute_bridge_reference_metrics,
    compute_metrics,
    compute_structure_metrics,
    parse_bridge_text,
)
from tmis.runtime import (
    build_collator,
    build_datasets,
    build_model,
    build_tokenizer_and_processor,
    choose_device,
)
from tmis.training import (
    autocast_context,
    generated_attention_mask,
    move_batch,
    resolve_amp_dtype,
)
from tmis.utils import load_checkpoint, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/twitter2015.yaml")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument(
        "--bertscore",
        action="store_true",
        help="Also compute BERTScore against reference bridge. Requires requirements-eval.txt.",
    )
    args = ap.parse_args()

    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)
    output_value = args.output_dir if args.output_dir is not None else cfg["output_dir"]
    cfg["output_dir"] = str(resolve_project_path(cfg, output_value))
    device = choose_device(cfg["device"])
    tokenizer, image_processor = build_tokenizer_and_processor(cfg)
    _, _, test_ds = build_datasets(cfg)
    collator = build_collator(cfg, tokenizer, image_processor)
    model = build_model(cfg, tokenizer).to(device)
    ckpt = args.checkpoint or str(Path(cfg["output_dir"]) / "best_joint.pt")
    load_checkpoint(ckpt, model, map_location=device)
    model.eval()
    amp_dtype = resolve_amp_dtype(cfg["training"], device)

    loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["eval_batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        persistent_workers=cfg["training"]["num_workers"] > 0,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    implicit: list[bool] = []
    predictions: list[dict] = []
    parsed_all = []
    generated_structured: list[dict[str, str]] = []
    reference_structured: list[dict[str, str] | None] = []

    with torch.no_grad():
        for raw in tqdm(loader, desc="test"):
            batch = move_batch(raw, device)
            # STRICT inference path: no gold reasoning tags are mixed and no gold
            # visual-evidence text is encoded into the prediction computation.
            with autocast_context(device, amp_dtype):
                core = model.encode_and_reason(
                    batch,
                    routing_gold_mix=0.0,
                    compute_visual_evidence_target=False,
                )
                ids = model.generate_bridge(core, cfg["evaluation"]["max_generation_length"])
                mask = generated_attention_mask(ids, model.eos_id, tokenizer.pad_token_id)
                logits, _ = model.classify_bridge(ids, mask)
            pred = logits.argmax(-1)
            texts = tokenizer.batch_decode(ids, skip_special_tokens=False)

            for j, text in enumerate(texts):
                parsed = parse_bridge_text(text)
                parsed_all.append(parsed)
                generated_structured.append(parsed.fields)
                ref = batch["reference_bridges"][j]
                reference_structured.append(ref)
                predictions.append(
                    {
                        "index": int(batch["indices"][j].item()),
                        "target": batch["targets"][j],
                        "image": batch["image_names"][j],
                        "reasoning_bridge": parsed.flattened,
                        "reasoning_bridge_structured": parsed.fields,
                        "bridge_structure_valid": parsed.structure_valid,
                        "bridge_structure_error": parsed.error,
                        "sentiment": ID_TO_SENTIMENT[int(pred[j].item())],
                        "gold_sentiment": batch["gold_sentiments"][j],
                        "implicit_reasoning_required": bool(batch["is_implicit"][j].item()),
                        "reference_reasoning_bridge": ref,
                        "route_probabilities": {
                            "explicit": float(core.tag_probs[j, 0].item()),
                            "implicit": float(core.tag_probs[j, 1].item()),
                            "cross_modal": float(core.tag_probs[j, 2].item()),
                        },
                        "route_weights": {
                            "direct": float(core.route_weights[j, 0].item()),
                            "implicit": float(core.route_weights[j, 1].item()),
                            "cross_modal": float(core.route_weights[j, 2].item()),
                        },
                    }
                )

            y_true.extend(batch["sentiment_labels"].cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            implicit.extend(batch["is_implicit"].cpu().tolist())

    sentiment_metrics = compute_metrics(y_true, y_pred, implicit)
    bridge_cfg = cfg.get("evaluation", {}).get("bridge_metrics", {})
    use_bertscore = bool(args.bertscore or bridge_cfg.get("bertscore", False))
    bridge_reference = compute_bridge_reference_metrics(
        generated_structured,
        reference_structured,
        compute_bertscore=use_bertscore,
        bertscore_model=bridge_cfg.get("bertscore_model"),
    )
    metrics = {
        "sentiment": sentiment_metrics,
        "bridge_structure": compute_structure_metrics(parsed_all),
        "bridge_reference": bridge_reference,
        "evaluation_protocol": {
            "sentiment_inference_uses_gold_evidence": False,
            "sentiment_inference_uses_gold_reasoning_tags": False,
            "sentiment_inference_uses_reference_bridge": False,
            "reference_bridge_used_only_for_offline_generation_metrics": True,
            "llm_judge_called_during_training": False,
        },
    }

    out = Path(cfg["output_dir"])
    write_json(out / "test_metrics.json", metrics)
    write_json(out / "test_predictions.json", predictions)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
