#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmis.config import load_config, resolve_project_path
from tmis.evaluation import compute_reasoning_tag_metrics, compute_text_evidence_token_metrics
from tmis.runtime import build_collator, build_datasets, build_model, build_tokenizer_and_processor, choose_device
from tmis.training import autocast_context, move_batch, resolve_amp_dtype
from tmis.utils import load_checkpoint, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate auxiliary evidence/tag heads. This is separate from clean sentiment inference.")
    ap.add_argument("--config", default="configs/twitter2015.yaml")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--checkpoint", default=None)
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

    tag_gold, tag_probs = [], []
    te_gold, te_probs = [], []
    route_rows = []
    visual_cos = []

    with torch.no_grad():
        for raw in tqdm(loader, desc="aux-eval"):
            batch = move_batch(raw, device)
            # Here visual target encoding is intentionally enabled because this
            # script evaluates the auxiliary supervision task itself, not final
            # sentiment inference.
            with autocast_context(device, amp_dtype):
                core = model.encode_and_reason(
                    batch,
                    routing_gold_mix=0.0,
                    compute_visual_evidence_target=True,
                )
            tag_gold.append(batch["reasoning_tag_labels"].cpu().numpy())
            tag_probs.append(core.tag_probs.cpu().numpy())
            te_gold.extend(batch["text_evidence_labels"].cpu().numpy())
            te_probs.extend(core.text_evidence_probs.cpu().numpy())

            mask = batch["has_visual_evidence"].bool()
            if mask.any() and core.visual_evidence_text_repr is not None:
                v = torch.nn.functional.normalize(core.h_ve[mask], dim=-1)
                t = torch.nn.functional.normalize(core.visual_evidence_text_repr[mask], dim=-1)
                visual_cos.extend((v * t).sum(-1).cpu().tolist())

            for j in range(core.route_weights.size(0)):
                route_rows.append(
                    {
                        "implicit_gold": bool(batch["is_implicit"][j].item()),
                        "direct": float(core.route_weights[j, 0].item()),
                        "implicit": float(core.route_weights[j, 1].item()),
                        "cross_modal": float(core.route_weights[j, 2].item()),
                    }
                )

    tags = compute_reasoning_tag_metrics(np.concatenate(tag_gold), np.concatenate(tag_probs))
    text_evidence = compute_text_evidence_token_metrics(te_gold, te_probs)

    def route_mean(rows):
        if not rows:
            return None
        return {k: float(np.mean([r[k] for r in rows])) for k in ("direct", "implicit", "cross_modal")}

    result = {
        "reasoning_tags": tags,
        "text_evidence_token": text_evidence,
        "visual_evidence_positive_cosine_mean": float(np.mean(visual_cos)) if visual_cos else None,
        "route_weight_means": {
            "full": route_mean(route_rows),
            "implicit": route_mean([r for r in route_rows if r["implicit_gold"]]),
            "non_implicit": route_mean([r for r in route_rows if not r["implicit_gold"]]),
        },
        "note": "Auxiliary metrics use gold auxiliary annotations only as evaluation targets; they are not fed into final sentiment inference.",
    }
    output = Path(cfg["output_dir"]) / "auxiliary_metrics.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
