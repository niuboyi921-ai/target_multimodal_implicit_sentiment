#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmis.config import load_config, resolve_project_path
from tmis.evaluation import compute_reasoning_tag_metrics
from tmis.runtime import (
    build_collator,
    build_datasets,
    build_model,
    build_tokenizer_and_processor,
    choose_device,
)
from tmis.training import autocast_context, move_batch, resolve_amp_dtype
from tmis.utils import load_checkpoint, write_json


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate reasoning tags, routes, and latent-selector collapse diagnostics."
    )
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

    tag_gold: list[np.ndarray] = []
    tag_probs: list[np.ndarray] = []
    route_rows: list[dict[str, float | bool]] = []
    text_ratios: list[float] = []
    visual_entropies: list[float] = []
    visual_top1: list[float] = []

    with torch.no_grad():
        for raw in tqdm(loader, desc="aux-eval"):
            batch = move_batch(raw, device)
            with autocast_context(device, amp_dtype):
                core = model.encode_and_reason(batch, routing_gold_mix=0.0)
            tag_gold.append(batch["reasoning_tag_labels"].cpu().numpy())
            tag_probs.append(core.tag_probs.float().cpu().numpy())

            mask = batch["text_token_mask"].float()
            text_ratio = (
                (core.text_selector_weights.float() * mask).sum(dim=1)
                / mask.sum(dim=1).clamp_min(1.0)
            )
            text_ratios.extend(text_ratio.cpu().tolist())

            visual = core.visual_selector_weights.float().clamp_min(1e-8)
            if visual.size(1) > 1:
                entropy = -(visual * visual.log()).sum(dim=1) / math.log(visual.size(1))
            else:
                entropy = torch.zeros(visual.size(0), device=visual.device)
            visual_entropies.extend(entropy.cpu().tolist())
            visual_top1.extend(visual.max(dim=1).values.cpu().tolist())

            for j in range(core.route_weights.size(0)):
                route_rows.append(
                    {
                        "implicit_gold": bool(batch["is_implicit"][j].item()),
                        "cross_gold": bool(batch["reasoning_tag_labels"][j, 2].item()),
                        "direct": float(core.route_weights[j, 0].item()),
                        "implicit": float(core.route_weights[j, 1].item()),
                        "cross_modal": float(core.route_weights[j, 2].item()),
                    }
                )

    tags = compute_reasoning_tag_metrics(
        np.concatenate(tag_gold), np.concatenate(tag_probs)
    )

    def route_mean(rows):
        if not rows:
            return None
        return {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in ("direct", "implicit", "cross_modal")
        }

    result = {
        "reasoning_tags": tags,
        "selector_diagnostics": {
            "text_mean_selection_ratio": _mean_or_none(text_ratios),
            "text_near_zero_rate": _mean_or_none([float(x < 0.01) for x in text_ratios]),
            "text_near_all_rate": _mean_or_none([float(x > 0.90) for x in text_ratios]),
            "visual_mean_normalized_entropy": _mean_or_none(visual_entropies),
            "visual_mean_top1_mass": _mean_or_none(visual_top1),
            "visual_near_uniform_rate": _mean_or_none(
                [float(x > 0.98) for x in visual_entropies]
            ),
            "visual_near_single_patch_rate": _mean_or_none(
                [float(x < 0.05) for x in visual_entropies]
            ),
        },
        "route_weight_means": {
            "full": route_mean(route_rows),
            "implicit": route_mean([r for r in route_rows if r["implicit_gold"]]),
            "non_implicit": route_mean([r for r in route_rows if not r["implicit_gold"]]),
            "cross_modal": route_mean([r for r in route_rows if r["cross_gold"]]),
            "non_cross_modal": route_mean([r for r in route_rows if not r["cross_gold"]]),
        },
        "note": (
            "Selectors are latent modules without artificial evidence labels. "
            "These diagnostics detect collapse but do not claim a unique gold rationale."
        ),
    }
    output = Path(cfg["output_dir"]) / "auxiliary_metrics.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
