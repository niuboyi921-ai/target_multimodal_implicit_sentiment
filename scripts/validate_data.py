#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmis.config import load_config, resolve_project_path
from tmis.data.dataset import load_json_records
from tmis.data.schema import normalize_record


EI_RAW_LABEL = re.compile(
    r"\b(?:positive|neutral|negative)\s+(?:sentiment|polarity|label|class|category)\b|"
    r"\b(?:sentiment|polarity)\s+(?:is|=)\s*(?:positive|neutral|negative)\b",
    flags=re.IGNORECASE,
)


def resolve_image(image_dir: Path, name: str, exts: list[str]) -> Path | None:
    p = image_dir / name
    if p.is_file():
        return p
    stem = Path(name).stem
    for ext in exts:
        p = image_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def validate_split(cfg, split: str):
    d = cfg["data"]
    path = resolve_project_path(cfg, d[f"{split}_file"])
    image_dir = resolve_project_path(cfg, d["image_dir"])
    raw = load_json_records(path)
    counts = Counter()
    errors: list[str] = []
    warnings: list[str] = []

    for i, item in enumerate(raw):
        try:
            r = normalize_record(item, i)
            counts[f"sentiment/{r.sentiment}"] += 1
            counts["implicit" if r.is_implicit else "non_implicit"] += 1
            counts["has_bridge" if r.reasoning_bridge else "missing_bridge"] += 1
            for tag_name, enabled in r.reasoning_tags.items():
                counts[f"reasoning_tag/{tag_name}/{str(enabled).lower()}"] += 1

            image_path = resolve_image(image_dir, r.image, d["image_extensions"])
            if image_path is None:
                errors.append(f"{split}[{i}] image not found: {r.image}")
            else:
                try:
                    with Image.open(image_path) as im:
                        im.verify()
                except Exception as exc:
                    errors.append(f"{split}[{i}] image cannot be decoded: {r.image}: {exc}")

            if r.reasoning_bridge:
                ei = r.reasoning_bridge["evaluative_implication"]
                if EI_RAW_LABEL.search(ei):
                    warnings.append(
                        f"{split}[{i}] evaluative_implication contains raw polarity/meta wording: {ei!r}"
                    )

            expected_name = str(d.get("dataset_name", "")).lower().replace("twitter", "twitter-")
            if r.source_dataset and expected_name:
                actual = r.source_dataset.lower().replace("_", "-")
                # Only warning: historical annotation may use Twitter-2015 while
                # config key is twitter2015.
                if expected_name not in actual and actual not in expected_name:
                    warnings.append(
                        f"{split}[{i}] source_dataset={r.source_dataset!r} differs from config dataset_name={d.get('dataset_name')!r}"
                    )
        except Exception as exc:
            errors.append(f"{split}[{i}] {exc}")

    if not raw:
        errors.append(f"{split}: split is empty; replace the placeholder JSON before training")

    print(f"\n[{split}] n={len(raw)}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    if warnings:
        print(f"  WARNINGS: {len(warnings)}")
        for w in warnings[:30]:
            print("   -", w)
    if errors:
        print(f"  ERRORS: {len(errors)}")
        for e in errors[:50]:
            print("   -", e)
    return errors, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/twitter2015.yaml")
    args = ap.parse_args()
    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)
    all_errors: list[str] = []
    all_warnings: list[str] = []
    for split in ["train", "dev", "test"]:
        errors, warnings = validate_split(cfg, split)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
    if all_errors:
        raise SystemExit(1)
    print(f"\nData validation passed with {len(all_warnings)} warning(s).")


if __name__ == "__main__":
    main()
