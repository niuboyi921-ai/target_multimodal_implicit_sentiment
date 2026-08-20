from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from tmis.data.schema import NormalizedRecord, normalize_record


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        root = json.load(f)
    if isinstance(root, list):
        return root
    if isinstance(root, dict):
        if isinstance(root.get("data"), list):
            return root["data"]
        if isinstance(root.get("records"), list):
            return root["records"]
    raise ValueError(f"{path}: JSON root must be list or contain data/records list")


class TwitterMultimodalDataset(Dataset):
    def __init__(
        self,
        json_path: str | Path,
        image_dir: str | Path,
        image_extensions: list[str],
        require_bridge: bool = False,
    ) -> None:
        self.json_path = Path(json_path)
        self.image_dir = Path(image_dir).resolve()
        self.image_extensions = image_extensions
        raw_records = load_json_records(self.json_path)
        records = [normalize_record(r, i) for i, r in enumerate(raw_records)]
        if require_bridge:
            records = [r for r in records if r.reasoning_bridge is not None]
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image(self, name: str) -> Path:
        def safe_candidate(relative_name: str) -> Path:
            candidate = (self.image_dir / relative_name).resolve()
            try:
                candidate.relative_to(self.image_dir)
            except ValueError as exc:
                raise ValueError(
                    f"image path escapes configured image_dir: {relative_name!r}"
                ) from exc
            return candidate

        candidate = safe_candidate(name)
        if candidate.is_file():
            return candidate
        stem = Path(name).stem
        for ext in self.image_extensions:
            candidate = safe_candidate(f"{stem}{ext}")
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"Image {name!r} not found in {self.image_dir}. "
            f"Tried configured extensions: {self.image_extensions}"
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r: NormalizedRecord = self.records[idx]
        image_path = self._resolve_image(r.image)
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        return {
            "index": r.index,
            "restored_text": r.restored_text,
            "target": r.target,
            "image": image,
            "image_name": r.image,
            "sentiment": r.sentiment,
            "sentiment_id": r.sentiment_id,
            "reasoning_tags": r.reasoning_tags,
            "reasoning_bridge": r.reasoning_bridge,
            "is_implicit": r.is_implicit,
        }
