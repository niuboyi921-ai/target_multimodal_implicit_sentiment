#!/usr/bin/env python3
"""Migrate the frozen Twitter-2015 reasoning tag to implicit sentiment presence.

The migration is intentionally limited to the reviewed dataset revision whose
SHA-256 digests are listed below. It performs three ordered operations:

1. Clear the implicit tag on every neutral record.
2. Preserve polar explicit+implicit co-occurrence; only clear the four explicit
   tags independently reviewed as lacking a target-specific cue.
3. Rename implicit_reasoning_required to implicit_sentiment_present.

An audit manifest and timestamped backup are written next to the dataset.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any


OLD_FIELD = "implicit_reasoning_required"
NEW_FIELD = "implicit_sentiment_present"

EXPECTED_INPUT_SHA256 = {
    "train": "c8100567a1ac05418754423324d7fa381bd578f9dc59326da447b9b41314f8c9",
    "dev": "b69fb11a40a5c39ddbe95d86b9e951ae4d483a31db1f8bf0066ca06ed4ce4437",
    "test": "0b94efeab9dfed9c6a5f7b1de18976da476933fde21c801572d49aad7473b5ab",
}

# These four records contain no obvious target-specific sentiment word. This
# judgment is independent of the obsolete explicit/implicit mutual-exclusion
# policy, so their explicit tag remains cleared.
EXPLICIT_FALSE_OVERRIDES: dict[tuple[str, int], dict[str, str]] = {
    ("train", 2488): {
        "target": "Darryl Dawkins",
        "reason": "Death is a negative event, not an explicit sentiment word.",
    },
    ("dev", 832): {
        "target": "Lily Collins",
        "reason": "'Indescribable' conveys intensity but has no explicit polarity.",
    },
    ("dev", 1056): {
        "target": "Tropical Cyclone # Joalane",
        "reason": "Storm size is factual; negative polarity follows from cyclone danger.",
    },
    ("test", 370): {
        "target": "FL",
        "reason": "The location description/invitation implies positivity without an explicit sentiment word.",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path: Path) -> list[dict[str, Any]]:
    root = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(root, list):
        raise ValueError(f"{path}: expected a JSON list")
    return root


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    if path.exists():
        path.chmod(path.stat().st_mode | stat.S_IWRITE)
    temporary.replace(path)


def migrate_split(split: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = {
        "records": len(rows),
        "neutral_implicit_cleared": 0,
        "initial_explicit_implicit_conflicts": 0,
        "independent_explicit_errors_cleared": 0,
        "explicit_implicit_cooccurrences_preserved": 0,
        "fields_renamed": 0,
        "final_implicit_positive": 0,
        "final_implicit_negative": 0,
        "final_implicit_neutral": 0,
        "final_explicit_implicit_conflicts": 0,
    }
    decisions: list[dict[str, Any]] = []

    for index, record in enumerate(rows):
        tags = record.get("reasoning_tags")
        if not isinstance(tags, dict):
            raise TypeError(f"{split}[{index}]: reasoning_tags must be an object")
        if OLD_FIELD not in tags:
            raise ValueError(f"{split}[{index}]: missing {OLD_FIELD}")
        if NEW_FIELD in tags:
            raise ValueError(f"{split}[{index}]: already contains {NEW_FIELD}")
        if not isinstance(tags[OLD_FIELD], bool) or not isinstance(
            tags.get("explicit_cue_present"), bool
        ):
            raise TypeError(f"{split}[{index}]: reasoning tags must be JSON booleans")

        sentiment = str(record.get("sentiment", "")).strip().lower()
        old_explicit = tags["explicit_cue_present"]
        old_implicit = tags[OLD_FIELD]
        was_conflict = old_explicit and old_implicit
        if was_conflict:
            summary["initial_explicit_implicit_conflicts"] += 1

        decision: str | None = None
        reason: str | None = None

        # Step 1: strict implicit sentiment is polar and cannot include neutral.
        if sentiment == "neutral" and tags[OLD_FIELD]:
            tags[OLD_FIELD] = False
            summary["neutral_implicit_cleared"] += 1
            decision = "clear_implicit"
            reason = "Neutral polarity cannot contain implicit polar sentiment."

        # Step 2: explicit and implicit are independent multi-label properties.
        # Preserve their polar co-occurrence, except for four explicit labels
        # independently reviewed as not containing a target-specific cue.
        if tags["explicit_cue_present"] and tags[OLD_FIELD]:
            override = EXPLICIT_FALSE_OVERRIDES.get((split, index))
            if override is not None:
                target = str(record.get("target") or record.get("targe") or "")
                if target != override["target"]:
                    raise ValueError(
                        f"{split}[{index}]: override fingerprint mismatch: {target!r}"
                    )
                tags["explicit_cue_present"] = False
                summary["independent_explicit_errors_cleared"] += 1
                decision = "clear_explicit"
                reason = override["reason"]
            else:
                summary["explicit_implicit_cooccurrences_preserved"] += 1
                decision = "preserve_cooccurrence"
                reason = (
                    "An explicit surface cue may coexist with implicit sentiment "
                    "or contextual reasoning."
                )

        if was_conflict:
            decisions.append(
                {
                    "split": split,
                    "index": index,
                    "target": record.get("target") or record.get("targe"),
                    "sentiment": sentiment,
                    "restored_text": record.get("restored_text") or record.get("text"),
                    "old_explicit_cue_present": old_explicit,
                    f"old_{OLD_FIELD}": old_implicit,
                    "decision": decision,
                    "reason": reason,
                    "new_explicit_cue_present": tags["explicit_cue_present"],
                    f"new_{NEW_FIELD}": tags[OLD_FIELD],
                }
            )

        # Step 3: preserve tag order while replacing the field name.
        migrated_tags: dict[str, bool] = {}
        for key, value in tags.items():
            migrated_tags[NEW_FIELD if key == OLD_FIELD else key] = value
        record["reasoning_tags"] = migrated_tags
        summary["fields_renamed"] += 1

        implicit = migrated_tags[NEW_FIELD]
        explicit = migrated_tags["explicit_cue_present"]
        if implicit:
            summary[f"final_implicit_{sentiment}"] += 1
        if explicit and implicit:
            summary["final_explicit_implicit_conflicts"] += 1

    if summary["final_implicit_neutral"] != 0:
        raise AssertionError(f"{split}: neutral implicit records remain")
    summary["conflict_decisions"] = decisions
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/twitter2015"))
    parser.add_argument("--apply", action="store_true", help="write migrated JSON files")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    paths = {split: data_dir / f"{split}.json" for split in ("train", "dev", "test")}
    input_hashes = {split: sha256_file(path) for split, path in paths.items()}
    if input_hashes != EXPECTED_INPUT_SHA256:
        raise SystemExit(
            "Input hashes do not match the reviewed frozen dataset revision.\n"
            f"Expected: {EXPECTED_INPUT_SHA256}\nActual:   {input_hashes}"
        )

    migrated: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for split, path in paths.items():
        migrated[split], summaries[split] = migrate_split(split, read_records(path))

    totals = {
        key: sum(int(summary[key]) for summary in summaries.values())
        for key in (
            "records",
            "neutral_implicit_cleared",
            "initial_explicit_implicit_conflicts",
            "independent_explicit_errors_cleared",
            "explicit_implicit_cooccurrences_preserved",
            "fields_renamed",
            "final_implicit_positive",
            "final_implicit_negative",
            "final_implicit_neutral",
            "final_explicit_implicit_conflicts",
        )
    }
    print(json.dumps({"splits": summaries, "totals": totals}, ensure_ascii=False, indent=2))

    if not args.apply:
        print("Dry run only. Re-run with --apply to write the migration.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = data_dir / f"_pre_implicit_sentiment_migration_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for split, path in paths.items():
        shutil.copy2(path, backup_dir / path.name)
        atomic_write_json(path, migrated[split])

    output_hashes = {split: sha256_file(path) for split, path in paths.items()}
    audit = {
        "migration_version": "2.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            NEW_FIELD: (
                "Positive or negative implicit sentiment that may coexist with an "
                "obvious target-specific cue."
            ),
            "constraints": [
                f"{NEW_FIELD}=true requires sentiment in [positive, negative]",
                f"explicit_cue_present and {NEW_FIELD} may both be true",
                "The two fields may both be false and are not logical complements.",
            ],
        },
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "backup_dir": backup_dir.name,
        "splits": summaries,
        "totals": totals,
    }
    atomic_write_json(data_dir / "IMPLICIT_SENTIMENT_MIGRATION_AUDIT.json", audit)
    print(f"Migration applied. Backup: {backup_dir}")
    print(f"Audit: {data_dir / 'IMPLICIT_SENTIMENT_MIGRATION_AUDIT.json'}")
    print(f"Output SHA-256: {output_hashes}")


if __name__ == "__main__":
    main()
