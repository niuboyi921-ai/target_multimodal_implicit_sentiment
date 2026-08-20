#!/usr/bin/env python3
"""Resolve polar records with neither explicit nor implicit sentiment tags.

An explicit cue is accepted only when a clear sentiment/evaluative expression
is directed at the current target entity. Cues about another entity, a general
event, or the surrounding scene do not make the current target explicit.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any


EXPECTED_INPUT_SHA256 = {
    "train": "91c75c17f349b70bc66758edd976d4e60b0402b12bab070a15ddc1a21e9727f0",
    "dev": "355387fbf2daf1c01c053ff8fe3dea0afdbb36cee8af9c327a5a2e4fa884881c",
    "test": "67b5c50c9989e3b6cb01f800912c8285616c772a95610bd0d9b6627810f3a110",
}

# Reviewed records where a clear cue is directed at the current target.
EXPLICIT_DECISIONS: dict[tuple[str, int], dict[str, str]] = {
    ("train", 271): {
        "target": "Barack",
        "cue": "supporting Barack",
        "reason": "The support expression is directly aimed at the current target.",
    },
    ("train", 272): {
        "target": "# Obama",
        "cue": "supporting Barack # Obama",
        "reason": "The support expression is directly aimed at the current target.",
    },
    ("train", 1595): {
        "target": "Serena Williams",
        "cue": "fat underachiever",
        "reason": "The derogatory evaluation is explicitly directed at the current target.",
    },
    ("train", 2009): {
        "target": "Jon",
        "cue": "Thank you Jon",
        "reason": "The expression of gratitude is explicitly directed at the current target.",
    },
    ("dev", 413): {
        "target": "Ted Cruz",
        "cue": "Worst. TedTalk. Ever.",
        "reason": "The visual-text insult is explicitly directed at the current target through the TedTalk pun.",
    },
    ("dev", 611): {
        "target": "# Hillary",
        "cue": "praising # Hillary",
        "reason": "The praise expression explicitly names the current target as its recipient.",
    },
    ("dev", 619): {
        "target": "Obama",
        "cue": "Obama lacks both",
        "reason": "The explicit deficiency judgment is directly asserted about the current target.",
    },
    ("test", 692): {
        "target": "UKIP",
        "cue": "agree with UKIP policy / support if you agree",
        "reason": "The agreement and support stance is explicitly directed at the current target's policy.",
    },
    ("test", 896): {
        "target": "Trump",
        "cue": "Stop supporting Trump",
        "reason": "The imperative explicitly withdraws support from the current target.",
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
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    if path.exists():
        path.chmod(path.stat().st_mode | stat.S_IWRITE)
    temporary.replace(path)


def target_of(record: dict[str, Any]) -> str:
    return str(record.get("target") or record.get("targe") or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/twitter2015"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    paths = {split: data_dir / f"{split}.json" for split in ("train", "dev", "test")}
    input_hashes = {split: sha256_file(path) for split, path in paths.items()}
    if input_hashes != EXPECTED_INPUT_SHA256:
        raise SystemExit(
            "Input hashes do not match the reviewed post-migration dataset.\n"
            f"Expected: {EXPECTED_INPUT_SHA256}\nActual:   {input_hashes}"
        )

    migrated: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    all_decisions: list[dict[str, Any]] = []

    for split, path in paths.items():
        rows = read_records(path)
        before = deepcopy(rows)
        gap_indices: list[int] = []
        explicit_count = 0
        implicit_count = 0

        for index, record in enumerate(rows):
            tags = record.get("reasoning_tags")
            if not isinstance(tags, dict):
                raise TypeError(f"{split}[{index}]: reasoning_tags must be an object")
            sentiment = str(record.get("sentiment", "")).strip().lower()
            is_gap = (
                sentiment in {"positive", "negative"}
                and tags.get("explicit_cue_present") is False
                and tags.get("implicit_sentiment_present") is False
            )
            if not is_gap:
                continue

            gap_indices.append(index)
            explicit = EXPLICIT_DECISIONS.get((split, index))
            if explicit is not None:
                if target_of(record) != explicit["target"]:
                    raise ValueError(f"{split}[{index}]: explicit decision fingerprint mismatch")
                tags["explicit_cue_present"] = True
                decision = "explicit_cue_present=true"
                cue = explicit["cue"]
                reason = explicit["reason"]
                explicit_count += 1
            else:
                tags["implicit_sentiment_present"] = True
                decision = "implicit_sentiment_present=true"
                cue = None
                reason = (
                    "No clear sentiment/evaluative expression is directed at the current target; "
                    "the polarity is inferred from an event, association, recommendation, or cue "
                    "aimed at another entity."
                )
                implicit_count += 1

            all_decisions.append(
                {
                    "split": split,
                    "index": index,
                    "target": target_of(record),
                    "sentiment": sentiment,
                    "restored_text": record.get("restored_text") or record.get("text"),
                    "decision": decision,
                    "target_specific_explicit_cue": cue,
                    "reason": reason,
                }
            )

        # Prove that no field outside the two requested tags changed.
        for index, (old, new) in enumerate(zip(before, rows)):
            old_copy = deepcopy(old)
            new_copy = deepcopy(new)
            old_tags = old_copy["reasoning_tags"]
            new_tags = new_copy["reasoning_tags"]
            old_tags.pop("explicit_cue_present")
            old_tags.pop("implicit_sentiment_present")
            new_tags.pop("explicit_cue_present")
            new_tags.pop("implicit_sentiment_present")
            if old_copy != new_copy:
                raise AssertionError(f"{split}[{index}]: a non-target field changed")

        remaining = sum(
            str(record.get("sentiment", "")).lower() in {"positive", "negative"}
            and not record["reasoning_tags"]["explicit_cue_present"]
            and not record["reasoning_tags"]["implicit_sentiment_present"]
            for record in rows
        )
        if remaining:
            raise AssertionError(f"{split}: {remaining} polar tag gaps remain")

        migrated[split] = rows
        summaries[split] = {
            "records": len(rows),
            "polar_tag_gaps_reviewed": len(gap_indices),
            "set_explicit_true": explicit_count,
            "set_implicit_true": implicit_count,
            "remaining_polar_tag_gaps": remaining,
        }

    reviewed_keys = {(item["split"], item["index"]) for item in all_decisions}
    unused_explicit = set(EXPLICIT_DECISIONS) - reviewed_keys
    if unused_explicit:
        raise AssertionError(f"explicit decisions were not matched: {sorted(unused_explicit)}")

    totals = {
        key: sum(summary[key] for summary in summaries.values())
        for key in (
            "polar_tag_gaps_reviewed",
            "set_explicit_true",
            "set_implicit_true",
            "remaining_polar_tag_gaps",
        )
    }
    preview = {"splits": summaries, "totals": totals, "decisions": all_decisions}
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    if not args.apply:
        print("Dry run only. Re-run with --apply to write the reviewed decisions.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = data_dir / f"_pre_polar_tag_gap_resolution_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for split, path in paths.items():
        shutil.copy2(path, backup_dir / path.name)
        atomic_write_json(path, migrated[split])

    output_hashes = {split: sha256_file(path) for split, path in paths.items()}
    audit = {
        "audit_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "explicit_definition": (
            "A clear sentiment/evaluative expression must be directed at the current target entity; "
            "a strong tendency or a cue directed at another entity is not explicit for this target."
        ),
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "backup_dir": backup_dir.name,
        **preview,
        "non_tag_fields_changed": 0,
    }
    atomic_write_json(data_dir / "POLAR_TAG_GAP_AUDIT.json", audit)
    print(f"Resolution applied. Backup: {backup_dir}")
    print(f"Audit: {data_dir / 'POLAR_TAG_GAP_AUDIT.json'}")
    print(f"Output SHA-256: {output_hashes}")


if __name__ == "__main__":
    main()
