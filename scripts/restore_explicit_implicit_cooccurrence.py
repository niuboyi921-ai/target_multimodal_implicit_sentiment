#!/usr/bin/env python3
"""Restore reviewed polar explicit+implicit co-occurrence from migration audit.

The former migration treated the two tags as mutually exclusive. Under the
current multi-label definition, an obvious cue may coexist with implicit
reasoning (for example, sarcasm). This script restores only records whose
implicit tag was cleared solely by that obsolete mutual-exclusion rule.

It deliberately does not restore:
- neutral implicit labels;
- four records whose explicit tag was independently reviewed as incorrect;
- any field outside the two named reasoning tags.
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


EXPLICIT = "explicit_cue_present"
IMPLICIT = "implicit_sentiment_present"
OLD_IMPLICIT = "implicit_reasoning_required"
SPLITS = ("train", "dev", "test")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def protected_digest(rows: list[dict[str, Any]]) -> str:
    """Hash everything except the two fields this restoration may change."""
    protected = deepcopy(rows)
    for record in protected:
        tags = record.get("reasoning_tags")
        if isinstance(tags, dict):
            tags.pop(EXPLICIT, None)
            tags.pop(IMPLICIT, None)
    payload = json.dumps(
        protected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/twitter2015"))
    parser.add_argument(
        "--migration-audit",
        type=Path,
        default=None,
        help="defaults to <data-dir>/IMPLICIT_SENTIMENT_MIGRATION_AUDIT.json",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    audit_path = (
        args.migration_audit.resolve()
        if args.migration_audit is not None
        else data_dir / "IMPLICIT_SENTIMENT_MIGRATION_AUDIT.json"
    )
    historical = read_json(audit_path)
    if historical.get("migration_version") != "1.0":
        raise ValueError("unsupported historical migration audit")

    paths = {split: data_dir / f"{split}.json" for split in SPLITS}
    original = {split: read_json(path) for split, path in paths.items()}
    restored = {split: deepcopy(rows) for split, rows in original.items()}
    protected_before = {
        split: protected_digest(rows) for split, rows in original.items()
    }
    changes: list[dict[str, Any]] = []
    skipped_independent_review = 0
    skipped_neutral = 0
    already_restored = 0

    for split in SPLITS:
        decisions = historical["splits"][split]["conflict_decisions"]
        for decision in decisions:
            sentiment = str(decision["sentiment"]).lower()
            if sentiment == "neutral":
                skipped_neutral += 1
                continue
            if decision.get("decision") == "clear_explicit":
                skipped_independent_review += 1
                continue
            if decision.get("decision") != "clear_implicit":
                raise ValueError(
                    f"{split}[{decision.get('index')}]: unexpected historical decision"
                )
            if not (
                decision.get("old_explicit_cue_present") is True
                and decision.get(f"old_{OLD_IMPLICIT}") is True
            ):
                raise ValueError(
                    f"{split}[{decision.get('index')}]: historical values were not double-true"
                )

            index = int(decision["index"])
            record = restored[split][index]
            target = record.get("target") or record.get("targe")
            text = record.get("restored_text") or record.get("text")
            if (
                target != decision.get("target")
                or text != decision.get("restored_text")
                or str(record.get("sentiment", "")).lower() != sentiment
            ):
                raise ValueError(f"{split}[{index}]: stable record fingerprint mismatch")

            tags = record.get("reasoning_tags")
            if not isinstance(tags, dict):
                raise TypeError(f"{split}[{index}]: reasoning_tags must be an object")
            before = {EXPLICIT: tags.get(EXPLICIT), IMPLICIT: tags.get(IMPLICIT)}
            if before == {EXPLICIT: True, IMPLICIT: True}:
                already_restored += 1
                continue
            if before != {EXPLICIT: True, IMPLICIT: False}:
                raise ValueError(
                    f"{split}[{index}]: current tag state is not the expected pre-restore state: {before}"
                )
            tags[EXPLICIT] = True
            tags[IMPLICIT] = True
            changes.append(
                {
                    "split": split,
                    "index": index,
                    "target": target,
                    "sentiment": sentiment,
                    "restored_text": text,
                    "before": before,
                    "after": {EXPLICIT: True, IMPLICIT: True},
                    "basis": "historical implicit tag was cleared only by the obsolete mutual-exclusion rule",
                }
            )

    protected_after = {
        split: protected_digest(rows) for split, rows in restored.items()
    }
    if protected_before != protected_after:
        raise AssertionError("a field outside the two authorized tags changed")

    summary = {
        "eligible_changes": len(changes),
        "already_restored": already_restored,
        "skipped_neutral_conflicts": skipped_neutral,
        "skipped_independently_reviewed_explicit_errors": skipped_independent_review,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.apply:
        print("Dry run only. Re-run with --apply to write the restoration.")
        return
    if not changes:
        print("No changes required.")
        return

    input_hashes = {split: sha256_file(path) for split, path in paths.items()}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = data_dir / f"_pre_explicit_implicit_restore_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for split, path in paths.items():
        shutil.copy2(path, backup_dir / path.name)
        atomic_write_json(path, restored[split])

    output_hashes = {split: sha256_file(path) for split, path in paths.items()}
    audit = {
        "restoration_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "explicit_implicit_may_cooccur": True,
            "implicit_neutral_remains_invalid": True,
            "authorized_record_fields": [
                f"reasoning_tags.{EXPLICIT}",
                f"reasoning_tags.{IMPLICIT}",
            ],
        },
        "source_migration_audit": audit_path.name,
        "backup_dir": backup_dir.name,
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "protected_content_sha256": protected_after,
        "summary": summary,
        "changes": changes,
    }
    output_audit = data_dir / "EXPLICIT_IMPLICIT_COOCCURRENCE_RESTORATION_AUDIT.json"
    atomic_write_json(output_audit, audit)
    print(f"Restoration applied. Backup: {backup_dir}")
    print(f"Audit: {output_audit}")


if __name__ == "__main__":
    main()
