# Training reports

This directory contains lightweight, reviewable evidence exported from server training runs. It intentionally does not contain model checkpoints or raw model caches.

## Layout

```text
reports/<dataset>/<run-id>/
├── run_manifest.json
├── RUN_SUMMARY.md
├── config.yaml
├── environment.json
├── dataset_summary.json
├── learning_curves.csv
├── checkpoint_manifest.json
└── artifacts/
    ├── run_state.json
    ├── *_epoch_*_train.json
    ├── *metrics*.json
    └── *predictions*.json
```

`run_manifest.json` is the authoritative index. Every listed artifact has a size and SHA256 digest. `checkpoint_manifest.json` records checkpoint names and sizes, but checkpoint binaries stay outside Git.

## Export and validation

```bash
python scripts/export_training_report.py \
  --config configs/twitter2015.yaml \
  --run-id twitter2015-run-001

python scripts/validate_training_report.py \
  --report-dir reports/twitter2015/twitter2015-run-001
```

Add `--hash-checkpoints` only when a full checkpoint SHA256 is required. Hashing multi-gigabyte files can take several minutes but never copies them into the report.

Console output is excluded by default because it may contain server paths or sensitive values. After manually reviewing `console_tail.txt`, pass `--include-console-tail` if it is genuinely needed for diagnosis.

## Interpretation rules

- JSON metrics and loss values are observed outputs.
- Architectural claims must be verified against the training commit recorded in `run_manifest.json`.
- A missing metric is unknown, not zero.
- Failed and interrupted runs are retained when useful for diagnosis, but must not be compared as completed experiments.
- Do not use checkpoint size or modification time as a proxy for model quality.
