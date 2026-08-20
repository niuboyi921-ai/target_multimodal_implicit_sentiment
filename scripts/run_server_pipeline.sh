#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/twitter2015.yaml}
RUN_ID=${2:-server-$(date -u +%Y%m%d-%H%M%S)}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
PUSH_REPORT=${PUSH_REPORT:-0}

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

DATASET_NAME=$(python -c 'import sys; from pathlib import Path; sys.path.insert(0,"src"); from tmis.config import load_config; c=load_config(Path(sys.argv[1])); print(c["data"]["dataset_name"])' "$CONFIG")
BASE_OUTPUT=$(python -c 'import sys; from pathlib import Path; sys.path.insert(0,"src"); from tmis.config import load_config, resolve_project_path; c=load_config(Path(sys.argv[1])); print(resolve_project_path(c,c["output_dir"]))' "$CONFIG")
OUTPUT_DIR="$BASE_OUTPUT/runs/$RUN_ID"
REPORT_DIR="reports/$DATASET_NAME/$RUN_ID"
mkdir -p "$OUTPUT_DIR"
CONSOLE_LOG="$OUTPUT_DIR/console.log"

export PYTHONUNBUFFERED=1

export_failed_report() {
  exit_code=$?
  failed_id="${RUN_ID}-failed-$(date -u +%Y%m%d-%H%M%S)"
  failed_report_dir="reports/$DATASET_NAME/$failed_id"
  if [[ -f "$CONSOLE_LOG" ]]; then
    tail -n 5000 "$CONSOLE_LOG" > "$OUTPUT_DIR/console_tail.txt" || true
  fi
  python scripts/export_training_report.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --report-dir "$failed_report_dir" \
    --run-id "$failed_id" \
    --status failed || true
  echo "Failure report: $failed_report_dir" >&2
  exit "$exit_code"
}
trap export_failed_report ERR INT TERM

python scripts/validate_data.py --config "$CONFIG"

TRAIN_ARGS=(scripts/train.py --config "$CONFIG" --output-dir "$OUTPUT_DIR" --run-id "$RUN_ID")
if [[ -n ${RESUME_CHECKPOINT:-} ]]; then
  TRAIN_ARGS+=(--resume "$RESUME_CHECKPOINT")
fi
if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "${TRAIN_ARGS[@]}" 2>&1 | tee "$CONSOLE_LOG"
else
  python "${TRAIN_ARGS[@]}" 2>&1 | tee "$CONSOLE_LOG"
fi

python scripts/evaluate.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint "$OUTPUT_DIR/best_joint.pt" \
  --result-tag best_joint \
  --also-write-canonical 2>&1 | tee -a "$CONSOLE_LOG"
python scripts/evaluate.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint "$OUTPUT_DIR/stage5_generated_only.pt" \
  --result-tag generated_only 2>&1 | tee -a "$CONSOLE_LOG"
python scripts/compare_checkpoint_evaluations.py \
  --output-dir "$OUTPUT_DIR" 2>&1 | tee -a "$CONSOLE_LOG"
python scripts/evaluate_auxiliary.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint "$OUTPUT_DIR/best_joint.pt" 2>&1 | tee -a "$CONSOLE_LOG"
tail -n 5000 "$CONSOLE_LOG" > "$OUTPUT_DIR/console_tail.txt"

python scripts/export_training_report.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --report-dir "$REPORT_DIR" \
  --run-id "$RUN_ID" \
  --status completed
python scripts/validate_training_report.py --report-dir "$REPORT_DIR"

trap - ERR INT TERM
if [[ "$PUSH_REPORT" == 1 ]]; then
  bash scripts/push_training_report.sh "$REPORT_DIR"
else
  echo "Report ready at $REPORT_DIR"
  echo "Set PUSH_REPORT=1 to create and push a runs/<run-id> branch automatically."
fi
