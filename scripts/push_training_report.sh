#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash scripts/push_training_report.sh REPORT_DIR [REMOTE] [BRANCH]"
  echo "Environment: NO_PUSH=1 creates the local result commit without pushing it."
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage
  exit 2
fi

REPORT_INPUT=$1
REMOTE=${2:-origin}

PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
  echo "Not inside a Git repository." >&2
  exit 1
}
cd "$PROJECT_ROOT"
git rev-parse --verify HEAD >/dev/null 2>&1 || {
  echo "The repository has no initial commit. Publish the project before pushing reports." >&2
  exit 1
}
START_BRANCH=$(git branch --show-current)
START_COMMIT=$(git rev-parse HEAD)
restore_start_ref() {
  if [[ -n "$START_BRANCH" ]]; then
    git switch "$START_BRANCH" >/dev/null 2>&1 || true
  else
    git switch --detach "$START_COMMIT" >/dev/null 2>&1 || true
  fi
}
trap restore_start_ref EXIT

REPORT_ABS=$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$REPORT_INPUT")
REPORT_REL=$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve().relative_to(Path(sys.argv[2]).resolve()).as_posix())' "$REPORT_ABS" "$PROJECT_ROOT") || {
  echo "Report directory must be inside the repository." >&2
  exit 1
}
if [[ "$REPORT_REL" != reports/* ]]; then
  echo "Only reports/ directories may be committed by this script." >&2
  exit 1
fi

python scripts/validate_training_report.py --report-dir "$REPORT_ABS"

if [[ ${NO_PUSH:-0} != 1 ]]; then
  git remote get-url "$REMOTE" >/dev/null 2>&1 || {
    echo "Git remote is not configured: $REMOTE" >&2
    exit 1
  }
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Tracked source changes are present. Commit or stash them before publishing a report." >&2
  exit 1
fi

RUN_ID=$(python -c 'import json, pathlib, sys; print(json.loads((pathlib.Path(sys.argv[1]) / "run_manifest.json").read_text(encoding="utf-8"))["run_id"])' "$REPORT_ABS")
SAFE_RUN_ID=$(python -c 'import re, sys; print(re.sub(r"[^A-Za-z0-9._-]+", "-", sys.argv[1]).strip("-"))' "$RUN_ID")
BRANCH=${3:-runs/$SAFE_RUN_ID}

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "Local branch already exists: $BRANCH" >&2
  exit 1
fi

git switch -c "$BRANCH"
git add -- "$REPORT_REL"

STAGED=$(git diff --cached --name-only)
if [[ -z "$STAGED" ]]; then
  echo "No report files were staged." >&2
  exit 1
fi
while IFS= read -r path; do
  if [[ "$path" != reports/* ]]; then
    echo "Refusing to commit non-report path: $path" >&2
    exit 1
  fi
done <<< "$STAGED"

git commit -m "Add training report $RUN_ID"
if [[ ${NO_PUSH:-0} == 1 ]]; then
  echo "Created local branch $BRANCH without pushing."
else
  git push --set-upstream "$REMOTE" "$BRANCH"
  echo "Pushed $BRANCH. Open a pull request for GPT/Codex analysis."
fi
