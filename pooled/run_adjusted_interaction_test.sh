#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Usage:
#   bash run_adjusted_interaction_test.sh smoke
#   bash run_adjusted_interaction_test.sh full
#
# Smoke mode restricts to a single dataset (eICU) via SMOKE_DATASETS to validate paths and
# output generation quickly. Full mode runs all 4 datasets (the spec'd default).

MODE="${1:-full}"
mkdir -p logs

if [[ "$MODE" == "smoke" ]]; then
  export SMOKE_DATASETS="${SMOKE_DATASETS:-eICU}"
  OUT="adjustedInteractionTest.smoke.ipynb"
elif [[ "$MODE" == "full" ]]; then
  export SMOKE_DATASETS="${SMOKE_DATASETS:-eICU,PMAP,MIMIC-IV,HYPERION}"
  OUT="adjustedInteractionTest.executed.ipynb"
else
  echo "Unknown mode: $MODE. Use smoke or full." >&2
  exit 2
fi

python _build_adjusted_interaction_test.py

jupyter nbconvert \
  --to notebook \
  --execute adjustedInteractionTest.ipynb \
  --output "$OUT" \
  --ExecutePreprocessor.timeout=-1 \
  2>&1 | tee "logs/adjustedInteractionTest_${MODE}.log"
