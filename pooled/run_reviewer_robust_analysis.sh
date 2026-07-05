#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Usage:
#   bash run_reviewer_robust_analysis.sh smoke
#   bash run_reviewer_robust_analysis.sh full
#
# Smoke mode checks the data paths, preprocessing, XGBoost, lasso, and CausalForestDML
# without spending cluster time on neural nets or BART.

MODE="${1:-full}"
mkdir -p reviewer_robust_outputs logs

export USE_TF_GPU="${USE_TF_GPU:-0}"
export NEURAL_BACKEND="${NEURAL_BACKEND:-sklearn}"
if [[ "$USE_TF_GPU" != "1" ]]; then
  export CUDA_VISIBLE_DEVICES="-1"
  export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
  export TF_XLA_FLAGS="${TF_XLA_FLAGS:---tf_xla_auto_jit=0}"
  export XLA_FLAGS="${XLA_FLAGS:-}"
fi

if [[ "$MODE" == "smoke" ]]; then
  export RUN_NEURAL_SLEARNER=0
  export RUN_BART_SLEARNER=0
  OUT="reviewerRobustAnalysis.smoke.ipynb"
elif [[ "$MODE" == "full" ]]; then
  export RUN_NEURAL_SLEARNER="${RUN_NEURAL_SLEARNER:-1}"
  export RUN_BART_SLEARNER="${RUN_BART_SLEARNER:-1}"
  OUT="reviewerRobustAnalysis.executed.ipynb"
else
  echo "Unknown mode: $MODE. Use smoke or full." >&2
  exit 2
fi

jupyter nbconvert \
  --to notebook \
  --execute reviewerRobustAnalysis.ipynb \
  --output "$OUT" \
  --ExecutePreprocessor.timeout=-1 \
  2>&1 | tee "logs/reviewerRobustAnalysis_${MODE}.log"
