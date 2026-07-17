#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

INPUT_CHECKPOINT="${1:-artifacts/checkpoints/bc_rule_v2_transformer_2000.pt}"
OUTPUT_CHECKPOINT="${2:-artifacts/checkpoints/ppo_v2_local_smoke.pt}"

if [[ ! -f "$INPUT_CHECKPOINT" ]]; then
  echo "Missing input checkpoint: $INPUT_CHECKPOINT" >&2
  echo "Pass a checkpoint path as the first argument." >&2
  exit 2
fi

python -m poketcg.rl.train_ppo \
  --input "$INPUT_CHECKPOINT" \
  --output "$OUTPUT_CHECKPOINT" \
  --iterations 2 \
  --games-per-iteration 32 \
  --ppo-epochs 1 \
  --batch-size 64 \
  --learning-rate 0.00005 \
  --device mps \
  --rollout-device cpu \
  --opponent rule \
  --checkpoint-every 1 \
  --seed 20260720
