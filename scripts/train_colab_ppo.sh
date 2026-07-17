#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

INPUT_CHECKPOINT="${1:-/content/drive/MyDrive/pokemonTCG/checkpoints/bc_rule_v2_transformer_2000.pt}"
OUTPUT_CHECKPOINT="${2:-/content/drive/MyDrive/pokemonTCG/checkpoints/ppo_v2_colab.pt}"
ROLLOUT_DEVICE="${POKETCG_ROLLOUT_DEVICE:-cpu}"

if [[ ! -f "$INPUT_CHECKPOINT" ]]; then
  echo "Missing input checkpoint: $INPUT_CHECKPOINT" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_CHECKPOINT")"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is not available"; print(torch.cuda.get_device_name(0))'

python -m poketcg.rl.train_ppo \
  --input "$INPUT_CHECKPOINT" \
  --output "$OUTPUT_CHECKPOINT" \
  --iterations 40 \
  --games-per-iteration 512 \
  --ppo-epochs 4 \
  --batch-size 512 \
  --learning-rate 0.00005 \
  --gamma 1.0 \
  --gae-lambda 1.0 \
  --device cuda \
  --rollout-device "$ROLLOUT_DEVICE" \
  --opponent population \
  --random-weight 0.1 \
  --rule-weight 0.5 \
  --initial-policy-weight 0.35 \
  --self-play-weight 0.35 \
  --snapshot-every 5 \
  --max-snapshots 4 \
  --adaptive-sampling win_rate \
  --adaptive-alpha 1.0 \
  --adaptive-min-multiplier 0.1 \
  --adaptive-ema-decay 0.95 \
  --adaptive-warmup-games 64 \
  --checkpoint-every 2 \
  --seed 20260721
