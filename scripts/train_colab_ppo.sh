#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

INPUT_CHECKPOINT="${1:-/content/drive/MyDrive/pokemonTCG/checkpoints/bc_rule_v2_transformer_2000.pt}"
OUTPUT_CHECKPOINT="${2:-/content/drive/MyDrive/pokemonTCG/checkpoints/ppo_v2_parallel_colab.pt}"
ROLLOUT_DEVICE="${POKETCG_ROLLOUT_DEVICE:-cpu}"
ROLLOUT_WORKERS="${POKETCG_ROLLOUT_WORKERS:-8}"
ITERATIONS="${POKETCG_ITERATIONS:-40}"
GAMES_PER_ITERATION="${POKETCG_GAMES_PER_ITERATION:-512}"
LEARNING_RATE="${POKETCG_LEARNING_RATE:-0.00005}"
GAE_LAMBDA="${POKETCG_GAE_LAMBDA:-0.95}"
VALUE_GAE_LAMBDA="${POKETCG_VALUE_GAE_LAMBDA:-0.95}"
REWARD_SHAPING="${POKETCG_REWARD_SHAPING:-none}"
REWARD_SHAPING_SCALE="${POKETCG_REWARD_SHAPING_SCALE:-1.0}"
POOL_CHECKPOINT="${POKETCG_POOL_CHECKPOINT:-}"
POOL_CHECKPOINT_WEIGHT="${POKETCG_POOL_CHECKPOINT_WEIGHT:-0.35}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-pokemon-tcg-ai-battle}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-ppo-v2-parallel-8w}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

if [[ ! -f "$INPUT_CHECKPOINT" ]]; then
  echo "Missing input checkpoint: $INPUT_CHECKPOINT" >&2
  exit 2
fi

if [[ -n "$POOL_CHECKPOINT" && ! -f "$POOL_CHECKPOINT" ]]; then
  echo "Missing pool checkpoint: $POOL_CHECKPOINT" >&2
  exit 4
fi

mkdir -p "$(dirname "$OUTPUT_CHECKPOINT")"

if [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is missing. Add it as a private Colab Secret." >&2
  exit 3
fi

WANDB_ARGS=(
  --wandb-mode "$WANDB_MODE"
  --wandb-project "$WANDB_PROJECT"
  --wandb-run-name "$WANDB_RUN_NAME"
)
if [[ -n "$WANDB_ENTITY" ]]; then
  WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

POOL_ARGS=()
if [[ -n "$POOL_CHECKPOINT" ]]; then
  POOL_ARGS+=(
    --pool-checkpoint "$POOL_CHECKPOINT"
    --pool-checkpoint-weight "$POOL_CHECKPOINT_WEIGHT"
  )
fi

python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is not available"; print(torch.cuda.get_device_name(0))'

python -m poketcg.rl.train_ppo \
  --input "$INPUT_CHECKPOINT" \
  --output "$OUTPUT_CHECKPOINT" \
  --iterations "$ITERATIONS" \
  --games-per-iteration "$GAMES_PER_ITERATION" \
  --ppo-epochs 4 \
  --batch-size 512 \
  --learning-rate "$LEARNING_RATE" \
  --gamma 1.0 \
  --gae-lambda "$GAE_LAMBDA" \
  --value-gae-lambda "$VALUE_GAE_LAMBDA" \
  --reward-shaping "$REWARD_SHAPING" \
  --reward-shaping-scale "$REWARD_SHAPING_SCALE" \
  --device cuda \
  --rollout-device "$ROLLOUT_DEVICE" \
  --rollout-workers "$ROLLOUT_WORKERS" \
  --worker-torch-threads 1 \
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
  --seed 20260721 \
  "${POOL_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
