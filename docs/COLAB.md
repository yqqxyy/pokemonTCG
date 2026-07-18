# Google Colab T4 workflow

Use a High-RAM GPU runtime (`Runtime -> Change runtime type -> T4 GPU`) and keep persistent
inputs under:

```text
MyDrive/pokemonTCG/
├── pokemon-tcg-ai-battle.zip
└── checkpoints/
    └── bc_rule_v2_transformer_2000.pt
```

The competition archive is restricted and remains in Drive. The notebook extracts only
`sample_submission/` into the ignored local `data/official/` directory. Never commit the archive,
native libraries, Kaggle credentials, datasets, or checkpoints.

Open [`notebooks/colab_ppo.ipynb`](../notebooks/colab_ppo.ipynb) in Colab and run it top to bottom.
The default training command uses:

- T4/CUDA for PPO updates;
- eight spawned CPU workers, each with an isolated official simulator;
- 512 games per iteration;
- batch size 512;
- GAE lambda 1.0 while the V2 critic is being established;
- PFSP population sampling with a reduced historical self-play weight;
- checkpoints written to Drive every two iterations;
- live experiment and system metrics in a W&B project named `pokemon-tcg-ai-battle`.

## W&B authentication

Create a new private Colab Secret named `WANDB_API_KEY`, paste the key into the secret value, and
grant notebook access. Never paste the key into a code or text cell. The notebook copies the secret
into the process environment without displaying it. The training script exits before training if
online logging is enabled and the environment variable is missing.

Only the parent training process initializes W&B. Rollout workers return metrics to the parent and
never authenticate or contact W&B. The dashboard records PPO losses, entropy, approximate KL, clip
fraction, explained variance, gradient norm, win rates, opponent sampling weights, rollout/update
timings, throughput, and W&B's automatic CPU/GPU system metrics.

Set `WANDB_MODE=offline` to keep local W&B logs without network sync, or `WANDB_MODE=disabled` to turn
tracking off:

```bash
WANDB_MODE=disabled bash scripts/train_colab_ppo.sh INPUT.pt OUTPUT.pt
```

## Parallel rollout

The native simulator holds battle state inside its process. The trainer therefore uses the safe
`spawn` start method rather than threads or `fork`, initializes one simulator per worker, and limits
each worker to one Torch/BLAS thread. The main process deterministically assigns PFSP opponents and
merges completed game trajectories before the CUDA PPO update.

Eight workers are appropriate for an eight-vCPU High-RAM runtime. Override the default when Colab
assigns a different CPU count:

```bash
POKETCG_ROLLOUT_WORKERS=4 bash scripts/train_colab_ppo.sh INPUT.pt OUTPUT.pt
```

Compare `performance/games_per_second` for 1, 2, 4, and 8 workers. More workers are not necessarily
faster once CPU cores, memory bandwidth, model replication, or process communication becomes the
bottleneck. Keep `POKETCG_ROLLOUT_DEVICE=cpu`: batch-one CUDA inference can be slower because every
game decision adds a synchronization point.

## Prize-progress potential shaping

Potential-based shaping is optional and disabled by default. Enable it for an experiment with:

```bash
POKETCG_REWARD_SHAPING=prize \
POKETCG_REWARD_SHAPING_SCALE=1.0 \
POKETCG_GAE_LAMBDA=0.95 \
WANDB_RUN_NAME=ppo-v2-prize-pbrs \
bash scripts/train_colab_ppo.sh INPUT.pt OUTPUT.pt
```

For learner `i`, the potential is the normalized relative prize progress
`Phi(s) = (opponent_prizes_remaining - own_prizes_remaining) / 6`. The shaped reward is
`scale * (gamma * Phi(next_state) - Phi(state))`; terminal potential is forced to zero. This gives
the actor immediate credit when prize progress improves while preserving the original optimal
policy under the usual potential-shaping assumptions. The critic deliberately keeps the unshaped
terminal win/loss target, so its value remains interpretable on `[-1, 1]`.

W&B adds `reward/mean_abs_shaping_reward` and `reward/mean_shaping_return`. The former should be
positive when prize counts change; with `gamma=1`, the latter should remain close to zero across
many games because potential differences telescope. Use the same starting checkpoint and seed with
`POKETCG_REWARD_SHAPING=none` for a controlled A/B comparison. The script also accepts
`POKETCG_ITERATIONS` and `POKETCG_GAMES_PER_ITERATION` overrides.

## Value calibration and trajectories

Run the diagnostic against the same deterministic RuleAgent panel used for checkpoint comparison:

```bash
python -m poketcg.rl.value_diagnostics \
  --checkpoint /content/drive/MyDrive/pokemonTCG/checkpoints/MODEL.pt \
  --opponent rule --games-per-seat 500 --device cpu \
  --output /content/drive/MyDrive/pokemonTCG/results/MODEL_value_rule500.json
```

The command writes the aggregate JSON plus `MODEL_value_rule500_trajectories.jsonl`. Interpret the
main fields as follows:

- lower `mae`, `rmse`, `brier_score`, and `expected_calibration_error` are better;
- positive correlation and explained variance indicate that values rank eventual outcomes;
- `calibration_slope` near one and `calibration_intercept` near zero indicate calibrated scale;
- a slope below one usually indicates over-dispersed/overconfident predictions, while a slope above
  one indicates predictions that are too compressed;
- late-game diagnostics should normally be stronger than early-game diagnostics; `setup` is kept
  separate because the official engine reports empty prize lists before cards are dealt;
- `mean_net_change_toward_outcome` should be positive, and high-confidence wrong-sign states should
  be uncommon;
- large Player 0/1 gaps reveal seat-conditioned bias even when aggregate metrics look acceptable.

Calibration rows are decision-weighted, so long games contribute more states. The `per_game` section
is included to check that a result is not caused only by a few unusually long trajectories. Add
`--stochastic` only when deliberately diagnosing the sampled training policy rather than reproducing
the deterministic fixed panel.

If the GitHub repository is private, authenticate the clone through a Colab Secret or clone it
manually. Do not put a personal access token directly in the notebook.
