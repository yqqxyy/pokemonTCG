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
- Actor and critic GAE lambda 0.95, independently configurable for ablations;
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
POKETCG_VALUE_GAE_LAMBDA=0.95 \
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
`POKETCG_ITERATIONS`, `POKETCG_GAMES_PER_ITERATION`, and `POKETCG_LEARNING_RATE` overrides.

Add one fixed historical policy to the population without expanding the full Python command:

```bash
POKETCG_POOL_CHECKPOINT=/content/drive/MyDrive/pokemonTCG/checkpoints/OLD_POLICY.pt \
POKETCG_POOL_CHECKPOINT_WEIGHT=0.35 \
POKETCG_LEARNING_RATE=0.00003 \
WANDB_RUN_NAME=ppo-v3-actions-v2-vs-old-policy \
bash scripts/train_colab_ppo.sh INPUT.pt OUTPUT.pt
```

If `POKETCG_POOL_CHECKPOINT` is unset, no extra historical checkpoint is added and the script keeps
its previous behavior. A configured path is validated before training starts.

## Decoupled actor and critic lambda

The Colab script defaults to `POKETCG_GAE_LAMBDA=0.95` for policy advantages and
`POKETCG_VALUE_GAE_LAMBDA=0.95` for critic targets. A completed ablation with critic lambda 1.0
produced noisier early-game targets and did not improve the fixed panel, so it is no longer the
recommended default. To reproduce that Monte Carlo-target ablation explicitly:

```bash
POKETCG_REWARD_SHAPING=none \
POKETCG_GAE_LAMBDA=0.95 \
POKETCG_VALUE_GAE_LAMBDA=1.0 \
WANDB_RUN_NAME=ppo-v2-terminal-value-mc \
bash scripts/train_colab_ppo.sh INPUT_ITER0030.pt OUTPUT.pt
```

The Python CLI remains backward compatible: if `--value-gae-lambda` is omitted, it inherits
`--gae-lambda`. Checkpoint `ppo_config` records both resolved numeric values.

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

## Decision coverage and input audit

Before changing the model, measure which official decisions the neural policy actually controls:

```bash
python -m poketcg.rl.coverage_diagnostics \
  --checkpoint /content/drive/MyDrive/pokemonTCG/checkpoints/MODEL.pt \
  --opponent rule --games-per-seat 500 --device cpu \
  --output /content/drive/MyDrive/pokemonTCG/results/MODEL_coverage_rule500.json
```

The report separates decisions into `forced` (only one legal result), `neural` (handled by the
policy), and `resolver` (handled by the deterministic fallback). Action Space V1 checkpoints leave
multi-select decisions in `resolver`; V2 checkpoints should reduce `resolver` to zero.
`strategic_neural_coverage` excludes forced choices and is the coverage number to watch.
Context-level win rates are observational diagnostics, not causal estimates. The command also
audits card/attack ID ranges, the active deck, and public engine logs; its companion
`_records.jsonl` file supports deeper analysis without another rollout.

## Train FeatureEncoder V3 before PPO

V3 adds runtime-derived structured card/attack semantics and up to 32 recent visible engine-event
tokens. Collect a fresh dataset because V1/V2 JSONL files do not contain these fields:

```bash
mkdir -p /content/drive/MyDrive/pokemonTCG/bc
python -m poketcg.rl.collect_bc \
  --games 2000 --encoder-version 3 --include-multiselect --seed 20260724 \
  --output /content/drive/MyDrive/pokemonTCG/bc/rule_v3_semantic_actions_v2_2000.jsonl
```

Train the selected semantic-only V3 checkpoint with Action Space V2 on T4:

```bash
python -m poketcg.rl.train_bc \
  --input /content/drive/MyDrive/pokemonTCG/bc/rule_v3_semantic_actions_v2_2000.jsonl \
  --output /content/drive/MyDrive/pokemonTCG/checkpoints/bc_rule_v3_semantic_actions_v2_2000.pt \
  --epochs 10 --batch-size 64 --learning-rate 0.0002 \
  --hidden-size 256 --model-type transformer_v3 \
  --num-layers 3 --num-heads 4 --dropout 0.1 \
  --disable-history \
  --device cuda --seed 20260724
```

The same JSONL supports all four controlled ablations: full V3, no semantics, no history, or neither.
Change the two disable flags and output filename as needed. Evaluate them on the same paired-seat
panel before PPO. Put the selected V3 checkpoint into the notebook's `input_checkpoint`; the PPO
trainer and spawned workers infer encoder V3 automatically from its model configuration.

`--include-multiselect` enables Action Space V2. It learns a cardinality-constrained set
distribution instead of enumerating every legal subset, and the same exact set log-probability is
used by BC, rollout workers, and PPO. Single-choice behavior is unchanged. Start a new PPO line from
this expanded BC checkpoint; old checkpoints intentionally remain Action Space V1 for reproducible
comparisons.

If the GitHub repository is private, authenticate the clone through a Colab Secret or clone it
manually. Do not put a personal access token directly in the notebook.
