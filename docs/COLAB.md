# Google Colab T4 workflow

Use a GPU runtime (`Runtime -> Change runtime type -> T4 GPU`) and keep persistent inputs under:

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
- CPU for sequential simulator rollout;
- 512 games per iteration;
- batch size 512;
- GAE lambda 1.0 while the V2 critic is being established;
- PFSP population sampling with a reduced historical self-play weight;
- checkpoints written to Drive every two iterations.

Colab's CPU still advances the native simulator one game at a time, so GPU capacity does not remove
the rollout bottleneck. Benchmark `POKETCG_ROLLOUT_DEVICE=cuda` before using it: batch-one GPU inference
can be slower than CPU because every game decision introduces a synchronization point.

If the GitHub repository is private, authenticate the clone through a Colab Secret or clone it
manually. Do not put a personal access token directly in the notebook.
