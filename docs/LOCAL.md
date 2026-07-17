# Local Mac workflow

Use this branch for simulator integration, feature changes, tests, and short MPS smoke runs. Formal
PPO training belongs on the `colab` branch.

```bash
conda activate poketcg
python -m pip install -e ".[dev]"
pytest -q
ruff check src tests
```

Run a small V2 PPO smoke test from an existing checkpoint:

```bash
./scripts/train_local_smoke.sh \
  artifacts/checkpoints/bc_rule_v2_transformer_2000.pt
```

The official simulator remains on CPU because each battle is sequential and its native library is
loaded by the host process. Batched PPO updates use MPS. Outputs stay under ignored `artifacts/` and
`results/` directories.
