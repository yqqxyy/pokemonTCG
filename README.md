# PokéAgent

Pokémon TCG AI Battle Challenge 的本地 Agent 开发与评估项目。当前包含 Random、
RuleAgent V1、软标签 Behavior Cloning，以及从 BC checkpoint 继续训练的 Masked PPO。

## Git 分支约定

- `main`：稳定的模型、评测与竞赛提交代码；只合入通过测试且可以复现的版本。
- `local`：Mac/MPS 小规模调试、冒烟训练和性能诊断。
- `colab`：Google Colab T4/CUDA 正式训练入口与运行配置。

数据集、官方模拟器、Kaggle token 和模型 checkpoint 不进入普通 Git 历史。模型文件建议保存到
Google Drive 或 GitHub Release，并在代码中记录训练配置与评测结果。详细同步方式见
[`docs/BRANCH_WORKFLOW.md`](docs/BRANCH_WORKFLOW.md)。

## 目录

```text
pokemonTCG/
├── data/
│   ├── raw/                  # Kaggle 原始压缩包，不提交
│   └── official/             # 官方引擎与 sample submission，不提交
├── src/poketcg/
│   ├── agents/                # Agent 实现
│   ├── engine.py              # 官方模拟器适配层
│   ├── match.py               # 单局对战逻辑
│   └── cli.py                 # 批量评估命令
├── tests/
├── environment.yml
└── pyproject.toml
```

`data/official` 中的竞赛引擎受官方许可条款约束，仅用于本竞赛，不要上传到 GitHub。

## 环境

已创建过环境时：

```bash
conda activate poketcg
python -m pip install -e ".[dev]"
```

从零复现时：

```bash
conda env create -f environment.yml
conda activate poketcg
```

## 运行 Random vs Random baseline

单局：

```bash
poketcg-evaluate --games 1 --seed 42
```

批量评估，并保存每局结果：

```bash
poketcg-evaluate --games 100 --seed 42 --output results/random_vs_random.jsonl
```

RuleAgent V1 对战 Random Agent：

```bash
poketcg-evaluate --games 100 --seed 42 --player0 rule --player1 random \
  --output results/rule_vs_random.jsonl
```

`--player0` 和 `--player1` 当前支持 `random`、`rule` 与 `bc`。为避免先后手造成误判，正式比较时应
再交换双方位置运行一组。

## RL 起步：Behavior Cloning

采集 RuleAgent 的单选决策。数据会对所有同分最优动作保存均匀软标签：

```bash
python -m poketcg.rl.collect_bc --games 2000 \
  --output artifacts/bc/rule_v1_soft_2000.jsonl
```

训练候选动作 Policy 和分类式 Value：

```bash
python -m poketcg.rl.train_bc \
  --input artifacts/bc/rule_v1_soft_2000.jsonl \
  --output artifacts/checkpoints/bc_rule_v1_soft_2000.pt \
  --epochs 15 --device mps
```

将训练后的 Policy 接回官方引擎：

```bash
python -m poketcg.cli --games 100 --player0 bc --player1 random \
  --checkpoint artifacts/checkpoints/bc_rule_v1_soft_2000.pt
```

软标签 Policy 可以在评测时按概率采样：

```bash
python -m poketcg.rl.evaluate_panel \
  --checkpoint artifacts/checkpoints/bc_rule_v1_soft_2000.pt \
  --games-per-seat 300 --stochastic \
  --output results/evaluation/bc_rule_v1_soft_2000_stochastic.json
```

## FeatureEncoder V2 + token attention

V1 数据只保存了压缩标量，无法恢复完整卡牌信息，因此 V2 必须从官方引擎重新采集：

```bash
python -m poketcg.rl.collect_bc \
  --games 2000 --encoder-version 2 --seed 20260720 \
  --output artifacts/bc/rule_v2_tokens_soft_2000.jsonl
```

V2 只编码当前玩家可见的信息：自己的手牌、双方 Active/Bench、弃牌、可见奖品、场地、
looking/deck selection、附着能量、工具和进化链。对手手牌和盖着的奖品不会进入 token。
每个 token 带有 card ID、区域、相对 owner、场上 slot、卡牌/能量类型、弱点、抗性和动态状态。

训练 3 层、256 hidden、4 head 的 token Transformer：

```bash
python -m poketcg.rl.train_bc \
  --input artifacts/bc/rule_v2_tokens_soft_2000.jsonl \
  --output artifacts/checkpoints/bc_rule_v2_transformer_2000.pt \
  --epochs 10 --batch-size 64 --learning-rate 0.0002 \
  --hidden-size 256 --model-type transformer_v2 \
  --num-layers 3 --num-heads 4 --dropout 0.1 \
  --device mps --seed 20260720
```

该模型有 4,110,694 个参数；训练会按 token 长度分桶，减少 attention padding。V1/V2
checkpoint 由同一套 Agent、PPO 和诊断入口自动识别。软标签 BC 应优先使用 stochastic 评测；
deterministic argmax 会固定选择大量同分动作中的一个，行为可能明显偏离 RuleAgent 的随机
tie-breaking。当前 V2 全量 Value MAE 为 0.652，V1 BC 为约 0.842；V2 PPO 管线已通过冒烟测试。

用 GAE + Masked PPO 继续对 RuleAgent 微调：

```bash
python -m poketcg.rl.train_ppo \
  --input artifacts/checkpoints/bc_rule_v1_soft_2000.pt \
  --output artifacts/checkpoints/ppo_rule_v1_soft_20.pt \
  --iterations 20 --games-per-iteration 128 \
  --device mps --rollout-device cpu \
  --checkpoint-every 5 --seed 20260717
```

`--device` 是 PPO 批量更新设备；`--rollout-device` 是官方引擎逐动作推理设备。当前小模型在
CPU 上逐动作推理通常更快，而 MPS 适合批量反向传播。训练会额外保存每 5 轮 checkpoint，
不要仅凭单轮 rollout 回报选择模型，应使用固定对手面板复评。

评估 checkpoint 的离线指标和固定对手面板：

```bash
python -m poketcg.rl.diagnose \
  --checkpoint artifacts/checkpoints/ppo_rule_v1_soft_20_iter0020.pt \
  --dataset artifacts/bc/rule_v1_soft_2000.jsonl \
  --output results/diagnostics/ppo_rule_v1_soft_20_iter0020.json

python -m poketcg.rl.evaluate_panel \
  --checkpoint artifacts/checkpoints/ppo_rule_v1_soft_20_iter0020.pt \
  --games-per-seat 500 --seed 20260717 \
  --output results/evaluation/ppo_rule_v1_soft_20_iter0020_final500.json
```

当前固定面板（每个对手、每个座位各 500 局）的结果：软标签 BC 对 RuleAgent 的双座位
平均胜率为 49.6%，PPO iter20 为 58.0%；对 Random 分别为 90.6% 和 90.3%。

## Population / historical self-play PPO

从固定 RuleAgent PPO 继续训练一个混合对手池：

```bash
python -m poketcg.rl.train_ppo \
  --input artifacts/checkpoints/ppo_rule_v1_soft_20_iter0020.pt \
  --output artifacts/checkpoints/ppo_population_v1_30.pt \
  --iterations 30 --games-per-iteration 128 \
  --device mps --rollout-device cpu \
  --opponent population \
  --random-weight 0.1 --rule-weight 0.4 \
  --initial-policy-weight 0.25 --self-play-weight 0.75 \
  --snapshot-every 5 --max-snapshots 4 \
  --checkpoint-every 5 --seed 20260718
```

`self-play-weight` 是所有历史快照共享的总权重，而不是每个快照的权重；保留的快照会均分该
权重，超过 `max-snapshots` 时淘汰最旧版本。每轮日志中的 `opponents` 会分别报告各池成员的
局数、胜负和平均回报。还可以重复传入 `--pool-checkpoint PATH` 加入额外历史策略。

checkpoint 交叉评测会自动交换先后手：

```bash
python -m poketcg.rl.evaluate_panel \
  --checkpoint artifacts/checkpoints/ppo_population_v1_30_iter0025.pt \
  --policy-opponent artifacts/checkpoints/ppo_rule_v1_soft_20_iter0020.pt \
  --games-per-seat 500 --seed 20260717 \
  --output results/evaluation/ppo_population_v1_30_iter0025_final500.json
```

当前选中的 population iter25 在 deterministic 固定面板中，对 RuleAgent 双座位平均 59.4%，
对 Random 平均 88.7%；与训练起点 PPO 交叉对战平均为 50.0%。它增加了策略多样性并略微提高
Rule 对局，但尚未形成对起点策略的显著支配，因此后续应保留固定面板做 checkpoint 选择。

### PFSP 自适应采样

扩大到每轮 256 局、降低历史 self-play 权重，并按对手 EMA 胜率动态采样：

```bash
python -m poketcg.rl.train_ppo \
  --input artifacts/checkpoints/ppo_population_v1_30_iter0025.pt \
  --output artifacts/checkpoints/ppo_population_v2_adaptive_30.pt \
  --iterations 30 --games-per-iteration 256 --batch-size 512 \
  --learning-rate 0.000075 --device mps --rollout-device cpu \
  --opponent population \
  --random-weight 0.1 --rule-weight 0.5 \
  --initial-policy-weight 0.35 --self-play-weight 0.35 \
  --snapshot-every 5 --max-snapshots 4 \
  --adaptive-sampling win_rate --adaptive-alpha 1.0 \
  --adaptive-min-multiplier 0.1 --adaptive-ema-decay 0.95 \
  --adaptive-warmup-games 32 \
  --checkpoint-every 5 --seed 20260719
```

胜率 PFSP 使用 `4p(1-p)` 作为竞争性乘数，其中 `p` 是 learner 对该对手的 EMA 胜率。
接近 50% 的对手获得最高权重，过强或过弱的对手会被降权；`adaptive-min-multiplier` 保证
任何对手都不会完全消失。训练日志中的 `sampling_weights` 是下一轮实际使用的有效权重。

当前选择 adaptive iter10。每个对手、每个座位各 500 局的 deterministic 结果为：Random
91.6%，RuleAgent 57.3%，对 population v1 起点 51.9%，对更早的 fixed-Rule PPO 52.8%。这些
提升幅度仍较小，但相较 v1 已从“与历史策略持平”变为对两个历史策略均取得正向点估计。

如果官方 sample submission 不在默认位置，可显式指定：

```bash
poketcg-evaluate --official-dir /absolute/path/to/sample_submission
```

默认位置为：

```text
data/official/sample_submission/sample_submission
```

## 测试与代码检查

```bash
pytest
ruff check .
```
