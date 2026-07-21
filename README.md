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

该模型有 4,110,694 个参数；训练会按 token 长度分桶，减少 attention padding。V1/V2/V3
checkpoint 由同一套 Agent、PPO 和诊断入口自动识别。软标签 BC 应优先使用 stochastic 评测；
deterministic argmax 会固定选择大量同分动作中的一个，行为可能明显偏离 RuleAgent 的随机
tie-breaking。当前 V2 全量 Value MAE 为 0.652，V1 BC 为约 0.842；V2 PPO 管线已通过冒烟测试。

## FeatureEncoder V3：结构化卡牌语义与事件历史

V3 保留 V2 的所有可见状态 token，并增加：

- 40 维卡牌/攻击效果语义，包括搜索、抽牌、弃牌、能量加速、伤害缩放、防伤、换位、状态、
  奖品和条件触发，以及印刷伤害、能耗和文本规模等数值；
- observation 中最近 32 条可见引擎事件，包括事件类型、相对玩家、源/目标卡牌、攻击、区域
  变化、数值及相对时序；
- option 级语义，使攻击和卡牌候选不只依赖离散 ID embedding。

语义在运行时从官方 catalog 解析，不生成或提交派生卡牌文本文件。重新采集 V3 BC 数据：

```bash
python -m poketcg.rl.collect_bc \
  --games 2000 --encoder-version 3 --include-multiselect --seed 20260724 \
  --output artifacts/bc/rule_v3_semantic_actions_v2_2000.jsonl
```

沿用当前表现最好的 semantic-only 配置训练 Action Space V2：

```bash
python -m poketcg.rl.train_bc \
  --input artifacts/bc/rule_v3_semantic_actions_v2_2000.jsonl \
  --output artifacts/checkpoints/bc_rule_v3_semantic_actions_v2_2000.pt \
  --epochs 10 --batch-size 64 --learning-rate 0.0002 \
  --hidden-size 256 --model-type transformer_v3 \
  --num-layers 3 --num-heads 4 --dropout 0.1 \
  --disable-history \
  --device mps --seed 20260724
```

同一份 V3 数据仍可做四组严格消融：完整模型、去语义、去历史，以及两者都去掉；checkpoint
会记录开关，之后 BC Agent、PPO、对手池、固定面板和 Value diagnostics 会自动恢复相同结构。

`--include-multiselect` 开启 Action Space V2。策略不再把一个合法集合枚举成一个巨大的离散
动作，而是给每个 option 输出一个 logit，并在 `minCount <= |S| <= maxCount` 的全部集合上定义
`P(S) ∝ exp(sum(i in S) logit_i)`。分区函数、采样、BC NLL、PPO ratio 和 entropy 都用动态规划
精确计算；单选是它的严格特例。新 checkpoint 会直接控制 `ATTACH_TO`、
`SETUP_BENCH_POKEMON`、`TO_HAND` 等多选决策；旧 checkpoint 未记录动作空间版本，仍自动按 V1
交给 RuleAgent，保持历史实验可复现。动作空间变了，因此需要重新采集 BC 数据并从新的 BC
checkpoint 开始一条 PPO 实验线，不能只给旧 checkpoint 改配置字段。

可用混合策略消融隔离多选策略的贡献：`--checkpoint` 只负责恰好选择一个 option 的决策，
`--multiselect-checkpoint` 负责其余集合决策。两个模型独立加载，不修改 checkpoint：

```bash
python -m poketcg.rl.evaluate_panel \
  --checkpoint artifacts/checkpoints/OLD_SINGLE_POLICY.pt \
  --multiselect-checkpoint artifacts/checkpoints/ACTION_V2_POLICY.pt \
  --games-per-seat 500 --stochastic \
  --output results/evaluation/hybrid_policy_final500.json
```

用 GAE + Masked PPO 继续对 RuleAgent 微调：

```bash
python -m poketcg.rl.train_ppo \
  --input artifacts/checkpoints/bc_rule_v3_semantic_actions_v2_2000.pt \
  --output artifacts/checkpoints/ppo_v3_semantic_actions_v2_rule_20.pt \
  --iterations 20 --games-per-iteration 128 \
  --device mps --rollout-device cpu \
  --checkpoint-every 5 --seed 20260717
```

`--device` 是 PPO 批量更新设备；`--rollout-device` 是官方引擎逐动作推理设备。当前小模型在
CPU 上逐动作推理通常更快，而 MPS 适合批量反向传播。训练会额外保存每 5 轮 checkpoint，
不要仅凭单轮 rollout 回报选择模型，应使用固定对手面板复评。

在多核 Linux/Colab 上可增加 `--rollout-workers 8 --worker-torch-threads 1`，用 `spawn`
启动进程隔离的官方模拟器。传入 `--wandb-mode online --wandb-project PROJECT` 可由主进程
实时记录 PPO、胜率、PFSP、吞吐和系统指标；API key 只应通过环境变量或 Colab Secret 提供。

可选的奖励卡进度 PBRS 使用
`--reward-shaping prize --reward-shaping-scale 1.0` 开启。势函数为双方剩余奖励卡数之差除以
6，每步奖励增加 `gamma * Phi(s') - Phi(s)`，并令终局势函数为 0。塑形只进入 policy
advantage；critic 仍拟合原始胜负回报，保持 `[-1, 1]` 的语义与支持范围。默认值 `none`
便于和同一起点、同一随机种子做严格 A/B。

Actor 和 critic 的 lambda 可分别用 `--gae-lambda` 与 `--value-gae-lambda` 控制。后者省略时
继承前者以兼容旧实验。完整终局目标的消融实验没有改善 early Value 或固定面板胜率，因此
当前推荐和 Colab 默认均恢复为 Actor `0.95`、critic `0.95`。保留独立参数用于后续实验。

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

对官方引擎中的完整对局做 on-policy Value calibration 与轨迹诊断：

```bash
python -m poketcg.rl.value_diagnostics \
  --checkpoint artifacts/checkpoints/ppo_v2_parallel_colab_iter0030.pt \
  --opponent rule --games-per-seat 500 --device cpu \
  --output results/diagnostics/ppo_v2_iter0030_value_rule500.json
```

该入口按总体、Player 0/1、奖励卡阶段和决策 context 报告 Value MAE、RMSE、Brier score、
Pearson correlation、explained variance、calibration slope/intercept 和 ECE；同时报告逐局
初始/最终 Value、朝最终结果方向的净变化、轨迹波动、符号翻转和高置信错误。指定 `--output`
时还会自动写出同名前缀的 `_trajectories.jsonl`，每行是一个模型实际参与的决策状态。
默认 deterministic 与固定评测面板一致；研究训练时的随机策略分布可增加 `--stochastic`。

在改动模型前，先检查神经策略究竟覆盖了多少官方决策，以及 V2 尚未使用的公开日志和卡牌
ID 是否存在问题：

```bash
python -m poketcg.rl.coverage_diagnostics \
  --checkpoint artifacts/checkpoints/bc_rule_v2_transformer_2000.pt \
  --opponent rule --games-per-seat 500 --device cpu \
  --output results/diagnostics/bc_rule_v2_coverage_rule500.json
```

报告把决策分为：只有一个合法结果的 `forced`、当前网络处理的 `neural`，以及仍由规则
fallback 处理的 `resolver`。Action Space V1 的 resolver 通常是多选；V2 checkpoint 应使其降为
0。判断覆盖率应看排除强制动作后的
`strategic_neural_coverage`。按 context 显示的胜率只是相关性诊断，不能当作该 context 的因果
影响。报告还会审计 card/attack embedding 的 ID 范围、当前牌组，以及 observation 中已有但
V2 尚未编码的公开事件日志；指定 `--output` 时同时保存逐决策 `_records.jsonl`。

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

## Policy-Value MCTS

当前官方 `cg.api` 已提供 `search_begin`、`search_step` 和 `search_end`。原生库为正式
`BattleStart` 与搜索 `AgentStart` 分配不同实例；`search_step(search_id, action)` 从指定父状态
复制子状态，因此不会直接改写正式对局状态。搜索树内部仍共享 search-side RNG 流，随机分支
并不是 common-random-number 配对样本。MCTS 使用现有 PPO checkpoint 提供 action prior 和
叶节点 Value，并用 PUCT 在官方模拟器的真实状态转移上向前搜索。

第一版实现有意保持保守：

- 只在 `SelectContext.MAIN` 作为根节点时搜索，卡牌效果内部的后续选择仍会进入搜索树；
- 单选节点按 Policy prior 展开至多 `max_actions` 个动作；
- Action Space V1 checkpoint 的多选节点只使用原 RuleAgent 动作，避免组合爆炸；
- Value 全部以根玩家视角累计，对手节点选择时最小化根玩家 Q，不能按 selection 深度翻转；
- 从双方完整牌组先验减去当前可见卡牌，采样隐藏 hand/deck/prize；可把固定总模拟预算拆到多棵
  独立 determinization 搜索树，再汇总根节点 visit/Q。

运行 16-simulation PUCT 对 RuleAgent：

```bash
python -m poketcg.cli \
  --games 50 --seed 20260901 \
  --player0 mcts --player1 rule \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --stochastic \
  --mcts-simulations 16 --mcts-max-depth 12 --mcts-max-actions 16 \
  --output results/evaluation/mcts16_as_player0_50.jsonl
```

正式比较必须交换 `player0/player1`。每座位 500 局的 8/16/32 simulations 消融结果分别为
59.0%、63.9%、63.9%，因此当前性价比拐点是 16。固定 16 总预算时，det=1/2/4 分别为
62.5%、59.1%、61.1%；把总预算增至 32、拆成两棵各 16 次的树也只有 62.7%，没有观察到
多次 determinization 的收益。完整区间、先后手和耗时见
[`docs/MCTS_EXPERIMENTS.md`](docs/MCTS_EXPERIMENTS.md)。

当前隐藏状态推断在离线同牌组评测中假设对手使用相同 deck。提交到公开 ladder 前必须加入
对手牌组 belief/候选 archetype，或者把搜索仅用于离线 Expert Iteration；否则搜索质量会受
错误的 opponent deck prior 限制。

双座位固定面板入口会在每个座位只加载一次 checkpoint，并报告 Wilson 95% 区间、搜索节点数、
深度和耗时。下面是 8/16/32 simulations 消融的单条命令模板：

```bash
python -m poketcg.rl.evaluate_mcts \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --games-per-seat 500 --seed 20261101 \
  --simulations 16 --determinizations 1 --torch-threads 1 \
  --output results/evaluation/mcts_sims16_rule_500x2.json
```

加入多次 determinization 和候选牌组 belief：

```bash
python -m poketcg.rl.evaluate_mcts \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --games-per-seat 500 --seed 20261103 \
  --simulations 16 --determinizations 4 --torch-threads 1 \
  --opponent-deck sample=data/official/sample_submission/sample_submission/deck.csv \
  --opponent-deck meta_a=configs/opponent_decks/meta_a.csv \
  --opponent-deck fishcat_v8=configs/opponent_decks/fishcat_v8.csv \
  --opponent-deck mcts_sample=configs/opponent_decks/mcts_sample.csv \
  --output results/evaluation/mcts_sims16_det04_belief_rule_500x2.json
```

belief 会累计本局事件日志中已经公开的对手卡牌，并结合当前场面/弃牌区的可见实体，按候选
牌组与证据的相容性更新后验；每局开始前清空证据。输出中的
`opponent_deck_samples` 可以检查实际 determinization 采样是否符合预期。候选表只是实验接口，
不是完整 meta；正式提交前应按最新公开牌组和本地对战记录更新。

验证 belief 时不要只打 sample mirror：`--actual-opponent-deck` 控制 RuleAgent 真正使用的牌组，
`--fixed-opponent-prior-deck` 控制不启用 belief 时 MCTS 假设的对手牌组。应在同一实际对手上比较
错误 fixed prior、正确 oracle prior、以及候选 belief 三臂；oracle 是性能上界，belief 的目标是
在不知道实际牌组时逼近它。

### Multi-deck × multi-agent meta panel

`evaluate_meta_panel` 用相同 checkpoint 同时评测直接 Policy 和 MCTS，并交叉多个实际对手牌组、
RuleAgent 和 Policy 对手。默认四个牌组为 official sample、Meta A、Fishcat V8 与公开 MCTS
notebook 的 deck snapshot。这里的 `mcts_sample` 只是牌组，不代表已经复现了 notebook Agent。

先在 Colab 做每格双座位各 50 局的 screen；`--workers` 按独立 matchup 并行，每个 worker 把
Torch 限制为一个 CPU thread：

```bash
python -m poketcg.rl.evaluate_meta_panel \
  --checkpoint /content/drive/MyDrive/pokemonTCG/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --games-per-seat 50 --workers 4 --torch-threads 1 \
  --candidate policy --candidate mcts \
  --opponent rule --opponent policy \
  --mcts-prior fixed-model --simulations 16 --determinizations 1 \
  --output /content/drive/MyDrive/pokemonTCG/results/meta_panel_screen50.json
```

`fixed-model` 精确复现第一版提交的关键假设：每个 MCTS Agent 都把对手当作使用自己的牌组。
定位问题后，可以保持其他配置不变，仅把 `--mcts-prior` 改成 `oracle` 或 `belief`。正式确认某个
差异时再把 `--games-per-seat` 提升到 500，不要一开始运行完整面板。

输出包含：

- `views.by_candidate`：Policy/MCTS 全面板汇总；
- `views.by_candidate_opponent`：按对手 Agent 切片；
- `views.by_candidate_deck`：按真实对手牌组切片；
- `views.comparisons.mcts_minus_policy.cells`：每个 Agent × deck cell 的双座位差值；
- `cells[*].seats[*].candidate_search`：搜索次数、深度及延迟的 P50/P95/P99/max。

可用重复的 `--opponent-deck NAME=PATH` 替换默认牌组；用
`--policy-opponent NAME=CHECKPOINT` 或 `--mcts-opponent NAME=CHECKPOINT` 加入历史模型。CLI seed
只控制 Agent 与 determinization RNG，官方 native battle RNG 不能固定，因此差值仍是独立评测，
不能当作严格 paired causal estimate。

## Expert Iteration：蒸馏 MCTS

`collect_expert_iteration` 运行当前 checkpoint 的 MCTS self-play。MAIN 根节点使用搜索子节点
visit count 作为软 Policy 标签；其他可学习选择使用原 Policy 分布作 replay，避免微调后遗忘
卡牌效果内部决策。所有样本在对局结束后从各自玩家视角回填 `-1/0/+1` Value target。

Colab Round 1 推荐先采集 2000 局。逐动作小模型推理在 CPU 上通常比 T4 更快，GPU 留给后续
批量训练：

```bash
cd /content/pokemonTCG
python -m poketcg.rl.collect_expert_iteration \
  --checkpoint /content/drive/MyDrive/pokemonTCG/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --games 2000 --simulations 16 --determinizations 1 \
  --target-temperature 1.0 --replay-temperature 1.0 \
  --device cpu --torch-threads 1 --seed 20260810 \
  --output /content/drive/MyDrive/pokemonTCG/expert_iteration/mcts16_round1_2000.jsonl
```

从原 iter0018 权重微调，不要随机初始化同一个 411 万参数模型：

```bash
python -m poketcg.rl.train_bc \
  --input /content/drive/MyDrive/pokemonTCG/expert_iteration/mcts16_round1_2000.jsonl \
  --initialize-from /content/drive/MyDrive/pokemonTCG/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --output /content/drive/MyDrive/pokemonTCG/checkpoints/expert_iter_round1_mcts16_2000.pt \
  --epochs 5 --batch-size 128 --learning-rate 0.00003 \
  --value-coefficient 0.25 --max-grad-norm 0.5 \
  --device cuda --seed 20260810
```

最后仍使用 MCTS16 做双座位固定面板，不能仅凭训练 loss 选模型：

```bash
python -m poketcg.rl.evaluate_mcts \
  --checkpoint /content/drive/MyDrive/pokemonTCG/checkpoints/expert_iter_round1_mcts16_2000.pt \
  --games-per-seat 500 --seed 20260811 \
  --simulations 16 --determinizations 1 --torch-threads 1 \
  --output /content/drive/MyDrive/pokemonTCG/results/expert_iter_round1_rule_500x2.json
```

本地 100 局采集 screen 产生 2439 条样本，其中 1891 条为 MCTS visit 标签。新 checkpoint
在 RuleAgent 100×2 面板上的点估计为 65.0%，原 iter0018 同配置为 62.5%；区间重叠，因此
只说明管线可用，不能替代上面的 500×2 正式评测。

如果官方 sample submission 不在默认位置，可显式指定：

```bash
poketcg-evaluate --official-dir /absolute/path/to/sample_submission
```

默认位置为：

```text
data/official/sample_submission/sample_submission
```

## 构建 Kaggle Agent 提交包

提交前把目标 checkpoint 放到本地或已挂载的 Drive 路径，然后生成自包含归档：

```bash
python -m poketcg.submission \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --output artifacts/submissions/ppo_v3_iter0018/submission.tar.gz
```

构建当前推荐的 MCTS 提交（16 simulations、单次 determinization）：

```bash
python -m poketcg.submission \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --mcts-simulations 16 --mcts-determinizations 1 \
  --mcts-c-puct 1.25 --mcts-max-depth 12 --mcts-max-actions 16 \
  --output artifacts/submissions/mcts16_ppo_v3_iter0018/submission.tar.gz
```

归档中的 `agent_config.json` 决定线上使用 direct policy 还是 MCTS。MCTS runtime 使用官方
Search API；初始化或搜索失败时依次降级到原 PPO policy、RuleAgent 和最小合法动作。

构建器会只保留 checkpoint 中的模型配置和参数，并把 `main.py`、`deck.csv`、项目推理代码及
官方 `cg` 运行库放在 tar 根目录。线上策略使用 stochastic 采样；模型异常时依次退回
RuleAgent 和最小合法动作。上传前应从归档解压到空目录并至少完成一局官方引擎冒烟测试。

确认归档后使用已认证的 Kaggle CLI 上传：

```bash
kaggle competitions submit pokemon-tcg-ai-battle \
  -f artifacts/submissions/ppo_v3_iter0018/submission.tar.gz \
  -m "PPO V3 semantic iter0018 baseline before MCTS"
```

## 测试与代码检查

```bash
pytest
ruff check .
```
