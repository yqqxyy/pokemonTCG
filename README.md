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

`evaluate_meta_panel` 用相同 checkpoint 同时评测直接 Policy、MCTS、显式规划器及其混合版本，
并交叉多个实际对手牌组、RuleAgent 和 Policy 对手。默认四个牌组为 official sample、Meta A、Fishcat V8 与公开 MCTS
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

经过人工检查的公开 Kaggle Agent 可以保留在忽略的 `artifacts/public_agents` 下，通过
`--external-opponent NAME=SOURCE,DECK` 接入；支持包含 `def agent(...)` 的 `.py`，以及用
`%%writefile main.py` 生成提交代码的 `.ipynb`。外部 Agent 会绑定自己的牌组，不与其他 deck
做无意义的交叉组合。Mega Lucario 官方样例的 50×2 screen：

```bash
python -m poketcg.rl.evaluate_meta_panel \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --games-per-seat 50 --workers 1 --torch-threads 1 \
  --candidate policy --candidate mcts --simulations 8 \
  --external-opponent mega_lucario=artifacts/public_agents/mega_lucario/a-sample-rule-based-agent-mega-lucario-ex-deck.ipynb,artifacts/public_agents/mega_lucario/deck.csv \
  --output results/mega_lucario_screen50.json
```

也可以用重复的 `--external-candidate NAME=SOURCE,DECK` 把多名已审计的公开 Agent 放到候选侧。
每名候选固定使用自己的 `deck.csv`，报告的 `candidate_decks` 和每个 cell 的 `candidate_deck`
会记录实际绑定，避免过去“换了 Agent 却仍使用我方 Mega 牌组”的失真评测。做公开 Agent
round-robin 时，需要在候选侧和对手侧分别列出同一组 Agent：

```bash
python -m poketcg.rl.evaluate_meta_panel \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --games-per-seat 20 --workers 1 --torch-threads 1 \
  --external-candidate agent_a=artifacts/public_agents/agent_a/main.py,artifacts/public_agents/agent_a/deck.csv \
  --external-candidate agent_b=artifacts/public_agents/agent_b/main.py,artifacts/public_agents/agent_b/deck.csv \
  --external-opponent agent_a=artifacts/public_agents/agent_a/main.py,artifacts/public_agents/agent_a/deck.csv \
  --external-opponent agent_b=artifacts/public_agents/agent_b/main.py,artifacts/public_agents/agent_b/deck.csv \
  --output results/evaluation/public_agents_round_robin_screen20x2.json
```

外部 Agent 若没有显式 episode reset hook，适配器会在每局开始前重新加载源码，防止模块全局
计划、回合计数等状态泄漏到下一局。先用 20×2 淘汰弱候选，再对晋级者和关键 meta 对手做
500×2 确认；round-robin 的简单平均不能替代按线上牌组占比计算的加权结果。

外部源码会在本机执行，因此必须先人工检查；本项目不复制或重新发布第三方 Agent 源码。若作者
未声明许可证，只把它作为本地比赛评测依赖使用。

下载线上 replay JSON 后，可按对手牌组、终局原因、Land Collapse 启动速度和最终牌库差做失败
诊断。`--input` 可以是单个 JSON，也可以是包含多局 replay 的目录：

```bash
python -m poketcg.rl.replay_diagnostics \
  --input artifacts/replays/54891713 \
  --team yqqxyy \
  --output results/evaluation/libraryout_54891713_loss_diagnostics.json
```

本地修改过并已审计的外部 `.py` Agent 可以独立打包，不需要伪造 checkpoint。打包器会检查
Python 语法、60 张牌、官方 `cg` 文件和归档根目录结构：

```bash
python -m poketcg.external_submission \
  --source artifacts/public_agents/libraryout_v2/main.py \
  --deck artifacts/public_agents/libraryout_v2/deck.csv \
  --output artifacts/submissions/libraryout_v2/submission.tar.gz
```

### Library-Out trajectory dataset 与 residual reranker

线上更强的 Library-Out V1 保持为生产基线。采集器在不改动公开源码的前提下，记录 V1 每次
决策的 V3 observation、合法动作、逐动作规则分数、实际动作、对手、座位和终局结果。规则分数
只在当前 selection 内做 z-score，不跨 context 比较。下面的 1400 局日程包含 7 类对手 × 双座位
各 100 局，适合 Colab 8 CPU 的第一轮：

```bash
python -m poketcg.rl.collect_libraryout_trajectories \
  --expert-source artifacts/public_agents/libraryout_1208/main.py \
  --expert-deck artifacts/public_agents/libraryout_1208/deck.csv \
  --mirror \
  --external-opponent strong_start=artifacts/public_agents/strong_start_v10/main.py,artifacts/public_agents/strong_start_v10/deck.csv \
  --external-opponent baseline1084=artifacts/public_agents/baseline_1084/main.py,artifacts/public_agents/baseline_1084/deck.csv \
  --external-opponent alakazam=artifacts/public_agents/alakazam_best5/main.py,artifacts/public_agents/alakazam_best5/deck.csv \
  --external-opponent field_alakazam=artifacts/public_agents/field_audited_alakazam_v8/main.py,artifacts/public_agents/field_audited_alakazam_v8/deck.csv \
  --external-opponent archaludon=artifacts/public_agents/archaludon_vs_starmie/main.py,artifacts/public_agents/archaludon_vs_starmie/deck.csv \
  --external-opponent mega_lucario=artifacts/public_agents/mega_lucario/a-sample-rule-based-agent-mega-lucario-ex-deck.ipynb,artifacts/public_agents/mega_lucario/deck.csv \
  --games 1400 --workers 8 --torch-threads 1 --encoder-version 3 \
  --output /content/drive/MyDrive/pokemonTCG/residual/libraryout_v1_trajectory_1400.jsonl
```

Round 0 使用 outcome-weighted imitation：胜局权重 1.0、和局 0.5、败局 0.25。Transformer
输出的是对规则 prior 的修正量，并用 L2 将残差约束在零附近；它不是重新从头学习完整策略。

```bash
python -m poketcg.rl.train_residual \
  --input /content/drive/MyDrive/pokemonTCG/residual/libraryout_v1_trajectory_1400.jsonl \
  --output /content/drive/MyDrive/pokemonTCG/checkpoints/libraryout_residual_round0.pt \
  --epochs 8 --batch-size 128 --learning-rate 0.0001 \
  --hidden-size 256 --num-layers 3 --num-heads 4 \
  --prior-strength 2.0 --residual-coefficient 0.01 \
  --override-margin 0.5 --minimum-confidence 0.65 \
  --device cuda --seed 20260722
```

先用 `--shadow` 只统计模型想覆盖哪些动作，不真正覆盖 V1；确认 proposed override 集中在合理
context 后再去掉 `--shadow`。第一轮 screen 不用于提交，只验证门控覆盖率和固定面板胜率：

```bash
python -m poketcg.rl.evaluate_residual \
  --checkpoint /content/drive/MyDrive/pokemonTCG/checkpoints/libraryout_residual_round0.pt \
  --baseline-source artifacts/public_agents/libraryout_1208/main.py \
  --baseline-deck artifacts/public_agents/libraryout_1208/deck.csv \
  --external-opponent mirror=artifacts/public_agents/libraryout_1208/main.py,artifacts/public_agents/libraryout_1208/deck.csv \
  --external-opponent alakazam=artifacts/public_agents/alakazam_best5/main.py,artifacts/public_agents/alakazam_best5/deck.csv \
  --external-opponent archaludon=artifacts/public_agents/archaludon_vs_starmie/main.py,artifacts/public_agents/archaludon_vs_starmie/deck.csv \
  --games-per-seat 100 --device cuda --shadow \
  --output /content/drive/MyDrive/pokemonTCG/results/libraryout_residual_round0_shadow100x2.json
```

这一轮只有专家动作和终局监督，不包含同一状态下所有替代动作的真实反事实收益。因此它的目标
是建立一个保守、可校准的 reranker 和数据管线，而不是立刻超过 V1。只有影子模型稳定后，才对
高分歧、低规则 margin 的局面追加 MCTS/counterfactual rollout 标签，形成真正能学习覆盖规则的
Round 1 数据。

#### 在线 paired one-step-deviation rollout

旧 trajectory JSONL 只有编码后的可见状态，不能恢复官方引擎的隐藏状态、RNG 和未结算效果。
`collect_paired_rollouts` 因此在 V1 正式对局进行中选择 MAIN 单选状态，立即调用官方 Search API
建立根状态；同一个 determinization 下，V1 和两个候选从相同 root 分叉，之后由双方原规则 Agent
继续到终局。正式对局始终执行 V1，不受搜索结果影响。

候选集合合并 V1、Round 0 logits、Rule top-k 和不同 option type。若根动作打开 TO_HAND、
ATTACH_TO、SWITCH 等强制选择，V1 会完成该短 option，并在记录中保存到返回 MAIN、换回合或终局
为止的 `option_sequence`。完整分支则继续到终局；只有超过 `--max-rollout-steps` 才使用 Round 0
Value bootstrap。

先用 50 状态 × 8 determinizations 检查错误率、paired 标准误和标签符号分布：

```bash
python -m poketcg.rl.collect_paired_rollouts \
  --expert-source artifacts/public_agents/libraryout_1208/main.py \
  --expert-deck artifacts/public_agents/libraryout_1208/deck.csv \
  --checkpoint artifacts/checkpoints/libraryout_residual_round0.pt \
  --mirror \
  --external-opponent strong_start=artifacts/public_agents/strong_start_v10/main.py,artifacts/public_agents/strong_start_v10/deck.csv \
  --external-opponent baseline1084=artifacts/public_agents/baseline_1084/main.py,artifacts/public_agents/baseline_1084/deck.csv \
  --external-opponent alakazam=artifacts/public_agents/alakazam_best5/main.py,artifacts/public_agents/alakazam_best5/deck.csv \
  --external-opponent field_alakazam=artifacts/public_agents/field_audited_alakazam_v8/main.py,artifacts/public_agents/field_audited_alakazam_v8/deck.csv \
  --external-opponent archaludon=artifacts/public_agents/archaludon_vs_starmie/main.py,artifacts/public_agents/archaludon_vs_starmie/deck.csv \
  --external-opponent mega_lucario=artifacts/public_agents/mega_lucario/a-sample-rule-based-agent-mega-lucario-ex-deck.ipynb,artifacts/public_agents/mega_lucario/deck.csv \
  --target-states 50 --max-games 350 --determinizations 8 \
  --max-states-per-game 1 --random-state-probability 0.15 \
  --workers 8 --torch-threads 1 \
  --output data/processed/libraryout_paired_screen50_det8.jsonl
```

要求 `search_failures == 0`、`rollout_errors` 为空且绝大多数分支到达 `terminal`。通过后把
`--target-states` 扩为 500、`--max-games` 扩为 1400、`--determinizations` 扩为 16。相邻
`.summary.json` 会额外报告 advantage 分位数、paired stderr，以及有候选满足 95% LCB > 0.05
的状态数。Round 1 advantage ensemble 只使用这份 paired 数据，不从旧 JSONL 重建状态。

#### Round 1 baseline-relative advantage ensemble

`train_advantage` 不再拟合 V1 动作，也不把神经 logits 直接叠加到规则分数。对每个已经 rollout
的候选，它回归同一 determinization 内的相对收益：

```text
ΔQ(I, o) = Q^V1(I, o) - Q^V1(I, o_v1)
```

模型预测同样使用候选 logit 与 V1 baseline logit 的差，因此对所有 logits 的公共平移不敏感。
loss 使用带噪声下限和上限的 inverse-variance 权重；第一轮默认冻结 V3 Transformer，只训练
policy head。三个成员从相同 Round 0 表示出发，但分别 bootstrap 训练状态，供后续用 ensemble
mean/std 做保守 LCB gate。

```bash
python -m poketcg.rl.train_advantage \
  --input data/processed/libraryout_paired_round1_500_det16.jsonl \
  --initialize-from artifacts/checkpoints/libraryout_residual_round0.pt \
  --output-dir artifacts/checkpoints/libraryout_advantage_round1 \
  --ensemble-size 3 --epochs 40 --patience 8 --batch-size 64 \
  --train-scope policy_head --learning-rate 0.0003 \
  --huber-delta 0.25 --noise-floor 0.15 --maximum-weight 20 \
  --gate-threshold 0.05 --uncertainty-multiplier 1.0 \
  --checkpoint-selection gain_lcb --selection-risk-multiplier 1.0 \
  --device cpu --seed 20260723
```

输出目录包含三个 `advantage_member*.pt` 和 `ensemble_manifest.json`。是否进入在线 shadow
不能只看 validation loss；还必须检查 `validation_lcb_metrics` 的 `selected_gain` 为正、
`harmful_override_rate` 足够低，并让覆盖率保持在保守区间。若这三项不满足，优先扩大 paired
状态数据，而不是把弱 reranker 接到 V1。

Advantage checkpoint 默认按固定验证集上的风险调整收益
`selected_gain_lcb = mean_gain - multiplier * standard_error` 选择，而不是按回归 loss 选择。
后者容易偏好把所有优势压到零附近的模型，虽然 MAE 较低，却不一定产生更好的覆盖动作。

扩大数据时必须更换 collector `--seed`。第一份 500-state 数据可固定为从不参与梯度更新的
验证集，新采集数据则全部用于 bootstrap 训练：

```bash
python -m poketcg.rl.train_advantage \
  --input data/processed/libraryout_paired_round1_train3000_det16.jsonl \
  --validation-input data/processed/libraryout_paired_round1_500_det16.jsonl \
  --initialize-from artifacts/checkpoints/libraryout_residual_round0.pt \
  --output-dir artifacts/checkpoints/libraryout_advantage_round1_3000 \
  --ensemble-size 3 --epochs 40 --patience 8 --batch-size 64 --device cpu
```

早期版本在每局遇到第一个分歧时就采样，导致绝大多数状态来自 Turn 1/2，终局标签方差过大。
后续高精度数据必须用 `--minimum-turn 3` 单独筛选中后期状态，并先以较小 screen 检查 turn
分布、paired stderr 和 estimated label reliability，再决定是否扩大。

#### Turn-gated advantage reranker

当前安全版本只在 Turn 4 以后评估与 paired collector 相同的候选集合，并通过动作语义白名单
限制覆盖。第一版只允许 `PLAY(7) -> END(14)` 和 `PLAY(7) -> ATTACK(13)`；其他模型建议全部
退回 Library-Out V1。下面的命令同时评测原 V1 和实际执行覆盖的 reranker：

```bash
python -m poketcg.rl.evaluate_advantage \
  --advantage-checkpoint artifacts/checkpoints/libraryout_advantage_turn4_1500_gain_selected/advantage_member00.pt \
  --round0-checkpoint artifacts/checkpoints/libraryout_residual_round0.pt \
  --baseline-source artifacts/public_agents/libraryout_1208/main.py \
  --baseline-deck artifacts/public_agents/libraryout_1208/deck.csv \
  --external-opponent mirror=artifacts/public_agents/libraryout_1208/main.py,artifacts/public_agents/libraryout_1208/deck.csv \
  --games-per-seat 200 --minimum-turn 4 --gate-threshold 0.05 \
  --allowed-transition '7->14' --allowed-transition '7->13' \
  --output results/evaluation/libraryout_advantage_turn4_semantic_confirm200x2.json
```

三对手 1200 局确认中，V1 为 56.33%，reranker 为 57.25%，实际覆盖率 6.26%；Wilson 区间
仍重叠，因此这只通过了本地安全性检查，不构成确定优于 V1 的统计证据。

构建对应的实验提交包时，必须同时打包 V1 源码、Library-Out 牌组和 Round 0 候选模型：

```bash
python -m poketcg.submission \
  --checkpoint artifacts/checkpoints/libraryout_advantage_turn4_1500_gain_selected/advantage_member00.pt \
  --deck artifacts/public_agents/libraryout_1208/deck.csv \
  --advantage-baseline-source artifacts/public_agents/libraryout_1208/main.py \
  --advantage-round0-checkpoint artifacts/checkpoints/libraryout_residual_round0.pt \
  --advantage-minimum-turn 4 --advantage-gate-threshold 0.05 \
  --advantage-allowed-transition '7->14' \
  --advantage-allowed-transition '7->13' \
  --output artifacts/submissions/libraryout_advantage_turn4_semantic/submission.tar.gz
```

该提交仍以 Library-Out V1 为默认动作；神经网络初始化、候选生成或推理失败时也会退回 V1，
不会把 advantage checkpoint 当成普通 policy 直接执行。

#### 完整回合协同收益诊断

one-step paired rollout 在候选根动作后立即把控制权交回 V1，因此看不到“动作 A 单独无效，
动作 B 单独无效，但 A→B 联合有效”的改进。`collect_turn_synergy` 在同一个在线 Search 根状态
和同一个 belief determinization 内同时比较：

1. V1 完整回合；
2. 只改变根动作、后续交回 V1；
3. 在当前玩家整个回合内持续展开的 beam search。

第三项会跨过 TO_HAND、ATTACH_TO、SWITCH 等 option context，并在返回 MAIN 后继续分支。
它为每个隐藏世界分别选取最佳完整计划，因此输出是验证组合策略上限的 oracle diagnostic，
不能直接当作线上 policy 的监督标签。

先跑 12 个状态 × 2 个 determinizations 的管线 screen：

```bash
python -m poketcg.rl.collect_turn_synergy \
  --expert-source artifacts/public_agents/libraryout_1208/main.py \
  --expert-deck artifacts/public_agents/libraryout_1208/deck.csv \
  --checkpoint artifacts/checkpoints/libraryout_residual_round0.pt \
  --mirror \
  --external-opponent strong_start=artifacts/public_agents/strong_start_v10/main.py,artifacts/public_agents/strong_start_v10/deck.csv \
  --external-opponent baseline1084=artifacts/public_agents/baseline_1084/main.py,artifacts/public_agents/baseline_1084/deck.csv \
  --target-states 12 --max-games 120 --determinizations 2 \
  --beam-width 8 --branch-width 4 --max-plan-steps 32 \
  --max-states-per-game 1 --minimum-turn 3 \
  --workers 4 --torch-threads 1 --seed 20260801 \
  --output data/processed/libraryout_turn_synergy_screen12_det2.jsonl
```

相邻的 `.summary.json` 重点检查 `search_failures`、`search_errors`、`branch_errors`、
`diagnostics.hidden_synergy_rate` 和 `diagnostics.joint_rescue_rate`。前三项原则上都应为零；
若仅 `branch_errors` 非零，collector 会跳过单个非法候选并保留其余
有效 beam 分支，但扩大实验前仍应先检查错误消息。后两项用于判断完整回合搜索是否发现了
one-step 标签系统性遗漏的组合收益。screen 通过后，用 7 个对手各约 20 个状态做确认：

```bash
python -m poketcg.rl.collect_turn_synergy \
  --expert-source artifacts/public_agents/libraryout_1208/main.py \
  --expert-deck artifacts/public_agents/libraryout_1208/deck.csv \
  --checkpoint artifacts/checkpoints/libraryout_residual_round0.pt \
  --mirror \
  --external-opponent strong_start=artifacts/public_agents/strong_start_v10/main.py,artifacts/public_agents/strong_start_v10/deck.csv \
  --external-opponent baseline1084=artifacts/public_agents/baseline_1084/main.py,artifacts/public_agents/baseline_1084/deck.csv \
  --external-opponent alakazam=artifacts/public_agents/alakazam_best5/main.py,artifacts/public_agents/alakazam_best5/deck.csv \
  --external-opponent field_alakazam=artifacts/public_agents/field_audited_alakazam_v8/main.py,artifacts/public_agents/field_audited_alakazam_v8/deck.csv \
  --external-opponent archaludon=artifacts/public_agents/archaludon_vs_starmie/main.py,artifacts/public_agents/archaludon_vs_starmie/deck.csv \
  --external-opponent mega_lucario=artifacts/public_agents/mega_lucario/a-sample-rule-based-agent-mega-lucario-ex-deck.ipynb,artifacts/public_agents/mega_lucario/deck.csv \
  --target-states 140 --max-games 980 --determinizations 8 \
  --beam-width 8 --branch-width 4 --max-plan-steps 32 \
  --max-states-per-game 1 --minimum-turn 3 \
  --workers 8 --torch-threads 1 --seed 20260805 \
  --output data/processed/libraryout_turn_synergy_confirm140_det8.jsonl
```

判读时以 `diagnostics.state_synergy_rate` 及其 Wilson 95% 区间为主，world-level rate 只作机制
诊断。若区间下界超过 10%，组合动作盲区有较强证据；若上界低于 10%，它不是当前主要瓶颈；
区间跨过 10% 时需要独立 seed 或 beam-width 消融。不能把这一批 oracle 分支直接送入训练器。

#### Held-out 语义 Turn Plan 评测

oracle 搜索证明“某个隐藏世界中存在更好的完整回合”，但线上 Agent 看不到对手手牌和牌库顺序，
也不能为每个隐藏世界选择不同动作。加入 `--heldout-semantic` 后，评测器会：

1. 仅在 proposal determinizations 上搜索候选回合计划；
2. 把候选动作转换为卡牌 ID、攻击 ID、区域、目标卡等语义指令，不保存易变的 option 下标；
3. 只根据 proposal 配对收益下界选择一条固定计划；
4. 将同一条计划重放到从未参与搜索和选择的 held-out determinizations；
5. 无法解析语义动作时立即退回 V1，并记录 replay/fallback 指标。

先运行小规模 screen：

```bash
python -m poketcg.rl.collect_turn_synergy \
  --heldout-semantic \
  --expert-source artifacts/public_agents/libraryout_1208/main.py \
  --expert-deck artifacts/public_agents/libraryout_1208/deck.csv \
  --checkpoint artifacts/checkpoints/libraryout_residual_round0.pt \
  --mirror \
  --external-opponent strong_start=artifacts/public_agents/strong_start_v10/main.py,artifacts/public_agents/strong_start_v10/deck.csv \
  --external-opponent baseline1084=artifacts/public_agents/baseline_1084/main.py,artifacts/public_agents/baseline_1084/deck.csv \
  --target-states 12 --max-games 120 \
  --proposal-determinizations 4 --heldout-determinizations 4 \
  --plan-pool-size 16 --selection-risk-multiplier 1.0 \
  --beam-width 8 --branch-width 4 --max-plan-steps 32 \
  --max-states-per-game 1 --minimum-turn 3 \
  --workers 4 --torch-threads 1 --seed 20260806 \
  --output data/processed/libraryout_heldout_turn_plan_screen12_p4h4.jsonl
```

重点看 `.summary.json` 中的 `mean_heldout_gain`、`mean_heldout_gain_ci95`、
`mean_optimism_gap`、`mean_replay_success_rate` 和 `accepted_heldout_rate`。proposal gain 高但
held-out gain 接近零，说明搜索在拟合少量 sampled worlds；replay success 低则说明当前计划
表示仍依赖不稳定的物理下标。只有 held-out 收益稳定为正、重放率接近 1 的计划，才适合进入
下一阶段的监督数据构建；`heldout_accepted` 是诊断标签，不会在同一批数据上重新选择计划。

### Mega Lucario 显式战术规划器

`TacticalPlannerAgent` 先枚举本回合可形成的攻击计划，再让后续附能、进化、换位、Boss、伤害
增益与攻击选择共同执行同一个计划。牌组身份放在 `DeckTacticalProfile`，通用部分则按击倒、
奖励卡、能量缺口、换位成本、Aura Jab 的弃牌区能量加速和攻击手饱和度评分。

`PlannerPolicyAgent` 有两种作用：明确由牌组知识覆盖的 context 直接交给规划器；其余决策仍由
神经网络执行，同时将规划器分数作为 MCTS 的 action prior。通用置信度路由默认开启，可用
`--no-planner-confidence-routing` 做显式 context-only 消融。下面一次运行得到四臂对照：

```bash
python -m poketcg.rl.evaluate_meta_panel \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --model-deck artifacts/public_agents/mega_lucario/deck.csv \
  --games-per-seat 100 --workers 1 --torch-threads 1 --deterministic \
  --candidate policy --candidate planner \
  --candidate planner-policy --candidate planner-mcts \
  --planner-threshold 0.8 --planner-weight 4.0 \
  --simulations 8 --determinizations 1 \
  --external-opponent mega_lucario=artifacts/public_agents/mega_lucario/a-sample-rule-based-agent-mega-lucario-ex-deck.ipynb,artifacts/public_agents/mega_lucario/deck.csv \
  --output results/evaluation/mega_planner_ablation_100x2.json
```

`planner` 是纯规则消融，`planner-policy` 是路由混合，`planner-mcts` 则把混合 logits 作为搜索
先验、继续使用 checkpoint 的 value。第一版实现不复制公开 Agent 源码；当前 screen 只证明运行
链路成立，尚未证明它优于原 policy，因此正式提交前仍需扩大对手和牌组面板。

逐 context 检查规划器、神经策略和影子专家在相同状态上的动作：

```bash
python -m poketcg.rl.planner_diagnostics \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --expert-source artifacts/public_agents/mega_lucario/a-sample-rule-based-agent-mega-lucario-ex-deck.ipynb \
  --deck artifacts/public_agents/mega_lucario/deck.csv \
  --games-per-seat 50 --planner-threshold 0.9 \
  --output results/evaluation/mega_planner_diagnostics_50x2.json
```

报告排除强制动作，并输出各 context 的 planner/policy/hybrid 专家一致率、规划器替换神经动作后
净增加的正确决策数，以及主要分歧动作。路由器会固定交给规划器处理搜索选牌、Aura Jab 附能
来源/目标与开局摆场；强制换上与普通换位仍优先保留给神经策略。

### 座位对称性与计划所有权

V3 Encoder 统一按“行动方、对手方”顺序编码标量状态、场上卡牌和弃牌 token。下面的审计会把
双方绝对标签、`yourIndex`、`firstPlayer`、结果和所有 `playerIndex` 一起交换，再比较编码、
option logits、确定性动作和 Value。当前 Value 是行动方视角，因此正确关系是
`V(relabel(s)) == V(s)`，而不是取负：

```bash
python -m poketcg.rl.symmetry_diagnostics \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --deck artifacts/public_agents/mega_lucario/deck.csv \
  --games-per-seat 10 \
  --output results/evaluation/player_symmetry_10x2.json
```

`turn-planner-policy` 在每个回合第一次 `MAIN` 决策时选定 Planner 或 Policy owner，并让该 owner
负责本回合后续所有 MAIN 和 resolver 选择。只有缓存计划的攻击手或目标失效时，Planner owner
才会向 Policy 做一次不可逆转移。报告中的 `candidate_policy.turn_ownership` 包含 owner 分布、
owner 切换、计划失效和两类 owner 的决策数。三臂 screen：

```bash
python -m poketcg.rl.evaluate_meta_panel \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --model-deck artifacts/public_agents/mega_lucario/deck.csv \
  --games-per-seat 100 --workers 1 --torch-threads 1 --deterministic \
  --candidate policy --candidate planner-policy \
  --candidate turn-planner-policy \
  --planner-threshold 0.8 --planner-weight 4.0 \
  --external-opponent mega_lucario=artifacts/public_agents/mega_lucario/a-sample-rule-based-agent-mega-lucario-ex-deck.ipynb,artifacts/public_agents/mega_lucario/deck.csv \
  --output results/evaluation/mega_turn_ownership_screen100x2.json
```

先确认 `owner_switches_per_turn` 接近 0，并分别检查 player0/player1；只有整回合版本相对两条基线
都不退步，才扩大到 500×2。当前原子动作 MCTS 尚不支持 turn ownership，两者需要等
plan-level MCTS 后再组合。

`commitment-planner-policy` 是更严格的 Options 版本。普通 MAIN 动作只建立临时
`resolver_chain`：触发的选牌、弃牌和目标选择沿用同一个 owner，一旦回到 MAIN 就释放。
只有 Planner 选择的动作确实命中当前攻击计划——计划内的附能、进化、换位、Boss、增伤或攻击——
才建立 `committed_turn`，控制剩余回合。抽牌 Ability 不再因为出现在回合开头而获得整回合
所有权。报告中的 `candidate_policy.commitment_ownership` 会分别统计 resolver chains、
committed turns、chain resolutions、计划失效和两类 owner 的决策数。

与旧路由和宽松整回合版本做四臂 screen：

```bash
python -m poketcg.rl.evaluate_meta_panel \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --model-deck artifacts/public_agents/mega_lucario/deck.csv \
  --games-per-seat 100 --workers 1 --torch-threads 1 --deterministic \
  --candidate policy --candidate planner-policy \
  --candidate turn-planner-policy --candidate commitment-planner-policy \
  --planner-threshold 0.8 --planner-weight 4.0 \
  --external-opponent mega_lucario=artifacts/public_agents/mega_lucario/a-sample-rule-based-agent-mega-lucario-ex-deck.ipynb,artifacts/public_agents/mega_lucario/deck.csv \
  --output results/evaluation/mega_commitment_ownership_screen100x2.json
```

### Plan-level MCTS V0

`plan-mcts` 不再搜索原子 action，而是在每个回合第一次非强制 MAIN 决策上比较两个完整宏执行器：

- `local_router_turn`：稳健的 PlannerPolicy 逐 context 路由执行到换手；
- `planner_turn`：TacticalPlanner 独占执行到换手。

每个宏分支从同一个官方 Search root 开始，在多次 determinization 下推进到换手、终局或
`max_macro_steps`，然后把 checkpoint Value 统一转换为根玩家视角并取均值。选中的 executor 会在
真实对局中持续拥有整个回合，即使两个宏分支第一步动作相同，后续也不会丢失搜索选择。V0 是
两分支的 root-level Monte Carlo exhaustive search；在证明宏选择有效前，不增加更深的 PUCT 树。

先用 oracle hidden-deck prior 做 Mega 专家 50×2 screen，隔离宏搜索本身是否有效：

```bash
python -m poketcg.rl.evaluate_meta_panel \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --model-deck artifacts/public_agents/mega_lucario/deck.csv \
  --games-per-seat 50 --workers 1 --torch-threads 1 --deterministic \
  --candidate planner-policy --candidate turn-planner-policy \
  --candidate plan-mcts \
  --mcts-prior oracle --plan-determinizations 4 --plan-max-steps 32 \
  --planner-threshold 0.8 --planner-weight 4.0 \
  --external-opponent mega_lucario=artifacts/public_agents/mega_lucario/a-sample-rule-based-agent-mega-lucario-ex-deck.ipynb,artifacts/public_agents/mega_lucario/deck.csv \
  --output results/evaluation/plan_mcts_mega_screen50x2.json
```

`candidate_search` 会输出两种宏的选择次数、平均 Value、selection margin、near-tie rate、首动作
相同率、宏步数、边界、延迟和错误。oracle 只用于诊断；通过后必须再评测 `belief` 或
`fixed-model`，才能构建可提交版本。

`belief` 会在四个面板牌组与我方牌组之间维护 Bayesian posterior，并根据整局已公开过的卡牌更新
假设；它不读取真实隐藏牌组，可用于提交前验证：

```bash
python -m poketcg.rl.evaluate_meta_panel \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --model-deck artifacts/public_agents/mega_lucario/deck.csv \
  --games-per-seat 50 --workers 4 --torch-threads 1 --deterministic \
  --candidate planner-policy --candidate plan-mcts \
  --opponent rule --opponent policy \
  --mcts-prior belief --plan-determinizations 4 --plan-max-steps 32 \
  --planner-threshold 0.8 --planner-weight 4.0 \
  --output results/evaluation/plan_mcts_belief_meta_screen50.json
```

## 外部专家策略蒸馏

### 原生 Mega Lucario 专家 parity

`MegaLucarioExpertAgent` 是公开 Mega Lucario notebook 的可重置原生实现。改动专家逻辑后，先让
notebook 驱动真实对局、原生版本逐状态 shadow，要求动作顺序严格一致：

```bash
python -m poketcg.rl.expert_parity \
  --expert-source artifacts/public_agents/mega_lucario/a-sample-rule-based-agent-mega-lucario-ex-deck.ipynb \
  --deck artifacts/public_agents/mega_lucario/deck.csv \
  --games-per-seat 10 \
  --output results/evaluation/mega_native_expert_parity10x2.json
```

`parity.exact_match_rate` 必须为 `1.0`，且 `context_mismatches` 为空。评测面板中的原生候选名为
`mega-expert`，可直接与同一个 external notebook 做镜像确认。

`collect_external_expert` 只记录经过人工检查的外部专家一方的 observation/action，并在对局结束
后从专家视角回填 Value target。采集日程对每种对手轮换两个座位，避免把对手类型和先后手混在
一起。每局都会重新执行一次专家源码，确保公开 Agent 保存在模块全局变量中的攻击计划不会泄漏
到下一局。支持三种对手：通用 `rule`、当前 `policy` 和同一外部专家的 `mirror`。

推荐从覆盖多选动作的 Actions V2 checkpoint 开始，让卡牌搜索、能量附着和弃牌选择也接受专家
监督。V3 JSONL 约为每局 0.8 MB，Colab 8 CPU 的第一轮先采集 1200 局（每种对手、每个座位
各 200 局），避免一开始产生约 5 GB 的 6000 局数据：

```bash
python -m poketcg.rl.collect_external_expert \
  --expert-source artifacts/public_agents/mega_lucario/a-sample-rule-based-agent-mega-lucario-ex-deck.ipynb \
  --expert-deck artifacts/public_agents/mega_lucario/deck.csv \
  --opponent rule --opponent policy --opponent mirror \
  --policy-checkpoint /content/drive/MyDrive/pokemonTCG/checkpoints/ppo_v3_actions_v2_best_response_iter0018_iter0006.pt \
  --games 1200 --workers 8 --torch-threads 1 \
  --encoder-version 3 --include-multiselect --seed 20260821 \
  --output /content/drive/MyDrive/pokemonTCG/expert_distillation/mega_lucario_expert_v3_1200.jsonl
```

如果从旧 `ppo_v3_semantic_population_iter0018.pt` 开始，则必须去掉
`--include-multiselect`，因为该 checkpoint 是 Action Space V1。采集器仍会执行专家的全部动作，
但只保存 checkpoint 能学习的单选决策。

从原 checkpoint 小学习率微调，并混入 25% 旧 Rule BC 样本以减轻只会 Mega 牌组的灾难性
遗忘。主数据和 replay 数据必须使用相同 Encoder/Action Space 版本：

```bash
python -m poketcg.rl.train_bc \
  --input /content/drive/MyDrive/pokemonTCG/expert_distillation/mega_lucario_expert_v3_1200.jsonl \
  --replay-input /content/drive/MyDrive/pokemonTCG/bc/rule_v3_semantic_actions_v2_2000.jsonl \
  --replay-fraction 0.25 \
  --initialize-from /content/drive/MyDrive/pokemonTCG/checkpoints/ppo_v3_actions_v2_best_response_iter0018_iter0006.pt \
  --output /content/drive/MyDrive/pokemonTCG/checkpoints/mega_lucario_distilled_round1.pt \
  --epochs 5 --batch-size 128 --learning-rate 0.00003 \
  --value-coefficient 0.1 --max-grad-norm 0.5 \
  --device cuda --seed 20260821
```

第一验收项不是 Rule 胜率，而是相同 Mega Lucario 牌组下对原专家的镜像胜率。Policy 从约 27%
升至 40% 以上才说明蒸馏获得了实质效果；接近 50% 表示大部分专家策略已经复现。之后再把新
checkpoint 作为 MCTS prior，判断搜索是否终于建立在有效策略之上。

## DAgger：在学生访问的状态上查询专家

普通 offline BC 只覆盖专家自己访问的状态。`collect_dagger` 让学生实际参与对局，同时在学生
一方的每个决策上调用一个影子 Mega Lucario 专家，并把“学生状态、专家动作”写入数据。影子
专家也会收到 forced selection，因此其跨决策的 `plan` 和 `ability_used` 能持续更新；只有学生
可学习的决策才进入 JSONL。动作以概率 `beta` 执行专家动作，否则执行学生动作。

Round 1 使用蒸馏模型、`beta=0.5` 和 600 局，三类对手 × 双座位各 100 局：

```bash
python -m poketcg.rl.collect_dagger \
  --student-checkpoint /content/drive/MyDrive/pokemonTCG/checkpoints/mega_lucario_distilled_round1.pt \
  --expert-source /content/drive/MyDrive/pokemonTCG/public_agents/mega_lucario/a-sample-rule-based-agent-mega-lucario-ex-deck.ipynb \
  --expert-deck /content/drive/MyDrive/pokemonTCG/public_agents/mega_lucario/deck.csv \
  --opponent rule --opponent policy --opponent mirror \
  --beta 0.5 --games 600 --workers 8 --torch-threads 1 \
  --seed 20260824 \
  --output /content/drive/MyDrive/pokemonTCG/expert_distillation/mega_dagger_round1_beta050_600.jsonl
```

采集摘要中的 `realized_beta` 应接近 0.5；`disagreement_rate` 是学生与专家在学生状态上的动作
分歧率，是后续轮次是否仍有新信息的核心指标。对局胜率包含专家干预，不能作为学生自身胜率。

以 DAgger 数据为主数据，同时 replay 原始专家轨迹和 Rule BC。Value target 来自混合执行策略，
所以这一轮降低 Value loss 权重，主要学习 Policy：

```bash
python -m poketcg.rl.train_bc \
  --input /content/drive/MyDrive/pokemonTCG/expert_distillation/mega_dagger_round1_beta050_600.jsonl \
  --replay-input /content/drive/MyDrive/pokemonTCG/expert_distillation/mega_lucario_expert_v3_1200.jsonl \
  --replay-input /content/drive/MyDrive/pokemonTCG/bc/rule_v3_semantic_actions_v2_2000.jsonl \
  --replay-fraction 0.5 \
  --initialize-from /content/drive/MyDrive/pokemonTCG/checkpoints/mega_lucario_distilled_round1.pt \
  --output /content/drive/MyDrive/pokemonTCG/checkpoints/mega_dagger_round1.pt \
  --epochs 4 --batch-size 128 --learning-rate 0.00002 \
  --value-coefficient 0.05 --max-grad-norm 0.5 \
  --device cuda --seed 20260824
```

Round 1 必须用 `--deterministic` 做 Mega 镜像 Policy 评测。若有提升，再依次用新 checkpoint
采集 `beta=0.25` 和 `beta=0.10`；每轮都聚合历史 DAgger 数据，而不是覆盖旧数据。

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

构建实验性的“显式规划 + 神经策略 + MCTS”提交：

```bash
python -m poketcg.submission \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --deck artifacts/public_agents/mega_lucario/deck.csv \
  --tactical-planner --planner-threshold 0.8 --planner-weight 4.0 \
  --mcts-simulations 8 --mcts-determinizations 1 \
  --output artifacts/submissions/mega_planner_mcts8/submission.tar.gz
```

构建不含 MCTS 的整回合 ownership 实验提交：

```bash
python -m poketcg.submission \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --deck artifacts/public_agents/mega_lucario/deck.csv \
  --planner-turn-ownership --planner-threshold 0.8 --planner-weight 4.0 \
  --output artifacts/submissions/mega_turn_ownership/submission.tar.gz
```

严格 commitment ownership 使用：

```bash
python -m poketcg.submission \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --deck artifacts/public_agents/mega_lucario/deck.csv \
  --planner-commitment-ownership \
  --planner-threshold 0.8 --planner-weight 4.0 \
  --output artifacts/submissions/mega_commitment_ownership/submission.tar.gz
```

构建使用 fixed-model prior 的 Plan-level MCTS V0：

```bash
python -m poketcg.submission \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --deck artifacts/public_agents/mega_lucario/deck.csv \
  --plan-mcts --plan-determinizations 4 --plan-max-steps 32 \
  --planner-threshold 0.8 --planner-weight 4.0 \
  --output artifacts/submissions/mega_plan_mcts_d4/submission.tar.gz
```

构建与 meta panel 相同的多牌组 belief 版本：

```bash
python -m poketcg.submission \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --deck artifacts/public_agents/mega_lucario/deck.csv \
  --plan-mcts --plan-mcts-prior belief \
  --plan-determinizations 4 --plan-max-steps 32 \
  --planner-threshold 0.8 --planner-weight 4.0 \
  --belief-deck sample=data/official/sample_submission/sample_submission/deck.csv \
  --belief-deck meta_a=configs/opponent_decks/meta_a.csv \
  --belief-deck fishcat_v8=configs/opponent_decks/fishcat_v8.csv \
  --belief-deck mcts_sample=configs/opponent_decks/mcts_sample.csv \
  --output artifacts/submissions/mega_plan_mcts_belief_d4/submission.tar.gz
```

每个 `--belief-deck NAME=PATH` 的60张卡ID会写入 `agent_config.json`，runtime 还会自动追加提交使用的
我方牌组为 `model` 假设。未显式选择 `--plan-mcts-prior belief` 时仍保持原来的 fixed-model 行为。

构建不使用神经决策的原生 Mega Lucario 专家基准：

```bash
python -m poketcg.submission \
  --checkpoint artifacts/checkpoints/ppo_v3_semantic_population_iter0018.pt \
  --deck artifacts/public_agents/mega_lucario/deck.csv \
  --mega-expert \
  --output artifacts/submissions/mega_native_expert/submission.tar.gz
```

checkpoint 仅保留为运行时异常的延迟 fallback；正常 `mega-expert` 模式不会初始化或调用神经模型。

归档中的 `agent_config.json` 决定线上使用 direct policy、planner-policy 还是 planner-MCTS。
MCTS runtime 使用官方 Search API；初始化或搜索失败时依次降级到混合/原 PPO policy、
RuleAgent 和最小合法动作。纯规划器消融可用 `--planner-only` 构建。

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
