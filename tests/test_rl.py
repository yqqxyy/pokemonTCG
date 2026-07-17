import torch

from poketcg.rl.collect_bc import optimal_policy_target
from poketcg.rl.data import BCExample, collate_bc
from poketcg.rl.features import OPTION_FEATURE_SIZE, STATE_FEATURE_SIZE, EncodedDecision
from poketcg.rl.model import (
    CandidatePolicyValueNet,
    TokenPolicyValueNet,
    categorical_value_targets,
)
from poketcg.rl.opponent_pool import OpponentPool
from poketcg.rl.train_ppo import compute_gae, summarize_opponent_results


def _example(option_count: int, action: int = 0) -> BCExample:
    return BCExample(
        decision=EncodedDecision(
            state=[0.0] * STATE_FEATURE_SIZE,
            select_type=0,
            context=0,
            options=[[0.0] * OPTION_FEATURE_SIZE for _ in range(option_count)],
            option_types=list(range(option_count)),
            areas=[0] * option_count,
            in_play_areas=[0] * option_count,
        ),
        action=action,
        value_target=1.0,
        player=0,
        game=0,
    )


def _v2_example(token_count: int, option_count: int = 3) -> BCExample:
    decision = _example(option_count).decision
    decision.version = 2
    decision.tokens = [[0.0] * 24 for _ in range(token_count)]
    decision.token_card_ids = [index + 1 for index in range(token_count)]
    decision.token_kinds = [1] * token_count
    decision.token_zones = [2] * token_count
    decision.token_owners = [1] * token_count
    decision.token_slots = [0] * token_count
    decision.token_card_types = [0] * token_count
    decision.token_energy_types = [0] * token_count
    decision.token_weaknesses = [0] * token_count
    decision.token_resistances = [0] * token_count
    decision.option_card_ids = list(range(1, option_count + 1))
    decision.option_attack_ids = [0] * option_count
    decision.option_special_conditions = [0] * option_count
    return BCExample(decision, action=0, value_target=1.0, player=0, game=0)


def test_collate_masks_variable_option_counts() -> None:
    batch = collate_bc([_example(2), _example(4, action=3)])

    assert batch["options"].shape == (2, 4, OPTION_FEATURE_SIZE)
    assert batch["option_mask"].tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]
    assert batch["policy_target"].tolist() == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_candidate_model_masks_padding_and_backpropagates() -> None:
    batch = collate_bc([_example(2), _example(4, action=3)])
    model = CandidatePolicyValueNet(hidden_size=32, value_bins=21)
    policy_logits, value_logits = model(batch)
    loss = torch.nn.functional.cross_entropy(policy_logits, batch["action"])
    loss.backward()

    assert policy_logits.shape == (2, 4)
    assert value_logits.shape == (2, 21)
    assert policy_logits[0, 2].item() < -1e20


def test_v2_attention_model_handles_token_and_option_padding() -> None:
    batch = collate_bc([_v2_example(0, 2), _v2_example(4, 3)])
    model = TokenPolicyValueNet(
        hidden_size=32,
        value_bins=21,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        card_vocab_size=64,
        attack_vocab_size=64,
    )
    policy_logits, value_logits = model(batch)
    (policy_logits[:, 0].sum() + value_logits.sum()).backward()

    assert batch["token_mask"].tolist() == [
        [False, False, False, False],
        [True, True, True, True],
    ]
    assert policy_logits.shape == (2, 3)
    assert value_logits.shape == (2, 21)
    assert policy_logits[0, 2].item() < -1e20


def test_categorical_targets_are_normalized() -> None:
    support = torch.linspace(-1.0, 1.0, 21)
    targets = categorical_value_targets(torch.tensor([-1.0, 0.0, 1.0]), support)

    assert torch.allclose(targets.sum(dim=-1), torch.ones(3))
    assert targets.argmax(dim=-1).tolist() == [0, 10, 20]


def test_gae_uses_terminal_reward_on_final_player_decision() -> None:
    advantages, targets = compute_gae([0.0, 0.0], 1.0, gamma=1.0, gae_lambda=0.5)

    assert advantages == [0.5, 1.0]
    assert targets == [0.5, 1.0]


def test_optimal_policy_target_is_uniform_over_ties() -> None:
    assert optimal_policy_target([2.0, 5.0, 5.0, 1.0]) == [0.0, 0.5, 0.5, 0.0]


def test_opponent_pool_splits_snapshot_group_weight_and_evicts_oldest() -> None:
    pool = OpponentPool(
        card_catalog={},
        attack_catalog={},
        seed=7,
        snapshot_weight=0.6,
        max_snapshots=2,
    )
    pool.add_random("random", 0.1)
    pool.add_rule("rule", 0.3)
    model = CandidatePolicyValueNet(hidden_size=32, value_bins=21)
    config = {"hidden_size": 32, "value_bins": 21}
    pool.add_snapshot("snapshot1", model, config)
    pool.add_snapshot("snapshot2", model, config)
    pool.add_snapshot("snapshot3", model, config)

    assert pool.manifest() == [
        {
            "name": "random",
            "kind": "random",
            "encoder_version": 1,
            "base_weight": 0.1,
            "effective_weight": 0.1,
            "games": 0,
            "win_rate": 0.5,
            "ema_win_rate": 0.5,
        },
        {
            "name": "rule",
            "kind": "rule",
            "encoder_version": 1,
            "base_weight": 0.3,
            "effective_weight": 0.3,
            "games": 0,
            "win_rate": 0.5,
            "ema_win_rate": 0.5,
        },
        {
            "name": "snapshot2",
            "kind": "snapshot",
            "encoder_version": 1,
            "base_weight": 0.3,
            "effective_weight": 0.3,
            "games": 0,
            "win_rate": 0.5,
            "ema_win_rate": 0.5,
        },
        {
            "name": "snapshot3",
            "kind": "snapshot",
            "encoder_version": 1,
            "base_weight": 0.3,
            "effective_weight": 0.3,
            "games": 0,
            "win_rate": 0.5,
            "ema_win_rate": 0.5,
        },
    ]


def test_win_rate_adaptive_sampling_downweights_easy_opponents() -> None:
    pool = OpponentPool(
        card_catalog={},
        attack_catalog={},
        seed=11,
        adaptive_sampling="win_rate",
        adaptive_min_multiplier=0.1,
        adaptive_ema_decay=0.0,
        adaptive_warmup_games=2,
    )
    pool.add_random("easy", 1.0)
    pool.add_rule("competitive", 1.0)
    pool.record_results(["easy", "easy"], [1.0, 1.0])
    pool.record_results(["competitive", "competitive"], [0.0, 0.0])

    assert pool.effective_weights() == {"easy": 0.1, "competitive": 1.0}


def test_summarize_opponent_results_separates_pool_members() -> None:
    assert summarize_opponent_results(
        ["rule", "random", "rule", "rule"], [1.0, -1.0, 1.0, -1.0]
    ) == {
        "random": {"games": 1, "wins": 0, "draws": 0, "losses": 1, "mean_return": -1.0},
        "rule": {"games": 3, "wins": 2, "draws": 0, "losses": 1, "mean_return": 0.333333},
    }
