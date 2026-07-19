import argparse
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from poketcg.rl.collect_bc import optimal_policy_target
from poketcg.rl.data import BCExample, collate_bc
from poketcg.rl.features import (
    HISTORY_FEATURE_SIZE,
    OPTION_FEATURE_SIZE,
    SEMANTIC_FEATURE_SIZE,
    SEMANTIC_TAGS,
    STATE_FEATURE_SIZE,
    EncodedDecision,
    expand_semantic_features,
    pack_semantic_features,
    structured_semantic_features,
)
from poketcg.rl.model import (
    CandidatePolicyValueNet,
    TokenPolicyValueNet,
    build_model,
    categorical_value_targets,
    encoder_version,
)
from poketcg.rl.opponent_pool import OpponentPool
from poketcg.rl.train_ppo import (
    IterationMetrics,
    PPOTransition,
    assign_episode_returns,
    compute_gae,
    explained_variance,
    initialize_wandb,
    iteration_log_values,
    partition_rollout_assignments,
    potential_shaping_rewards,
    prize_potential,
    summarize_opponent_results,
)


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


def _v3_example(
    token_count: int,
    option_count: int = 3,
    history_count: int = 2,
) -> BCExample:
    example = _v2_example(token_count, option_count)
    decision = example.decision
    decision.version = 3
    decision.token_semantics = [
        [float(index == 0)] * SEMANTIC_FEATURE_SIZE for index in range(token_count)
    ]
    decision.option_semantics = [
        [float(index == 1)] * SEMANTIC_FEATURE_SIZE for index in range(option_count)
    ]
    decision.history_features = [
        [index / max(history_count, 1)] * HISTORY_FEATURE_SIZE
        for index in range(history_count)
    ]
    decision.history_types = list(range(history_count))
    decision.history_owners = [1] * history_count
    decision.history_card_ids = list(range(1, history_count + 1))
    decision.history_target_card_ids = [0] * history_count
    decision.history_attack_ids = [0] * history_count
    decision.history_from_zones = [2] * history_count
    decision.history_to_zones = [4] * history_count
    return example


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


def test_v3_semantic_history_model_handles_independent_padding() -> None:
    batch = collate_bc([_v3_example(0, 2, 0), _v3_example(4, 3, 2)])
    model = TokenPolicyValueNet(
        hidden_size=32,
        value_bins=21,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        card_vocab_size=64,
        attack_vocab_size=64,
        semantic_feature_size=SEMANTIC_FEATURE_SIZE,
        history_feature_size=HISTORY_FEATURE_SIZE,
    )

    policy_logits, value_logits = model(batch)
    (policy_logits[:, 0].sum() + value_logits.sum()).backward()

    assert batch["history_mask"].tolist() == [[False, False], [True, True]]
    assert batch["token_semantics"].shape == (2, 4, SEMANTIC_FEATURE_SIZE)
    assert batch["option_semantics"].shape == (2, 3, SEMANTIC_FEATURE_SIZE)
    assert policy_logits.shape == (2, 3)
    assert value_logits.shape == (2, 21)
    assert policy_logits[0, 2].item() < -1e20


@pytest.mark.parametrize(
    ("use_card_semantics", "use_history"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_v3_model_supports_semantic_history_ablations(
    use_card_semantics: bool,
    use_history: bool,
) -> None:
    batch = collate_bc([_v3_example(2, 3, 2)])
    config = {
        "model_type": "transformer_v3",
        "hidden_size": 32,
        "value_bins": 21,
        "num_layers": 1,
        "num_heads": 4,
        "dropout": 0.0,
        "card_vocab_size": 64,
        "attack_vocab_size": 64,
        "use_card_semantics": use_card_semantics,
        "use_history": use_history,
    }

    policy_logits, value_logits = build_model(config)(batch)

    assert encoder_version(config) == 3
    assert policy_logits.shape == (1, 3)
    assert value_logits.shape == (1, 21)


def test_structured_semantics_extract_card_effects_without_catalog_artifacts() -> None:
    skill = SimpleNamespace(
        text="Search your deck for up to 3 Pokemon ex, reveal them, and put them into your hand. "
        "Then, shuffle your deck."
    )
    attack = SimpleNamespace(
        text="Discard 2 Energy from this Pokemon. This attack does 20 more damage.",
        damage=130,
        energies=[3, 3, 0],
    )

    features = structured_semantic_features(SimpleNamespace(skills=[skill]), [attack])
    tags = dict(zip(SEMANTIC_TAGS, features[: len(SEMANTIC_TAGS)], strict=True))

    assert len(features) == SEMANTIC_FEATURE_SIZE
    assert tags["search_deck"] == 1.0
    assert tags["reveal"] == 1.0
    assert tags["shuffle"] == 1.0
    assert tags["discard"] == 1.0
    assert tags["damage_scaling"] == 1.0
    assert features[len(SEMANTIC_TAGS) + 1] == pytest.approx(130 / 400)
    assert features[len(SEMANTIC_TAGS) + 4] == pytest.approx(3 / 5)
    assert expand_semantic_features(pack_semantic_features(features)) == features


def test_categorical_targets_are_normalized() -> None:
    support = torch.linspace(-1.0, 1.0, 21)
    targets = categorical_value_targets(torch.tensor([-1.0, 0.0, 1.0]), support)

    assert torch.allclose(targets.sum(dim=-1), torch.ones(3))
    assert targets.argmax(dim=-1).tolist() == [0, 10, 20]


def test_gae_uses_terminal_reward_on_final_player_decision() -> None:
    advantages, targets = compute_gae([0.0, 0.0], 1.0, gamma=1.0, gae_lambda=0.5)

    assert advantages == [0.5, 1.0]
    assert targets == [0.5, 1.0]


def test_gae_accepts_dense_per_step_rewards() -> None:
    advantages, targets = compute_gae(
        [0.0, 0.0],
        rewards=[0.2, 0.8],
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert advantages == pytest.approx([1.0, 0.8])
    assert targets == pytest.approx([1.0, 0.8])


def test_prize_potential_is_relative_progress_from_learning_player() -> None:
    observation = {
        "current": {
            "players": [
                {"prize": [1, 2, 3, 4]},
                {"prize": [1, 2, 3, 4, 5, 6]},
            ]
        }
    }

    assert prize_potential(observation, 0) == pytest.approx(2.0 / 6.0)
    assert prize_potential(observation, 1) == pytest.approx(-2.0 / 6.0)


def test_prize_shaping_telescopes_to_zero_with_terminal_zero_potential() -> None:
    rewards = potential_shaping_rewards(
        [0.0, 1.0 / 6.0, 2.0 / 6.0],
        gamma=1.0,
        scale=1.0,
    )

    assert rewards == pytest.approx([1.0 / 6.0, 1.0 / 6.0, -2.0 / 6.0])
    assert sum(rewards) == pytest.approx(0.0)


def test_prize_shaping_changes_actor_advantage_but_not_critic_target() -> None:
    episode = [
        PPOTransition(_example(2).decision, 0, 0.0, 0.0, potential=0.0),
        PPOTransition(_example(2).decision, 0, 0.0, 0.0, potential=1.0 / 6.0),
    ]
    assign_episode_returns(
        episode,
        terminal_return=1.0,
        gamma=1.0,
        gae_lambda=0.5,
        value_gae_lambda=0.5,
        reward_shaping="prize",
        reward_shaping_scale=1.0,
    )

    assert [item.shaping_reward for item in episode] == pytest.approx(
        [1.0 / 6.0, -1.0 / 6.0]
    )
    assert [item.advantage for item in episode] == pytest.approx(
        [7.0 / 12.0, 5.0 / 6.0]
    )
    assert [item.value_target for item in episode] == pytest.approx([0.5, 1.0])


def test_value_lambda_can_use_monte_carlo_targets_with_shorter_policy_gae() -> None:
    episode = [
        PPOTransition(_example(2).decision, 0, 0.0, 0.0),
        PPOTransition(_example(2).decision, 0, 0.0, 0.0),
    ]
    assign_episode_returns(
        episode,
        terminal_return=1.0,
        gamma=1.0,
        gae_lambda=0.5,
        value_gae_lambda=1.0,
        reward_shaping="none",
        reward_shaping_scale=1.0,
    )

    assert [item.advantage for item in episode] == pytest.approx([0.5, 1.0])
    assert [item.value_target for item in episode] == pytest.approx([1.0, 1.0])


def test_explained_variance_handles_perfect_and_constant_targets() -> None:
    assert explained_variance([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]) == 1.0
    assert explained_variance([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_partition_rollout_assignments_balances_and_preserves_order() -> None:
    names = [f"opponent-{index}" for index in range(10)]
    shards = partition_rollout_assignments(names, workers=3)

    assert [len(indices) for indices, _ in shards] == [4, 3, 3]
    assert [index for indices, _ in shards for index in indices] == list(range(10))
    assert [name for _, shard_names in shards for name in shard_names] == names


def test_iteration_log_values_flattens_training_and_opponent_metrics() -> None:
    metrics = IterationMetrics(
        iteration=3,
        games=8,
        wins=5,
        draws=1,
        losses=2,
        transitions=120,
        mean_return=0.375,
        policy_loss=-0.01,
        value_loss=2.0,
        entropy=0.8,
        approximate_kl=0.005,
        clip_fraction=0.1,
        explained_variance=0.4,
        gradient_norm=1.5,
        rollout_seconds=2.0,
        update_seconds=0.5,
        games_per_second=4.0,
        mean_abs_shaping_reward=0.02,
        mean_shaping_return=0.0,
        opponents={
            "rule": {
                "games": 4,
                "wins": 3,
                "draws": 0,
                "losses": 1,
                "mean_return": 0.5,
            }
        },
        sampling_weights={"rule": 0.5},
    )

    values = iteration_log_values(metrics)

    assert values["rollout/win_rate"] == 0.625
    assert values["opponent/rule/win_rate"] == 0.75
    assert values["sampling_weight/rule"] == 0.5
    assert values["time/iteration_seconds"] == 2.5


def test_initialize_wandb_passes_safe_configuration(monkeypatch) -> None:
    captured: dict = {}

    class FakeWandb:
        @staticmethod
        def init(**options):
            captured.update(options)
            return object()

    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        return FakeWandb if name == "wandb" else real_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    args = argparse.Namespace(
        wandb_mode="offline",
        wandb_project="poketcg-test",
        wandb_entity=None,
        wandb_run_name="smoke",
        wandb_run_id=None,
        input=Path("checkpoint.pt"),
    )

    initialize_wandb(args)

    assert captured["mode"] == "offline"
    assert captured["project"] == "poketcg-test"
    assert captured["config"]["input"] == "checkpoint.pt"
    assert "wandb_run_id" not in captured["config"]


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
    worker_state = pool.worker_state()
    assert [item["name"] for item in worker_state] == [
        "random",
        "rule",
        "snapshot2",
        "snapshot3",
    ]
    assert worker_state[2]["model_config"] == config
    assert "model_state_dict" in worker_state[2]


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
