from __future__ import annotations

from pathlib import Path

import pytest

from poketcg.rl.collect_external_expert import (
    _validate_configuration,
    scheduled_matchup,
)
from poketcg.rl.data import BCExample, write_jsonl
from poketcg.rl.features import OPTION_FEATURE_SIZE, STATE_FEATURE_SIZE, EncodedDecision
from poketcg.rl.train_bc import load_training_dataset


def _example(game: int, action: int = 0) -> BCExample:
    return BCExample(
        decision=EncodedDecision(
            state=[0.0] * STATE_FEATURE_SIZE,
            select_type=0,
            context=0,
            options=[[0.0] * OPTION_FEATURE_SIZE for _ in range(2)],
            option_types=[0, 1],
            areas=[0, 0],
            in_play_areas=[0, 0],
        ),
        action=action,
        value_target=1.0,
        player=0,
        game=game,
    )


def test_expert_schedule_crosses_each_opponent_with_both_seats() -> None:
    opponents = ("rule", "policy", "mirror")

    matchups = [scheduled_matchup(opponents, game) for game in range(6)]

    assert matchups == [
        ("rule", 0),
        ("rule", 1),
        ("policy", 0),
        ("policy", 1),
        ("mirror", 0),
        ("mirror", 1),
    ]


def test_policy_opponent_requires_checkpoint() -> None:
    with pytest.raises(ValueError, match="policy opponent"):
        _validate_configuration(("rule", "policy"), None)


def test_replay_mix_reaches_fraction_and_separates_game_ids(tmp_path: Path) -> None:
    primary_path = tmp_path / "expert.jsonl"
    replay_path = tmp_path / "rule.jsonl"
    write_jsonl(primary_path, [_example(game) for game in range(6)])
    write_jsonl(replay_path, [_example(game, action=1) for game in range(6)])

    dataset, summary = load_training_dataset(
        primary_path,
        replay_paths=[replay_path],
        replay_fraction=0.25,
        seed=7,
    )

    assert len(dataset) == 8
    assert summary["primary_examples"] == 6
    assert summary["replay_examples"] == 2
    assert summary["actual_replay_fraction"] == 0.25
    assert all(example.game >= 6 for example in dataset.examples[6:])


def test_replay_fraction_requires_replay_input(tmp_path: Path) -> None:
    primary_path = tmp_path / "expert.jsonl"
    write_jsonl(primary_path, [_example(0), _example(1)])

    with pytest.raises(ValueError, match="requires at least one replay"):
        load_training_dataset(primary_path, replay_fraction=0.2)
