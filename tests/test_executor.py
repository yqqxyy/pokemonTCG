from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from poketcg.rl.action_space import subset_log_probability
from poketcg.rl.executor_data import (
    EXECUTOR_CONDITION_SIZE,
    ExecutorDataset,
    ExecutorExample,
    encode_executor_condition,
    load_executor_dataset,
    sanitize_public_decision,
    split_executor_dataset,
)
from poketcg.rl.features import (
    HISTORY_FEATURE_SIZE,
    OPTION_FEATURE_SIZE,
    SEMANTIC_FEATURE_SIZE,
    STATE_FEATURE_SIZE,
    TOKEN_FEATURE_SIZE,
    EncodedDecision,
    FeatureEncoderV3,
)
from poketcg.rl.train_executor import expected_subset_nll


def _decision(
    option_count: int = 3,
    *,
    minimum: int = 1,
    maximum: int = 1,
) -> EncodedDecision:
    return EncodedDecision(
        state=[0.0] * STATE_FEATURE_SIZE,
        select_type=0,
        context=3,
        options=[[0.0] * OPTION_FEATURE_SIZE for _ in range(option_count)],
        option_types=list(range(option_count)),
        areas=[0] * option_count,
        in_play_areas=[0] * option_count,
        version=3,
        tokens=[[0.0] * TOKEN_FEATURE_SIZE],
        token_card_ids=[58],
        token_kinds=[1],
        token_zones=[1],
        token_owners=[1],
        token_slots=[0],
        token_card_types=[1],
        token_energy_types=[0],
        token_weaknesses=[0],
        token_resistances=[0],
        option_card_ids=[0] * option_count,
        option_attack_ids=[0] * option_count,
        option_special_conditions=[0] * option_count,
        token_semantics=[[0.0] * SEMANTIC_FEATURE_SIZE],
        option_semantics=[
            [0.0] * SEMANTIC_FEATURE_SIZE for _ in range(option_count)
        ],
        history_features=[[0.0] * HISTORY_FEATURE_SIZE],
        history_types=[0],
        history_owners=[1],
        history_card_ids=[0],
        history_target_card_ids=[0],
        history_attack_ids=[0],
        history_from_zones=[0],
        history_to_zones=[0],
        minimum=minimum,
        maximum=maximum,
    )


def _plan(plan_type: str = "mill_four_now") -> dict:
    return {
        "plan_id": "plan-1",
        "plan_type": plan_type,
        "root_action": {"context": 0, "options": []},
        "primary_card_id": 58,
        "target_card_id": None,
        "attack_id": 62,
        "desired_tags": ["mill_deck", "discard"],
        "preserve_card_ids": [1185],
        "require_attack": True,
        "termination": "turn_end",
        "maximum_steps": 32,
        "source_action": [0],
        "sources": ["test"],
        "strategy_version": "libraryout_v2",
        "preferred_card_ids": [1185],
        "preferred_attack_ids": [62],
        "preconditions": ["ancient_supporter_played"],
        "success_conditions": ["land_collapse_used"],
        "public_signals": ["active=58"],
        "feasibility_score": 0.75,
    }


def _progress(decisions: int = 0) -> dict:
    return {
        "owner_player": 0,
        "start_turn": 3,
        "decisions": decisions,
        "contexts": [0] * decisions,
        "option_types": [7] * decisions,
        "played_card_ids": [1185] if decisions else [],
        "attack_ids": [],
        "plan_hits": decisions,
    }


def _row(world: int, action: list[int], *, group: str = "game-1") -> dict:
    return {
        "schema_version": 3,
        "example_type": "macro_executor_action",
        "state_id": f"state-{group}",
        "split_group": group,
        "plan_id": "plan-1",
        "plan_type": "mill_four_now",
        "determinization_id": world,
        "opponent": "mirror",
        "turn": 3,
        "executor_input": {
            "decision": _decision().to_dict(),
            "plan": _plan(),
            "progress": _progress(),
        },
        "target_action": action,
    }


def test_executor_condition_has_stable_schema_and_tracks_progress() -> None:
    start = encode_executor_condition(_plan(), _progress())
    advanced = encode_executor_condition(_plan(), _progress(decisions=2))

    assert len(start) == EXECUTOR_CONDITION_SIZE
    assert len(advanced) == EXECUTOR_CONDITION_SIZE
    assert start != advanced


def test_hidden_world_actions_become_one_public_soft_target(tmp_path: Path) -> None:
    path = tmp_path / "executor.jsonl"
    rows = [_row(0, [0]), _row(1, [1]), _row(2, [1])]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    dataset, summary = load_executor_dataset(path)
    example = dataset.examples[0]

    assert summary["raw_rows"] == 3
    assert summary["public_inputs"] == 1
    assert example.world_count == 3
    assert example.modal_action == [1]
    assert example.inclusion_target == pytest.approx([1 / 3, 2 / 3, 0.0])
    assert example.consensus_rate == pytest.approx(2 / 3)
    assert example.action_distribution == [
        ([0], pytest.approx(1 / 3)),
        ([1], pytest.approx(2 / 3)),
    ]


def test_public_sanitizer_removes_determinized_prize_identity() -> None:
    value = _decision().to_dict()
    value["tokens"].append([1.0] * TOKEN_FEATURE_SIZE)
    for name, item in (
        ("token_card_ids", 999),
        ("token_kinds", 1),
        ("token_zones", 6),
        ("token_owners", 2),
        ("token_slots", 0),
        ("token_card_types", 1),
        ("token_energy_types", 0),
        ("token_weaknesses", 0),
        ("token_resistances", 0),
    ):
        value[name].append(item)
    value["token_semantics"].append([1.0] * SEMANTIC_FEATURE_SIZE)

    decision = sanitize_public_decision(value)

    assert decision.token_zones == [1]
    assert decision.token_card_ids == [58]
    assert len(decision.tokens or []) == 1
    assert len(decision.token_semantics or []) == 1


def test_feature_encoder_never_exposes_face_down_prize_identity() -> None:
    player = {
        "deckCount": 40,
        "handCount": 5,
        "prize": [{"id": 999}],
        "bench": [],
        "benchMax": 5,
        "active": [],
        "discard": [],
        "hand": [],
    }
    observation = {
        "current": {
            "turn": 3,
            "turnActionCount": 0,
            "firstPlayer": 0,
            "yourIndex": 0,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "stadium": [],
            "players": [player, {**player, "prize": [{"id": 998}]}],
        },
        "select": {
            "type": 0,
            "context": 3,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {
                    "type": 1,
                    "area": 6,
                    "playerIndex": 0,
                    "index": 0,
                    "cardId": 999,
                }
            ],
        },
        "logs": [],
    }

    decision = FeatureEncoderV3({999: {}}, {}).encode(observation)

    assert 999 not in (decision.token_card_ids or [])
    assert decision.option_card_ids == [0]
    assert decision.option_semantics == [[0.0] * 9]


def test_expected_subset_nll_matches_explicit_multiselect_distribution() -> None:
    logits = torch.tensor([[0.2, -0.1, 0.4]])
    batch = {
        "option_mask": torch.tensor([[True, True, True]]),
        "inclusion_target": torch.tensor([[0.75, 0.0, 0.75]]),
        "minimum": torch.tensor([0]),
        "maximum": torch.tensor([2]),
    }
    actual = expected_subset_nll(logits, batch)
    empty = torch.tensor([False, False, False])
    pair = torch.tensor([True, False, True])
    expected = -(
        0.25 * subset_log_probability(logits[0], empty, 0, 2)
        + 0.75 * subset_log_probability(logits[0], pair, 0, 2)
    )

    assert actual.item() == pytest.approx(expected.item())


def test_executor_split_never_crosses_game_groups() -> None:
    examples = []
    for index in range(9):
        examples.append(
            ExecutorExample(
                decision=_decision(),
                condition=encode_executor_condition(_plan(), _progress()),
                modal_action=[0],
                inclusion_target=[1.0, 0.0, 0.0],
                action_distribution=[([0], 1.0)],
                consensus_rate=1.0,
                normalized_entropy=0.0,
                example_weight=1.0,
                observation_count=2,
                world_count=2,
                split_group=f"group-{index}",
                state_id=f"state-{index}",
                input_fingerprint=f"fingerprint-{index}",
                plan_type="mill_four_now",
                phase=("early", "mid", "late")[index % 3],
                context=3,
                opponent=("mirror", "rule", "strong")[index % 3],
            )
        )

    train, validation, test, manifest = split_executor_dataset(
        ExecutorDataset(examples),
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )
    train_groups = {example.split_group for example in train.examples}
    validation_groups = {example.split_group for example in validation.examples}
    test_groups = {example.split_group for example in test.examples}

    assert train_groups
    assert validation_groups
    assert test_groups
    assert not train_groups.intersection(validation_groups)
    assert not train_groups.intersection(test_groups)
    assert not validation_groups.intersection(test_groups)
    assert set(manifest) == {"train", "validation", "test"}
