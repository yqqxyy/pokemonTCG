from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from poketcg.rl.features import OPTION_FEATURE_SIZE, STATE_FEATURE_SIZE, EncodedDecision
from poketcg.rl.model import CandidatePolicyValueNet
from poketcg.rl.residual_data import (
    ResidualDataset,
    ResidualExample,
    collate_residual,
    normalize_rule_scores,
)
from poketcg.rl.train_residual import outcome_weights, residual_loss


def _example(value: float = 1.0) -> ResidualExample:
    return ResidualExample(
        decision=EncodedDecision(
            state=[0.0] * STATE_FEATURE_SIZE,
            select_type=1,
            context=2,
            options=[[0.0] * OPTION_FEATURE_SIZE for _ in range(3)],
            option_types=[0, 1, 2],
            areas=[0, 0, 0],
            in_play_areas=[0, 0, 0],
        ),
        baseline_action=[1],
        target_action=[1],
        rule_scores=normalize_rule_scores([1.0, 5.0, -2.0]),
        value_target=value,
        player=0,
        game=4,
        decision_index=7,
        opponent="mirror",
    )


def test_rule_score_normalization_preserves_order_and_ties() -> None:
    scores = normalize_rule_scores([10.0, -5.0, 10.0, 2.0])

    assert scores[0] == scores[2]
    assert scores[0] > scores[3] > scores[1]
    assert normalize_rule_scores([7.0, 7.0]) == [0.0, 0.0]


def test_residual_jsonl_round_trip_and_collation(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    example = _example()
    path.write_text(json.dumps(example.to_dict()) + "\n", encoding="utf-8")

    dataset = ResidualDataset.from_jsonl(path)
    batch = collate_residual(dataset.examples)

    assert len(dataset) == 1
    assert batch["rule_scores"].shape == (1, 3)
    assert batch["baseline_action_mask"].tolist() == [[False, True, False]]


def test_residual_loss_is_finite_and_outcomes_are_weighted() -> None:
    model = CandidatePolicyValueNet(hidden_size=32, value_bins=11)
    batch = collate_residual([_example(1.0), _example(-1.0)])

    loss, combined, value_logits, policy_loss = residual_loss(
        model,
        batch,
        prior_strength=2.0,
        value_coefficient=0.2,
        residual_coefficient=0.01,
        win_weight=1.0,
        draw_weight=0.5,
        loss_weight=0.25,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(policy_loss)
    assert combined.shape == (2, 3)
    assert value_logits.shape == (2, 11)
    assert outcome_weights(
        torch.tensor([1.0, 0.0, -1.0]),
        win_weight=1.0,
        draw_weight=0.5,
        loss_weight=0.25,
    ).tolist() == pytest.approx([1.0, 0.5, 0.25])
