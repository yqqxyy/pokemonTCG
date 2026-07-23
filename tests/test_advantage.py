from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from poketcg.rl.advantage_data import (
    AdvantageDataset,
    AdvantageExample,
    collate_advantage,
)
from poketcg.rl.features import OPTION_FEATURE_SIZE, STATE_FEATURE_SIZE, EncodedDecision
from poketcg.rl.model import CandidatePolicyValueNet
from poketcg.rl.train_advantage import advantage_loss, predicted_advantage


def _paired_row() -> dict:
    decision = EncodedDecision(
        state=[0.0] * STATE_FEATURE_SIZE,
        select_type=0,
        context=0,
        options=[[0.0] * OPTION_FEATURE_SIZE for _ in range(3)],
        option_types=[1, 2, 3],
        areas=[0, 0, 0],
        in_play_areas=[0, 0, 0],
    )
    return {
        "state_id": "w0-g0-d1",
        "game": 0,
        "opponent": "mirror",
        "player": 0,
        "selection_reason": "round0_disagreement",
        "decision": decision.to_dict(),
        "rule_action": [1],
        "rollout": {
            "candidates": [
                {
                    "action": [1],
                    "paired_advantage": 0.0,
                    "paired_stderr": 0.0,
                    "effective_pairs": 16,
                },
                {
                    "action": [0],
                    "paired_advantage": 0.375,
                    "paired_stderr": 0.25,
                    "effective_pairs": 16,
                },
                {
                    "action": [2],
                    "paired_advantage": -0.125,
                    "paired_stderr": 0.4,
                    "effective_pairs": 15,
                },
            ]
        },
    }


def test_paired_advantage_loading_and_collation(tmp_path: Path) -> None:
    path = tmp_path / "paired.jsonl"
    path.write_text(json.dumps(_paired_row()) + "\n", encoding="utf-8")

    dataset = AdvantageDataset.from_jsonl(path)
    batch = collate_advantage(dataset.examples)

    assert len(dataset) == 1
    assert batch["baseline_index"].tolist() == [1]
    assert batch["advantage_mask"].tolist() == [[True, False, True]]
    assert batch["advantage_target"].tolist()[0] == pytest.approx([0.375, 0.0, -0.125])
    assert batch["effective_pairs"].tolist() == [[16, 0, 15]]


def test_predicted_advantage_is_invariant_to_common_logit_shift() -> None:
    baseline = torch.tensor([1])
    first = predicted_advantage(torch.tensor([[0.2, 0.5, -0.1]]), baseline)
    shifted = predicted_advantage(torch.tensor([[9.2, 9.5, 8.9]]), baseline)

    assert torch.allclose(first, shifted, atol=1e-6)
    assert first[0, 1] == 0.0


def test_advantage_loss_is_finite() -> None:
    row = _paired_row()
    batch = collate_advantage([AdvantageExample.from_paired_row(row)])
    model = CandidatePolicyValueNet(hidden_size=32, value_bins=11)

    loss, prediction, weights = advantage_loss(
        model,
        batch,
        noise_floor=0.15,
        maximum_weight=20.0,
        huber_delta=0.25,
    )

    assert torch.isfinite(loss)
    assert prediction.shape == (1, 3)
    assert weights[0, 0] > weights[0, 2]
