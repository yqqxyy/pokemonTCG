from __future__ import annotations

import pytest
import torch

from poketcg.agents.advantage_agent import conservative_candidate_scores
from poketcg.rl.evaluate_advantage import _parse_transition


def test_conservative_candidate_scores_are_baseline_relative() -> None:
    scores = conservative_candidate_scores(
        [
            torch.tensor([1.0, 1.4, 0.7]),
            torch.tensor([3.0, 3.2, 2.8]),
        ],
        baseline_index=0,
        candidate_indices=[1, 2],
        uncertainty_multiplier=1.0,
    )

    assert scores[1] == pytest.approx(0.2)
    assert scores[2] == pytest.approx(-0.3)


def test_single_member_has_zero_ensemble_penalty() -> None:
    scores = conservative_candidate_scores(
        [torch.tensor([0.5, 0.8])],
        baseline_index=0,
        candidate_indices=[1],
        uncertainty_multiplier=4.0,
    )

    assert scores[1] == pytest.approx(0.3)


def test_transition_parser() -> None:
    assert _parse_transition("7->14") == (7, 14)
