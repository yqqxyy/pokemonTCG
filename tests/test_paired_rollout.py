from __future__ import annotations

import random

import pytest

from poketcg.mcts import HiddenStateGuess, SearchPosition
from poketcg.rl.collect_paired_rollouts import root_candidates, selection_reason
from poketcg.rl.paired_rollout import (
    PairedRolloutEvaluator,
    RootCandidate,
    paired_summary,
)


def _observation(*, result: int = -1, context: int = 0, turn: int = 1) -> dict:
    return {
        "current": {"result": result, "yourIndex": 0, "turn": turn},
        "select": {
            "context": context,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 1}, {"type": 2}, {"type": 1}],
        },
    }


class _Determinizer:
    def sample(self, observation: dict) -> HiddenStateGuess:
        return HiddenStateGuess([], [], [], [], [], [], "fixed")


class _Policy:
    def choose_action(self, observation: dict) -> list[int]:
        return [0]


class _TerminalBackend:
    def __init__(self) -> None:
        self.begin_calls = 0
        self.end_calls = 0

    def begin(self, observation: dict, hidden: HiddenStateGuess) -> SearchPosition:
        self.begin_calls += 1
        return SearchPosition(0, observation)

    def step(self, search_id: int, action: list[int]) -> SearchPosition:
        winner = 0 if action == [0] else 1
        return SearchPosition(search_id + 1, _observation(result=winner))

    def end(self) -> None:
        self.end_calls += 1


def test_paired_rollout_reuses_one_root_per_determinization() -> None:
    backend = _TerminalBackend()
    evaluator = PairedRolloutEvaluator(
        _Determinizer(),
        _Policy,
        _Policy,
        determinizations=4,
        backend=backend,
    )

    result = evaluator.evaluate(
        _observation(),
        [
            RootCandidate((0,), ("rule_choice",)),
            RootCandidate((1,), ("round0_top",)),
        ],
    )

    assert backend.begin_calls == 4
    assert backend.end_calls == 4
    assert result["candidates"][0]["paired_advantage"] == 0.0
    assert result["candidates"][1]["paired_advantage"] == -2.0
    assert result["candidates"][1]["effective_pairs"] == 4


def test_paired_summary_uses_difference_within_each_pair() -> None:
    result = paired_summary([1.0, -1.0, 1.0], [-1.0, -1.0, 1.0])

    assert result["paired_advantage"] == pytest.approx(2.0 / 3.0)
    assert result["positive_pair_rate"] == pytest.approx(1.0 / 3.0)


def test_candidate_generation_combines_rule_model_and_type_diversity() -> None:
    candidates = root_candidates(
        [0],
        [3.0, 2.0, 1.0, 0.0],
        [0.0, 1.0, 4.0, 3.0],
        [1, 1, 1, 2],
    )

    assert candidates[0].action == (0,)
    assert (2,) in {candidate.action for candidate in candidates}
    assert (3,) in {candidate.action for candidate in candidates}
    assert "diverse_type" in next(
        candidate.sources for candidate in candidates if candidate.action == (3,)
    )


def test_state_selection_prioritizes_disagreement_then_low_margin() -> None:
    assert (
        selection_reason(
            [0],
            [1.0, 0.0],
            [0.0, 1.0],
            low_margin_threshold=0.25,
            random_probability=0.0,
            rng=random.Random(1),
        )
        == "round0_disagreement"
    )
    assert (
        selection_reason(
            [0],
            [1.0, 0.9],
            [1.0, 0.0],
            low_margin_threshold=0.25,
            random_probability=0.0,
            rng=random.Random(1),
        )
        == "low_rule_margin"
    )
