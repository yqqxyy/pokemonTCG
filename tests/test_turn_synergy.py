from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from poketcg.mcts import HiddenStateGuess, SearchPosition
from poketcg.rl.collect_turn_synergy import (
    summarize_heldout_turn_plans,
    summarize_turn_synergy,
)
from poketcg.rl.heldout_turn_plan import HeldoutTurnPlanEvaluator
from poketcg.rl.paired_rollout import RootCandidate
from poketcg.rl.semantic_plan import (
    resolve_semantic_action,
    semantic_action,
)
from poketcg.rl.turn_synergy import TurnSynergyEvaluator


def _observation(state: str, *, result: int = -1) -> dict:
    return {
        "state": state,
        "current": {
            "result": result,
            "yourIndex": 0,
            "turn": 1,
        },
        "select": {
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": 7, "cardId": 10},
                {"type": 7, "cardId": 20},
            ],
        },
    }


class FakeBackend:
    def begin(self, observation: dict, hidden: HiddenStateGuess) -> SearchPosition:
        return SearchPosition(1, _observation("root"))

    def step(self, search_id: int, action: list[int]) -> SearchPosition:
        transitions = {
            (1, (0,)): SearchPosition(3, _observation("baseline_loss", result=1)),
            (1, (1,)): SearchPosition(2, _observation("setup")),
            (2, (0,)): SearchPosition(4, _observation("followup_loss", result=1)),
            (2, (1,)): SearchPosition(5, _observation("joint_win", result=0)),
        }
        return transitions[(search_id, tuple(action))]

    def end(self) -> None:
        return None


class FakeDeterminizer:
    def sample(self, observation: dict) -> HiddenStateGuess:
        return HiddenStateGuess([], [], [], [], [], [], "fake")


@dataclass
class AlwaysZeroPolicy:
    def choose_action(self, observation: dict) -> list[int]:
        return [0]


def _candidates(observation: dict) -> list[RootCandidate]:
    return [
        RootCandidate((0,), ("baseline",)),
        RootCandidate((1,), ("search",)),
    ]


def test_turn_search_finds_joint_change_missed_by_one_step() -> None:
    evaluator = TurnSynergyEvaluator(
        FakeDeterminizer(),
        AlwaysZeroPolicy,
        AlwaysZeroPolicy,
        _candidates,
        determinizations=2,
        beam_width=4,
        branch_width=2,
        max_plan_steps=4,
        backend=FakeBackend(),
    )

    result = evaluator.evaluate(_observation("root"))

    assert result["mean_baseline_return"] == -1.0
    assert result["mean_best_one_step_return"] == -1.0
    assert result["mean_best_full_turn_return"] == 1.0
    assert result["mean_synergy_gain"] == 2.0
    assert result["hidden_synergy_rate"] == 1.0
    assert result["joint_rescue_rate"] == 1.0
    assert result["different_continuation_rate"] == 1.0
    assert result["errors"] == {}
    assert result["samples"][0]["best_full_turn"]["sequence"] == [[1], [1]]


def test_semantic_action_survives_option_reordering() -> None:
    source = _observation("source")
    source["select"]["option"] = [
        {"type": 7, "cardId": 100},
        {"type": 7, "cardId": 200},
    ]
    directive = semantic_action(source, [0])
    replay = _observation("replay")
    replay["select"]["option"] = [
        {"type": 7, "cardId": 200},
        {"type": 7, "cardId": 100},
    ]

    assert resolve_semantic_action(replay, directive) == [1]


def test_heldout_evaluator_replays_one_fixed_joint_plan() -> None:
    evaluator = HeldoutTurnPlanEvaluator(
        FakeDeterminizer(),
        AlwaysZeroPolicy,
        AlwaysZeroPolicy,
        _candidates,
        proposal_determinizations=2,
        heldout_determinizations=2,
        plan_pool_size=4,
        beam_width=4,
        branch_width=2,
        max_plan_steps=4,
        backend=FakeBackend(),
    )

    result = evaluator.evaluate(_observation("root"))

    assert result["candidate_plans"] >= 2
    assert result["proposal_selected"]["paired_advantage"] == 2.0
    assert result["heldout_selected"]["paired_advantage"] == 2.0
    assert result["heldout_selected"]["replay_success_rate"] == 1.0
    assert result["heldout_accepted"] is True
    assert len(result["selected_plan"]["actions"]) == 2


def test_summary_clusters_synergy_by_state(tmp_path: Path) -> None:
    output = tmp_path / "synergy.jsonl"
    rows = [
        {
            "opponent": "mirror",
            "turn": 4,
            "diagnostic": {
                "mean_synergy_gain": 1.0,
                "samples": [
                    {
                        "synergy_gain": 2.0,
                        "one_step_gain": 0.0,
                        "full_turn_gain": 2.0,
                        "joint_rescue": True,
                        "joint_deviation_count": 2,
                    },
                    {
                        "synergy_gain": 0.0,
                        "one_step_gain": 0.0,
                        "full_turn_gain": 0.0,
                        "joint_rescue": False,
                        "joint_deviation_count": 0,
                    },
                ],
            },
        },
        {
            "opponent": "mirror",
            "turn": 4,
            "diagnostic": {
                "mean_synergy_gain": 0.0,
                "samples": [
                    {
                        "synergy_gain": 0.0,
                        "one_step_gain": 0.0,
                        "full_turn_gain": 0.0,
                        "joint_rescue": False,
                        "joint_deviation_count": 0,
                    }
                ],
            },
        },
    ]
    output.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    summary = summarize_turn_synergy(output)

    assert summary["states"] == 2
    assert summary["effective_hidden_worlds"] == 3
    assert summary["state_synergy_rate"] == 0.5
    assert summary["state_joint_rescue_rate"] == 0.5
    assert summary["by_opponent"]["mirror"]["state_synergy_rate"] == 0.5
    lower, upper = summary["state_synergy_wilson95"]
    assert lower < 0.5 < upper


def test_heldout_summary_reports_optimism_and_replay(tmp_path: Path) -> None:
    output = tmp_path / "heldout.jsonl"
    rows = [
        {
            "opponent": "mirror",
            "turn": 4,
            "diagnostic": {
                "candidate_plans": 8,
                "heldout_accepted": True,
                "proposal_selected": {"paired_advantage": 1.0},
                "heldout_selected": {
                    "paired_advantage": 0.5,
                    "replay_success_rate": 1.0,
                    "mean_resolved_fraction": 1.0,
                },
            },
        },
        {
            "opponent": "mirror",
            "turn": 5,
            "diagnostic": {
                "candidate_plans": 4,
                "heldout_accepted": False,
                "proposal_selected": {"paired_advantage": 1.5},
                "heldout_selected": {
                    "paired_advantage": -0.5,
                    "replay_success_rate": 0.5,
                    "mean_resolved_fraction": 0.75,
                },
            },
        },
    ]
    output.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    summary = summarize_heldout_turn_plans(output)

    assert summary["states"] == 2
    assert summary["positive_heldout_rate"] == 0.5
    assert summary["accepted_heldout_rate"] == 0.5
    assert summary["mean_proposal_gain"] == 1.25
    assert summary["mean_heldout_gain"] == 0.0
    assert summary["mean_optimism_gap"] == 1.25
    assert summary["mean_replay_success_rate"] == 0.75
    assert summary["mean_resolved_fraction"] == 0.875
    assert summary["mean_candidate_plans"] == 6.0
