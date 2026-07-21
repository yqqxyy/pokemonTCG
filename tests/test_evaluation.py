from pathlib import Path

import pytest

from poketcg.agents import HybridPolicyAgent
from poketcg.match import MatchResult
from poketcg.rl.evaluate_meta_panel import _named_path, _report_views, _summary
from poketcg.rl.evaluate_panel import wilson_interval


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_interval(90, 100)

    assert low < 0.9 < high
    assert 0.8 < low < 0.9
    assert 0.9 < high < 1.0


def test_wilson_interval_handles_boundaries() -> None:
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0


class _RecordingPolicy:
    def __init__(self, action: list[int]) -> None:
        self.action = action
        self.calls = 0

    def choose_action(self, observation: dict) -> list[int]:
        self.calls += 1
        return self.action


def test_hybrid_policy_routes_single_and_multiselect_decisions() -> None:
    single = _RecordingPolicy([1])
    multiselect = _RecordingPolicy([0, 2])
    agent = HybridPolicyAgent.__new__(HybridPolicyAgent)
    agent._single_policy = single
    agent._multiselect_policy = multiselect

    assert agent.choose_action({"select": {"minCount": 1, "maxCount": 1}}) == [1]
    assert agent.choose_action({"select": {"minCount": 0, "maxCount": 2}}) == [0, 2]
    assert single.calls == 1
    assert multiselect.calls == 1


def _match(winner: int) -> MatchResult:
    return MatchResult(
        game=0,
        winner=winner,
        turns=3,
        decisions=7,
        elapsed_ms=10.0,
        player0="candidate",
        player1="opponent",
        agent_seed0=1,
        agent_seed1=2,
    )


def test_meta_panel_reports_per_cell_candidate_delta() -> None:
    policy_player0 = _summary([(_match(1), 0)])
    policy_player1 = _summary([(_match(0), 1)])
    mcts_player0 = _summary([(_match(0), 0)])
    mcts_player1 = _summary([(_match(1), 1)])
    cells = [
        {
            "candidate": "policy",
            "opponent": "rule",
            "opponent_deck": "sample",
            "overall": _summary([(_match(1), 0), (_match(0), 1)]),
            "seats": {
                "as_player0": policy_player0,
                "as_player1": policy_player1,
            },
        },
        {
            "candidate": "mcts",
            "opponent": "rule",
            "opponent_deck": "sample",
            "overall": _summary([(_match(0), 0), (_match(1), 1)]),
            "seats": {
                "as_player0": mcts_player0,
                "as_player1": mcts_player1,
            },
        },
    ]

    views = _report_views(cells, ["policy", "mcts"])

    comparison = views["comparisons"]["mcts_minus_policy"]
    assert comparison["overall_win_rate_delta"] == 1.0
    assert comparison["cells_improved"] == 1
    assert comparison["cells_worsened"] == 0
    assert comparison["cells"][0]["player0_delta"] == 1.0
    assert comparison["cells"][0]["player1_delta"] == 1.0


def test_meta_panel_named_path_parser(tmp_path: Path) -> None:
    name, path = _named_path(f"meta={tmp_path / 'deck.csv'}", "--opponent-deck")

    assert name == "meta"
    assert path == (tmp_path / "deck.csv").resolve()
    with pytest.raises(ValueError, match="NAME=PATH"):
        _named_path("missing-separator", "--opponent-deck")
