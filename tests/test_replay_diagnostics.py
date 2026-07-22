from __future__ import annotations

import json
from pathlib import Path

from poketcg.rl.replay_diagnostics import analyze_replay, build_report


def _replay(*, loss: bool = True) -> dict:
    our_deck = [58] * 4 + [345] * 4 + [1185] * 4 + list(range(48))
    opponent_deck = [344] * 4 + [345] * 4 + [999] * 52
    players = [
        {
            "deck": [
                {"id": card, "name": "Great Tusk" if card == 58 else "Card"} for card in our_deck
            ],
            "hand": [],
            "active": [{"id": 58}],
            "bench": [],
            "discard": [],
            "prize": [None] * (2 if loss else 0),
            "deckCount": 20,
        },
        {
            "deck": [
                {
                    "id": card,
                    "name": "Dwebble" if card == 344 else ("Crustle" if card == 345 else "Card"),
                }
                for card in opponent_deck
            ],
            "hand": [],
            "active": [{"id": 345}],
            "bench": [],
            "discard": [],
            "prize": [] if loss else [None] * 3,
            "deckCount": 14,
        },
    ]
    initial = {
        "action": [our_deck, opponent_deck],
        "current": {"players": players, "turn": 1},
    }
    observation = {
        "current": {"players": players, "turn": 5},
        "select": {
            "option": [
                {"type": 13, "attackId": 62},
                {"type": 14},
            ]
        },
    }
    return {
        "configuration": {"seed": 42},
        "info": {"EpisodeId": 7, "TeamNames": ["Yqqxyy ", "Opponent"]},
        "rewards": [-1, 1] if loss else [1, -1],
        "steps": [
            [
                {"action": [], "observation": {}, "visualize": [initial]},
                {"action": [], "observation": {}, "visualize": []},
            ],
            [
                {"action": [0], "observation": observation, "visualize": []},
                {"action": [], "observation": {}, "visualize": []},
            ],
            [
                {
                    "action": [],
                    "observation": {},
                    "visualize": [{"current": {"players": players, "turn": 5}}],
                },
                {"action": [], "observation": {}, "visualize": []},
            ],
        ],
    }


def test_analyze_replay_detects_wall_loss_and_mill_metrics(tmp_path: Path) -> None:
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(_replay()))

    result = analyze_replay(path)

    assert result["opponent_archetype"] == "crustle_wall"
    assert result["outcome"] == "loss"
    assert result["strategy"]["land_collapse_uses"] == 1
    assert result["strategy"]["first_land_collapse_turn"] == 5
    assert "wall_or_mirror_matchup" in result["failure_tags"]
    assert "slow_mill_setup" in result["failure_tags"]


def test_build_report_groups_archetypes(tmp_path: Path) -> None:
    loss_path = tmp_path / "loss.json"
    win_path = tmp_path / "win.json"
    loss_path.write_text(json.dumps(_replay(loss=True)))
    win_path.write_text(json.dumps(_replay(loss=False)))

    report = build_report([tmp_path])

    assert report["summary"]["games"] == 2
    assert report["summary"]["wins"] == 1
    assert report["summary"]["by_archetype"]["crustle_wall"]["games"] == 2
