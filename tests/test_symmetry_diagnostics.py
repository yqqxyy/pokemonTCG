from __future__ import annotations

from poketcg.rl.features import FeatureEncoderV3
from poketcg.rl.symmetry_diagnostics import (
    encoded_symmetry_differences,
    relabel_players,
)


def _observation() -> dict:
    return {
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {
                    "type": 1,
                    "playerIndex": 0,
                    "area": 2,
                    "index": 0,
                    "inPlayArea": 4,
                    "inPlayIndex": 0,
                },
                {"type": 3, "attackId": 20},
            ],
            "deck": None,
        },
        "logs": [{"type": 3, "playerIndex": 1, "cardId": 11, "value": 30}],
        "current": {
            "turn": 4,
            "turnActionCount": 2,
            "yourIndex": 0,
            "firstPlayer": 0,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "result": -1,
            "stadium": [],
            "looking": None,
            "players": [
                {
                    "active": [
                        {
                            "id": 10,
                            "serial": 100,
                            "hp": 100,
                            "maxHp": 120,
                            "energies": [6],
                            "energyCards": [],
                            "tools": [],
                            "preEvolution": [],
                        }
                    ],
                    "bench": [],
                    "hand": [{"id": 12, "serial": 102, "playerIndex": 0}],
                    "handCount": 1,
                    "deckCount": 55,
                    "benchMax": 3,
                    "discard": [],
                    "prize": [None] * 4,
                },
                {
                    "active": [
                        {
                            "id": 11,
                            "serial": 101,
                            "hp": 80,
                            "maxHp": 100,
                            "energies": [7, 7],
                            "energyCards": [],
                            "tools": [],
                            "preEvolution": [],
                        }
                    ],
                    "bench": [],
                    "hand": None,
                    "handCount": 5,
                    "deckCount": 52,
                    "benchMax": 3,
                    "discard": [],
                    "prize": [None] * 3,
                },
            ],
        },
    }


def test_relabel_players_swaps_absolute_labels_and_outcome() -> None:
    observation = _observation()
    observation["current"]["result"] = 0

    relabeled = relabel_players(observation)

    assert relabeled["current"]["yourIndex"] == 1
    assert relabeled["current"]["firstPlayer"] == 1
    assert relabeled["current"]["result"] == 1
    assert relabeled["current"]["players"][1]["active"][0]["id"] == 10
    assert relabeled["select"]["option"][0]["playerIndex"] == 1
    assert relabeled["logs"][0]["playerIndex"] == 0


def test_feature_encoder_v3_is_invariant_to_player_relabeling() -> None:
    cards = {
        10: {"cardType": 0, "hp": 120, "attacks": [20], "basic": True},
        11: {"cardType": 0, "hp": 100, "attacks": [], "basic": True},
        12: {"cardType": 4, "attacks": []},
    }
    attacks = {20: {"damage": 60, "energies": [6], "text": ""}}
    encoder = FeatureEncoderV3(cards, attacks)
    observation = _observation()

    original = encoder.encode(observation)
    relabeled = encoder.encode(relabel_players(observation))

    assert encoded_symmetry_differences(original, relabeled) == {}
