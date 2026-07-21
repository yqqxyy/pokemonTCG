from __future__ import annotations

import torch

from poketcg.agents.bc_agent import PolicyValueEvaluation
from poketcg.agents.rule_agent import AreaType, OptionType, SelectContext
from poketcg.agents.tactical_planner import PlannerPolicyAgent, TacticalPlannerAgent


def _catalogs():
    cards = {
        6: {"cardType": 5, "attacks": []},
        677: {"cardType": 0, "hp": 80, "attacks": [981], "basic": True},
        678: {
            "cardType": 0,
            "hp": 340,
            "attacks": [982, 983],
            "megaEx": True,
        },
        900: {"cardType": 0, "hp": 250, "attacks": [], "weakness": None},
    }
    attacks = {
        981: {"damage": 30, "energies": [6]},
        982: {"damage": 130, "energies": [6]},
        983: {"damage": 270, "energies": [6, 6]},
    }
    return cards, attacks


def _pokemon(card_id: int, serial: int, hp: int, energies: int) -> dict:
    return {
        "id": card_id,
        "serial": serial,
        "hp": hp,
        "maxHp": hp,
        "energies": [6] * energies,
        "energyCards": [],
        "tools": [],
        "preEvolution": [],
    }


def _observation(options: list[dict], *, energies: int = 2, target_hp: int = 250) -> dict:
    return {
        "select": {
            "context": int(SelectContext.MAIN),
            "minCount": 1,
            "maxCount": 1,
            "option": options,
            "deck": None,
        },
        "logs": [],
        "current": {
            "turn": 5,
            "turnActionCount": 0,
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
                    "active": [_pokemon(678, 10, 340, energies)],
                    "bench": [],
                    "hand": [{"id": 6, "serial": 50, "playerIndex": 0}],
                    "handCount": 1,
                    "discard": [],
                    "prize": [None] * 4,
                },
                {
                    "active": [_pokemon(900, 20, target_hp, 2)],
                    "bench": [],
                    "hand": None,
                    "handCount": 5,
                    "discard": [],
                    "prize": [None] * 4,
                },
            ],
        },
    }


def test_planner_selects_immediate_knockout_attack() -> None:
    cards, attacks = _catalogs()
    observation = _observation(
        [
            {"type": int(OptionType.ATTACK), "attackId": 982},
            {"type": int(OptionType.ATTACK), "attackId": 983},
            {"type": int(OptionType.END)},
        ]
    )
    planner = TacticalPlannerAgent(card_catalog=cards, attack_catalog=attacks, seed=1)

    evaluation = planner.evaluate(observation, persist=True)

    assert evaluation.action == (1,)
    assert evaluation.plan is not None
    assert evaluation.plan.attack_id == 983
    assert evaluation.plan.knockout
    assert evaluation.confidence >= 0.7


def test_planner_attaches_energy_to_enable_knockout() -> None:
    cards, attacks = _catalogs()
    observation = _observation(
        [
            {
                "type": int(OptionType.ATTACH),
                "area": int(AreaType.HAND),
                "index": 0,
                "inPlayArea": int(AreaType.ACTIVE),
                "inPlayIndex": 0,
            },
            {"type": int(OptionType.ATTACK), "attackId": 982},
            {"type": int(OptionType.END)},
        ],
        energies=1,
    )
    planner = TacticalPlannerAgent(card_catalog=cards, attack_catalog=attacks, seed=1)

    evaluation = planner.evaluate(observation, persist=True)

    assert evaluation.plan is not None
    assert evaluation.plan.attack_id == 983
    assert evaluation.plan.energy_missing == 1
    assert evaluation.action == (0,)


def test_planner_develops_an_unready_attacker_instead_of_overstacking() -> None:
    cards, attacks = _catalogs()
    observation = _observation(
        [
            {
                "type": int(OptionType.ATTACH),
                "area": int(AreaType.HAND),
                "index": 0,
                "inPlayArea": int(AreaType.BENCH),
                "inPlayIndex": 0,
            },
            {
                "type": int(OptionType.ATTACH),
                "area": int(AreaType.HAND),
                "index": 0,
                "inPlayArea": int(AreaType.BENCH),
                "inPlayIndex": 1,
            },
            {"type": int(OptionType.END)},
        ]
    )
    observation["current"]["players"][0]["bench"] = [
        _pokemon(678, 11, 340, 2),
        _pokemon(677, 12, 80, 0),
    ]
    planner = TacticalPlannerAgent(card_catalog=cards, attack_catalog=attacks, seed=1)

    evaluation = planner.evaluate(observation, persist=True)

    assert evaluation.action == (1,)


def test_planner_resolves_main_play_from_implicit_hand_area() -> None:
    cards, attacks = _catalogs()
    observation = _observation(
        [
            {"type": int(OptionType.PLAY), "index": 0},
            {"type": int(OptionType.END)},
        ]
    )
    observation["current"]["players"][0]["hand"] = [{"id": 1102, "serial": 50}]
    planner = TacticalPlannerAgent(card_catalog=cards, attack_catalog=attacks, seed=1)

    assert planner._option_card_id(observation, observation["select"]["option"][0]) == 1102


def test_planner_takes_maximum_energy_for_attach_to_effect() -> None:
    cards, attacks = _catalogs()
    observation = _observation([])
    observation["current"]["players"][0]["discard"] = [
        {"id": 6, "serial": 60 + index} for index in range(3)
    ]
    observation["select"] = {
        "context": int(SelectContext.ATTACH_TO),
        "minCount": 0,
        "maxCount": 3,
        "option": [
            {
                "type": int(OptionType.CARD),
                "area": int(AreaType.DISCARD),
                "index": index,
            }
            for index in range(3)
        ],
        "deck": None,
    }
    planner = TacticalPlannerAgent(card_catalog=cards, attack_catalog=attacks, seed=1)

    evaluation = planner.evaluate(observation, persist=True)

    assert evaluation.action == (0, 1, 2)
    assert (
        planner.routing_reason(observation, evaluation, threshold=0.9)
        == "context:ATTACH_TO"
    )


class _RecordingPolicy:
    action_space_version = 2

    def __init__(self) -> None:
        self.calls = 0

    def choose_action(self, observation: dict) -> list[int]:
        self.calls += 1
        return [2]

    def evaluate(self, observation: dict) -> PolicyValueEvaluation:
        return PolicyValueEvaluation(
            logits=torch.zeros(len(observation["select"]["option"])),
            value=0.25,
            minimum=1,
            maximum=1,
        )


def test_planner_policy_routes_high_confidence_plan() -> None:
    cards, attacks = _catalogs()
    observation = _observation(
        [
            {"type": int(OptionType.ATTACK), "attackId": 982},
            {"type": int(OptionType.ATTACK), "attackId": 983},
            {"type": int(OptionType.END)},
        ]
    )
    policy = _RecordingPolicy()
    hybrid = PlannerPolicyAgent(
        policy,  # type: ignore[arg-type]
        TacticalPlannerAgent(card_catalog=cards, attack_catalog=attacks, seed=1),
        planner_threshold=0.7,
        confidence_routing=True,
    )

    assert hybrid.choose_action(observation) == [1]
    assert policy.calls == 0
    assert hybrid.metrics()["planner_route_rate"] == 1.0


def test_planner_policy_exposes_planner_biased_mcts_logits() -> None:
    cards, attacks = _catalogs()
    observation = _observation(
        [
            {"type": int(OptionType.ATTACK), "attackId": 982},
            {"type": int(OptionType.ATTACK), "attackId": 983},
            {"type": int(OptionType.END)},
        ]
    )
    hybrid = PlannerPolicyAgent(
        _RecordingPolicy(),  # type: ignore[arg-type]
        TacticalPlannerAgent(card_catalog=cards, attack_catalog=attacks, seed=1),
        planner_weight=4.0,
    )

    evaluation = hybrid.evaluate(observation)

    assert evaluation.logits.argmax().item() == 1
    assert evaluation.value == 0.25
