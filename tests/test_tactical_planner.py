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


def test_planner_keeps_the_same_valid_plan_across_main_decisions() -> None:
    cards, attacks = _catalogs()
    first = _observation(
        [
            {"type": int(OptionType.ATTACK), "attackId": 982},
            {"type": int(OptionType.ATTACK), "attackId": 983},
            {"type": int(OptionType.END)},
        ]
    )
    planner = TacticalPlannerAgent(card_catalog=cards, attack_catalog=attacks, seed=1)

    initial = planner.evaluate(first, persist=True)
    repeated = planner.evaluate(first, persist=True)

    assert initial.plan is not None
    assert repeated.plan is initial.plan


def test_turn_ownership_keeps_planner_for_follow_up_context() -> None:
    cards, attacks = _catalogs()
    main = _observation(
        [
            {"type": int(OptionType.ATTACK), "attackId": 982},
            {"type": int(OptionType.ATTACK), "attackId": 983},
            {"type": int(OptionType.END)},
        ]
    )
    follow_up = _observation([])
    follow_up["select"] = {
        "context": int(SelectContext.DISCARD),
        "minCount": 1,
        "maxCount": 1,
        "option": [
            {"type": int(OptionType.CARD), "area": int(AreaType.HAND), "index": i}
            for i in range(3)
        ],
        "deck": None,
    }
    follow_up["current"]["players"][0]["hand"] = [
        {"id": 6, "serial": 50},
        {"id": 677, "serial": 51},
        {"id": 1102, "serial": 52},
    ]
    policy = _RecordingPolicy()
    hybrid = PlannerPolicyAgent(
        policy,  # type: ignore[arg-type]
        TacticalPlannerAgent(card_catalog=cards, attack_catalog=attacks, seed=1),
        planner_threshold=0.7,
        confidence_routing=True,
        turn_ownership=True,
    )

    assert hybrid.choose_action(main) == [1]
    hybrid.choose_action(follow_up)

    metrics = hybrid.metrics()["turn_ownership"]
    assert policy.calls == 0
    assert metrics["planner_owned_turns"] == 1
    assert metrics["planner_owned_decisions"] == 2
    assert metrics["owner_switches"] == 0


def test_turn_ownership_keeps_policy_even_when_follow_up_is_planner_priority() -> None:
    cards, attacks = _catalogs()
    main = _observation(
        [
            {"type": int(OptionType.ATTACK), "attackId": 982},
            {"type": int(OptionType.ATTACK), "attackId": 983},
            {"type": int(OptionType.END)},
        ]
    )
    follow_up = _observation([])
    follow_up["select"] = {
        "context": int(SelectContext.TO_HAND),
        "minCount": 1,
        "maxCount": 1,
        "option": [
            {"type": int(OptionType.CARD), "area": int(AreaType.DECK), "index": i}
            for i in range(3)
        ],
        "deck": [
            {"id": 6, "serial": 60},
            {"id": 677, "serial": 61},
            {"id": 1102, "serial": 62},
        ],
    }
    policy = _RecordingPolicy()
    hybrid = PlannerPolicyAgent(
        policy,  # type: ignore[arg-type]
        TacticalPlannerAgent(card_catalog=cards, attack_catalog=attacks, seed=1),
        confidence_routing=False,
        turn_ownership=True,
    )

    assert hybrid.choose_action(main) == [2]
    assert hybrid.choose_action(follow_up) == [2]

    metrics = hybrid.metrics()["turn_ownership"]
    assert policy.calls == 2
    assert metrics["policy_owned_turns"] == 1
    assert metrics["policy_owned_decisions"] == 2
    assert metrics["owner_switches"] == 0


def test_invalid_planner_plan_transfers_remaining_turn_to_policy() -> None:
    cards, attacks = _catalogs()
    main = _observation(
        [
            {"type": int(OptionType.ATTACK), "attackId": 982},
            {"type": int(OptionType.ATTACK), "attackId": 983},
            {"type": int(OptionType.END)},
        ]
    )
    invalidated = _observation(
        [
            {"type": int(OptionType.ATTACK), "attackId": 982},
            {"type": int(OptionType.ATTACK), "attackId": 983},
            {"type": int(OptionType.END)},
        ]
    )
    invalidated["current"]["players"][1]["active"] = []
    policy = _RecordingPolicy()
    hybrid = PlannerPolicyAgent(
        policy,  # type: ignore[arg-type]
        TacticalPlannerAgent(card_catalog=cards, attack_catalog=attacks, seed=1),
        planner_threshold=0.7,
        confidence_routing=True,
        turn_ownership=True,
    )

    assert hybrid.choose_action(main) == [1]
    assert hybrid.choose_action(invalidated) == [2]

    metrics = hybrid.metrics()["turn_ownership"]
    assert policy.calls == 1
    assert metrics["owner_switches"] == 1
    assert metrics["plan_invalidations"] == 1
    assert metrics["policy_owned_decisions"] == 1


def test_commitment_ownership_treats_draw_ability_as_resolver_chain() -> None:
    cards, attacks = _catalogs()
    cards[675] = {"cardType": 0, "hp": 70, "attacks": [], "basic": True}
    ability = _observation(
        [
            {
                "type": int(OptionType.ABILITY),
                "area": int(AreaType.BENCH),
                "index": 0,
            },
            {"type": int(OptionType.END)},
        ]
    )
    ability["current"]["players"][0]["bench"] = [_pokemon(675, 30, 70, 0)]
    attack = _observation(
        [
            {"type": int(OptionType.ATTACK), "attackId": 982},
            {"type": int(OptionType.ATTACK), "attackId": 983},
            {"type": int(OptionType.END)},
        ]
    )
    attack["current"]["players"][0]["bench"] = [_pokemon(675, 30, 70, 0)]
    policy = _RecordingPolicy()
    planner = TacticalPlannerAgent(card_catalog=cards, attack_catalog=attacks, seed=1)
    hybrid = PlannerPolicyAgent(
        policy,  # type: ignore[arg-type]
        planner,
        planner_threshold=0.7,
        confidence_routing=True,
        commitment_ownership=True,
    )

    ability_evaluation = planner.evaluate(ability, persist=False)
    assert planner.routing_reason(
        ability,
        ability_evaluation,
        threshold=0.7,
        allow_confidence=True,
    ) == "main:draw-engine-ability"
    assert not planner.is_plan_commitment(ability, ability_evaluation)

    assert hybrid.choose_action(ability) == [0]
    assert hybrid.choose_action(attack) == [1]

    metrics = hybrid.metrics()["commitment_ownership"]
    assert policy.calls == 0
    assert metrics["resolver_chains"] == 1
    assert metrics["committed_turns"] == 1
    assert metrics["chain_resolutions"] == 1
    assert metrics["planner_segments"] == 2


def test_commitment_ownership_keeps_policy_for_its_resolver_chain_only() -> None:
    cards, attacks = _catalogs()
    main = _observation(
        [
            {"type": int(OptionType.ATTACK), "attackId": 982},
            {"type": int(OptionType.ATTACK), "attackId": 983},
            {"type": int(OptionType.END)},
        ]
    )
    follow_up = _observation([])
    follow_up["select"] = {
        "context": int(SelectContext.TO_HAND),
        "minCount": 1,
        "maxCount": 1,
        "option": [
            {"type": int(OptionType.CARD), "area": int(AreaType.DECK), "index": i}
            for i in range(3)
        ],
        "deck": [
            {"id": 6, "serial": 60},
            {"id": 677, "serial": 61},
            {"id": 1102, "serial": 62},
        ],
    }
    policy = _RecordingPolicy()
    hybrid = PlannerPolicyAgent(
        policy,  # type: ignore[arg-type]
        TacticalPlannerAgent(card_catalog=cards, attack_catalog=attacks, seed=1),
        confidence_routing=False,
        commitment_ownership=True,
    )

    assert hybrid.choose_action(main) == [2]
    assert hybrid.choose_action(follow_up) == [2]
    assert hybrid.choose_action(main) == [2]

    metrics = hybrid.metrics()["commitment_ownership"]
    assert policy.calls == 3
    assert metrics["resolver_chains"] == 2
    assert metrics["committed_turns"] == 0
    assert metrics["chain_resolutions"] == 1
    assert metrics["policy_segments"] == 2
