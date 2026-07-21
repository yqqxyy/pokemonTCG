"""Diagnose tactical-planner decisions against a shadow external expert."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from poketcg.agents import BCPolicyAgent, ExternalPythonAgent, TacticalPlannerAgent
from poketcg.agents.rule_agent import AreaType, OptionType, SelectContext
from poketcg.engine import OfficialEngine
from poketcg.match import _validate_action
from poketcg.paths import resolve_official_dir

from .action_space import deterministic_subset, legal_action_set_count


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Planner, policy, and expert choices on one non-forced student state."""

    player: int
    context: int
    context_name: str
    planner_handled: bool
    planner_confidence: float
    routed_to_planner: bool
    route_reason: str | None
    planner_agrees: bool
    policy_agrees: bool
    hybrid_agrees: bool
    planner_action: str
    policy_action: str
    expert_action: str


def _enum_name(enum_type: type, value: int) -> str:
    try:
        return enum_type(value).name
    except ValueError:
        return str(value)


def _record_id(record: dict | None) -> int | None:
    if not record or record.get("id") is None:
        return None
    return int(record["id"])


def _option_card_id(observation: dict, option: dict) -> int | None:
    if option.get("cardId") is not None:
        return int(option["cardId"])
    if option.get("index") is None:
        return None
    index = int(option["index"])
    area = option.get("area")
    state = observation["current"]
    raw_player = option.get("playerIndex")
    player_index = int(state["yourIndex"] if raw_player is None else raw_player)
    if area is None and int(option["type"]) in {
        int(OptionType.PLAY),
        int(OptionType.ATTACH),
        int(OptionType.EVOLVE),
    }:
        records = state["players"][player_index].get("hand")
    elif area == int(AreaType.DECK):
        records = observation["select"].get("deck")
    elif area == int(AreaType.HAND):
        records = state["players"][player_index].get("hand")
    elif area == int(AreaType.DISCARD):
        records = state["players"][player_index].get("discard")
    elif area == int(AreaType.ACTIVE):
        records = state["players"][player_index].get("active")
    elif area == int(AreaType.BENCH):
        records = state["players"][player_index].get("bench")
    elif area == int(AreaType.PRIZE):
        records = state["players"][player_index].get("prize")
    else:
        records = None
    if records is None or not 0 <= index < len(records):
        return None
    return _record_id(records[index])


def _target_card_id(observation: dict, option: dict) -> int | None:
    area = option.get("inPlayArea")
    index = option.get("inPlayIndex")
    if area not in {int(AreaType.ACTIVE), int(AreaType.BENCH)} or index is None:
        return None
    state = observation["current"]
    raw_player = option.get("playerIndex")
    player_index = int(state["yourIndex"] if raw_player is None else raw_player)
    player = state["players"][player_index]
    records = player.get("active") if area == int(AreaType.ACTIVE) else player.get("bench")
    if records is None or not 0 <= int(index) < len(records):
        return None
    return _record_id(records[int(index)])


def _action_label(observation: dict, action: list[int] | tuple[int, ...]) -> str:
    labels = []
    for index in sorted(action):
        option = observation["select"]["option"][index]
        option_type = int(option["type"])
        label = _enum_name(OptionType, option_type)
        if option.get("attackId") is not None:
            label += f":attack={int(option['attackId'])}"
        card_id = _option_card_id(observation, option)
        if card_id is not None:
            label += f":card={card_id}"
        target_id = _target_card_id(observation, option)
        if target_id is not None:
            label += f":target={target_id}"
        labels.append(label)
    return "+".join(labels) if labels else "<empty>"


def _same(left: list[int] | tuple[int, ...], right: list[int] | tuple[int, ...]) -> bool:
    return sorted(left) == sorted(right)


def summarize_decisions(records: list[DecisionRecord]) -> dict[str, Any]:
    """Aggregate agreement and counterfactual replacement counts by context."""

    def summarize(items: list[DecisionRecord]) -> dict[str, Any]:
        count = len(items)
        routed = sum(item.routed_to_planner for item in items)
        planner_correct = sum(item.planner_agrees for item in items)
        policy_correct = sum(item.policy_agrees for item in items)
        hybrid_correct = sum(item.hybrid_agrees for item in items)
        planner_only = sum(item.planner_agrees and not item.policy_agrees for item in items)
        policy_only = sum(item.policy_agrees and not item.planner_agrees for item in items)
        routed_planner_only = sum(
            item.routed_to_planner and item.planner_agrees and not item.policy_agrees
            for item in items
        )
        routed_policy_only = sum(
            item.routed_to_planner and item.policy_agrees and not item.planner_agrees
            for item in items
        )

        def rate(value: int) -> float:
            return round(value / count, 6) if count else 0.0

        return {
            "decisions": count,
            "planner_handled_rate": rate(sum(item.planner_handled for item in items)),
            "planner_route_rate": rate(routed),
            "planner_route_reasons": dict(
                sorted(Counter(item.route_reason for item in items if item.route_reason).items())
            ),
            "planner_expert_agreement": rate(planner_correct),
            "policy_expert_agreement": rate(policy_correct),
            "hybrid_expert_agreement": rate(hybrid_correct),
            "planner_only_correct": planner_only,
            "policy_only_correct": policy_only,
            "planner_minus_policy_correct": planner_only - policy_only,
            "routed_planner_only_correct": routed_planner_only,
            "routed_policy_only_correct": routed_policy_only,
            "routed_net_replacements": routed_planner_only - routed_policy_only,
        }

    contexts = {
        name: summarize([item for item in records if item.context_name == name])
        for name in sorted({item.context_name for item in records})
    }
    seats = {
        f"player{player}": summarize([item for item in records if item.player == player])
        for player in (0, 1)
    }
    patterns = Counter(
        (
            item.context_name,
            item.policy_action,
            item.planner_action,
            item.expert_action,
            item.routed_to_planner,
        )
        for item in records
        if not (item.policy_agrees and item.planner_agrees)
    )
    return {
        "overall": summarize(records),
        "seats": seats,
        "contexts": contexts,
        "top_disagreement_patterns": [
            {
                "count": count,
                "context": key[0],
                "policy_action": key[1],
                "planner_action": key[2],
                "expert_action": key[3],
                "routed_to_planner": key[4],
            }
            for key, count in patterns.most_common(30)
        ],
    }


def run_planner_diagnostics(
    checkpoint: str | Path,
    expert_source: str | Path,
    deck_path: str | Path,
    *,
    games_per_seat: int,
    planner_threshold: float = 0.9,
    planner_confidence_routing: bool = True,
    seed: int = 20_260_720,
    official_dir: str | Path | None = None,
) -> dict[str, Any]:
    if games_per_seat <= 0:
        raise ValueError("games_per_seat must be positive")
    if not 0.0 <= planner_threshold <= 1.0:
        raise ValueError("planner_threshold must be in [0, 1]")
    engine = OfficialEngine(resolve_official_dir(official_dir))
    deck = engine.load_deck(deck_path)
    cards = engine.card_catalog()
    attacks = engine.attack_catalog()
    policy = BCPolicyAgent(
        checkpoint,
        card_catalog=cards,
        attack_catalog=attacks,
        deterministic=True,
        seed=seed,
    )
    planner = TacticalPlannerAgent(
        card_catalog=cards,
        attack_catalog=attacks,
        seed=seed + 1,
    )
    records: list[DecisionRecord] = []
    seat_results: dict[str, dict[str, int | float]] = {}

    for candidate_player in (0, 1):
        wins = 0
        decisions = 0
        for game in range(games_per_seat):
            planner.reset_episode()
            opponent = ExternalPythonAgent(
                expert_source,
                deck_path,
                name=f"diagnostic-opponent-{candidate_player}-{game}",
                expected_deck=deck,
            )
            shadow = ExternalPythonAgent(
                expert_source,
                deck_path,
                name=f"diagnostic-shadow-{candidate_player}-{game}",
                expected_deck=deck,
            )
            observation, start_data = engine.start(deck, deck)
            if observation is None:
                raise RuntimeError(
                    "Official simulator failed to start "
                    f"(errorPlayer={start_data.errorPlayer}, errorType={start_data.errorType})."
                )
            try:
                while int(observation["current"]["result"]) == -1:
                    player = int(observation["current"]["yourIndex"])
                    if player != candidate_player:
                        action = opponent.choose_action(observation)
                        _validate_action(observation, action)
                        observation = engine.select(action)
                        continue

                    expert_action = shadow.choose_action(observation)
                    planned = planner.evaluate(observation, persist=True)
                    neural = policy.evaluate(observation)
                    policy_action = deterministic_subset(
                        neural.logits,
                        neural.minimum,
                        neural.maximum,
                    )
                    route_reason = planner.routing_reason(
                        observation,
                        planned,
                        threshold=planner_threshold,
                        allow_confidence=planner_confidence_routing,
                    )
                    routed = route_reason is not None
                    hybrid_action = list(planned.action) if routed else policy_action
                    _validate_action(observation, expert_action)
                    _validate_action(observation, hybrid_action)
                    selection = observation["select"]
                    action_count = legal_action_set_count(
                        len(selection["option"]),
                        int(selection["minCount"]),
                        int(selection["maxCount"]),
                    )
                    if action_count > 1:
                        context = int(selection["context"])
                        records.append(
                            DecisionRecord(
                                player=candidate_player,
                                context=context,
                                context_name=_enum_name(SelectContext, context),
                                planner_handled=planned.handled,
                                planner_confidence=round(planned.confidence, 6),
                                routed_to_planner=routed,
                                route_reason=route_reason,
                                planner_agrees=_same(planned.action, expert_action),
                                policy_agrees=_same(policy_action, expert_action),
                                hybrid_agrees=_same(hybrid_action, expert_action),
                                planner_action=_action_label(
                                    observation, list(planned.action)
                                ),
                                policy_action=_action_label(observation, policy_action),
                                expert_action=_action_label(observation, expert_action),
                            )
                        )
                    decisions += 1
                    observation = engine.select(hybrid_action)
                wins += int(int(observation["current"]["result"]) == candidate_player)
            finally:
                engine.finish()
        seat_results[f"as_player{candidate_player}"] = {
            "games": games_per_seat,
            "wins": wins,
            "win_rate": round(wins / games_per_seat, 6),
            "decisions": decisions,
        }

    return {
        "format": "poketcg-planner-diagnostics-v1",
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "expert_source": str(Path(expert_source).expanduser().resolve()),
        "deck": str(Path(deck_path).expanduser().resolve()),
        "games_per_seat": games_per_seat,
        "planner_threshold": planner_threshold,
        "planner_confidence_routing": planner_confidence_routing,
        "seed": seed,
        "seats": seat_results,
        "summary": summarize_decisions(records),
        "records": [asdict(item) for item in records],
        "notes": [
            "Forced decisions are executed but excluded from agreement metrics.",
            (
                "The shadow expert is queried on every candidate decision to keep "
                "its plan synchronized."
            ),
            "Agreement is diagnostic and does not prove causal win-rate impact.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expert-source", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--games-per-seat", type=int, default=50)
    parser.add_argument("--planner-threshold", type=float, default=0.9)
    confidence_group = parser.add_mutually_exclusive_group()
    confidence_group.add_argument(
        "--planner-confidence-routing",
        dest="planner_confidence_routing",
        action="store_true",
        default=True,
    )
    confidence_group.add_argument(
        "--no-planner-confidence-routing",
        dest="planner_confidence_routing",
        action="store_false",
    )
    parser.add_argument("--seed", type=int, default=20_260_720)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    report = run_planner_diagnostics(
        args.checkpoint,
        args.expert_source,
        args.deck,
        games_per_seat=args.games_per_seat,
        planner_threshold=args.planner_threshold,
        planner_confidence_routing=args.planner_confidence_routing,
        seed=args.seed,
        official_dir=args.official_dir,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "summary": report["summary"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
