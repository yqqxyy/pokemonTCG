"""Audit player-label symmetry in the encoder and policy/value model."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from poketcg.agents import BCPolicyAgent, RuleAgent
from poketcg.engine import OfficialEngine
from poketcg.match import _validate_action
from poketcg.paths import resolve_official_dir

from .action_space import deterministic_subset
from .features import EncodedDecision


def _swap_index(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, int):
        return value
    return 1 - value if value in {0, 1} else value


def relabel_players(observation: dict) -> dict:
    """Swap absolute player labels while preserving the acting-player viewpoint.

    This is a representation audit, not a simulator transition. Option ordering is
    intentionally retained, so a correctly relative encoder should produce the same
    encoded decision and the same option logits.
    """

    swapped = copy.deepcopy(observation)

    def swap_player_indices(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "playerIndex":
                    value[key] = _swap_index(child)
                else:
                    swap_player_indices(child)
        elif isinstance(value, list):
            for child in value:
                swap_player_indices(child)

    swap_player_indices(swapped)
    state = swapped.get("current")
    if not isinstance(state, dict):
        raise ValueError("Observation must contain a current state")
    players = state.get("players")
    if not isinstance(players, list) or len(players) != 2:
        raise ValueError("Player symmetry audit requires exactly two players")
    state["players"] = [players[1], players[0]]
    state["yourIndex"] = _swap_index(state.get("yourIndex"))
    state["firstPlayer"] = _swap_index(state.get("firstPlayer"))
    state["result"] = _swap_index(state.get("result"))
    return swapped


def _max_abs_difference(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0 if left is right else float("inf")
    if isinstance(left, (int, float, bool)) and isinstance(
        right, (int, float, bool)
    ):
        return abs(float(left) - float(right))
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return float("inf")
        return max(
            (_max_abs_difference(a, b) for a, b in zip(left, right, strict=True)),
            default=0.0,
        )
    return 0.0 if left == right else float("inf")


def encoded_symmetry_differences(
    original: EncodedDecision,
    relabeled: EncodedDecision,
) -> dict[str, float]:
    """Return only encoded fields that change under a player-label relabeling."""
    differences = {}
    for field in fields(EncodedDecision):
        difference = _max_abs_difference(
            getattr(original, field.name), getattr(relabeled, field.name)
        )
        if difference != 0.0:
            differences[field.name] = difference
    return differences


def run_symmetry_diagnostics(
    checkpoint: str | Path,
    *,
    games_per_seat: int,
    seed: int = 20_260_721,
    official_dir: str | Path | None = None,
    deck_path: str | Path | None = None,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Collect real decisions and compare original vs relabeled inference."""
    if games_per_seat <= 0:
        raise ValueError("games_per_seat must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    engine = OfficialEngine(resolve_official_dir(official_dir))
    deck = engine.load_deck(deck_path or engine.sample_deck_path)
    cards = engine.card_catalog()
    attacks = engine.attack_catalog()
    policy = BCPolicyAgent(
        checkpoint,
        card_catalog=cards,
        attack_catalog=attacks,
        deterministic=True,
        seed=seed,
    )
    field_maxima: dict[str, float] = {}
    max_logit_difference = 0.0
    max_value_difference = 0.0
    action_mismatches = 0
    decisions = 0
    seat_decisions = {"player0": 0, "player1": 0}

    for candidate_player in (0, 1):
        rule = RuleAgent(
            card_catalog=cards,
            attack_catalog=attacks,
            seed=seed + candidate_player + 1,
        )
        for _game in range(games_per_seat):
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
                        action = rule.choose_action(observation)
                    else:
                        relabeled = relabel_players(observation)
                        encoded = policy.encode(observation)
                        relabeled_encoded = policy.encode(relabeled)
                        for name, difference in encoded_symmetry_differences(
                            encoded, relabeled_encoded
                        ).items():
                            field_maxima[name] = max(
                                field_maxima.get(name, 0.0), difference
                            )

                        original_eval = policy.evaluate(observation)
                        relabeled_eval = policy.evaluate(relabeled)
                        logit_difference = float(
                            (original_eval.logits - relabeled_eval.logits).abs().max()
                        )
                        max_logit_difference = max(
                            max_logit_difference, logit_difference
                        )
                        max_value_difference = max(
                            max_value_difference,
                            abs(original_eval.value - relabeled_eval.value),
                        )
                        original_action = deterministic_subset(
                            original_eval.logits,
                            original_eval.minimum,
                            original_eval.maximum,
                        )
                        relabeled_action = deterministic_subset(
                            relabeled_eval.logits,
                            relabeled_eval.minimum,
                            relabeled_eval.maximum,
                        )
                        action_mismatches += int(original_action != relabeled_action)
                        decisions += 1
                        seat_decisions[f"player{candidate_player}"] += 1
                        action = original_action
                    _validate_action(observation, action)
                    observation = engine.select(action)
            finally:
                engine.finish()

    finite_field_maxima = {
        name: value for name, value in field_maxima.items() if value != float("inf")
    }
    structural_mismatches = sorted(
        name for name, value in field_maxima.items() if value == float("inf")
    )
    max_encoded_difference = max(finite_field_maxima.values(), default=0.0)
    passed = (
        not structural_mismatches
        and max_encoded_difference <= tolerance
        and max_logit_difference <= tolerance
        and max_value_difference <= tolerance
        and action_mismatches == 0
    )
    return {
        "format": "poketcg-player-symmetry-v1",
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "games_per_seat": games_per_seat,
        "decisions": decisions,
        "seat_decisions": seat_decisions,
        "tolerance": tolerance,
        "passed": passed,
        "encoded": {
            "max_abs_difference": max_encoded_difference,
            "field_maxima": finite_field_maxima,
            "structural_mismatches": structural_mismatches,
        },
        "policy": {
            "max_logit_abs_difference": max_logit_difference,
            "action_mismatches": action_mismatches,
        },
        "value": {
            "perspective": "acting-player-relative",
            "expected_relation": "V(relabel_players(s)) == V(s)",
            "max_abs_difference": max_value_difference,
        },
        "notes": [
            "This swaps absolute labels, both player arrays, firstPlayer, and result.",
            "Option order is preserved; policy logits therefore compare index-for-index.",
            "This audit detects representation asymmetry, not legitimate first-player advantage.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games-per-seat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_260_721)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--deck", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    report = run_symmetry_diagnostics(
        args.checkpoint,
        games_per_seat=args.games_per_seat,
        seed=args.seed,
        official_dir=args.official_dir,
        deck_path=args.deck,
        tolerance=args.tolerance,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
