"""Compare the native Mega Lucario expert with the public notebook policy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from poketcg.agents import (
    ExternalPythonAgent,
    MegaLucarioExpertAgent,
    RuleAgent,
)
from poketcg.agents.rule_agent import OptionType, SelectContext
from poketcg.engine import OfficialEngine
from poketcg.match import _validate_action
from poketcg.paths import resolve_official_dir

from .action_space import legal_action_set_count


def _enum_name(enum_type: type, value: int) -> str:
    try:
        return enum_type(value).name
    except ValueError:
        return str(value)


def _action_signature(observation: dict, action: list[int]) -> list[dict[str, Any]]:
    options = observation["select"]["option"]
    signature = []
    for index in action:
        option = options[index]
        signature.append(
            {
                "index": index,
                "type": _enum_name(OptionType, int(option["type"])),
                "card_id": option.get("cardId"),
                "attack_id": option.get("attackId"),
                "area": option.get("area"),
                "option_index": option.get("index"),
                "in_play_area": option.get("inPlayArea"),
                "in_play_index": option.get("inPlayIndex"),
            }
        )
    return signature


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def summarize_parity(
    *,
    decisions: int,
    non_forced_decisions: int,
    exact_matches: int,
    set_matches: int,
    context_counts: Counter[str],
    context_mismatches: Counter[str],
    mismatches: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create the stable summary used by CLI output and tests."""
    return {
        "decisions": decisions,
        "non_forced_decisions": non_forced_decisions,
        "exact_matches": exact_matches,
        "exact_match_rate": _rate(exact_matches, decisions),
        "set_matches": set_matches,
        "set_match_rate": _rate(set_matches, decisions),
        "context_counts": dict(sorted(context_counts.items())),
        "context_mismatches": dict(sorted(context_mismatches.items())),
        "mismatches": mismatches,
    }


def run_expert_parity(
    expert_source: str | Path,
    deck_path: str | Path,
    *,
    games_per_seat: int,
    seed: int = 20_260_721,
    official_dir: str | Path | None = None,
    mismatch_limit: int = 30,
) -> dict[str, Any]:
    """Drive games with the notebook expert and shadow every expert decision."""
    if games_per_seat <= 0:
        raise ValueError("games_per_seat must be positive")
    if mismatch_limit < 0:
        raise ValueError("mismatch_limit must be non-negative")
    engine = OfficialEngine(resolve_official_dir(official_dir))
    deck = engine.load_deck(deck_path)
    cards = engine.card_catalog()
    attacks = engine.attack_catalog()
    decisions = 0
    non_forced_decisions = 0
    exact_matches = 0
    set_matches = 0
    context_counts: Counter[str] = Counter()
    context_mismatches: Counter[str] = Counter()
    mismatches: list[dict[str, Any]] = []
    seats: dict[str, dict[str, Any]] = {}

    for expert_player in (0, 1):
        wins = 0
        seat_decisions = 0
        seat_exact = 0
        for game in range(games_per_seat):
            notebook = ExternalPythonAgent(
                expert_source,
                deck_path,
                name=f"parity-notebook-{expert_player}-{game}",
                expected_deck=deck,
            )
            native = MegaLucarioExpertAgent(card_catalog=cards, deck=deck)
            opponent = RuleAgent(
                card_catalog=cards,
                attack_catalog=attacks,
                seed=seed + expert_player * 100_000 + game,
            )
            observation, start_data = engine.start(deck, deck)
            if observation is None:
                raise RuntimeError(
                    "Official simulator failed to start "
                    f"(errorPlayer={start_data.errorPlayer}, "
                    f"errorType={start_data.errorType})."
                )
            try:
                while int(observation["current"]["result"]) == -1:
                    player = int(observation["current"]["yourIndex"])
                    if player != expert_player:
                        action = opponent.choose_action(observation)
                        _validate_action(observation, action)
                        observation = engine.select(action)
                        continue

                    notebook_action = notebook.choose_action(observation)
                    native_action = native.choose_action(observation)
                    _validate_action(observation, notebook_action)
                    _validate_action(observation, native_action)
                    selection = observation["select"]
                    context = int(selection["context"])
                    context_name = _enum_name(SelectContext, context)
                    context_counts[context_name] += 1
                    action_count = legal_action_set_count(
                        len(selection["option"]),
                        int(selection["minCount"]),
                        int(selection["maxCount"]),
                    )
                    if action_count > 1:
                        non_forced_decisions += 1
                    exact = notebook_action == native_action
                    set_equal = sorted(notebook_action) == sorted(native_action)
                    exact_matches += int(exact)
                    set_matches += int(set_equal)
                    decisions += 1
                    seat_decisions += 1
                    seat_exact += int(exact)
                    if not exact:
                        context_mismatches[context_name] += 1
                        if len(mismatches) < mismatch_limit:
                            mismatches.append(
                                {
                                    "game": game,
                                    "expert_player": expert_player,
                                    "turn": int(observation["current"]["turn"]),
                                    "context": context_name,
                                    "notebook_action": _action_signature(
                                        observation, notebook_action
                                    ),
                                    "native_action": _action_signature(
                                        observation, native_action
                                    ),
                                }
                            )
                    observation = engine.select(notebook_action)
                wins += int(
                    int(observation["current"]["result"]) == expert_player
                )
            finally:
                engine.finish()
        seats[f"as_player{expert_player}"] = {
            "games": games_per_seat,
            "wins": wins,
            "win_rate": _rate(wins, games_per_seat),
            "decisions": seat_decisions,
            "exact_match_rate": _rate(seat_exact, seat_decisions),
        }

    return {
        "format": "poketcg-mega-expert-parity-v1",
        "expert_source": str(Path(expert_source).expanduser().resolve()),
        "deck": str(Path(deck_path).expanduser().resolve()),
        "games_per_seat": games_per_seat,
        "seed": seed,
        "seats": seats,
        "parity": summarize_parity(
            decisions=decisions,
            non_forced_decisions=non_forced_decisions,
            exact_matches=exact_matches,
            set_matches=set_matches,
            context_counts=context_counts,
            context_mismatches=context_mismatches,
            mismatches=mismatches,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-source", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--games-per-seat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_260_721)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--mismatch-limit", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_expert_parity(
        args.expert_source,
        args.deck,
        games_per_seat=args.games_per_seat,
        seed=args.seed,
        official_dir=args.official_dir,
        mismatch_limit=args.mismatch_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
