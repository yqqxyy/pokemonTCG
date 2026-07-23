"""Evaluate a turn-gated Library-Out advantage reranker against external agents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from poketcg.agents import AdvantageRerankerAgent, ExternalPythonAgent
from poketcg.engine import OfficialEngine
from poketcg.match import MatchResult, play_match

from .collect_libraryout_trajectories import parse_external_opponent
from .evaluate_panel import wilson_interval


def _parse_transition(value: str) -> tuple[int, int]:
    try:
        source, target = value.split("->", maxsplit=1)
        return int(source), int(target)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("transition must look like 7->14") from error


def _summary(results: list[tuple[MatchResult, int]]) -> dict[str, Any]:
    games = len(results)
    wins = sum(result.winner == player for result, player in results)
    draws = sum(result.winner == 2 for result, _ in results)
    low, high = wilson_interval(wins, games)
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": games - wins - draws,
        "win_rate": round(wins / games, 6),
        "win_rate_ci95": [round(low, 6), round(high, 6)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--advantage-checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--round0-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--baseline-deck", type=Path, required=True)
    parser.add_argument("--external-opponent", action="append", required=True)
    parser.add_argument("--games-per-seat", type=int, default=20)
    parser.add_argument("--minimum-turn", type=int, default=4)
    parser.add_argument("--gate-threshold", type=float, default=0.05)
    parser.add_argument("--uncertainty-multiplier", type=float, default=0.0)
    parser.add_argument(
        "--allowed-transition",
        action="append",
        type=_parse_transition,
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.games_per_seat <= 0:
        raise SystemExit("--games-per-seat must be positive")

    engine = OfficialEngine(args.official_dir)
    baseline_deck = engine.load_deck(args.baseline_deck)
    card_catalog = engine.card_catalog()
    attack_catalog = engine.attack_catalog()
    opponents = [parse_external_opponent(value) for value in args.external_opponent]
    baseline = ExternalPythonAgent(
        args.baseline_source,
        args.baseline_deck,
        name="libraryout-baseline",
        expected_deck=baseline_deck,
    )
    reranker = AdvantageRerankerAgent(
        args.advantage_checkpoint,
        args.round0_checkpoint,
        args.baseline_source,
        args.baseline_deck,
        card_catalog=card_catalog,
        attack_catalog=attack_catalog,
        expected_deck=baseline_deck,
        device=args.device,
        minimum_turn=args.minimum_turn,
        gate_threshold=args.gate_threshold,
        uncertainty_multiplier=args.uncertainty_multiplier,
        allowed_transitions=(
            set(args.allowed_transition) if args.allowed_transition else None
        ),
        shadow=args.shadow,
    )
    candidates = {"baseline": baseline, "advantage": reranker}
    cells = {}
    totals: dict[str, list[tuple[MatchResult, int]]] = {
        name: [] for name in candidates
    }
    for opponent_spec in opponents:
        opponent_deck = engine.load_deck(opponent_spec.deck)
        for candidate_name, candidate in candidates.items():
            for candidate_player in (0, 1):
                results = []
                for game in range(args.games_per_seat):
                    opponent = ExternalPythonAgent(
                        opponent_spec.source,
                        opponent_spec.deck,
                        name=opponent_spec.name,
                        expected_deck=opponent_deck,
                    )
                    for agent in (candidate, opponent):
                        reset = getattr(agent, "reset_episode", None)
                        if callable(reset):
                            reset()
                    agents = (
                        (candidate, opponent)
                        if candidate_player == 0
                        else (opponent, candidate)
                    )
                    decks = (
                        (baseline_deck, opponent_deck)
                        if candidate_player == 0
                        else (opponent_deck, baseline_deck)
                    )
                    result = play_match(
                        engine,
                        decks[0],
                        decks[1],
                        agents[0],
                        agents[1],
                        game=game,
                    )
                    item = (result, candidate_player)
                    results.append(item)
                    totals[candidate_name].append(item)
                cells[
                    f"{candidate_name}_vs_{opponent_spec.name}_as_player{candidate_player}"
                ] = _summary(results)
    output = {
        "advantage_checkpoints": [str(path.resolve()) for path in args.advantage_checkpoint],
        "round0_checkpoint": str(args.round0_checkpoint.resolve()),
        "games_per_seat": args.games_per_seat,
        "minimum_turn": args.minimum_turn,
        "gate_threshold": args.gate_threshold,
        "uncertainty_multiplier": args.uncertainty_multiplier,
        "allowed_transitions": (
            [f"{source}->{target}" for source, target in args.allowed_transition]
            if args.allowed_transition
            else None
        ),
        "shadow": args.shadow,
        "overall": {name: _summary(results) for name, results in totals.items()},
        "cells": cells,
        "routing": reranker.metrics(),
    }
    rendered = json.dumps(output, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
