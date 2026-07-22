"""Compare Library-Out baseline and residual reranker on an external panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from poketcg.agents import ExternalPythonAgent, ResidualRerankerAgent
from poketcg.engine import OfficialEngine
from poketcg.match import MatchResult, play_match

from .collect_libraryout_trajectories import (
    ExternalOpponentSpec,
    parse_external_opponent,
)
from .evaluate_panel import wilson_interval


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
        "mean_turns": round(sum(result.turns for result, _ in results) / games, 3),
    }


def evaluate_residual(
    checkpoint: str | Path,
    baseline_source: str | Path,
    baseline_deck_path: str | Path,
    opponents: list[ExternalOpponentSpec],
    *,
    games_per_seat: int,
    official_dir: str | Path | None = None,
    device: str = "cpu",
    shadow: bool = False,
    override_margin: float | None = None,
    minimum_confidence: float | None = None,
) -> dict[str, Any]:
    if games_per_seat <= 0:
        raise ValueError("games_per_seat must be positive")
    engine = OfficialEngine(official_dir)
    baseline_deck = engine.load_deck(baseline_deck_path)
    card_catalog = engine.card_catalog()
    attack_catalog = engine.attack_catalog()
    candidates: dict[str, Any] = {
        "baseline": ExternalPythonAgent(
            baseline_source,
            baseline_deck_path,
            name="libraryout-baseline",
            expected_deck=baseline_deck,
        ),
        "residual": ResidualRerankerAgent(
            checkpoint,
            baseline_source,
            baseline_deck_path,
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            expected_deck=baseline_deck,
            device=device,
            shadow=shadow,
            override_margin=override_margin,
            minimum_confidence=minimum_confidence,
        ),
    }
    cells = {}
    candidate_totals: dict[str, list[tuple[MatchResult, int]]] = {
        name: [] for name in candidates
    }
    for opponent_spec in opponents:
        opponent_deck = engine.load_deck(opponent_spec.deck)
        for candidate_name, candidate in candidates.items():
            for candidate_player in (0, 1):
                opponent = ExternalPythonAgent(
                    opponent_spec.source,
                    opponent_spec.deck,
                    name=opponent_spec.name,
                    expected_deck=opponent_deck,
                )
                results = []
                for game in range(games_per_seat):
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
                    candidate_totals[candidate_name].append(item)
                key = (
                    f"{candidate_name}_vs_{opponent_spec.name}_"
                    f"as_player{candidate_player}"
                )
                cells[key] = _summary(results)
    return {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "baseline_source": str(Path(baseline_source).expanduser().resolve()),
        "baseline_deck": str(Path(baseline_deck_path).expanduser().resolve()),
        "games_per_seat": games_per_seat,
        "shadow": shadow,
        "overall": {
            name: _summary(results) for name, results in candidate_totals.items()
        },
        "cells": cells,
        "routing": candidates["residual"].metrics(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--baseline-deck", type=Path, required=True)
    parser.add_argument(
        "--external-opponent",
        action="append",
        required=True,
        metavar="NAME=SOURCE,DECK",
    )
    parser.add_argument("--games-per-seat", type=int, default=100)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--override-margin", type=float)
    parser.add_argument("--minimum-confidence", type=float)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_residual(
        args.checkpoint,
        args.baseline_source,
        args.baseline_deck,
        [parse_external_opponent(value) for value in args.external_opponent],
        games_per_seat=args.games_per_seat,
        official_dir=args.official_dir,
        device=args.device,
        shadow=args.shadow,
        override_margin=args.override_margin,
        minimum_confidence=args.minimum_confidence,
    )
    rendered = json.dumps(result, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
