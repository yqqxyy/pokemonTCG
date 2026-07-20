"""Paired-seat fixed-panel evaluation for policy-value MCTS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from poketcg.agents import BCPolicyAgent, RuleAgent
from poketcg.engine import OfficialEngine
from poketcg.match import MatchResult, play_match
from poketcg.mcts import (
    DeckDeterminizer,
    DeckHypothesis,
    MCTSConfig,
    OpponentDeckBelief,
    PolicyValueMCTSAgent,
)
from poketcg.rl.evaluate_panel import wilson_interval


def _outcome_summary(results: list[MatchResult], model_player: int) -> dict:
    wins = sum(result.winner == model_player for result in results)
    draws = sum(result.winner == 2 for result in results)
    low, high = wilson_interval(wins, len(results))
    return {
        "games": len(results),
        "wins": wins,
        "draws": draws,
        "losses": len(results) - wins - draws,
        "win_rate": round(wins / len(results), 6),
        "win_rate_ci95": [round(low, 6), round(high, 6)],
        "mean_turns": round(sum(result.turns for result in results) / len(results), 3),
        "mean_decisions": round(
            sum(result.decisions for result in results) / len(results), 3
        ),
        "mean_elapsed_ms": round(
            sum(result.elapsed_ms for result in results) / len(results), 3
        ),
    }


def evaluate_mcts(
    checkpoint: str | Path,
    *,
    games_per_seat: int,
    seed: int,
    simulations: int,
    determinizations: int = 1,
    c_puct: float = 1.25,
    max_depth: int = 12,
    max_actions: int = 16,
    official_dir: str | Path | None = None,
    deck_path: str | Path | None = None,
    actual_opponent_deck_path: str | Path | None = None,
    fixed_opponent_prior_deck_path: str | Path | None = None,
    stochastic: bool = True,
    torch_threads: int = 1,
    opponent_deck_hypotheses: list[DeckHypothesis] | None = None,
) -> dict:
    if games_per_seat <= 0:
        raise ValueError("games_per_seat must be positive")
    if torch_threads <= 0:
        raise ValueError("torch_threads must be positive")
    torch.set_num_threads(torch_threads)
    engine = OfficialEngine(official_dir)
    model_deck_path = Path(deck_path or engine.sample_deck_path)
    opponent_deck_path = Path(actual_opponent_deck_path or model_deck_path)
    fixed_prior_path = Path(fixed_opponent_prior_deck_path or opponent_deck_path)
    model_deck = engine.load_deck(model_deck_path)
    actual_opponent_deck = engine.load_deck(opponent_deck_path)
    fixed_opponent_prior_deck = engine.load_deck(fixed_prior_path)
    card_catalog = engine.card_catalog()
    attack_catalog = engine.attack_catalog()
    basic_card_ids = {
        card_id
        for card_id, card in card_catalog.items()
        if bool(getattr(card, "basic", False))
    }
    opponent_belief = (
        OpponentDeckBelief(opponent_deck_hypotheses)
        if opponent_deck_hypotheses
        else None
    )
    config = MCTSConfig(
        simulations=simulations,
        determinizations=determinizations,
        c_puct=c_puct,
        max_depth=max_depth,
        max_actions=max_actions,
    )
    matchups = {}
    total_wins = 0
    total_draws = 0
    total_games = 0
    started = perf_counter()

    for model_player in (0, 1):
        pairing_seed = seed + model_player * 10_000
        decks = (
            (model_deck, actual_opponent_deck)
            if model_player == 0
            else (actual_opponent_deck, model_deck)
        )
        determinizer_decks = (
            (model_deck, fixed_opponent_prior_deck)
            if model_player == 0
            else (fixed_opponent_prior_deck, model_deck)
        )
        policy = BCPolicyAgent(
            checkpoint,
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            seed=pairing_seed,
            deterministic=not stochastic,
        )
        determinizer = DeckDeterminizer(
            determinizer_decks[0],
            determinizer_decks[1],
            basic_card_ids=basic_card_ids,
            seed=pairing_seed + 300_000,
            opponent_belief=opponent_belief,
        )
        model = PolicyValueMCTSAgent(
            policy,
            determinizer,
            config=config,
            seed=pairing_seed + 400_000,
        )
        baseline = RuleAgent(
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            seed=pairing_seed + 1,
        )
        results = []
        for game in range(games_per_seat):
            agents = (model, baseline) if model_player == 0 else (baseline, model)
            model.reset_episode()
            results.append(
                play_match(
                    engine,
                    decks[0],
                    decks[1],
                    agents[0],
                    agents[1],
                    game=game,
                    agent_seed0=pairing_seed,
                    agent_seed1=pairing_seed + 1,
                )
            )
        summary = _outcome_summary(results, model_player)
        summary["search"] = model.metrics()
        matchups[f"vs_rule_as_player{model_player}"] = summary
        total_wins += int(summary["wins"])
        total_draws += int(summary["draws"])
        total_games += games_per_seat

    low, high = wilson_interval(total_wins, total_games)
    return {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "model_deck": str(model_deck_path.expanduser().resolve()),
        "actual_opponent_deck": str(opponent_deck_path.expanduser().resolve()),
        "fixed_opponent_prior_deck": str(fixed_prior_path.expanduser().resolve()),
        "games_per_seat": games_per_seat,
        "seed": seed,
        "action_selection": "stochastic" if stochastic else "deterministic",
        "opponent_deck_hypotheses": [
            {"name": item.name, "prior": item.prior}
            for item in opponent_deck_hypotheses or []
        ],
        "mcts_config": {
            "simulations": simulations,
            "determinizations": determinizations,
            "c_puct": c_puct,
            "max_depth": max_depth,
            "max_actions": max_actions,
            "root_contexts": [0],
        },
        "overall": {
            "games": total_games,
            "wins": total_wins,
            "draws": total_draws,
            "losses": total_games - total_wins - total_draws,
            "win_rate": round(total_wins / total_games, 6),
            "win_rate_ci95": [round(low, 6), round(high, 6)],
        },
        "wall_seconds": round(perf_counter() - started, 3),
        "matchups": matchups,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games-per-seat", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20_260_901)
    parser.add_argument("--simulations", type=int, required=True)
    parser.add_argument("--determinizations", type=int, default=1)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--deck", type=Path)
    parser.add_argument(
        "--actual-opponent-deck",
        type=Path,
        help="Deck actually played by RuleAgent; defaults to --deck.",
    )
    parser.add_argument(
        "--fixed-opponent-prior-deck",
        type=Path,
        help="MCTS fixed opponent prior; defaults to the actual opponent deck.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--opponent-deck",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeat to define equal-prior opponent deck hypotheses.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hypotheses = []
    for specification in args.opponent_deck:
        try:
            name, raw_path = specification.split("=", 1)
        except ValueError as error:
            raise SystemExit("--opponent-deck must use NAME=PATH") from error
        if not name or not raw_path:
            raise SystemExit("--opponent-deck must use non-empty NAME=PATH")
        hypotheses.append(
            DeckHypothesis(
                name=name,
                deck=tuple(OfficialEngine.load_deck(Path(raw_path))),
            )
        )
    result = evaluate_mcts(
        args.checkpoint,
        games_per_seat=args.games_per_seat,
        seed=args.seed,
        simulations=args.simulations,
        determinizations=args.determinizations,
        c_puct=args.c_puct,
        max_depth=args.max_depth,
        max_actions=args.max_actions,
        official_dir=args.official_dir,
        deck_path=args.deck,
        actual_opponent_deck_path=args.actual_opponent_deck,
        fixed_opponent_prior_deck_path=args.fixed_opponent_prior_deck,
        stochastic=not args.deterministic,
        torch_threads=args.torch_threads,
        opponent_deck_hypotheses=hypotheses,
    )
    rendered = json.dumps(result, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
