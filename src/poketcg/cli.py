"""Command-line batch evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .agents import Agent, BCPolicyAgent, RandomAgent, RuleAgent
from .engine import OfficialEngine
from .match import MatchResult, play_match


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate local Pokémon TCG agents.")
    parser.add_argument("--games", type=int, default=10, help="Number of matches to play.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for agent decisions.")
    parser.add_argument("--player0", choices=("random", "rule", "bc"), default="random")
    parser.add_argument("--player1", choices=("random", "rule", "bc"), default="random")
    parser.add_argument("--checkpoint", type=Path, help="Required when either player is bc.")
    parser.add_argument("--deck", type=Path, help="Deck CSV for both players; defaults to sample.")
    parser.add_argument(
        "--official-dir",
        type=Path,
        help="Directory containing the official cg package and deck.csv.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSONL path for per-match results.")
    parser.add_argument("--max-decisions", type=int, default=10_000)
    return parser


def summarize(results: list[MatchResult]) -> dict:
    winners = Counter(result.winner for result in results)
    total = len(results)
    return {
        "games": total,
        "player0": results[0].player0,
        "player1": results[0].player1,
        "player0_wins": winners[0],
        "player1_wins": winners[1],
        "draws": winners[2],
        "player0_win_rate": round(winners[0] / total, 4),
        "player1_win_rate": round(winners[1] / total, 4),
        "draw_rate": round(winners[2] / total, 4),
        "mean_turns": round(sum(result.turns for result in results) / total, 3),
        "mean_decisions": round(sum(result.decisions for result in results) / total, 3),
        "mean_elapsed_ms": round(sum(result.elapsed_ms for result in results) / total, 3),
    }


def build_agent(
    name: str,
    seed: int,
    *,
    card_catalog: dict[int, object],
    attack_catalog: dict[int, object],
    checkpoint: Path | None,
) -> Agent:
    if name == "random":
        return RandomAgent(seed)
    if name == "rule":
        return RuleAgent(card_catalog=card_catalog, attack_catalog=attack_catalog, seed=seed)
    if name == "bc":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for the bc agent")
        return BCPolicyAgent(
            checkpoint,
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            seed=seed,
        )
    raise ValueError(f"Unknown agent: {name}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.games <= 0:
        raise SystemExit("--games must be greater than zero")

    engine = OfficialEngine(args.official_dir)
    deck_path = args.deck or engine.sample_deck_path
    deck = engine.load_deck(deck_path)
    card_catalog = engine.card_catalog()
    attack_catalog = engine.attack_catalog()
    results: list[MatchResult] = []

    for game in range(args.games):
        seed0 = args.seed + game * 2
        seed1 = seed0 + 1
        result = play_match(
            engine,
            deck,
            deck,
            build_agent(
                args.player0,
                seed0,
                card_catalog=card_catalog,
                attack_catalog=attack_catalog,
                checkpoint=args.checkpoint,
            ),
            build_agent(
                args.player1,
                seed1,
                card_catalog=card_catalog,
                attack_catalog=attack_catalog,
                checkpoint=args.checkpoint,
            ),
            game=game,
            agent_seed0=seed0,
            agent_seed1=seed1,
            max_decisions=args.max_decisions,
        )
        results.append(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            for result in results:
                stream.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    print(json.dumps(summarize(results), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
