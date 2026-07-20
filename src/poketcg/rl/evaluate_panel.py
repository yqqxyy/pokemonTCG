"""Paired-seat evaluation against fixed baseline opponents."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from poketcg.agents import BCPolicyAgent, HybridPolicyAgent, RandomAgent, RuleAgent
from poketcg.engine import OfficialEngine
from poketcg.match import MatchResult, play_match


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial success rate."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _summarize(results: list[MatchResult], model_player: int) -> dict:
    wins = sum(result.winner == model_player for result in results)
    draws = sum(result.winner == 2 for result in results)
    losses = len(results) - wins - draws
    low, high = wilson_interval(wins, len(results))
    return {
        "games": len(results),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": round(wins / len(results), 6),
        "win_rate_ci95": [round(low, 6), round(high, 6)],
        "mean_turns": round(sum(item.turns for item in results) / len(results), 3),
        "mean_decisions": round(sum(item.decisions for item in results) / len(results), 3),
    }


def _opponent(
    name: str,
    seed: int,
    card_catalog: dict[int, object],
    attack_catalog: dict[int, object],
):
    if name == "random":
        return RandomAgent(seed)
    return RuleAgent(card_catalog=card_catalog, attack_catalog=attack_catalog, seed=seed)


def evaluate_panel(
    checkpoint: str | Path,
    *,
    games_per_seat: int,
    seed: int,
    official_dir: str | Path | None = None,
    deck_path: str | Path | None = None,
    stochastic: bool = False,
    policy_opponents: list[str | Path] | None = None,
    multiselect_checkpoint: str | Path | None = None,
) -> dict:
    engine = OfficialEngine(official_dir)
    deck = engine.load_deck(deck_path or engine.sample_deck_path)
    card_catalog = engine.card_catalog()
    attack_catalog = engine.attack_catalog()
    matchups = {}
    all_results: list[tuple[MatchResult, int]] = []

    opponent_specs: list[tuple[str, Path | None]] = [("random", None), ("rule", None)]
    opponent_specs.extend(
        (f"policy_{Path(path).stem}", Path(path)) for path in policy_opponents or []
    )
    for opponent_index, (opponent_name, opponent_checkpoint) in enumerate(opponent_specs):
        for model_player in (0, 1):
            pairing_seed = seed + opponent_index * 100_000 + model_player * 10_000
            agent_options = {
                "card_catalog": card_catalog,
                "attack_catalog": attack_catalog,
                "seed": pairing_seed,
                "deterministic": not stochastic,
            }
            if multiselect_checkpoint is None:
                model = BCPolicyAgent(checkpoint, **agent_options)
            else:
                model = HybridPolicyAgent(
                    checkpoint,
                    multiselect_checkpoint,
                    **agent_options,
                )
            if opponent_checkpoint is None:
                baseline = _opponent(
                    opponent_name,
                    pairing_seed + 1,
                    card_catalog,
                    attack_catalog,
                )
            else:
                baseline = BCPolicyAgent(
                    opponent_checkpoint,
                    card_catalog=card_catalog,
                    attack_catalog=attack_catalog,
                    seed=pairing_seed + 1,
                    deterministic=not stochastic,
                )
            results = []
            for game in range(games_per_seat):
                agents = (model, baseline) if model_player == 0 else (baseline, model)
                result = play_match(
                    engine,
                    deck,
                    deck,
                    agents[0],
                    agents[1],
                    game=game,
                    agent_seed0=pairing_seed,
                    agent_seed1=pairing_seed + 1,
                )
                results.append(result)
                all_results.append((result, model_player))
            matchups[f"vs_{opponent_name}_as_player{model_player}"] = _summarize(
                results, model_player
            )

    total_wins = sum(result.winner == player for result, player in all_results)
    total_draws = sum(result.winner == 2 for result, _ in all_results)
    low, high = wilson_interval(total_wins, len(all_results))
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "multiselect_checkpoint": (
            str(Path(multiselect_checkpoint).resolve())
            if multiselect_checkpoint is not None
            else None
        ),
        "games_per_seat": games_per_seat,
        "action_selection": "stochastic" if stochastic else "deterministic",
        "policy_opponents": [str(Path(path).resolve()) for path in policy_opponents or []],
        "overall": {
            "games": len(all_results),
            "wins": total_wins,
            "draws": total_draws,
            "losses": len(all_results) - total_wins - total_draws,
            "win_rate": round(total_wins / len(all_results), 6),
            "win_rate_ci95": [round(low, 6), round(high, 6)],
        },
        "matchups": matchups,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint against fixed baselines.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games-per-seat", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20_260_717)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--deck", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--policy-opponent", type=Path, action="append", default=[])
    parser.add_argument(
        "--multiselect-checkpoint",
        type=Path,
        help=(
            "Use --checkpoint for exact-one decisions and this Action Space V2 "
            "checkpoint for multi-select decisions."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_panel(
        args.checkpoint,
        games_per_seat=args.games_per_seat,
        seed=args.seed,
        official_dir=args.official_dir,
        deck_path=args.deck,
        stochastic=args.stochastic,
        policy_opponents=args.policy_opponent,
        multiselect_checkpoint=args.multiselect_checkpoint,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
