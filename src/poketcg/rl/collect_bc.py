"""Collect RuleAgent decisions for behavior cloning."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from poketcg.agents import RuleAgent
from poketcg.engine import OfficialEngine

from .action_space import neural_selection
from .data import BCExample, write_jsonl
from .features import build_feature_encoder


def optimal_policy_target(scores: list[float], tolerance: float = 1e-6) -> list[float]:
    """Put uniform probability on every option tied for the maximum score."""
    if not scores:
        raise ValueError("Cannot build a policy target without options.")
    maximum = max(scores)
    optimal = [math.isclose(score, maximum, rel_tol=0.0, abs_tol=tolerance) for score in scores]
    probability = 1.0 / sum(optimal)
    return [probability if is_optimal else 0.0 for is_optimal in optimal]


def collect_examples(
    engine: OfficialEngine,
    deck: list[int],
    *,
    games: int,
    seed: int,
    encoder_version: int = 1,
    include_multiselect: bool = False,
) -> tuple[list[BCExample], dict]:
    card_catalog = engine.card_catalog()
    attack_catalog = engine.attack_catalog()
    encoder = build_feature_encoder(encoder_version, card_catalog, attack_catalog)
    examples: list[BCExample] = []
    winners: Counter[int] = Counter()
    contexts: Counter[int] = Counter()
    optimal_action_counts: Counter[int] = Counter()
    skipped = 0
    multiselect_examples = 0

    for game in range(games):
        agents = (
            RuleAgent(
                card_catalog=card_catalog,
                attack_catalog=attack_catalog,
                seed=seed + game * 2,
            ),
            RuleAgent(
                card_catalog=card_catalog,
                attack_catalog=attack_catalog,
                seed=seed + game * 2 + 1,
            ),
        )
        observation, start_data = engine.start(deck, deck)
        if observation is None:
            raise RuntimeError(
                "Official simulator failed to start "
                f"(errorPlayer={start_data.errorPlayer}, errorType={start_data.errorType})."
            )

        pending: list[tuple[object, list[int], int, list[float] | None]] = []
        try:
            while int(observation["current"]["result"]) == -1:
                player = int(observation["current"]["yourIndex"])
                action = agents[player].choose_action(observation)
                selection = observation["select"]
                learnable = neural_selection(
                    selection, 2 if include_multiselect else 1
                )
                if learnable:
                    decision = encoder.encode(observation)
                    single_choice = (
                        int(selection["minCount"]) == int(selection["maxCount"]) == 1
                    )
                    policy_target = (
                        optimal_policy_target(agents[player].score_options(observation))
                        if single_choice
                        else None
                    )
                    if policy_target is not None:
                        optimal_action_counts[sum(value > 0 for value in policy_target)] += 1
                    else:
                        multiselect_examples += 1
                    pending.append((decision, list(action), player, policy_target))
                    contexts[int(selection["context"])] += 1
                else:
                    skipped += 1
                observation = engine.select(action)

            winner = int(observation["current"]["result"])
            winners[winner] += 1
            for decision, action, player, policy_target in pending:
                value = 0.0 if winner == 2 else (1.0 if winner == player else -1.0)
                examples.append(
                    BCExample(
                        decision=decision,
                        action=action,
                        value_target=value,
                        player=player,
                        game=game,
                        policy_target=policy_target,
                    )
                )
        finally:
            engine.finish()

    summary = {
        "games": games,
        "examples": len(examples),
        "skipped_decisions": skipped,
        "winner_counts": dict(sorted(winners.items())),
        "context_counts": dict(sorted(contexts.items())),
        "optimal_action_count_histogram": dict(sorted(optimal_action_counts.items())),
        "encoder_version": encoder_version,
        "include_multiselect": include_multiselect,
        "multiselect_examples": multiselect_examples,
    }
    return examples, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect RuleAgent behavior-cloning data.")
    parser.add_argument("--games", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deck", type=Path)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-version", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--include-multiselect", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.games <= 0:
        raise SystemExit("--games must be greater than zero")
    engine = OfficialEngine(args.official_dir)
    deck = engine.load_deck(args.deck or engine.sample_deck_path)
    examples, summary = collect_examples(
        engine,
        deck,
        games=args.games,
        seed=args.seed,
        encoder_version=args.encoder_version,
        include_multiselect=args.include_multiselect,
    )
    write_jsonl(args.output, examples)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
