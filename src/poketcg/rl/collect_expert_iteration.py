"""Collect MCTS visit targets and policy replay examples for Expert Iteration."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch

from poketcg.agents import BCPolicyAgent
from poketcg.engine import OfficialEngine
from poketcg.mcts import DeckDeterminizer, MCTSConfig, PolicyValueMCTSAgent

from .action_space import neural_selection
from .data import BCExample, write_jsonl
from .features import build_feature_encoder
from .model import encoder_version


def root_visit_policy_target(
    search: dict,
    option_count: int,
    *,
    temperature: float = 1.0,
) -> list[float]:
    """Convert root child visits into an option probability distribution."""
    if option_count <= 0:
        raise ValueError("option_count must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    counts = [0.0] * option_count
    for child in search.get("children") or []:
        action = child.get("action") or []
        if len(action) != 1:
            raise ValueError("MCTS soft targets currently require single-option roots")
        index = int(action[0])
        if not 0 <= index < option_count:
            raise ValueError("MCTS child action is outside the root option range")
        counts[index] += max(float(child.get("visits", 0)), 0.0)
    exponent = 1.0 / temperature
    weights = [count**exponent for count in counts]
    total = sum(weights)
    if total <= 0:
        selected = search.get("selected_action") or []
        if len(selected) != 1:
            raise ValueError("MCTS search has neither visits nor one selected action")
        weights[int(selected[0])] = 1.0
        total = 1.0
    return [weight / total for weight in weights]


def _policy_replay_target(
    policy: BCPolicyAgent,
    observation: dict,
    *,
    temperature: float,
) -> list[float]:
    if temperature <= 0:
        raise ValueError("replay temperature must be positive")
    option_count = len(observation["select"]["option"])
    logits = policy.evaluate(observation).logits[:option_count] / temperature
    return logits.softmax(dim=0).tolist()


def _entropy(probabilities: list[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0)


def collect_expert_examples(
    engine: OfficialEngine,
    deck: list[int],
    checkpoint: str | Path,
    *,
    games: int,
    seed: int,
    simulations: int = 16,
    determinizations: int = 1,
    c_puct: float = 1.25,
    max_depth: int = 12,
    max_actions: int = 16,
    target_temperature: float = 1.0,
    replay_temperature: float = 1.0,
    device: str = "cpu",
    torch_threads: int = 1,
) -> tuple[list[BCExample], dict]:
    """Run MCTS self-play and return terminal-labelled training examples."""
    if games <= 0:
        raise ValueError("games must be positive")
    if torch_threads <= 0:
        raise ValueError("torch_threads must be positive")
    torch.set_num_threads(torch_threads)
    card_catalog = engine.card_catalog()
    attack_catalog = engine.attack_catalog()
    basic_card_ids = {
        card_id
        for card_id, card in card_catalog.items()
        if bool(getattr(card, "basic", False))
    }
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    feature_version = encoder_version(checkpoint_data["model_config"])
    encoder = build_feature_encoder(feature_version, card_catalog, attack_catalog)
    config = MCTSConfig(
        simulations=simulations,
        determinizations=determinizations,
        c_puct=c_puct,
        max_depth=max_depth,
        max_actions=max_actions,
    )
    policies = tuple(
        BCPolicyAgent(
            checkpoint,
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            seed=seed + player * 100_000,
            device=device,
            deterministic=False,
        )
        for player in (0, 1)
    )
    agents = tuple(
        PolicyValueMCTSAgent(
            policies[player],
            DeckDeterminizer(
                deck,
                deck,
                basic_card_ids=basic_card_ids,
                seed=seed + 200_000 + player * 100_000,
            ),
            config=config,
            seed=seed + 400_000 + player * 100_000,
        )
        for player in (0, 1)
    )

    examples: list[BCExample] = []
    winners: Counter[int] = Counter()
    contexts: Counter[int] = Counter()
    source_counts: Counter[str] = Counter()
    target_entropies: list[float] = []
    top_probabilities: list[float] = []
    skipped = 0
    started = perf_counter()

    for game in range(games):
        for agent in agents:
            agent.reset_episode()
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
                selection = observation["select"]
                action = agents[player].choose_action(observation)
                if neural_selection(selection, policies[player].action_space_version):
                    decision = encoder.encode(observation)
                    exact_one = (
                        int(selection["minCount"])
                        == int(selection["maxCount"])
                        == 1
                    )
                    target: list[float] | None = None
                    if agents[player].last_search is not None:
                        if not exact_one:
                            raise RuntimeError("MCTS searched a non-single-choice root")
                        target = root_visit_policy_target(
                            agents[player].last_search,
                            len(selection["option"]),
                            temperature=target_temperature,
                        )
                        source_counts["mcts_visit"] += 1
                    elif exact_one:
                        target = _policy_replay_target(
                            policies[player],
                            observation,
                            temperature=replay_temperature,
                        )
                        source_counts["policy_replay"] += 1
                    else:
                        source_counts["hard_replay"] += 1
                    if target is not None:
                        target_entropies.append(_entropy(target))
                        top_probabilities.append(max(target))
                    pending.append((decision, list(action), player, target))
                    contexts[int(selection["context"])] += 1
                else:
                    skipped += 1
                observation = engine.select(action)

            winner = int(observation["current"]["result"])
            winners[winner] += 1
            for decision, action, player, target in pending:
                value = 0.0 if winner == 2 else (1.0 if winner == player else -1.0)
                examples.append(
                    BCExample(
                        decision=decision,
                        action=action,
                        value_target=value,
                        player=player,
                        game=game,
                        policy_target=target,
                    )
                )
        finally:
            engine.finish()

    elapsed = perf_counter() - started
    search_metrics = [agent.metrics() for agent in agents]
    summary = {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "games": games,
        "examples": len(examples),
        "skipped_decisions": skipped,
        "winner_counts": dict(sorted(winners.items())),
        "context_counts": dict(sorted(contexts.items())),
        "target_source_counts": dict(sorted(source_counts.items())),
        "mean_target_entropy": round(sum(target_entropies) / len(target_entropies), 6),
        "mean_top_target_probability": round(
            sum(top_probabilities) / len(top_probabilities), 6
        ),
        "encoder_version": feature_version,
        "mcts_config": {
            "simulations": simulations,
            "determinizations": determinizations,
            "c_puct": c_puct,
            "max_depth": max_depth,
            "max_actions": max_actions,
        },
        "elapsed_seconds": round(elapsed, 3),
        "games_per_second": round(games / elapsed, 3),
        "search_metrics": search_metrics,
    }
    return examples, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20_260_720)
    parser.add_argument("--deck", type=Path)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--determinizations", type=int, default=1)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--replay-temperature", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = OfficialEngine(args.official_dir)
    deck = engine.load_deck(args.deck or engine.sample_deck_path)
    examples, summary = collect_expert_examples(
        engine,
        deck,
        args.checkpoint,
        games=args.games,
        seed=args.seed,
        simulations=args.simulations,
        determinizations=args.determinizations,
        c_puct=args.c_puct,
        max_depth=args.max_depth,
        max_actions=args.max_actions,
        target_temperature=args.target_temperature,
        replay_temperature=args.replay_temperature,
        device=args.device,
        torch_threads=args.torch_threads,
    )
    write_jsonl(args.output, examples)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
