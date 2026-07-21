"""Collect hard behavior-cloning targets from an inspected external expert agent."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from poketcg.agents import BCPolicyAgent, ExternalPythonAgent, RuleAgent
from poketcg.engine import OfficialEngine

from .action_space import neural_selection
from .data import BCExample
from .features import build_feature_encoder

OPPONENT_KINDS = ("rule", "policy", "mirror")


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    expert_source: str
    expert_deck: str
    opponent_deck: str
    opponents: tuple[str, ...]
    policy_checkpoint: str | None
    policy_stochastic: bool
    encoder_version: int
    include_multiselect: bool
    seed: int
    official_dir: str
    torch_threads: int


def scheduled_matchup(opponents: Sequence[str], game: int) -> tuple[str, int]:
    """Cycle through every opponent and both expert seats without confounding them."""
    if not opponents:
        raise ValueError("At least one opponent is required")
    schedule = tuple((opponent, player) for opponent in opponents for player in (0, 1))
    return schedule[game % len(schedule)]


def _validate_configuration(
    opponents: Sequence[str], policy_checkpoint: str | Path | None
) -> tuple[str, ...]:
    normalized = tuple(str(value).lower() for value in opponents)
    if not normalized:
        raise ValueError("At least one opponent is required")
    unknown = sorted(set(normalized) - set(OPPONENT_KINDS))
    if unknown:
        raise ValueError(f"Unknown opponent kinds: {', '.join(unknown)}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Opponent kinds must not be repeated")
    if "policy" in normalized and policy_checkpoint is None:
        raise ValueError("The policy opponent requires --policy-checkpoint")
    return normalized


def _validate_action(observation: dict, action: list[int]) -> None:
    selection = observation["select"]
    if not isinstance(action, list) or not all(isinstance(index, int) for index in action):
        raise TypeError("Expert action must be list[int]")
    if len(action) != len(set(action)):
        raise ValueError("Expert action contains duplicate option indices")
    minimum = int(selection["minCount"])
    maximum = int(selection["maxCount"])
    if not minimum <= len(action) <= maximum:
        raise ValueError("Expert action violates the selection cardinality")
    option_count = len(selection["option"])
    if any(index < 0 or index >= option_count for index in action):
        raise IndexError("Expert action references an option outside the legal range")


def _new_external_expert(
    config: CollectionConfig,
    expert_deck: list[int],
    *,
    name: str,
) -> ExternalPythonAgent:
    # A fresh namespace per game is intentional. Public agents frequently keep
    # turn plans in module globals but do not expose reset_episode().
    return ExternalPythonAgent(
        config.expert_source,
        config.expert_deck,
        name=name,
        expected_deck=expert_deck,
    )


def _new_statistics(opponents: Sequence[str]) -> dict[str, Any]:
    return {
        "games": 0,
        "examples": 0,
        "skipped_decisions": 0,
        "context_counts": Counter(),
        "expert_outcomes": Counter(),
        "opponents": {
            opponent: {
                "games": 0,
                "examples": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "as_player0_games": 0,
                "as_player0_wins": 0,
                "as_player1_games": 0,
                "as_player1_wins": 0,
            }
            for opponent in opponents
        },
    }


def _finalize_statistics(statistics: dict[str, Any]) -> dict[str, Any]:
    opponents = {}
    for name, raw in statistics["opponents"].items():
        item = dict(raw)
        games = int(item["games"])
        item["win_rate"] = round(item["wins"] / games, 6) if games else 0.0
        for player in (0, 1):
            seat_games = int(item[f"as_player{player}_games"])
            seat_wins = int(item[f"as_player{player}_wins"])
            item[f"as_player{player}_win_rate"] = (
                round(seat_wins / seat_games, 6) if seat_games else 0.0
            )
        opponents[name] = item
    return {
        "games": int(statistics["games"]),
        "examples": int(statistics["examples"]),
        "skipped_decisions": int(statistics["skipped_decisions"]),
        "context_counts": dict(sorted(statistics["context_counts"].items())),
        "expert_outcomes": dict(sorted(statistics["expert_outcomes"].items())),
        "opponents": opponents,
    }


def _merge_statistics(items: Iterable[dict[str, Any]], opponents: Sequence[str]) -> dict[str, Any]:
    merged = _new_statistics(opponents)
    for item in items:
        for key in ("games", "examples", "skipped_decisions"):
            merged[key] += int(item[key])
        merged["context_counts"].update(item["context_counts"])
        merged["expert_outcomes"].update(item["expert_outcomes"])
        for opponent in opponents:
            source = item["opponents"][opponent]
            target = merged["opponents"][opponent]
            for key in target:
                target[key] += int(source[key])
    return merged


def _collect_games(
    engine: OfficialEngine,
    config: CollectionConfig,
    game_ids: Iterable[int],
    emit: Callable[[BCExample], None],
) -> dict[str, Any]:
    opponents = _validate_configuration(config.opponents, config.policy_checkpoint)
    torch.set_num_threads(config.torch_threads)
    expert_deck = engine.load_deck(config.expert_deck)
    opponent_deck = engine.load_deck(config.opponent_deck)
    card_catalog = engine.card_catalog()
    attack_catalog = engine.attack_catalog()
    encoder = build_feature_encoder(config.encoder_version, card_catalog, attack_catalog)
    action_version = 2 if config.include_multiselect else 1
    policy = None
    if "policy" in opponents:
        policy = BCPolicyAgent(
            config.policy_checkpoint,
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            seed=config.seed + 900_000,
            deterministic=not config.policy_stochastic,
        )

    statistics = _new_statistics(opponents)
    for game in game_ids:
        opponent_kind, expert_player = scheduled_matchup(opponents, game)
        expert = _new_external_expert(
            config,
            expert_deck,
            name=f"teacher-{game}",
        )
        if opponent_kind == "mirror":
            opponent = _new_external_expert(
                config,
                expert_deck,
                name=f"mirror-{game}",
            )
            current_opponent_deck = expert_deck
        elif opponent_kind == "rule":
            opponent = RuleAgent(
                card_catalog=card_catalog,
                attack_catalog=attack_catalog,
                seed=config.seed + game,
            )
            current_opponent_deck = opponent_deck
        else:
            if policy is None:
                raise AssertionError("Policy opponent was not initialized")
            opponent = policy
            current_opponent_deck = opponent_deck

        decks = (
            (expert_deck, current_opponent_deck)
            if expert_player == 0
            else (current_opponent_deck, expert_deck)
        )
        agents = (expert, opponent) if expert_player == 0 else (opponent, expert)
        observation, start_data = engine.start(decks[0], decks[1])
        if observation is None:
            raise RuntimeError(
                "Official simulator failed to start "
                f"(errorPlayer={start_data.errorPlayer}, errorType={start_data.errorType})."
            )

        pending: list[tuple[object, list[int]]] = []
        try:
            while int(observation["current"]["result"]) == -1:
                player = int(observation["current"]["yourIndex"])
                action = agents[player].choose_action(observation)
                _validate_action(observation, action)
                if player == expert_player:
                    selection = observation["select"]
                    if neural_selection(selection, action_version):
                        pending.append((encoder.encode(observation), list(action)))
                        statistics["context_counts"][int(selection["context"])] += 1
                    else:
                        statistics["skipped_decisions"] += 1
                observation = engine.select(action)

            winner = int(observation["current"]["result"])
            outcome = "draw" if winner == 2 else ("win" if winner == expert_player else "loss")
            value_target = 0.0 if outcome == "draw" else (1.0 if outcome == "win" else -1.0)
            for decision, action in pending:
                emit(
                    BCExample(
                        decision=decision,
                        action=action,
                        value_target=value_target,
                        player=expert_player,
                        game=game,
                    )
                )
        finally:
            engine.finish()

        statistics["games"] += 1
        statistics["examples"] += len(pending)
        statistics["expert_outcomes"][outcome] += 1
        opponent_stats = statistics["opponents"][opponent_kind]
        opponent_stats["games"] += 1
        opponent_stats["examples"] += len(pending)
        outcome_key = {"win": "wins", "draw": "draws", "loss": "losses"}[outcome]
        opponent_stats[outcome_key] += 1
        opponent_stats[f"as_player{expert_player}_games"] += 1
        if outcome == "win":
            opponent_stats[f"as_player{expert_player}_wins"] += 1
    return statistics


def collect_external_expert_examples(
    engine: OfficialEngine,
    expert_source: str | Path,
    expert_deck: str | Path,
    *,
    games: int,
    seed: int,
    opponents: Sequence[str] = ("rule", "mirror"),
    opponent_deck: str | Path | None = None,
    policy_checkpoint: str | Path | None = None,
    policy_stochastic: bool = True,
    encoder_version: int = 3,
    include_multiselect: bool = False,
    torch_threads: int = 1,
) -> tuple[list[BCExample], dict[str, Any]]:
    """Collect expert-only examples in-process; mainly useful for tests and screens."""
    if games <= 0:
        raise ValueError("games must be positive")
    if torch_threads <= 0:
        raise ValueError("torch_threads must be positive")
    normalized = _validate_configuration(opponents, policy_checkpoint)
    resolved_expert_deck = Path(expert_deck).expanduser().resolve()
    config = CollectionConfig(
        expert_source=str(Path(expert_source).expanduser().resolve()),
        expert_deck=str(resolved_expert_deck),
        opponent_deck=str(Path(opponent_deck or resolved_expert_deck).expanduser().resolve()),
        opponents=normalized,
        policy_checkpoint=(
            str(Path(policy_checkpoint).expanduser().resolve())
            if policy_checkpoint is not None
            else None
        ),
        policy_stochastic=policy_stochastic,
        encoder_version=encoder_version,
        include_multiselect=include_multiselect,
        seed=seed,
        official_dir=str(engine.official_dir),
        torch_threads=torch_threads,
    )
    examples: list[BCExample] = []
    started = perf_counter()
    statistics = _collect_games(engine, config, range(games), examples.append)
    summary = _finalize_statistics(statistics)
    summary.update(
        {
            "expert_source": config.expert_source,
            "expert_deck": config.expert_deck,
            "opponent_deck": config.opponent_deck,
            "encoder_version": encoder_version,
            "action_space_version": 2 if include_multiselect else 1,
            "elapsed_seconds": round(perf_counter() - started, 3),
            "fresh_expert_namespace_per_game": True,
        }
    )
    return examples, summary


def _collect_shard(
    config: CollectionConfig,
    game_ids: tuple[int, ...],
    output: str,
) -> dict[str, Any]:
    engine = OfficialEngine(config.official_dir)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:

        def emit(example: BCExample) -> None:
            stream.write(json.dumps(example.to_dict(), separators=(",", ":")) + "\n")

        return _collect_games(engine, config, game_ids, emit)


def _combine_parts(parts: Sequence[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target:
        for part in parts:
            with part.open(encoding="utf-8") as source:
                for line in source:
                    target.write(line)
    for part in parts:
        part.unlink(missing_ok=True)


def _parallel_collect(
    config: CollectionConfig,
    *,
    games: int,
    workers: int,
    output: Path,
) -> dict[str, Any]:
    worker_count = min(workers, games)
    assignments = [tuple(range(worker, games, worker_count)) for worker in range(worker_count)]
    parts = [
        output.with_name(f".{output.name}.part{worker:02d}.jsonl")
        for worker in range(worker_count)
    ]
    if worker_count == 1:
        statistics = _collect_shard(config, assignments[0], str(parts[0]))
        _combine_parts(parts, output)
        return statistics
    statistics: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            futures = {
                executor.submit(_collect_shard, config, assignment, str(part)): worker
                for worker, (assignment, part) in enumerate(zip(assignments, parts, strict=True))
            }
            for future in as_completed(futures):
                statistics.append(future.result())
        _combine_parts(parts, output)
    except BaseException:
        for part in parts:
            part.unlink(missing_ok=True)
        raise
    return _merge_statistics(statistics, config.opponents)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-source", type=Path, required=True)
    parser.add_argument("--expert-deck", type=Path, required=True)
    parser.add_argument(
        "--opponent",
        action="append",
        choices=OPPONENT_KINDS,
        help="Repeat to mix rule, current policy, and expert mirror opponents.",
    )
    parser.add_argument(
        "--opponent-deck",
        type=Path,
        help="Deck used by rule/policy opponents; defaults to the expert deck.",
    )
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--deterministic-policy-opponent", action="store_true")
    parser.add_argument("--games", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20_260_821)
    parser.add_argument("--encoder-version", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--include-multiselect", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.games <= 0:
        raise SystemExit("--games must be greater than zero")
    if args.workers <= 0:
        raise SystemExit("--workers must be greater than zero")
    if args.torch_threads <= 0:
        raise SystemExit("--torch-threads must be greater than zero")
    opponents = _validate_configuration(
        args.opponent or ("rule", "mirror"), args.policy_checkpoint
    )
    engine = OfficialEngine(args.official_dir)
    expert_source = args.expert_source.expanduser().resolve()
    expert_deck = args.expert_deck.expanduser().resolve()
    opponent_deck = (args.opponent_deck or expert_deck).expanduser().resolve()
    # Validate paths and deck shape before spawning workers.
    engine.load_deck(expert_deck)
    engine.load_deck(opponent_deck)
    if not expert_source.is_file():
        raise FileNotFoundError(f"External expert source not found: {expert_source}")
    policy_checkpoint = (
        args.policy_checkpoint.expanduser().resolve()
        if args.policy_checkpoint is not None
        else None
    )
    if policy_checkpoint is not None and not policy_checkpoint.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {policy_checkpoint}")
    config = CollectionConfig(
        expert_source=str(expert_source),
        expert_deck=str(expert_deck),
        opponent_deck=str(opponent_deck),
        opponents=opponents,
        policy_checkpoint=str(policy_checkpoint) if policy_checkpoint is not None else None,
        policy_stochastic=not args.deterministic_policy_opponent,
        encoder_version=args.encoder_version,
        include_multiselect=args.include_multiselect,
        seed=args.seed,
        official_dir=str(engine.official_dir),
        torch_threads=args.torch_threads,
    )
    started = perf_counter()
    statistics = _parallel_collect(
        config,
        games=args.games,
        workers=args.workers,
        output=args.output.expanduser().resolve(),
    )
    elapsed = perf_counter() - started
    summary = _finalize_statistics(statistics)
    summary.update(
        {
            "expert_source": config.expert_source,
            "expert_deck": config.expert_deck,
            "opponent_deck": config.opponent_deck,
            "policy_checkpoint": config.policy_checkpoint,
            "policy_stochastic": config.policy_stochastic,
            "encoder_version": config.encoder_version,
            "action_space_version": 2 if config.include_multiselect else 1,
            "workers": min(args.workers, args.games),
            "torch_threads_per_worker": config.torch_threads,
            "fresh_expert_namespace_per_game": True,
            "elapsed_seconds": round(elapsed, 3),
            "games_per_second": round(args.games / elapsed, 3),
            "output": str(args.output.expanduser().resolve()),
        }
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
