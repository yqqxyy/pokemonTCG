"""Collect scored Library-Out trajectories against a diverse external pool."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from poketcg.agents import ExternalPythonAgent
from poketcg.engine import OfficialEngine

from .action_space import neural_selection
from .features import build_feature_encoder
from .residual_data import ResidualExample, normalize_rule_scores


@dataclass(frozen=True, slots=True)
class ExternalOpponentSpec:
    name: str
    source: str
    deck: str


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    expert_source: str
    expert_deck: str
    opponents: tuple[ExternalOpponentSpec, ...]
    encoder_version: int
    seed: int
    official_dir: str
    torch_threads: int
    target_source: str


def parse_external_opponent(specification: str) -> ExternalOpponentSpec:
    try:
        name, raw_paths = specification.split("=", 1)
        raw_source, raw_deck = raw_paths.rsplit(",", 1)
    except ValueError as error:
        raise ValueError("--external-opponent must use NAME=SOURCE,DECK") from error
    name = name.strip()
    source = Path(raw_source.strip()).expanduser().resolve()
    deck = Path(raw_deck.strip()).expanduser().resolve()
    if not name:
        raise ValueError("external opponent name cannot be empty")
    if not source.is_file():
        raise FileNotFoundError(f"External opponent source not found: {source}")
    if not deck.is_file():
        raise FileNotFoundError(f"External opponent deck not found: {deck}")
    return ExternalOpponentSpec(name=name, source=str(source), deck=str(deck))


def scheduled_matchup(
    opponents: Sequence[ExternalOpponentSpec], game: int
) -> tuple[ExternalOpponentSpec, int]:
    if not opponents:
        raise ValueError("At least one opponent is required")
    schedule = tuple((opponent, player) for opponent in opponents for player in (0, 1))
    return schedule[game % len(schedule)]


def _validate_action(observation: dict, action: list[int]) -> None:
    selection = observation["select"]
    option_count = len(selection["option"])
    minimum = int(selection["minCount"])
    maximum = int(selection["maxCount"])
    if len(action) != len(set(action)):
        raise ValueError("Agent action contains duplicate option indices")
    if not minimum <= len(action) <= maximum:
        raise ValueError("Agent action violates selection cardinality")
    if any(index < 0 or index >= option_count for index in action):
        raise IndexError("Agent action references an invalid option")


def _new_agent(source: str, deck: str, name: str, expected_deck: list[int]):
    return ExternalPythonAgent(
        source,
        deck,
        name=name,
        expected_deck=expected_deck,
    )


def _empty_statistics(opponents: Sequence[ExternalOpponentSpec]) -> dict[str, Any]:
    return {
        "games": 0,
        "examples": 0,
        "skipped_decisions": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "contexts": Counter(),
        "opponents": {
            opponent.name: {
                "games": 0,
                "examples": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "player0_games": 0,
                "player0_wins": 0,
                "player1_games": 0,
                "player1_wins": 0,
            }
            for opponent in opponents
        },
    }


def _merge_statistics(
    items: Iterable[dict[str, Any]], opponents: Sequence[ExternalOpponentSpec]
) -> dict[str, Any]:
    merged = _empty_statistics(opponents)
    for item in items:
        for key in (
            "games",
            "examples",
            "skipped_decisions",
            "wins",
            "draws",
            "losses",
        ):
            merged[key] += int(item[key])
        merged["contexts"].update(item["contexts"])
        for opponent in opponents:
            source = item["opponents"][opponent.name]
            target = merged["opponents"][opponent.name]
            for key in target:
                target[key] += int(source[key])
    return merged


def _finalize_statistics(statistics: dict[str, Any]) -> dict[str, Any]:
    result = dict(statistics)
    result["contexts"] = dict(sorted(statistics["contexts"].items()))
    opponents = {}
    for name, raw in statistics["opponents"].items():
        item = dict(raw)
        games = int(item["games"])
        item["win_rate"] = round(item["wins"] / games, 6) if games else 0.0
        opponents[name] = item
    result["opponents"] = opponents
    games = int(statistics["games"])
    result["win_rate"] = round(statistics["wins"] / games, 6) if games else 0.0
    return result


def _collect_games(
    config: CollectionConfig,
    game_ids: Iterable[int],
    output: Path,
) -> dict[str, Any]:
    torch.set_num_threads(config.torch_threads)
    engine = OfficialEngine(config.official_dir)
    expert_deck = engine.load_deck(config.expert_deck)
    opponent_decks = {
        opponent.name: engine.load_deck(opponent.deck) for opponent in config.opponents
    }
    encoder = build_feature_encoder(
        config.encoder_version, engine.card_catalog(), engine.attack_catalog()
    )
    statistics = _empty_statistics(config.opponents)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for game in game_ids:
            opponent_spec, expert_player = scheduled_matchup(config.opponents, game)
            opponent_deck = opponent_decks[opponent_spec.name]
            expert = _new_agent(
                config.expert_source,
                config.expert_deck,
                f"libraryout-teacher-{game}",
                expert_deck,
            )
            opponent = _new_agent(
                opponent_spec.source,
                opponent_spec.deck,
                f"{opponent_spec.name}-{game}",
                opponent_deck,
            )
            decks = (
                (expert_deck, opponent_deck)
                if expert_player == 0
                else (opponent_deck, expert_deck)
            )
            observation, start_data = engine.start(decks[0], decks[1])
            if observation is None:
                raise RuntimeError(
                    "Official simulator failed to start "
                    f"(errorPlayer={start_data.errorPlayer}, "
                    f"errorType={start_data.errorType})."
                )
            pending: list[ResidualExample] = []
            expert_decision = 0
            try:
                while int(observation["current"]["result"]) == -1:
                    player = int(observation["current"]["yourIndex"])
                    if player == expert_player:
                        action, raw_scores = expert.choose_action_with_scores(observation)
                        _validate_action(observation, action)
                        selection = observation["select"]
                        if neural_selection(selection, action_space_version=2):
                            pending.append(
                                ResidualExample(
                                    decision=encoder.encode(observation),
                                    baseline_action=list(action),
                                    target_action=list(action),
                                    rule_scores=normalize_rule_scores(raw_scores),
                                    value_target=0.0,
                                    player=expert_player,
                                    game=game,
                                    decision_index=expert_decision,
                                    opponent=opponent_spec.name,
                                    target_source=config.target_source,
                                )
                            )
                            statistics["contexts"][int(selection["context"])] += 1
                        else:
                            statistics["skipped_decisions"] += 1
                        expert_decision += 1
                    else:
                        action = opponent.choose_action(observation)
                        _validate_action(observation, action)
                    observation = engine.select(action)

                winner = int(observation["current"]["result"])
                if winner == 2:
                    outcome = "draws"
                    value_target = 0.0
                elif winner == expert_player:
                    outcome = "wins"
                    value_target = 1.0
                else:
                    outcome = "losses"
                    value_target = -1.0
                for example in pending:
                    example.value_target = value_target
                    stream.write(
                        json.dumps(example.to_dict(), separators=(",", ":")) + "\n"
                    )
            finally:
                engine.finish()

            statistics["games"] += 1
            statistics["examples"] += len(pending)
            statistics[outcome] += 1
            opponent_stats = statistics["opponents"][opponent_spec.name]
            opponent_stats["games"] += 1
            opponent_stats["examples"] += len(pending)
            opponent_stats[outcome] += 1
            opponent_stats[f"player{expert_player}_games"] += 1
            if outcome == "wins":
                opponent_stats[f"player{expert_player}_wins"] += 1
    return statistics


def _combine_parts(parts: Sequence[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target:
        for part in parts:
            with part.open(encoding="utf-8") as source:
                for line in source:
                    target.write(line)
    for part in parts:
        part.unlink(missing_ok=True)


def collect_parallel(
    config: CollectionConfig, games: int, workers: int, output: Path
) -> dict[str, Any]:
    worker_count = min(games, workers)
    assignments = [tuple(range(worker, games, worker_count)) for worker in range(worker_count)]
    parts = [
        output.with_name(f".{output.name}.part{worker:02d}")
        for worker in range(worker_count)
    ]
    if worker_count == 1:
        statistics = _collect_games(config, assignments[0], parts[0])
        _combine_parts(parts, output)
        return statistics
    collected = []
    context = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            futures = [
                executor.submit(
                    _collect_games,
                    config,
                    assignment,
                    part,
                )
                for assignment, part in zip(assignments, parts, strict=True)
            ]
            for future in as_completed(futures):
                collected.append(future.result())
        _combine_parts(parts, output)
    except BaseException:
        for part in parts:
            part.unlink(missing_ok=True)
        raise
    return _merge_statistics(collected, config.opponents)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-source", type=Path, required=True)
    parser.add_argument("--expert-deck", type=Path, required=True)
    parser.add_argument(
        "--external-opponent",
        action="append",
        default=[],
        metavar="NAME=SOURCE,DECK",
    )
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--games", type=int, default=1_200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--encoder-version", type=int, choices=(2, 3), default=3)
    parser.add_argument("--seed", type=int, default=20_260_722)
    parser.add_argument("--target-source", default="libraryout_v1")
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.games <= 0 or args.workers <= 0 or args.torch_threads <= 0:
        raise SystemExit("--games, --workers and --torch-threads must be positive")
    engine = OfficialEngine(args.official_dir)
    expert_source = args.expert_source.expanduser().resolve()
    expert_deck = args.expert_deck.expanduser().resolve()
    engine.load_deck(expert_deck)
    opponents = [parse_external_opponent(value) for value in args.external_opponent]
    if args.mirror:
        opponents.append(
            ExternalOpponentSpec(
                name="libraryout_mirror",
                source=str(expert_source),
                deck=str(expert_deck),
            )
        )
    names = [opponent.name for opponent in opponents]
    if not opponents:
        raise SystemExit("Add --external-opponent and/or --mirror")
    if len(names) != len(set(names)):
        raise SystemExit("Opponent names must be unique")
    for opponent in opponents:
        engine.load_deck(opponent.deck)
    config = CollectionConfig(
        expert_source=str(expert_source),
        expert_deck=str(expert_deck),
        opponents=tuple(opponents),
        encoder_version=args.encoder_version,
        seed=args.seed,
        official_dir=str(engine.official_dir),
        torch_threads=args.torch_threads,
        target_source=args.target_source,
    )
    output = args.output.expanduser().resolve()
    started = perf_counter()
    raw_statistics = collect_parallel(config, args.games, args.workers, output)
    elapsed = perf_counter() - started
    summary = _finalize_statistics(raw_statistics)
    summary.update(
        {
            "expert_source": config.expert_source,
            "expert_deck": config.expert_deck,
            "encoder_version": config.encoder_version,
            "action_space_version": 2,
            "target_source": config.target_source,
            "workers": min(args.workers, args.games),
            "elapsed_seconds": round(elapsed, 3),
            "games_per_second": round(args.games / elapsed, 3),
            "output": str(output),
        }
    )
    summary_output = (
        args.summary_output.expanduser().resolve()
        if args.summary_output
        else output.with_suffix(".summary.json")
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
