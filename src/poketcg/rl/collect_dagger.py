"""Collect DAgger labels by querying an external expert on student-visited states."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
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
from .collect_external_expert import (
    OPPONENT_KINDS,
    _combine_parts,
    _validate_action,
    scheduled_matchup,
)
from .data import BCExample
from .features import build_feature_encoder
from .model import action_space_version, encoder_version


@dataclass(frozen=True, slots=True)
class DAggerConfig:
    student_checkpoint: str
    expert_source: str
    expert_deck: str
    opponent_deck: str
    opponents: tuple[str, ...]
    opponent_policy_checkpoint: str | None
    encoder_version: int
    action_space_version: int
    beta: float
    stochastic_student: bool
    stochastic_policy_opponent: bool
    seed: int
    official_dir: str
    torch_threads: int


def choose_dagger_action(
    student_action: list[int],
    expert_action: list[int],
    *,
    beta: float,
    rng: random.Random,
) -> tuple[list[int], str]:
    """Execute the expert with probability beta and otherwise execute the student."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if rng.random() < beta:
        return list(expert_action), "expert"
    return list(student_action), "student"


def _validate_configuration(
    opponents: Sequence[str],
    *,
    beta: float,
) -> tuple[str, ...]:
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    normalized = tuple(str(value).lower() for value in opponents)
    if not normalized:
        raise ValueError("At least one opponent is required")
    unknown = sorted(set(normalized) - set(OPPONENT_KINDS))
    if unknown:
        raise ValueError(f"Unknown opponent kinds: {', '.join(unknown)}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Opponent kinds must not be repeated")
    return normalized


def _checkpoint_versions(checkpoint: str | Path) -> tuple[int, int]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = saved.get("model_config")
    if not isinstance(model_config, dict):
        raise TypeError("Student checkpoint must contain model_config")
    return encoder_version(model_config), action_space_version(model_config)


def _new_expert(
    config: DAggerConfig,
    expert_deck: list[int],
    *,
    name: str,
) -> ExternalPythonAgent:
    return ExternalPythonAgent(
        config.expert_source,
        config.expert_deck,
        name=name,
        expected_deck=expert_deck,
    )


def _new_opponent_statistics(opponents: Sequence[str]) -> dict[str, dict[str, int]]:
    return {
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
    }


def _new_statistics(opponents: Sequence[str]) -> dict[str, Any]:
    return {
        "games": 0,
        "examples": 0,
        "skipped_decisions": 0,
        "student_decisions": 0,
        "expert_executions": 0,
        "student_executions": 0,
        "disagreements": 0,
        "student_outcomes": Counter(),
        "contexts": {},
        "opponents": _new_opponent_statistics(opponents),
    }


def _update_context(
    statistics: dict[str, Any],
    context: int,
    *,
    disagreed: bool,
    execution_source: str,
) -> None:
    item = statistics["contexts"].setdefault(
        context,
        {
            "count": 0,
            "disagreements": 0,
            "expert_executions": 0,
            "student_executions": 0,
        },
    )
    item["count"] += 1
    item["disagreements"] += int(disagreed)
    item[f"{execution_source}_executions"] += 1


def _finalize_statistics(statistics: dict[str, Any]) -> dict[str, Any]:
    decisions = int(statistics["student_decisions"])
    contexts = {}
    for context, raw in sorted(statistics["contexts"].items()):
        count = int(raw["count"])
        item = dict(raw)
        item["disagreement_rate"] = round(item["disagreements"] / count, 6)
        item["realized_beta"] = round(item["expert_executions"] / count, 6)
        contexts[str(context)] = item

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
        "student_decisions": decisions,
        "expert_executions": int(statistics["expert_executions"]),
        "student_executions": int(statistics["student_executions"]),
        "realized_beta": (
            round(statistics["expert_executions"] / decisions, 6) if decisions else 0.0
        ),
        "disagreements": int(statistics["disagreements"]),
        "disagreement_rate": (
            round(statistics["disagreements"] / decisions, 6) if decisions else 0.0
        ),
        "student_outcomes": dict(sorted(statistics["student_outcomes"].items())),
        "contexts": contexts,
        "opponents": opponents,
    }


def _merge_statistics(
    items: Iterable[dict[str, Any]], opponents: Sequence[str]
) -> dict[str, Any]:
    merged = _new_statistics(opponents)
    scalar_keys = (
        "games",
        "examples",
        "skipped_decisions",
        "student_decisions",
        "expert_executions",
        "student_executions",
        "disagreements",
    )
    for source in items:
        for key in scalar_keys:
            merged[key] += int(source[key])
        merged["student_outcomes"].update(source["student_outcomes"])
        for context, raw in source["contexts"].items():
            target = merged["contexts"].setdefault(
                int(context),
                {
                    "count": 0,
                    "disagreements": 0,
                    "expert_executions": 0,
                    "student_executions": 0,
                },
            )
            for key in target:
                target[key] += int(raw[key])
        for opponent in opponents:
            source_opponent = source["opponents"][opponent]
            target_opponent = merged["opponents"][opponent]
            for key in target_opponent:
                target_opponent[key] += int(source_opponent[key])
    return merged


def _collect_games(
    engine: OfficialEngine,
    config: DAggerConfig,
    game_ids: Iterable[int],
    emit: Callable[[BCExample], None],
) -> dict[str, Any]:
    opponents = _validate_configuration(config.opponents, beta=config.beta)
    torch.set_num_threads(config.torch_threads)
    expert_deck = engine.load_deck(config.expert_deck)
    opponent_deck = engine.load_deck(config.opponent_deck)
    card_catalog = engine.card_catalog()
    attack_catalog = engine.attack_catalog()
    encoder = build_feature_encoder(
        config.encoder_version, card_catalog, attack_catalog
    )
    student = BCPolicyAgent(
        config.student_checkpoint,
        card_catalog=card_catalog,
        attack_catalog=attack_catalog,
        seed=config.seed + 700_000,
        deterministic=not config.stochastic_student,
    )
    policy_opponent = None
    if "policy" in opponents:
        policy_opponent = BCPolicyAgent(
            config.opponent_policy_checkpoint or config.student_checkpoint,
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            seed=config.seed + 800_000,
            deterministic=not config.stochastic_policy_opponent,
        )

    statistics = _new_statistics(opponents)
    for game in game_ids:
        opponent_kind, student_player = scheduled_matchup(opponents, game)
        shadow_expert = _new_expert(
            config,
            expert_deck,
            name=f"shadow-teacher-{game}",
        )
        if opponent_kind == "mirror":
            opponent = _new_expert(
                config,
                expert_deck,
                name=f"mirror-opponent-{game}",
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
            if policy_opponent is None:
                raise AssertionError("Policy opponent was not initialized")
            opponent = policy_opponent
            current_opponent_deck = opponent_deck

        decks = (
            (expert_deck, current_opponent_deck)
            if student_player == 0
            else (current_opponent_deck, expert_deck)
        )
        rng = random.Random(config.seed + game * 1_000_003)
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
                if player != student_player:
                    action = opponent.choose_action(observation)
                    _validate_action(observation, action)
                    observation = engine.select(action)
                    continue

                # Query the shadow expert on every student decision, including forced
                # selections. This keeps its cross-decision plan synchronized.
                expert_action = shadow_expert.choose_action(observation)
                student_action = student.choose_action(observation)
                _validate_action(observation, expert_action)
                _validate_action(observation, student_action)
                selection = observation["select"]
                if neural_selection(selection, config.action_space_version):
                    decision = encoder.encode(observation)
                    pending.append((decision, list(expert_action)))
                    action, source = choose_dagger_action(
                        student_action,
                        expert_action,
                        beta=config.beta,
                        rng=rng,
                    )
                    disagreed = sorted(student_action) != sorted(expert_action)
                    statistics["student_decisions"] += 1
                    statistics[f"{source}_executions"] += 1
                    statistics["disagreements"] += int(disagreed)
                    _update_context(
                        statistics,
                        int(selection["context"]),
                        disagreed=disagreed,
                        execution_source=source,
                    )
                else:
                    statistics["skipped_decisions"] += 1
                    action = student_action
                observation = engine.select(action)

            winner = int(observation["current"]["result"])
            outcome = "draw" if winner == 2 else ("win" if winner == student_player else "loss")
            value_target = 0.0 if outcome == "draw" else (1.0 if outcome == "win" else -1.0)
            for decision, expert_action in pending:
                emit(
                    BCExample(
                        decision=decision,
                        action=expert_action,
                        value_target=value_target,
                        player=student_player,
                        game=game,
                    )
                )
        finally:
            engine.finish()

        statistics["games"] += 1
        statistics["examples"] += len(pending)
        statistics["student_outcomes"][outcome] += 1
        opponent_stats = statistics["opponents"][opponent_kind]
        opponent_stats["games"] += 1
        opponent_stats["examples"] += len(pending)
        outcome_key = {"win": "wins", "draw": "draws", "loss": "losses"}[outcome]
        opponent_stats[outcome_key] += 1
        opponent_stats[f"as_player{student_player}_games"] += 1
        if outcome == "win":
            opponent_stats[f"as_player{student_player}_wins"] += 1
    return statistics


def collect_dagger_examples(
    engine: OfficialEngine,
    student_checkpoint: str | Path,
    expert_source: str | Path,
    expert_deck: str | Path,
    *,
    games: int,
    beta: float,
    seed: int,
    opponents: Sequence[str] = ("rule", "policy", "mirror"),
    opponent_deck: str | Path | None = None,
    opponent_policy_checkpoint: str | Path | None = None,
    stochastic_student: bool = False,
    stochastic_policy_opponent: bool = True,
    torch_threads: int = 1,
) -> tuple[list[BCExample], dict[str, Any]]:
    """Collect an in-process DAgger screen and return expert-labelled examples."""
    if games <= 0:
        raise ValueError("games must be positive")
    if torch_threads <= 0:
        raise ValueError("torch_threads must be positive")
    normalized = _validate_configuration(opponents, beta=beta)
    resolved_expert_deck = Path(expert_deck).expanduser().resolve()
    feature_version, action_version = _checkpoint_versions(student_checkpoint)
    config = DAggerConfig(
        student_checkpoint=str(Path(student_checkpoint).expanduser().resolve()),
        expert_source=str(Path(expert_source).expanduser().resolve()),
        expert_deck=str(resolved_expert_deck),
        opponent_deck=str(Path(opponent_deck or resolved_expert_deck).expanduser().resolve()),
        opponents=normalized,
        opponent_policy_checkpoint=(
            str(Path(opponent_policy_checkpoint).expanduser().resolve())
            if opponent_policy_checkpoint is not None
            else None
        ),
        encoder_version=feature_version,
        action_space_version=action_version,
        beta=beta,
        stochastic_student=stochastic_student,
        stochastic_policy_opponent=stochastic_policy_opponent,
        seed=seed,
        official_dir=str(engine.official_dir),
        torch_threads=torch_threads,
    )
    examples: list[BCExample] = []
    started = perf_counter()
    statistics = _collect_games(engine, config, range(games), examples.append)
    summary = _finalize_statistics(statistics)
    summary.update(_summary_metadata(config, elapsed=perf_counter() - started, workers=1))
    return examples, summary


def _summary_metadata(
    config: DAggerConfig,
    *,
    elapsed: float,
    workers: int,
) -> dict[str, Any]:
    return {
        "student_checkpoint": config.student_checkpoint,
        "expert_source": config.expert_source,
        "expert_deck": config.expert_deck,
        "opponent_deck": config.opponent_deck,
        "opponent_policy_checkpoint": (
            config.opponent_policy_checkpoint or config.student_checkpoint
        ),
        "encoder_version": config.encoder_version,
        "action_space_version": config.action_space_version,
        "beta": config.beta,
        "student_action_selection": (
            "stochastic" if config.stochastic_student else "deterministic"
        ),
        "policy_opponent_action_selection": (
            "stochastic" if config.stochastic_policy_opponent else "deterministic"
        ),
        "workers": workers,
        "torch_threads_per_worker": config.torch_threads,
        "shadow_expert_queried_on_every_student_decision": True,
        "fresh_expert_namespace_per_game": True,
        "elapsed_seconds": round(elapsed, 3),
    }


def _collect_shard(
    config: DAggerConfig,
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


def _parallel_collect(
    config: DAggerConfig,
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
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-source", type=Path, required=True)
    parser.add_argument("--expert-deck", type=Path, required=True)
    parser.add_argument(
        "--opponent",
        action="append",
        choices=OPPONENT_KINDS,
        help="Repeat to mix rule, policy, and external-expert mirror opponents.",
    )
    parser.add_argument("--opponent-deck", type=Path)
    parser.add_argument(
        "--opponent-policy-checkpoint",
        type=Path,
        help="Defaults to the student checkpoint.",
    )
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--stochastic-student", action="store_true")
    parser.add_argument("--deterministic-policy-opponent", action="store_true")
    parser.add_argument("--games", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20_260_824)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _resolved_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.games <= 0:
        raise SystemExit("--games must be greater than zero")
    if args.workers <= 0:
        raise SystemExit("--workers must be greater than zero")
    if args.torch_threads <= 0:
        raise SystemExit("--torch-threads must be greater than zero")
    try:
        opponents = _validate_configuration(
            args.opponent or ("rule", "policy", "mirror"), beta=args.beta
        )
        student_checkpoint = _resolved_file(args.student_checkpoint, "Student checkpoint")
        expert_source = _resolved_file(args.expert_source, "External expert source")
        expert_deck = _resolved_file(args.expert_deck, "External expert deck")
        opponent_deck = _resolved_file(
            args.opponent_deck or expert_deck, "Opponent deck"
        )
        opponent_policy_checkpoint = (
            _resolved_file(args.opponent_policy_checkpoint, "Opponent policy checkpoint")
            if args.opponent_policy_checkpoint is not None
            else None
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    engine = OfficialEngine(args.official_dir)
    engine.load_deck(expert_deck)
    engine.load_deck(opponent_deck)
    feature_version, action_version = _checkpoint_versions(student_checkpoint)
    config = DAggerConfig(
        student_checkpoint=str(student_checkpoint),
        expert_source=str(expert_source),
        expert_deck=str(expert_deck),
        opponent_deck=str(opponent_deck),
        opponents=opponents,
        opponent_policy_checkpoint=(
            str(opponent_policy_checkpoint)
            if opponent_policy_checkpoint is not None
            else None
        ),
        encoder_version=feature_version,
        action_space_version=action_version,
        beta=args.beta,
        stochastic_student=args.stochastic_student,
        stochastic_policy_opponent=not args.deterministic_policy_opponent,
        seed=args.seed,
        official_dir=str(engine.official_dir),
        torch_threads=args.torch_threads,
    )
    started = perf_counter()
    output = args.output.expanduser().resolve()
    statistics = _parallel_collect(
        config,
        games=args.games,
        workers=args.workers,
        output=output,
    )
    elapsed = perf_counter() - started
    summary = _finalize_statistics(statistics)
    summary.update(
        _summary_metadata(
            config,
            elapsed=elapsed,
            workers=min(args.workers, args.games),
        )
    )
    summary["games_per_second"] = round(args.games / elapsed, 3)
    summary["output"] = str(output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
