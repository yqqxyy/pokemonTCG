"""Collect online paired one-step-deviation labels for Library-Out V1."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from poketcg.agents import BCPolicyAgent, ExternalPythonAgent
from poketcg.engine import OfficialEngine
from poketcg.mcts import DeckDeterminizer

from .advantage_candidates import root_candidates
from .collect_libraryout_trajectories import (
    ExternalOpponentSpec,
    parse_external_opponent,
    scheduled_matchup,
)
from .features import build_feature_encoder
from .paired_rollout import PairedRolloutEvaluator
from .residual_data import normalize_rule_scores


@dataclass(frozen=True, slots=True)
class CollectorConfig:
    expert_source: str
    expert_deck: str
    checkpoint: str
    opponents: tuple[ExternalOpponentSpec, ...]
    official_dir: str
    determinizations: int
    max_rollout_steps: int
    max_states_per_game: int
    minimum_turn: int
    random_state_probability: float
    low_margin_threshold: float
    seed: int
    torch_threads: int


def selection_reason(
    baseline_action: list[int],
    normalized_rule_scores: list[float],
    model_logits: list[float],
    *,
    low_margin_threshold: float,
    random_probability: float,
    rng: random.Random,
) -> str | None:
    """Choose online-search states without using their future game outcome."""
    if len(baseline_action) != 1 or len(normalized_rule_scores) < 2:
        return None
    model_choice = max(range(len(model_logits)), key=model_logits.__getitem__)
    if model_choice != baseline_action[0]:
        return "round0_disagreement"
    ordered = sorted(normalized_rule_scores, reverse=True)
    if ordered[0] - ordered[1] <= low_margin_threshold:
        return "low_rule_margin"
    if rng.random() < random_probability:
        return "random_calibration"
    return None


def _new_external(
    source: str, deck: str, name: str, expected_deck: list[int]
) -> ExternalPythonAgent:
    return ExternalPythonAgent(
        source, deck, name=name, expected_deck=expected_deck
    )


def _collect_shard(
    config: CollectorConfig,
    *,
    worker_id: int,
    worker_count: int,
    target_states: int,
    max_games: int,
    output: str,
) -> dict[str, Any]:
    torch.set_num_threads(config.torch_threads)
    engine = OfficialEngine(config.official_dir)
    expert_deck = engine.load_deck(config.expert_deck)
    opponent_decks = {
        opponent.name: engine.load_deck(opponent.deck) for opponent in config.opponents
    }
    cards = engine.card_catalog()
    attacks = engine.attack_catalog()
    basic_card_ids = {
        card_id for card_id, card in cards.items() if bool(getattr(card, "basic", False))
    }
    encoder = build_feature_encoder(3, cards, attacks)
    value_policy = BCPolicyAgent(
        config.checkpoint,
        card_catalog=cards,
        attack_catalog=attacks,
        deterministic=True,
        device="cpu",
    )
    rng = random.Random(config.seed + worker_id * 1_000_003)
    statistics: dict[str, Any] = {
        "games": 0,
        "states": 0,
        "formal_wins": 0,
        "formal_losses": 0,
        "formal_draws": 0,
        "search_attempts": 0,
        "search_failures": 0,
        "branches_requested": 0,
        "selection_reasons": Counter(),
        "rollout_boundaries": Counter(),
        "rollout_errors": Counter(),
        "opponents": Counter(),
        "turns": Counter(),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for local_game in range(max_games):
            if statistics["states"] >= target_states:
                break
            game = worker_id + local_game * worker_count
            opponent_spec, expert_player = scheduled_matchup(config.opponents, game)
            opponent_deck = opponent_decks[opponent_spec.name]
            expert = _new_external(
                config.expert_source,
                config.expert_deck,
                f"formal-libraryout-{worker_id}-{game}",
                expert_deck,
            )
            opponent = _new_external(
                opponent_spec.source,
                opponent_spec.deck,
                f"formal-{opponent_spec.name}-{worker_id}-{game}",
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
            game_records = []
            searched_this_game = 0
            decision_index = 0
            try:
                while int(observation["current"]["result"]) == -1:
                    player = int(observation["current"]["yourIndex"])
                    if player == expert_player:
                        action, raw_scores = expert.choose_action_with_scores(observation)
                        selection = observation["select"]
                        exact_main = (
                            int(selection["context"]) == 0
                            and int(selection["minCount"]) == 1
                            and int(selection["maxCount"]) == 1
                            and len(selection["option"]) >= 2
                        )
                        turn = int(observation["current"]["turn"])
                        if (
                            exact_main
                            and turn >= config.minimum_turn
                            and statistics["states"] + len(game_records) < target_states
                            and searched_this_game < config.max_states_per_game
                        ):
                            evaluation = value_policy.evaluate(observation)
                            option_count = len(selection["option"])
                            model_logits = evaluation.logits[:option_count].tolist()
                            normalized_scores = normalize_rule_scores(raw_scores)
                            reason = selection_reason(
                                action,
                                normalized_scores,
                                model_logits,
                                low_margin_threshold=config.low_margin_threshold,
                                random_probability=config.random_state_probability,
                                rng=rng,
                            )
                            if reason is not None:
                                candidates = root_candidates(
                                    action,
                                    normalized_scores,
                                    model_logits,
                                    [int(option["type"]) for option in selection["option"]],
                                )
                                if len(candidates) >= 2:
                                    statistics["search_attempts"] += 1
                                    determinizer = DeckDeterminizer(
                                        list(decks[0]),
                                        list(decks[1]),
                                        basic_card_ids=basic_card_ids,
                                        seed=(
                                            config.seed
                                            + worker_id * 10_000_019
                                            + game * 10_007
                                            + decision_index
                                        ),
                                    )
                                    evaluator = PairedRolloutEvaluator(
                                        determinizer,
                                        partial(
                                            _new_external,
                                            config.expert_source,
                                            config.expert_deck,
                                            "rollout-libraryout",
                                            expert_deck,
                                        ),
                                        partial(
                                            _new_external,
                                            opponent_spec.source,
                                            opponent_spec.deck,
                                            f"rollout-{opponent_spec.name}",
                                            opponent_deck,
                                        ),
                                        determinizations=config.determinizations,
                                        max_rollout_steps=config.max_rollout_steps,
                                        value_policy=value_policy,
                                    )
                                    statistics["branches_requested"] += (
                                        len(candidates) * config.determinizations
                                    )
                                    try:
                                        rollout = evaluator.evaluate(
                                            observation, candidates
                                        )
                                    except Exception as error:
                                        statistics["search_failures"] += 1
                                        statistics["rollout_errors"][
                                            type(error).__name__
                                        ] += 1
                                    else:
                                        for sample in rollout["samples"]:
                                            for branch in sample["branches"]:
                                                if "boundary" in branch:
                                                    statistics["rollout_boundaries"][
                                                        branch["boundary"]
                                                    ] += 1
                                        statistics["rollout_errors"].update(
                                            rollout["errors"]
                                        )
                                        record = {
                                            "schema_version": 2,
                                            "state_id": (
                                                f"s{config.seed}-w{worker_id}-"
                                                f"g{game}-d{decision_index}"
                                            ),
                                            "collector_seed": config.seed,
                                            "game": game,
                                            "decision_index": decision_index,
                                            "opponent": opponent_spec.name,
                                            "player": expert_player,
                                            "turn": turn,
                                            "selection_reason": reason,
                                            "decision": encoder.encode(
                                                observation
                                            ).to_dict(),
                                            "rule_action": list(action),
                                            "raw_rule_scores": raw_scores,
                                            "normalized_rule_scores": normalized_scores,
                                            "round0_logits": model_logits,
                                            "round0_value": evaluation.value,
                                            "rollout": rollout,
                                        }
                                        game_records.append(record)
                                        searched_this_game += 1
                                        statistics["selection_reasons"][reason] += 1
                                        statistics["turns"][turn] += 1
                    else:
                        action = opponent.choose_action(observation)
                    observation = engine.select(action)
                    decision_index += 1

                winner = int(observation["current"]["result"])
                if winner == 2:
                    outcome = "draws"
                    behavior_return = 0.0
                elif winner == expert_player:
                    outcome = "wins"
                    behavior_return = 1.0
                else:
                    outcome = "losses"
                    behavior_return = -1.0
                statistics[f"formal_{outcome}"] += 1
                for record in game_records:
                    record["formal_behavior_return"] = behavior_return
                    stream.write(
                        json.dumps(record, separators=(",", ":")) + "\n"
                    )
            finally:
                engine.finish()
            statistics["games"] += 1
            statistics["states"] += len(game_records)
            statistics["opponents"][opponent_spec.name] += len(game_records)
    return statistics


def _merge_parts(parts: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target:
        for part in parts:
            with part.open(encoding="utf-8") as source:
                for line in source:
                    target.write(line)
    for part in parts:
        part.unlink(missing_ok=True)


def _merge_statistics(items: list[dict[str, Any]]) -> dict[str, Any]:
    counters = {
        "selection_reasons",
        "rollout_boundaries",
        "rollout_errors",
        "opponents",
        "turns",
    }
    merged: dict[str, Any] = {
        key: Counter() if key in counters else 0 for key in items[0]
    }
    for item in items:
        for key, value in item.items():
            if key in counters:
                merged[key].update(value)
            else:
                merged[key] += int(value)
    return merged


def summarize_paired_dataset(path: Path) -> dict[str, Any]:
    """Summarize advantage coverage without loading encoded states into memory."""
    states = 0
    candidate_count = Counter()
    advantages = []
    stderrs = []
    positive_states = 0
    conservative_positive_states = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            states += 1
            candidates = record["rollout"]["candidates"]
            candidate_count[len(candidates)] += 1
            state_positive = False
            state_conservative = False
            for candidate in candidates[1:]:
                advantage = candidate.get("paired_advantage")
                stderr = candidate.get("paired_stderr")
                ci95 = candidate.get("paired_ci95")
                if advantage is None:
                    continue
                advantages.append(float(advantage))
                if stderr is not None:
                    stderrs.append(float(stderr))
                state_positive |= float(advantage) > 0.0
                state_conservative |= bool(ci95 and float(ci95[0]) > 0.05)
            positive_states += int(state_positive)
            conservative_positive_states += int(state_conservative)

    def quantile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        return round(ordered[round((len(ordered) - 1) * fraction)], 6)

    return {
        "states": states,
        "candidate_count": dict(sorted(candidate_count.items())),
        "nonbaseline_candidates": len(advantages),
        "positive_advantage_candidates": sum(value > 0 for value in advantages),
        "negative_advantage_candidates": sum(value < 0 for value in advantages),
        "zero_advantage_candidates": sum(value == 0 for value in advantages),
        "states_with_positive_candidate": positive_states,
        "states_with_lcb95_above_0_05": conservative_positive_states,
        "advantage_quantiles": {
            "p10": quantile(advantages, 0.10),
            "p25": quantile(advantages, 0.25),
            "p50": quantile(advantages, 0.50),
            "p75": quantile(advantages, 0.75),
            "p90": quantile(advantages, 0.90),
        },
        "mean_paired_stderr": (
            round(sum(stderrs) / len(stderrs), 6) if stderrs else None
        ),
    }


def collect_parallel(
    config: CollectorConfig,
    *,
    target_states: int,
    max_games: int,
    workers: int,
    output: Path,
) -> dict[str, Any]:
    worker_count = min(workers, target_states)
    targets = [
        target_states // worker_count + int(worker < target_states % worker_count)
        for worker in range(worker_count)
    ]
    games = [
        max_games // worker_count + int(worker < max_games % worker_count)
        for worker in range(worker_count)
    ]
    parts = [
        output.with_name(f".{output.name}.part{worker:02d}")
        for worker in range(worker_count)
    ]
    if worker_count == 1:
        result = _collect_shard(
            config,
            worker_id=0,
            worker_count=1,
            target_states=targets[0],
            max_games=games[0],
            output=str(parts[0]),
        )
        _merge_parts(parts, output)
        return result
    context = multiprocessing.get_context("spawn")
    results = []
    try:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            futures = [
                executor.submit(
                    _collect_shard,
                    config,
                    worker_id=worker,
                    worker_count=worker_count,
                    target_states=targets[worker],
                    max_games=games[worker],
                    output=str(parts[worker]),
                )
                for worker in range(worker_count)
            ]
            for future in as_completed(futures):
                results.append(future.result())
        _merge_parts(parts, output)
    except BaseException:
        for part in parts:
            part.unlink(missing_ok=True)
        raise
    return _merge_statistics(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-source", type=Path, required=True)
    parser.add_argument("--expert-deck", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--external-opponent",
        action="append",
        default=[],
        metavar="NAME=SOURCE,DECK",
    )
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--target-states", type=int, default=500)
    parser.add_argument("--max-games", type=int, default=1_400)
    parser.add_argument("--determinizations", type=int, default=16)
    parser.add_argument("--max-rollout-steps", type=int, default=1_000)
    parser.add_argument("--max-states-per-game", type=int, default=2)
    parser.add_argument("--minimum-turn", type=int, default=1)
    parser.add_argument("--random-state-probability", type=float, default=0.15)
    parser.add_argument("--low-margin-threshold", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20_260_723)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    positive = {
        "target-states": args.target_states,
        "max-games": args.max_games,
        "determinizations": args.determinizations,
        "max-rollout-steps": args.max_rollout_steps,
        "max-states-per-game": args.max_states_per_game,
        "minimum-turn": args.minimum_turn,
        "workers": args.workers,
        "torch-threads": args.torch_threads,
    }
    for name, value in positive.items():
        if value <= 0:
            raise SystemExit(f"--{name} must be positive")
    if not 0.0 <= args.random_state_probability <= 1.0:
        raise SystemExit("--random-state-probability must be in [0, 1]")
    engine = OfficialEngine(args.official_dir)
    expert_source = args.expert_source.expanduser().resolve()
    expert_deck = args.expert_deck.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    engine.load_deck(expert_deck)
    if not expert_source.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("expert source and checkpoint must exist")
    opponents = [parse_external_opponent(value) for value in args.external_opponent]
    if args.mirror:
        opponents.append(
            ExternalOpponentSpec(
                "libraryout_mirror", str(expert_source), str(expert_deck)
            )
        )
    if not opponents:
        raise SystemExit("Add --external-opponent and/or --mirror")
    names = [opponent.name for opponent in opponents]
    if len(names) != len(set(names)):
        raise SystemExit("Opponent names must be unique")
    for opponent in opponents:
        engine.load_deck(opponent.deck)
    config = CollectorConfig(
        expert_source=str(expert_source),
        expert_deck=str(expert_deck),
        checkpoint=str(checkpoint),
        opponents=tuple(opponents),
        official_dir=str(engine.official_dir),
        determinizations=args.determinizations,
        max_rollout_steps=args.max_rollout_steps,
        max_states_per_game=args.max_states_per_game,
        minimum_turn=args.minimum_turn,
        random_state_probability=args.random_state_probability,
        low_margin_threshold=args.low_margin_threshold,
        seed=args.seed,
        torch_threads=args.torch_threads,
    )
    output = args.output.expanduser().resolve()
    started = perf_counter()
    raw = collect_parallel(
        config,
        target_states=args.target_states,
        max_games=args.max_games,
        workers=args.workers,
        output=output,
    )
    elapsed = perf_counter() - started
    label_diagnostics = summarize_paired_dataset(output)
    for key in (
        "selection_reasons",
        "rollout_boundaries",
        "rollout_errors",
        "opponents",
        "turns",
    ):
        raw[key] = dict(sorted(raw[key].items()))
    summary = {
        **raw,
        "target_states": args.target_states,
        "determinizations": args.determinizations,
        "minimum_turn": args.minimum_turn,
        "seed": args.seed,
        "workers": min(args.workers, args.target_states),
        "elapsed_seconds": round(elapsed, 3),
        "states_per_second": round(raw["states"] / elapsed, 6),
        "label_diagnostics": label_diagnostics,
        "output": str(output),
    }
    summary_path = (
        args.summary_output.expanduser().resolve()
        if args.summary_output
        else output.with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
