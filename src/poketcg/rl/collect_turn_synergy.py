"""Collect online full-turn synergy diagnostics around the Library-Out policy."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import random
import statistics
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

from .collect_libraryout_trajectories import (
    ExternalOpponentSpec,
    parse_external_opponent,
    scheduled_matchup,
)
from .collect_paired_rollouts import selection_reason
from .features import build_feature_encoder
from .heldout_turn_plan import HeldoutTurnPlanEvaluator
from .residual_data import normalize_rule_scores
from .turn_synergy import TurnSynergyEvaluator, turn_candidates


@dataclass(frozen=True, slots=True)
class SynergyCollectorConfig:
    expert_source: str
    expert_deck: str
    checkpoint: str
    opponents: tuple[ExternalOpponentSpec, ...]
    official_dir: str
    determinizations: int
    beam_width: int
    branch_width: int
    max_plan_steps: int
    max_rollout_steps: int
    max_states_per_game: int
    minimum_turn: int
    random_state_probability: float
    low_margin_threshold: float
    seed: int
    torch_threads: int
    heldout_semantic: bool = False
    proposal_determinizations: int = 4
    heldout_determinizations: int = 4
    plan_pool_size: int = 16
    selection_risk_multiplier: float = 1.0


def _new_external(
    source: str, deck: str, name: str, expected_deck: list[int]
) -> ExternalPythonAgent:
    return ExternalPythonAgent(
        source, deck, name=name, expected_deck=expected_deck
    )


def _collect_shard(
    config: SynergyCollectorConfig,
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
    candidate_scorer = _new_external(
        config.expert_source,
        config.expert_deck,
        f"synergy-candidate-scorer-{worker_id}",
        expert_deck,
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
        "effective_determinizations": 0,
        "hidden_synergy_worlds": 0,
        "joint_rescue_worlds": 0,
        "different_continuation_worlds": 0,
        "proposal_worlds": 0,
        "heldout_worlds": 0,
        "heldout_accepted_states": 0,
        "heldout_positive_states": 0,
        "selection_reasons": Counter(),
        "search_errors": Counter(),
        "branch_errors": Counter(),
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
                f"formal-libraryout-synergy-{worker_id}-{game}",
                expert_deck,
            )
            opponent = _new_external(
                opponent_spec.source,
                opponent_spec.deck,
                f"formal-synergy-{opponent_spec.name}-{worker_id}-{game}",
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

                                def candidate_factory(branch_observation: dict):
                                    branch_action, branch_raw_scores = (
                                        candidate_scorer.choose_action_with_scores(
                                            branch_observation
                                        )
                                    )
                                    branch_evaluation = value_policy.evaluate(
                                        branch_observation
                                    )
                                    branch_selection = branch_observation["select"]
                                    count = len(branch_selection["option"])
                                    return turn_candidates(
                                        branch_action,
                                        normalize_rule_scores(branch_raw_scores),
                                        branch_evaluation.logits[:count].tolist(),
                                        branch_selection,
                                        maximum=config.branch_width,
                                    )

                                evaluator_type = (
                                    HeldoutTurnPlanEvaluator
                                    if config.heldout_semantic
                                    else TurnSynergyEvaluator
                                )
                                evaluator_kwargs = {
                                    "beam_width": config.beam_width,
                                    "branch_width": config.branch_width,
                                    "max_plan_steps": config.max_plan_steps,
                                    "max_rollout_steps": config.max_rollout_steps,
                                    "value_policy": value_policy,
                                }
                                if config.heldout_semantic:
                                    evaluator_kwargs.update(
                                        {
                                            "proposal_determinizations": (
                                                config.proposal_determinizations
                                            ),
                                            "heldout_determinizations": (
                                                config.heldout_determinizations
                                            ),
                                            "plan_pool_size": config.plan_pool_size,
                                            "selection_risk_multiplier": (
                                                config.selection_risk_multiplier
                                            ),
                                        }
                                    )
                                else:
                                    evaluator_kwargs["determinizations"] = (
                                        config.determinizations
                                    )
                                evaluator = evaluator_type(
                                    determinizer,
                                    partial(
                                        _new_external,
                                        config.expert_source,
                                        config.expert_deck,
                                        "rollout-libraryout-synergy",
                                        expert_deck,
                                    ),
                                    partial(
                                        _new_external,
                                        opponent_spec.source,
                                        opponent_spec.deck,
                                        f"rollout-synergy-{opponent_spec.name}",
                                        opponent_deck,
                                    ),
                                    candidate_factory,
                                    **evaluator_kwargs,
                                )
                                try:
                                    diagnostic = evaluator.evaluate(observation)
                                except Exception as error:
                                    statistics["search_failures"] += 1
                                    statistics["search_errors"][
                                        type(error).__name__
                                    ] += 1
                                else:
                                    if config.heldout_semantic:
                                        proposal_pairs = int(
                                            diagnostic["proposal_selected"][
                                                "effective_pairs"
                                            ]
                                        )
                                        heldout_pairs = int(
                                            diagnostic["heldout_selected"][
                                                "effective_pairs"
                                            ]
                                        )
                                        statistics["proposal_worlds"] += proposal_pairs
                                        statistics["heldout_worlds"] += heldout_pairs
                                        statistics[
                                            "heldout_accepted_states"
                                        ] += int(diagnostic["heldout_accepted"])
                                        heldout_gain = diagnostic[
                                            "heldout_selected"
                                        ].get("paired_advantage")
                                        statistics[
                                            "heldout_positive_states"
                                        ] += int(
                                            heldout_gain is not None
                                            and float(heldout_gain) > 0.0
                                        )
                                    else:
                                        effective = int(
                                            diagnostic[
                                                "effective_determinizations"
                                            ]
                                        )
                                        statistics[
                                            "effective_determinizations"
                                        ] += effective
                                        statistics[
                                            "hidden_synergy_worlds"
                                        ] += round(
                                            float(
                                                diagnostic[
                                                    "hidden_synergy_rate"
                                                ]
                                            )
                                            * effective
                                        )
                                        statistics[
                                            "joint_rescue_worlds"
                                        ] += round(
                                            float(
                                                diagnostic["joint_rescue_rate"]
                                            )
                                            * effective
                                        )
                                        statistics[
                                            "different_continuation_worlds"
                                        ] += round(
                                            float(
                                                diagnostic[
                                                    "different_continuation_rate"
                                                ]
                                            )
                                            * effective
                                        )
                                    statistics["search_errors"].update(
                                        diagnostic["errors"]
                                    )
                                    statistics["branch_errors"].update(
                                        diagnostic["branch_errors"]
                                    )
                                    record = {
                                        "schema_version": 1,
                                        "diagnostic_kind": diagnostic.get(
                                            "diagnostic_kind",
                                            "hidden_world_full_turn_oracle",
                                        ),
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
                                        "diagnostic": diagnostic,
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
        "search_errors",
        "branch_errors",
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


def collect_parallel(
    config: SynergyCollectorConfig,
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
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=context
        ) as executor:
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


def summarize_turn_synergy(path: Path) -> dict[str, Any]:
    """Aggregate world-level oracle gains without treating them as policy labels."""
    states = 0
    worlds = 0
    synergy_worlds = 0
    rescue_worlds = 0
    multi_worlds = 0
    synergy_sum = 0.0
    one_step_gain_sum = 0.0
    full_gain_sum = 0.0
    synergy_states = 0
    rescue_states = 0
    by_opponent: dict[str, Counter[str]] = {}
    by_turn: dict[int, Counter[str]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            states += 1
            diagnostic = record["diagnostic"]
            valid = [
                sample
                for sample in diagnostic["samples"]
                if "error" not in sample
            ]
            opponent_counts = by_opponent.setdefault(record["opponent"], Counter())
            turn_counts = by_turn.setdefault(int(record["turn"]), Counter())
            state_has_synergy = any(
                float(sample["synergy_gain"]) > 0 for sample in valid
            )
            state_has_rescue = any(
                bool(sample["joint_rescue"]) for sample in valid
            )
            synergy_states += int(state_has_synergy)
            rescue_states += int(state_has_rescue)
            for counts in (opponent_counts, turn_counts):
                counts["states"] += 1
                counts["synergy_states"] += int(state_has_synergy)
                counts["rescue_states"] += int(state_has_rescue)
            for sample in valid:
                worlds += 1
                synergy = float(sample["synergy_gain"])
                synergy_sum += synergy
                one_step_gain_sum += float(sample["one_step_gain"])
                full_gain_sum += float(sample["full_turn_gain"])
                synergy_positive = int(synergy > 0)
                rescued = int(bool(sample["joint_rescue"]))
                multi = int(int(sample["joint_deviation_count"]) > 0)
                synergy_worlds += synergy_positive
                rescue_worlds += rescued
                multi_worlds += multi
                for counts in (opponent_counts, turn_counts):
                    counts["worlds"] += 1
                    counts["synergy"] += synergy_positive
                    counts["joint_rescue"] += rescued

    def rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 6) if denominator else 0.0

    def wilson95(successes: int, total: int) -> list[float]:
        if not total:
            return [0.0, 0.0]
        z = 1.959963984540054
        proportion = successes / total
        denominator = 1.0 + z**2 / total
        center = (proportion + z**2 / (2.0 * total)) / denominator
        radius = (
            z
            * math.sqrt(
                proportion * (1.0 - proportion) / total
                + z**2 / (4.0 * total**2)
            )
            / denominator
        )
        return [round(center - radius, 6), round(center + radius, 6)]

    def group_summary(groups: dict) -> dict:
        return {
            str(name): {
                "states": counts["states"],
                "state_synergy_rate": rate(
                    counts["synergy_states"], counts["states"]
                ),
                "state_joint_rescue_rate": rate(
                    counts["rescue_states"], counts["states"]
                ),
                "worlds": counts["worlds"],
                "hidden_synergy_rate": rate(counts["synergy"], counts["worlds"]),
                "joint_rescue_rate": rate(
                    counts["joint_rescue"], counts["worlds"]
                ),
            }
            for name, counts in sorted(groups.items())
        }

    return {
        "states": states,
        "effective_hidden_worlds": worlds,
        "mean_one_step_oracle_gain": round(
            one_step_gain_sum / worlds, 6
        ) if worlds else 0.0,
        "mean_full_turn_oracle_gain": round(
            full_gain_sum / worlds, 6
        ) if worlds else 0.0,
        "mean_synergy_gain": round(synergy_sum / worlds, 6) if worlds else 0.0,
        "hidden_synergy_rate": rate(synergy_worlds, worlds),
        "joint_rescue_rate": rate(rescue_worlds, worlds),
        "different_continuation_rate": rate(multi_worlds, worlds),
        "states_with_positive_mean_synergy": synergy_states,
        "state_synergy_rate": rate(synergy_states, states),
        "state_synergy_wilson95": wilson95(synergy_states, states),
        "states_with_joint_rescue": rescue_states,
        "state_joint_rescue_rate": rate(rescue_states, states),
        "state_joint_rescue_wilson95": wilson95(rescue_states, states),
        "by_opponent": group_summary(by_opponent),
        "by_turn": group_summary(by_turn),
        "warning": (
            "Full-turn plans are selected separately inside each sampled hidden "
            "world; these are oracle diagnostics, not deployable training labels."
        ),
    }


def summarize_heldout_turn_plans(path: Path) -> dict[str, Any]:
    """Aggregate state-level gains from plans selected without held-out worlds."""
    states = 0
    proposal_gains: list[float] = []
    heldout_gains: list[float] = []
    replay_successes: list[float] = []
    resolved_fractions: list[float] = []
    candidate_counts: list[int] = []
    positive = 0
    accepted = 0
    by_opponent: dict[str, Counter] = {}
    by_turn: dict[int, Counter] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            diagnostic = record["diagnostic"]
            proposal = diagnostic["proposal_selected"]
            heldout = diagnostic["heldout_selected"]
            proposal_gain = float(proposal.get("paired_advantage") or 0.0)
            heldout_gain = float(heldout.get("paired_advantage") or 0.0)
            replay_success = float(heldout["replay_success_rate"])
            resolved_fraction = float(heldout["mean_resolved_fraction"])
            is_positive = heldout_gain > 0.0
            is_accepted = bool(diagnostic["heldout_accepted"])
            states += 1
            proposal_gains.append(proposal_gain)
            heldout_gains.append(heldout_gain)
            replay_successes.append(replay_success)
            resolved_fractions.append(resolved_fraction)
            candidate_counts.append(int(diagnostic["candidate_plans"]))
            positive += int(is_positive)
            accepted += int(is_accepted)
            opponent_counts = by_opponent.setdefault(record["opponent"], Counter())
            turn_counts = by_turn.setdefault(int(record["turn"]), Counter())
            for counts in (opponent_counts, turn_counts):
                counts["states"] += 1
                counts["positive"] += int(is_positive)
                counts["accepted"] += int(is_accepted)
                counts["gain_sum"] += heldout_gain
                counts["replay_success_sum"] += replay_success

    def rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 6) if denominator else 0.0

    def mean(values: list[float]) -> float:
        return round(statistics.mean(values), 6) if values else 0.0

    def mean_ci95(values: list[float]) -> list[float]:
        if not values:
            return [0.0, 0.0]
        center = statistics.mean(values)
        stderr = (
            statistics.stdev(values) / math.sqrt(len(values))
            if len(values) > 1
            else 0.0
        )
        return [
            round(center - 1.96 * stderr, 6),
            round(center + 1.96 * stderr, 6),
        ]

    def groups(values: dict) -> dict:
        return {
            str(name): {
                "states": counts["states"],
                "positive_rate": rate(counts["positive"], counts["states"]),
                "accepted_rate": rate(counts["accepted"], counts["states"]),
                "mean_heldout_gain": round(
                    counts["gain_sum"] / max(1, counts["states"]), 6
                ),
                "mean_replay_success_rate": round(
                    counts["replay_success_sum"] / max(1, counts["states"]),
                    6,
                ),
            }
            for name, counts in sorted(values.items())
        }

    optimism = [
        proposal - heldout
        for proposal, heldout in zip(
            proposal_gains, heldout_gains, strict=True
        )
    ]
    return {
        "states": states,
        "positive_heldout_states": positive,
        "positive_heldout_rate": rate(positive, states),
        "accepted_heldout_states": accepted,
        "accepted_heldout_rate": rate(accepted, states),
        "mean_proposal_gain": mean(proposal_gains),
        "mean_heldout_gain": mean(heldout_gains),
        "mean_heldout_gain_ci95": mean_ci95(heldout_gains),
        "mean_optimism_gap": mean(optimism),
        "mean_replay_success_rate": mean(replay_successes),
        "mean_resolved_fraction": mean(resolved_fractions),
        "mean_candidate_plans": mean(
            [float(value) for value in candidate_counts]
        ),
        "by_opponent": groups(by_opponent),
        "by_turn": groups(by_turn),
        "warning": (
            "Plans are selected only on proposal determinizations; held-out gain "
            "uses one fixed semantic plan and is the deployment-relevant metric."
        ),
    }


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
    parser.add_argument("--target-states", type=int, default=30)
    parser.add_argument("--max-games", type=int, default=240)
    parser.add_argument("--determinizations", type=int, default=4)
    parser.add_argument("--heldout-semantic", action="store_true")
    parser.add_argument("--proposal-determinizations", type=int, default=4)
    parser.add_argument("--heldout-determinizations", type=int, default=4)
    parser.add_argument("--plan-pool-size", type=int, default=16)
    parser.add_argument("--selection-risk-multiplier", type=float, default=1.0)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--branch-width", type=int, default=4)
    parser.add_argument("--max-plan-steps", type=int, default=32)
    parser.add_argument("--max-rollout-steps", type=int, default=1_000)
    parser.add_argument("--max-states-per-game", type=int, default=1)
    parser.add_argument("--minimum-turn", type=int, default=3)
    parser.add_argument("--random-state-probability", type=float, default=0.25)
    parser.add_argument("--low-margin-threshold", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20_260_801)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    positive = {
        "target-states": args.target_states,
        "max-games": args.max_games,
        "determinizations": (
            args.proposal_determinizations + args.heldout_determinizations
            if args.heldout_semantic
            else args.determinizations
        ),
        "proposal-determinizations": args.proposal_determinizations,
        "heldout-determinizations": args.heldout_determinizations,
        "plan-pool-size": args.plan_pool_size,
        "beam-width": args.beam_width,
        "branch-width": args.branch_width,
        "max-plan-steps": args.max_plan_steps,
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
    if args.selection_risk_multiplier < 0:
        raise SystemExit("--selection-risk-multiplier must be non-negative")
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
    config = SynergyCollectorConfig(
        expert_source=str(expert_source),
        expert_deck=str(expert_deck),
        checkpoint=str(checkpoint),
        opponents=tuple(opponents),
        official_dir=str(engine.official_dir),
        determinizations=args.determinizations,
        beam_width=args.beam_width,
        branch_width=args.branch_width,
        max_plan_steps=args.max_plan_steps,
        max_rollout_steps=args.max_rollout_steps,
        max_states_per_game=args.max_states_per_game,
        minimum_turn=args.minimum_turn,
        random_state_probability=args.random_state_probability,
        low_margin_threshold=args.low_margin_threshold,
        seed=args.seed,
        torch_threads=args.torch_threads,
        heldout_semantic=args.heldout_semantic,
        proposal_determinizations=args.proposal_determinizations,
        heldout_determinizations=args.heldout_determinizations,
        plan_pool_size=args.plan_pool_size,
        selection_risk_multiplier=args.selection_risk_multiplier,
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
    for key in (
        "selection_reasons",
        "search_errors",
        "branch_errors",
        "opponents",
        "turns",
    ):
        raw[key] = dict(sorted(raw[key].items()))
    summary = {
        **raw,
        "diagnostic_mode": (
            "heldout_semantic" if args.heldout_semantic else "oracle"
        ),
        "target_states": args.target_states,
        "determinizations": (
            args.proposal_determinizations + args.heldout_determinizations
            if args.heldout_semantic
            else args.determinizations
        ),
        "proposal_determinizations": args.proposal_determinizations,
        "heldout_determinizations": args.heldout_determinizations,
        "plan_pool_size": args.plan_pool_size,
        "beam_width": args.beam_width,
        "branch_width": args.branch_width,
        "minimum_turn": args.minimum_turn,
        "seed": args.seed,
        "workers": min(args.workers, args.target_states),
        "elapsed_seconds": round(elapsed, 3),
        "states_per_second": round(raw["states"] / elapsed, 6),
        "diagnostics": (
            summarize_heldout_turn_plans(output)
            if args.heldout_semantic
            else summarize_turn_synergy(output)
        ),
        "output": str(output),
    }
    summary_path = (
        args.summary_output.expanduser().resolve()
        if args.summary_output
        else output.with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
