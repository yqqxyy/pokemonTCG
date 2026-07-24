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
from .heldout_option import HeldoutCardEffectEvaluator
from .heldout_turn_plan import HeldoutTurnPlanEvaluator
from .macro_oracle import MacroPlanOracleEvaluator
from .macro_plan import MacroPlanGenerator
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
    heldout_option: bool = False
    macro_oracle: bool = False
    build_determinizations: int = 8
    calibration_determinizations: int = 8
    proposal_determinizations: int = 4
    heldout_determinizations: int = 4
    plan_pool_size: int = 16
    selection_risk_multiplier: float = 1.0
    max_option_steps: int = 12
    option_gate_multiplier: float = 1.96
    option_min_calibration_pairs: int = 8
    option_min_coverage: float = 0.9
    option_familywise_alpha: float = 0.05
    macro_alignment_weight: float = 0.05
    compact_macro_oracle: bool = False
    phase_quotas: tuple[int, int, int] | None = None
    early_max_turn: int = 4
    mid_max_turn: int = 7


TURN_PHASES = ("early", "mid", "late")


def turn_phase(
    turn: int,
    *,
    early_max_turn: int = 4,
    mid_max_turn: int = 7,
) -> str:
    """Map one public turn to the configured collection phase."""
    if early_max_turn >= mid_max_turn:
        raise ValueError("early_max_turn must be smaller than mid_max_turn")
    if turn <= early_max_turn:
        return "early"
    if turn <= mid_max_turn:
        return "mid"
    return "late"


def split_phase_quotas(
    quotas: tuple[int, int, int],
    workers: int,
) -> list[dict[str, int]]:
    """Distribute every phase quota exactly across worker shards."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    if any(value < 0 for value in quotas):
        raise ValueError("phase quotas must be non-negative")
    result = [dict.fromkeys(TURN_PHASES, 0) for _ in range(workers)]
    cursor = 0
    for phase, total in zip(TURN_PHASES, quotas, strict=True):
        base, remainder = divmod(total, workers)
        for worker in range(workers):
            result[worker][phase] = base
        for offset in range(remainder):
            worker = (cursor + offset) % workers
            result[worker][phase] += 1
        cursor = (cursor + remainder) % workers
    return result


def compact_macro_diagnostic(
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    """Drop non-teacher beam leaves after search without changing winners."""
    if diagnostic.get("diagnostic_kind") != "macro_plan_oracle_v2_libraryout":
        raise ValueError("compact mode requires macro_plan_oracle_v2_libraryout")
    beam_trajectories = 0
    beam_executor_rows = 0
    retained_trajectories = 0
    retained_executor_rows = 0
    for sample in diagnostic["samples"]:
        if "error" in sample:
            continue
        for plan_result in sample["plans"]:
            trajectories = plan_result.pop("trajectories")
            best = plan_result["best_trajectory"]
            best_index = next(
                (
                    index
                    for index, trajectory in enumerate(trajectories)
                    if trajectory == best
                ),
                None,
            )
            if best_index is None:
                raise ValueError(
                    "best_trajectory is absent from its source beam"
                )
            plan_result["best_trajectory_index"] = best_index
            beam_trajectories += len(trajectories)
            beam_executor_rows += sum(
                int(trajectory["decision_count"])
                for trajectory in trajectories
            )
            retained_trajectories += 1
            retained_executor_rows += int(best["decision_count"])
    diagnostic["serialization"] = {
        "mode": "best_per_plan_compact",
        "search_quality_preserved": True,
        "beam_trajectories": beam_trajectories,
        "beam_executor_rows": beam_executor_rows,
        "retained_teacher_trajectories": retained_trajectories,
        "retained_executor_rows": retained_executor_rows,
    }
    return diagnostic


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
    phase_targets: dict[str, int] | None,
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
        "build_worlds": 0,
        "calibration_worlds": 0,
        "heldout_worlds": 0,
        "heldout_accepted_states": 0,
        "heldout_positive_states": 0,
        "selection_reasons": Counter(),
        "search_errors": Counter(),
        "branch_errors": Counter(),
        "opponents": Counter(),
        "turns": Counter(),
        "phases": Counter(),
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
                            and (
                                not config.heldout_option
                                or any(
                                    int(option["type"]) == 7
                                    for option in selection["option"]
                                )
                            )
                        )
                        turn = int(observation["current"]["turn"])
                        phase = turn_phase(
                            turn,
                            early_max_turn=config.early_max_turn,
                            mid_max_turn=config.mid_max_turn,
                        )
                        phase_open = (
                            phase_targets is None
                            or statistics["phases"][phase]
                            < phase_targets[phase]
                        )
                        if (
                            exact_main
                            and turn >= config.minimum_turn
                            and phase_open
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
                                        maximum=max(
                                            config.branch_width,
                                            config.plan_pool_size,
                                        ),
                                    )

                                common_kwargs = {
                                    "beam_width": config.beam_width,
                                    "branch_width": config.branch_width,
                                    "max_rollout_steps": config.max_rollout_steps,
                                    "value_policy": value_policy,
                                }
                                if config.macro_oracle:
                                    evaluator_type = MacroPlanOracleEvaluator
                                    evaluator_kwargs = {
                                        **common_kwargs,
                                        "determinizations": (
                                            config.determinizations
                                        ),
                                        "max_plan_steps": (
                                            config.max_plan_steps
                                        ),
                                        "plan_generator": MacroPlanGenerator(
                                            cards,
                                            attacks,
                                            maximum_steps=(
                                                config.max_plan_steps
                                            ),
                                        ),
                                        "decision_encoder": encoder.encode,
                                        "plan_pool_size": (
                                            config.plan_pool_size
                                        ),
                                        "alignment_weight": (
                                            config.macro_alignment_weight
                                        ),
                                    }
                                elif config.heldout_option:
                                    evaluator_type = HeldoutCardEffectEvaluator
                                    evaluator_kwargs = {
                                        **common_kwargs,
                                        "build_determinizations": (
                                            config.build_determinizations
                                        ),
                                        "calibration_determinizations": (
                                            config.calibration_determinizations
                                        ),
                                        "heldout_determinizations": (
                                            config.heldout_determinizations
                                        ),
                                        "root_candidate_limit": (
                                            config.plan_pool_size
                                        ),
                                        "selection_risk_multiplier": (
                                            config.option_gate_multiplier
                                        ),
                                        "max_option_steps": (
                                            config.max_option_steps
                                        ),
                                        "minimum_calibration_pairs": (
                                            config.option_min_calibration_pairs
                                        ),
                                        "minimum_closed_loop_coverage": (
                                            config.option_min_coverage
                                        ),
                                        "familywise_alpha": (
                                            config.option_familywise_alpha
                                        ),
                                    }
                                elif config.heldout_semantic:
                                    evaluator_type = HeldoutTurnPlanEvaluator
                                    evaluator_kwargs = {
                                        **common_kwargs,
                                        "max_plan_steps": config.max_plan_steps,
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
                                else:
                                    evaluator_type = TurnSynergyEvaluator
                                    evaluator_kwargs = {
                                        **common_kwargs,
                                        "max_plan_steps": config.max_plan_steps,
                                        "determinizations": config.determinizations,
                                    }
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
                                        f"{type(error).__name__}: {error}"
                                    ] += 1
                                else:
                                    if config.heldout_option:
                                        calibration_pairs = int(
                                            diagnostic[
                                                "calibration_selected"
                                            ]["effective_pairs"]
                                        )
                                        heldout_pairs = int(
                                            diagnostic["heldout_selected"][
                                                "effective_pairs"
                                            ]
                                        )
                                        statistics["build_worlds"] += int(
                                            diagnostic[
                                                "build_determinizations"
                                            ]
                                        )
                                        statistics[
                                            "calibration_worlds"
                                        ] += calibration_pairs
                                        statistics[
                                            "heldout_worlds"
                                        ] += heldout_pairs
                                        statistics[
                                            "heldout_accepted_states"
                                        ] += int(
                                            diagnostic["heldout_accepted"]
                                        )
                                        statistics[
                                            "heldout_positive_states"
                                        ] += int(
                                            float(
                                                diagnostic[
                                                    "deployable_heldout_gain"
                                                ]
                                            )
                                            > 0.0
                                        )
                                    elif config.heldout_semantic:
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
                                    if (
                                        config.compact_macro_oracle
                                        and config.macro_oracle
                                    ):
                                        diagnostic = compact_macro_diagnostic(
                                            diagnostic
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
                                    statistics["phases"][phase] += 1
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
        "phases",
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
    phase_targets = (
        split_phase_quotas(config.phase_quotas, worker_count)
        if config.phase_quotas is not None
        else [None] * worker_count
    )
    if config.phase_quotas is not None:
        targets = [
            sum(worker_targets.values())
            for worker_targets in phase_targets
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
            phase_targets=phase_targets[0],
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
                    phase_targets=phase_targets[worker],
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


def summarize_macro_oracle(path: Path) -> dict[str, Any]:
    """Aggregate macro upper bounds and search-teacher dataset volume."""
    summary = summarize_turn_synergy(path)
    states = 0
    candidate_plans = 0
    beam_trajectories = 0
    beam_executor_rows = 0
    teacher_trajectories = 0
    teacher_executor_rows = 0
    positive_trajectories = 0
    synergy_trajectories = 0
    best_plan_types: Counter[str] = Counter()
    proposed_plan_types: Counter[str] = Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            diagnostic = record["diagnostic"]
            states += 1
            candidate_plans += int(diagnostic["candidate_plans"])
            serialization = diagnostic.get("serialization") or {}
            compact = serialization.get("mode") == "best_per_plan_compact"
            if compact:
                beam_trajectories += int(
                    serialization["beam_trajectories"]
                )
                beam_executor_rows += int(
                    serialization["beam_executor_rows"]
                )
            for plan in diagnostic["plans"]:
                proposed_plan_types[str(plan["plan_type"])] += 1
            for sample in diagnostic["samples"]:
                if "error" in sample:
                    continue
                best_plan_types[
                    str(sample["best_macro"]["plan_type"])
                ] += 1
                for item in sample["plans"]:
                    best = item["best_trajectory"]
                    teacher_trajectories += 1
                    teacher_executor_rows += int(best["decision_count"])
                    positive_trajectories += int(
                        float(best["paired_advantage"]) > 0.0
                    )
                    synergy_trajectories += int(
                        float(best["macro_synergy"]) > 0.0
                    )
                    if not compact:
                        for trajectory in item["trajectories"]:
                            beam_trajectories += 1
                            beam_executor_rows += int(
                                trajectory["decision_count"]
                            )

    def rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 6) if denominator else 0.0

    return {
        **summary,
        "mean_candidate_plans": round(
            candidate_plans / states, 6
        ) if states else 0.0,
        "beam_trajectories": beam_trajectories,
        "beam_executor_rows": beam_executor_rows,
        "teacher_trajectories": teacher_trajectories,
        "estimated_executor_rows": teacher_executor_rows,
        "positive_teacher_trajectory_rate": rate(
            positive_trajectories, teacher_trajectories
        ),
        "synergistic_teacher_trajectory_rate": rate(
            synergy_trajectories, teacher_trajectories
        ),
        "proposed_plan_type_counts": dict(
            sorted(proposed_plan_types.items())
        ),
        "best_plan_type_counts": dict(sorted(best_plan_types.items())),
        "warning": (
            "Each macro candidate receives its own full-turn beam, but the best "
            "continuation is still selected separately per hidden world. This "
            "is an oracle upper bound and search-teacher dataset, not a "
            "deployable macro selector."
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


def summarize_heldout_options(path: Path) -> dict[str, Any]:
    """Aggregate deployable gains and coverage of closed-loop effect policies."""
    states = 0
    gated = 0
    raw_positive = 0
    deployable_positive = 0
    accepted = 0
    calibration_gains: list[float] = []
    heldout_gains: list[float] = []
    deployable_gains: list[float] = []
    gated_heldout_gains: list[float] = []
    coverages: list[float] = []
    compiled_counts: list[float] = []
    fallback_steps = 0
    continuation_steps = 0
    continuation_worlds = 0
    states_with_continuation = 0
    by_opponent: dict[str, Counter] = {}
    by_turn: dict[int, Counter] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            diagnostic = record["diagnostic"]
            calibration = diagnostic.get(
                "calibration_selected",
                diagnostic.get("proposal_selected"),
            )
            if calibration is None:
                raise KeyError("Missing calibration/proposal selected summary")
            heldout = diagnostic["heldout_selected"]
            calibration_gain = float(
                calibration.get("paired_advantage") or 0.0
            )
            heldout_gain = float(heldout.get("paired_advantage") or 0.0)
            coverage = float(heldout["mean_closed_loop_coverage"])
            state_continuation_steps = int(heldout["continuation_steps"])
            state_continuation_worlds = int(heldout["continuation_worlds"])
            is_gated = bool(
                diagnostic.get(
                    "calibration_gate_passed",
                    diagnostic.get("proposal_gate_passed", False),
                )
            )
            deployable_gain = float(
                diagnostic.get(
                    "deployable_heldout_gain",
                    heldout_gain if is_gated else 0.0,
                )
            )
            is_raw_positive = heldout_gain > 0.0
            is_deployable_positive = deployable_gain > 0.0
            is_accepted = bool(diagnostic["heldout_accepted"])
            states += 1
            gated += int(is_gated)
            raw_positive += int(is_raw_positive)
            deployable_positive += int(is_deployable_positive)
            accepted += int(is_accepted)
            calibration_gains.append(calibration_gain)
            heldout_gains.append(heldout_gain)
            deployable_gains.append(deployable_gain)
            if is_gated:
                gated_heldout_gains.append(heldout_gain)
            if state_continuation_worlds:
                coverages.append(coverage)
            compiled_counts.append(float(diagnostic["compiled_policies"]))
            fallback_steps += int(heldout["fallback_steps"])
            continuation_steps += state_continuation_steps
            continuation_worlds += state_continuation_worlds
            states_with_continuation += int(state_continuation_worlds > 0)
            opponent_counts = by_opponent.setdefault(record["opponent"], Counter())
            turn_counts = by_turn.setdefault(int(record["turn"]), Counter())
            for counts in (opponent_counts, turn_counts):
                counts["states"] += 1
                counts["gated"] += int(is_gated)
                counts["raw_positive"] += int(is_raw_positive)
                counts["deployable_positive"] += int(
                    is_deployable_positive
                )
                counts["accepted"] += int(is_accepted)
                counts["raw_gain_sum"] += heldout_gain
                counts["deployable_gain_sum"] += deployable_gain
                if state_continuation_worlds:
                    counts["coverage_sum"] += coverage
                    counts["coverage_states"] += 1

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
                "calibration_gate_rate": rate(
                    counts["gated"], counts["states"]
                ),
                "raw_positive_heldout_rate": rate(
                    counts["raw_positive"], counts["states"]
                ),
                "deployable_positive_rate": rate(
                    counts["deployable_positive"], counts["states"]
                ),
                "accepted_rate": rate(
                    counts["accepted"], counts["states"]
                ),
                "mean_raw_heldout_gain": round(
                    counts["raw_gain_sum"] / max(1, counts["states"]), 6
                ),
                "mean_deployable_heldout_gain": round(
                    counts["deployable_gain_sum"]
                    / max(1, counts["states"]),
                    6,
                ),
                "mean_closed_loop_coverage": round(
                    counts["coverage_sum"]
                    / max(1, counts["coverage_states"]),
                    6,
                ),
            }
            for name, counts in sorted(values.items())
        }

    calibration_optimism = [
        calibration - heldout
        for calibration, heldout in zip(
            calibration_gains, heldout_gains, strict=True
        )
    ]
    return {
        "states": states,
        "calibration_gate_states": gated,
        "calibration_gate_rate": rate(gated, states),
        "raw_positive_heldout_states": raw_positive,
        "raw_positive_heldout_rate": rate(raw_positive, states),
        "deployable_positive_states": deployable_positive,
        "deployable_positive_rate": rate(deployable_positive, states),
        "accepted_heldout_states": accepted,
        "accepted_heldout_rate": rate(accepted, states),
        "mean_calibration_gain": mean(calibration_gains),
        "mean_raw_heldout_gain": mean(heldout_gains),
        "mean_raw_heldout_gain_ci95": mean_ci95(heldout_gains),
        "mean_deployable_heldout_gain": mean(deployable_gains),
        "mean_deployable_heldout_gain_ci95": mean_ci95(
            deployable_gains
        ),
        "mean_gated_heldout_gain": mean(gated_heldout_gains),
        "mean_gated_heldout_gain_ci95": mean_ci95(
            gated_heldout_gains
        ),
        "mean_calibration_optimism_gap": mean(
            calibration_optimism
        ),
        "states_with_continuation": states_with_continuation,
        "continuation_worlds": continuation_worlds,
        "continuation_steps": continuation_steps,
        "mean_closed_loop_coverage": mean(coverages),
        "heldout_fallback_steps": fallback_steps,
        "closed_loop_step_coverage": round(
            1.0 - fallback_steps / max(1, continuation_steps), 6
        ),
        "mean_compiled_policies": mean(compiled_counts),
        "by_opponent": groups(by_opponent),
        "by_turn": groups(by_turn),
        "warning": (
            "Build worlds compile policies, calibration worlds select and gate "
            "them, and held-out worlds only evaluate. Deployable gain is zero "
            "whenever the independent calibration gate rejects an override."
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
    parser.add_argument("--heldout-option", action="store_true")
    parser.add_argument("--macro-oracle", action="store_true")
    parser.add_argument(
        "--compact-macro-oracle",
        action="store_true",
        help=(
            "After full macro search, omit non-best beam trajectories from "
            "JSON without changing search or teacher selection."
        ),
    )
    parser.add_argument("--build-determinizations", type=int, default=8)
    parser.add_argument("--calibration-determinizations", type=int, default=8)
    parser.add_argument("--proposal-determinizations", type=int, default=4)
    parser.add_argument("--heldout-determinizations", type=int, default=4)
    parser.add_argument("--plan-pool-size", type=int, default=16)
    parser.add_argument("--selection-risk-multiplier", type=float, default=1.0)
    parser.add_argument("--option-gate-multiplier", type=float, default=1.96)
    parser.add_argument("--option-min-calibration-pairs", type=int, default=8)
    parser.add_argument("--option-min-coverage", type=float, default=0.9)
    parser.add_argument("--option-familywise-alpha", type=float, default=0.05)
    parser.add_argument("--macro-alignment-weight", type=float, default=0.05)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--branch-width", type=int, default=4)
    parser.add_argument("--max-plan-steps", type=int, default=32)
    parser.add_argument("--max-option-steps", type=int, default=12)
    parser.add_argument("--max-rollout-steps", type=int, default=1_000)
    parser.add_argument("--max-states-per-game", type=int, default=1)
    parser.add_argument("--minimum-turn", type=int, default=3)
    parser.add_argument("--early-states", type=int, default=0)
    parser.add_argument("--mid-states", type=int, default=0)
    parser.add_argument("--late-states", type=int, default=0)
    parser.add_argument("--early-max-turn", type=int, default=4)
    parser.add_argument("--mid-max-turn", type=int, default=7)
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
            args.build_determinizations
            + args.calibration_determinizations
            + args.heldout_determinizations
            if args.heldout_option
            else args.proposal_determinizations
            + args.heldout_determinizations
            if args.heldout_semantic
            else args.determinizations
        ),
        "build-determinizations": args.build_determinizations,
        "calibration-determinizations": args.calibration_determinizations,
        "proposal-determinizations": args.proposal_determinizations,
        "heldout-determinizations": args.heldout_determinizations,
        "plan-pool-size": args.plan_pool_size,
        "beam-width": args.beam_width,
        "branch-width": args.branch_width,
        "max-plan-steps": args.max_plan_steps,
        "max-option-steps": args.max_option_steps,
        "option-min-calibration-pairs": args.option_min_calibration_pairs,
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
    if args.option_gate_multiplier < 0:
        raise SystemExit("--option-gate-multiplier must be non-negative")
    if args.macro_alignment_weight < 0:
        raise SystemExit("--macro-alignment-weight must be non-negative")
    if not 0.0 <= args.option_min_coverage <= 1.0:
        raise SystemExit("--option-min-coverage must be in [0, 1]")
    if not 0.0 < args.option_familywise_alpha < 1.0:
        raise SystemExit("--option-familywise-alpha must be in (0, 1)")
    if args.early_max_turn >= args.mid_max_turn:
        raise SystemExit("--early-max-turn must be smaller than --mid-max-turn")
    raw_phase_quotas = (
        args.early_states,
        args.mid_states,
        args.late_states,
    )
    if any(value < 0 for value in raw_phase_quotas):
        raise SystemExit("phase state quotas must be non-negative")
    phase_quotas = (
        raw_phase_quotas if any(raw_phase_quotas) else None
    )
    if phase_quotas is not None and sum(phase_quotas) != args.target_states:
        raise SystemExit(
            "--early-states + --mid-states + --late-states must equal "
            "--target-states"
        )
    if (
        phase_quotas is not None
        and phase_quotas[0] > 0
        and args.minimum_turn > args.early_max_turn
    ):
        raise SystemExit("early quota is unreachable above --minimum-turn")
    if (
        phase_quotas is not None
        and phase_quotas[1] > 0
        and args.minimum_turn > args.mid_max_turn
    ):
        raise SystemExit("mid quota is unreachable above --minimum-turn")
    diagnostic_modes = sum(
        (
            args.heldout_semantic,
            args.heldout_option,
            args.macro_oracle,
        )
    )
    if diagnostic_modes > 1:
        raise SystemExit(
            "--heldout-semantic, --heldout-option, and --macro-oracle are "
            "mutually exclusive"
        )
    if args.compact_macro_oracle and not args.macro_oracle:
        raise SystemExit("--compact-macro-oracle requires --macro-oracle")
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
        heldout_option=args.heldout_option,
        macro_oracle=args.macro_oracle,
        build_determinizations=args.build_determinizations,
        calibration_determinizations=args.calibration_determinizations,
        proposal_determinizations=args.proposal_determinizations,
        heldout_determinizations=args.heldout_determinizations,
        plan_pool_size=args.plan_pool_size,
        selection_risk_multiplier=args.selection_risk_multiplier,
        max_option_steps=args.max_option_steps,
        option_gate_multiplier=args.option_gate_multiplier,
        option_min_calibration_pairs=args.option_min_calibration_pairs,
        option_min_coverage=args.option_min_coverage,
        option_familywise_alpha=args.option_familywise_alpha,
        macro_alignment_weight=args.macro_alignment_weight,
        compact_macro_oracle=args.compact_macro_oracle,
        phase_quotas=phase_quotas,
        early_max_turn=args.early_max_turn,
        mid_max_turn=args.mid_max_turn,
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
        "phases",
    ):
        raw[key] = dict(sorted(raw[key].items()))
    requested_phases = (
        dict(zip(TURN_PHASES, phase_quotas, strict=True))
        if phase_quotas is not None
        else None
    )
    phase_shortfalls = (
        {
            phase: max(
                0,
                requested - int(raw["phases"].get(phase, 0)),
            )
            for phase, requested in requested_phases.items()
        }
        if requested_phases is not None
        else None
    )
    summary = {
        **raw,
        "diagnostic_mode": (
            "macro_oracle"
            if args.macro_oracle
            else "heldout_option"
            if args.heldout_option
            else "heldout_semantic"
            if args.heldout_semantic
            else "oracle"
        ),
        "target_states": args.target_states,
        "determinizations": (
            args.build_determinizations
            + args.calibration_determinizations
            + args.heldout_determinizations
            if args.heldout_option
            else args.proposal_determinizations
            + args.heldout_determinizations
            if args.heldout_semantic
            else args.determinizations
        ),
        "build_determinizations": args.build_determinizations,
        "calibration_determinizations": args.calibration_determinizations,
        "proposal_determinizations": args.proposal_determinizations,
        "heldout_determinizations": args.heldout_determinizations,
        "plan_pool_size": args.plan_pool_size,
        "beam_width": args.beam_width,
        "branch_width": args.branch_width,
        "max_option_steps": args.max_option_steps,
        "option_gate_multiplier": args.option_gate_multiplier,
        "option_min_calibration_pairs": args.option_min_calibration_pairs,
        "option_min_coverage": args.option_min_coverage,
        "option_familywise_alpha": args.option_familywise_alpha,
        "macro_alignment_weight": args.macro_alignment_weight,
        "minimum_turn": args.minimum_turn,
        "compact_macro_oracle": args.compact_macro_oracle,
        "turn_phase_boundaries": {
            "early": [args.minimum_turn, args.early_max_turn],
            "mid": [args.early_max_turn + 1, args.mid_max_turn],
            "late": [args.mid_max_turn + 1, None],
        },
        "requested_phase_states": requested_phases,
        "phase_shortfalls": phase_shortfalls,
        "seed": args.seed,
        "workers": min(args.workers, args.target_states),
        "elapsed_seconds": round(elapsed, 3),
        "states_per_second": round(raw["states"] / elapsed, 6),
        "diagnostics": (
            summarize_macro_oracle(output)
            if args.macro_oracle
            else summarize_heldout_options(output)
            if args.heldout_option
            else summarize_heldout_turn_plans(output)
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
