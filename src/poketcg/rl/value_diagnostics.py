"""On-policy value calibration and trajectory diagnostics in the official simulator."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Categorical

from poketcg.agents import RandomAgent, RuleAgent
from poketcg.agents.rule_agent import SelectContext
from poketcg.engine import OfficialEngine
from poketcg.match import play_match

from .data import BCExample, collate_bc
from .features import build_feature_encoder
from .model import build_model, encoder_version
from .train_bc import resolve_device


@dataclass(slots=True)
class ValueRecord:
    game: int
    player: int
    decision: int
    turn: int
    turn_action_count: int
    context: int
    context_name: str
    option_count: int
    chosen_action: int
    action_probability: float
    policy_entropy: float
    predicted_return: float
    outcome: float
    own_prizes_remaining: int
    opponent_prizes_remaining: int
    phase: str


def _context_name(context: int) -> str:
    try:
        return SelectContext(context).name
    except ValueError:
        return f"CONTEXT_{context}"


def prize_phase(own_remaining: int, opponent_remaining: int) -> str:
    """Bucket a state by the leading player's prize progress."""
    if own_remaining == 0 and opponent_remaining == 0:
        return "setup"
    progress = max(6 - own_remaining, 6 - opponent_remaining)
    if progress <= 1:
        return "early"
    if progress <= 3:
        return "mid"
    return "late"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        raise ValueError("correlation inputs must have equal length")
    if not first:
        return 0.0
    first_mean = _mean(first)
    second_mean = _mean(second)
    covariance = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first, second, strict=True)
    )
    first_variance = sum((value - first_mean) ** 2 for value in first)
    second_variance = sum((value - second_mean) ** 2 for value in second)
    denominator = math.sqrt(first_variance * second_variance)
    return covariance / denominator if denominator > 0.0 else 0.0


def calibration_bins(records: list[ValueRecord], bins: int) -> list[dict[str, Any]]:
    if bins < 2:
        raise ValueError("bins must be at least two")
    grouped: list[list[ValueRecord]] = [[] for _ in range(bins)]
    for record in records:
        normalized = min(max((record.predicted_return + 1.0) / 2.0, 0.0), 1.0)
        index = min(int(normalized * bins), bins - 1)
        grouped[index].append(record)

    result = []
    for index, group in enumerate(grouped):
        lower = -1.0 + 2.0 * index / bins
        upper = -1.0 + 2.0 * (index + 1) / bins
        predictions = [record.predicted_return for record in group]
        outcomes = [record.outcome for record in group]
        predicted = _mean(predictions)
        observed = _mean(outcomes)
        result.append(
            {
                "lower": round(lower, 6),
                "upper": round(upper, 6),
                "count": len(group),
                "mean_predicted_return": round(predicted, 6),
                "observed_mean_return": round(observed, 6),
                "calibration_gap": round(predicted - observed, 6),
                "observed_win_rate": round(
                    sum(record.outcome > 0.0 for record in group) / len(group), 6
                )
                if group
                else 0.0,
                "observed_draw_rate": round(
                    sum(record.outcome == 0.0 for record in group) / len(group), 6
                )
                if group
                else 0.0,
            }
        )
    return result


def summarize_calibration(records: list[ValueRecord], *, bins: int) -> dict[str, Any]:
    if not records:
        return {
            "decisions": 0,
            "games": 0,
            "mean_predicted_return": 0.0,
            "observed_mean_return": 0.0,
            "bias": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "brier_score": 0.0,
            "pearson_correlation": 0.0,
            "explained_variance": 0.0,
            "calibration_intercept": 0.0,
            "calibration_slope": 0.0,
            "expected_calibration_error": 0.0,
            "bins": calibration_bins([], bins),
        }

    predictions = [record.predicted_return for record in records]
    outcomes = [record.outcome for record in records]
    errors = [
        prediction - outcome
        for prediction, outcome in zip(predictions, outcomes, strict=True)
    ]
    residuals = [-error for error in errors]
    prediction_mean = _mean(predictions)
    outcome_mean = _mean(outcomes)
    outcome_variance = _mean([(outcome - outcome_mean) ** 2 for outcome in outcomes])
    residual_mean = _mean(residuals)
    residual_variance = _mean(
        [(residual - residual_mean) ** 2 for residual in residuals]
    )
    prediction_variance_sum = sum(
        (prediction - prediction_mean) ** 2 for prediction in predictions
    )
    covariance_sum = sum(
        (prediction - prediction_mean) * (outcome - outcome_mean)
        for prediction, outcome in zip(predictions, outcomes, strict=True)
    )
    slope = covariance_sum / prediction_variance_sum if prediction_variance_sum > 0.0 else 0.0
    intercept = outcome_mean - slope * prediction_mean
    bin_values = calibration_bins(records, bins)
    calibration_error = sum(
        item["count"] * abs(item["calibration_gap"]) for item in bin_values
    ) / len(records)

    return {
        "decisions": len(records),
        "games": len({(record.player, record.game) for record in records}),
        "mean_predicted_return": round(prediction_mean, 6),
        "observed_mean_return": round(outcome_mean, 6),
        "bias": round(_mean(errors), 6),
        "mae": round(_mean([abs(error) for error in errors]), 6),
        "rmse": round(math.sqrt(_mean([error**2 for error in errors])), 6),
        "brier_score": round(
            _mean(
                [
                    ((prediction + 1.0) / 2.0 - (outcome + 1.0) / 2.0) ** 2
                    for prediction, outcome in zip(predictions, outcomes, strict=True)
                ]
            ),
            6,
        ),
        "pearson_correlation": round(_pearson(predictions, outcomes), 6),
        "explained_variance": round(
            1.0 - residual_variance / outcome_variance if outcome_variance > 0.0 else 0.0,
            6,
        ),
        "calibration_intercept": round(intercept, 6),
        "calibration_slope": round(slope, 6),
        "expected_calibration_error": round(calibration_error, 6),
        "bins": bin_values,
    }


def summarize_trajectories(records: list[ValueRecord]) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[ValueRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.player, record.game)].append(record)

    games = []
    for (player, game), trajectory in sorted(grouped.items()):
        trajectory.sort(key=lambda record: record.decision)
        values = [record.predicted_return for record in trajectory]
        deltas = [right - left for left, right in zip(values, values[1:], strict=False)]
        outcome = trajectory[0].outcome
        games.append(
            {
                "game": game,
                "player": player,
                "outcome": outcome,
                "decisions": len(trajectory),
                "initial_value": values[0],
                "final_value": values[-1],
                "net_value_change": values[-1] - values[0],
                "net_change_toward_outcome": outcome * (values[-1] - values[0]),
                "mean_absolute_step_change": _mean([abs(delta) for delta in deltas]),
                "sign_flips": sum(
                    left * right < 0.0
                    for left, right in zip(values, values[1:], strict=False)
                ),
            }
        )

    def game_mean(key: str) -> float:
        return _mean([float(game[key]) for game in games])

    by_outcome = {}
    for label, outcome in (("loss", -1.0), ("draw", 0.0), ("win", 1.0)):
        selected = [game for game in games if game["outcome"] == outcome]
        by_outcome[label] = {
            "games": len(selected),
            "mean_initial_value": round(
                _mean([float(game["initial_value"]) for game in selected]), 6
            ),
            "mean_final_value": round(
                _mean([float(game["final_value"]) for game in selected]), 6
            ),
            "mean_net_change_toward_outcome": round(
                _mean([float(game["net_change_toward_outcome"]) for game in selected]), 6
            ),
        }
    return {
        "games": len(games),
        "mean_initial_value": round(game_mean("initial_value"), 6),
        "mean_final_value": round(game_mean("final_value"), 6),
        "mean_net_change_toward_outcome": round(game_mean("net_change_toward_outcome"), 6),
        "mean_absolute_step_change": round(game_mean("mean_absolute_step_change"), 6),
        "mean_sign_flips": round(game_mean("sign_flips"), 6),
        "by_outcome": by_outcome,
        "per_game": games,
    }


def trajectory_endpoints(
    records: list[ValueRecord],
) -> tuple[list[ValueRecord], list[ValueRecord]]:
    grouped: dict[tuple[int, int], list[ValueRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.player, record.game)].append(record)
    initial = []
    final = []
    for trajectory in grouped.values():
        trajectory.sort(key=lambda record: record.decision)
        initial.append(trajectory[0])
        final.append(trajectory[-1])
    return initial, final


def high_confidence_errors(
    records: list[ValueRecord], *, threshold: float, limit: int
) -> list[dict[str, Any]]:
    selected = [
        record
        for record in records
        if abs(record.predicted_return) >= threshold
        and record.predicted_return * record.outcome < 0.0
    ]
    selected.sort(
        key=lambda record: abs(record.predicted_return - record.outcome), reverse=True
    )
    return [asdict(record) for record in selected[:limit]]


class DiagnosticPolicyAgent:
    name = "diagnostic-policy"

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        card_catalog: dict[int, object],
        attack_catalog: dict[int, object],
        seed: int,
        device: torch.device,
        deterministic: bool,
    ) -> None:
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        self._model = build_model(saved["model_config"]).to(device)
        self._model.load_state_dict(saved["model_state_dict"])
        self._model.eval()
        self._encoder = build_feature_encoder(
            encoder_version(saved["model_config"]),
            card_catalog,
            attack_catalog,
        )
        self._fallback = RuleAgent(
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            seed=seed,
        )
        self._device = device
        self._deterministic = deterministic
        self._rng = random.Random(seed)
        self._game = 0
        self._decision = 0
        self._game_start = 0
        self.records: list[ValueRecord] = []

    def start_game(self, game: int) -> None:
        self._game = game
        self._decision = 0
        self._game_start = len(self.records)

    def finish_game(self, outcome: float) -> None:
        for record in self.records[self._game_start :]:
            record.outcome = outcome

    def choose_action(self, observation: dict) -> list[int]:
        selection = observation.get("select")
        if selection is None:
            raise ValueError("DiagnosticPolicyAgent received a deck-selection observation")
        learnable = (
            len(selection["option"]) > 1
            and int(selection["minCount"]) == 1
            and int(selection["maxCount"]) == 1
        )
        if not learnable:
            return self._fallback.choose_action(observation)

        state = observation["current"]
        player = int(state["yourIndex"])
        decision = self._encoder.encode(observation)
        example = BCExample(decision, action=0, value_target=0.0, player=player, game=self._game)
        batch = {
            key: value.to(self._device) for key, value in collate_bc([example]).items()
        }
        with torch.no_grad():
            policy_logits, value_logits = self._model(batch)
            distribution = Categorical(logits=policy_logits)
            predicted_return = float(self._model.expected_value(value_logits).item())
        probabilities = policy_logits.softmax(dim=-1).squeeze(0).cpu().tolist()
        if self._deterministic:
            chosen_action = max(range(len(probabilities)), key=probabilities.__getitem__)
        else:
            chosen_action = self._rng.choices(
                range(len(probabilities)), weights=probabilities, k=1
            )[0]
        own_prizes = len(state["players"][player]["prize"])
        opponent_prizes = len(state["players"][1 - player]["prize"])
        self.records.append(
            ValueRecord(
                game=self._game,
                player=player,
                decision=self._decision,
                turn=int(state["turn"]),
                turn_action_count=int(state["turnActionCount"]),
                context=int(selection["context"]),
                context_name=_context_name(int(selection["context"])),
                option_count=len(selection["option"]),
                chosen_action=chosen_action,
                action_probability=round(probabilities[chosen_action], 6),
                policy_entropy=round(float(distribution.entropy().item()), 6),
                predicted_return=round(predicted_return, 6),
                outcome=0.0,
                own_prizes_remaining=own_prizes,
                opponent_prizes_remaining=opponent_prizes,
                phase=prize_phase(own_prizes, opponent_prizes),
            )
        )
        self._decision += 1
        return [chosen_action]


def _baseline_agent(
    opponent: str,
    *,
    seed: int,
    card_catalog: dict[int, object],
    attack_catalog: dict[int, object],
):
    if opponent == "random":
        return RandomAgent(seed)
    return RuleAgent(
        card_catalog=card_catalog,
        attack_catalog=attack_catalog,
        seed=seed,
    )


def _group_summaries(
    records: list[ValueRecord], key, *, bins: int, include_bins: bool = True
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[ValueRecord]] = defaultdict(list)
    for record in records:
        groups[str(key(record))].append(record)
    result = {}
    for name, group in sorted(groups.items()):
        summary = summarize_calibration(group, bins=bins)
        if not include_bins:
            summary.pop("bins")
        result[name] = summary
    return result


def diagnose_value_trajectories(
    checkpoint: str | Path,
    *,
    opponent: str,
    games_per_seat: int,
    seed: int,
    bins: int,
    confidence_threshold: float,
    error_limit: int,
    official_dir: str | Path | None = None,
    deck_path: str | Path | None = None,
    device_name: str = "cpu",
    stochastic: bool = False,
) -> tuple[dict[str, Any], list[ValueRecord]]:
    if games_per_seat < 1:
        raise ValueError("games_per_seat must be positive")
    if opponent not in {"rule", "random"}:
        raise ValueError("opponent must be 'rule' or 'random'")
    if bins < 2:
        raise ValueError("bins must be at least two")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if error_limit < 0:
        raise ValueError("error_limit must be non-negative")
    device = resolve_device(device_name)
    engine = OfficialEngine(official_dir)
    deck = engine.load_deck(deck_path or engine.sample_deck_path)
    card_catalog = engine.card_catalog()
    attack_catalog = engine.attack_catalog()
    records: list[ValueRecord] = []
    outcomes: list[float] = []

    for model_player in (0, 1):
        pairing_seed = seed + model_player * 10_000
        model_agent = DiagnosticPolicyAgent(
            checkpoint,
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            seed=pairing_seed,
            device=device,
            deterministic=not stochastic,
        )
        baseline = _baseline_agent(
            opponent,
            seed=pairing_seed + 1,
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
        )
        for game in range(games_per_seat):
            model_agent.start_game(game)
            agents = (model_agent, baseline) if model_player == 0 else (baseline, model_agent)
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
            outcome = (
                0.0
                if result.winner == 2
                else (1.0 if result.winner == model_player else -1.0)
            )
            model_agent.finish_game(outcome)
            outcomes.append(outcome)
        records.extend(model_agent.records)

    trajectory_summary = summarize_trajectories(records)
    initial_records, final_records = trajectory_endpoints(records)
    confidence_errors = high_confidence_errors(
        records, threshold=confidence_threshold, limit=error_limit
    )
    confidence_error_count = sum(
        abs(record.predicted_return) >= confidence_threshold
        and record.predicted_return * record.outcome < 0.0
        for record in records
    )
    result = {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "opponent": opponent,
        "device": str(device),
        "action_selection": "stochastic" if stochastic else "deterministic",
        "games_per_seat": games_per_seat,
        "games": len(outcomes),
        "wins": sum(outcome > 0.0 for outcome in outcomes),
        "draws": sum(outcome == 0.0 for outcome in outcomes),
        "losses": sum(outcome < 0.0 for outcome in outcomes),
        "win_rate": round(sum(outcome > 0.0 for outcome in outcomes) / len(outcomes), 6),
        "decisions": len(records),
        "overall": summarize_calibration(records, bins=bins),
        "by_player": _group_summaries(records, lambda record: record.player, bins=bins),
        "by_phase": _group_summaries(records, lambda record: record.phase, bins=bins),
        "by_context": _group_summaries(
            records,
            lambda record: record.context_name,
            bins=bins,
            include_bins=False,
        ),
        "endpoint_calibration": {
            "initial_decision": summarize_calibration(initial_records, bins=bins),
            "final_decision": summarize_calibration(final_records, bins=bins),
        },
        "trajectory": {
            key: value
            for key, value in trajectory_summary.items()
            if key != "per_game"
        },
        "per_game": trajectory_summary["per_game"],
        "high_confidence_error_count": confidence_error_count,
        "high_confidence_errors": confidence_errors,
    }
    return result, records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure on-policy value calibration over complete official-engine games."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opponent", choices=("rule", "random"), default="rule")
    parser.add_argument("--games-per-seat", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20_260_718)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--error-limit", type=int, default=50)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--deck", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trajectory-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, records = diagnose_value_trajectories(
        args.checkpoint,
        opponent=args.opponent,
        games_per_seat=args.games_per_seat,
        seed=args.seed,
        bins=args.bins,
        confidence_threshold=args.confidence_threshold,
        error_limit=args.error_limit,
        official_dir=args.official_dir,
        deck_path=args.deck,
        device_name=args.device,
        stochastic=args.stochastic,
    )
    if args.trajectory_output:
        trajectory_path = args.trajectory_output
    elif args.output:
        trajectory_path = args.output.with_name(f"{args.output.stem}_trajectories.jsonl")
    else:
        trajectory_path = None
    if trajectory_path:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        with trajectory_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")
        result["trajectory_output"] = str(trajectory_path.expanduser().resolve())
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        compact = {
            key: result[key]
            for key in (
                "checkpoint",
                "opponent",
                "device",
                "action_selection",
                "games_per_seat",
                "games",
                "wins",
                "draws",
                "losses",
                "win_rate",
                "decisions",
            )
        }
        compact["overall"] = {
            key: value for key, value in result["overall"].items() if key != "bins"
        }
        compact["by_player"] = {
            player: {key: value for key, value in summary.items() if key != "bins"}
            for player, summary in result["by_player"].items()
        }
        compact["by_phase"] = {
            phase: {key: value for key, value in summary.items() if key != "bins"}
            for phase, summary in result["by_phase"].items()
        }
        compact["trajectory"] = result["trajectory"]
        compact["high_confidence_error_count"] = result["high_confidence_error_count"]
        compact["output"] = str(args.output.expanduser().resolve())
        compact["trajectory_output"] = result.get("trajectory_output")
        print(json.dumps(compact, indent=2))
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
