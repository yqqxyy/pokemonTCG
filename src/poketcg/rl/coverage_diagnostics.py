"""Measure neural-policy coverage and currently unused public history signals."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from poketcg.agents import BCPolicyAgent, RandomAgent, RuleAgent
from poketcg.engine import OfficialEngine
from poketcg.match import play_match

from .train_bc import resolve_device


@dataclass(frozen=True, slots=True)
class DecisionShape:
    classification: str
    option_count: int
    minimum: int
    maximum: int
    valid_action_count: int


@dataclass(slots=True)
class CoverageRecord:
    game: int
    player: int
    context: int
    context_name: str
    select_type: int
    select_type_name: str
    classification: str
    option_count: int
    minimum: int
    maximum: int
    valid_action_count: int
    log_types: list[int]
    outcome: float = 0.0


def classify_selection(selection: dict) -> DecisionShape:
    """Classify a legal selection by whether the current neural policy controls it."""
    option_count = len(selection["option"])
    minimum = int(selection["minCount"])
    maximum = int(selection["maxCount"])
    if not 0 <= minimum <= maximum <= option_count:
        raise ValueError("selection bounds are inconsistent with its option count")
    valid_action_count = sum(
        math.comb(option_count, count) for count in range(minimum, maximum + 1)
    )
    if valid_action_count == 1:
        classification = "forced"
    elif option_count > 1 and minimum == maximum == 1:
        classification = "neural"
    else:
        classification = "resolver"
    return DecisionShape(
        classification=classification,
        option_count=option_count,
        minimum=minimum,
        maximum=maximum,
        valid_action_count=valid_action_count,
    )


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def summarize_coverage(records: list[CoverageRecord]) -> dict[str, Any]:
    counts = Counter(record.classification for record in records)
    strategic = counts["neural"] + counts["resolver"]
    game_outcomes: dict[tuple[int, int], float] = {}
    for record in records:
        game_outcomes[(record.player, record.game)] = record.outcome
    log_events = sum(len(record.log_types) for record in records)
    return {
        "decisions": len(records),
        "games_observed": len(game_outcomes),
        "forced_decisions": counts["forced"],
        "neural_decisions": counts["neural"],
        "resolver_decisions": counts["resolver"],
        "strategic_decisions": strategic,
        "raw_neural_coverage": round(_safe_rate(counts["neural"], len(records)), 6),
        "strategic_neural_coverage": round(
            _safe_rate(counts["neural"], strategic), 6
        ),
        "strategic_resolver_coverage": round(
            _safe_rate(counts["resolver"], strategic), 6
        ),
        "mean_option_count": round(
            _safe_rate(sum(record.option_count for record in records), len(records)), 6
        ),
        "mean_log10_valid_actions": round(
            _safe_rate(
                sum(math.log10(max(record.valid_action_count, 1)) for record in records),
                len(records),
            ),
            6,
        ),
        "max_valid_action_count": max(
            (record.valid_action_count for record in records), default=0
        ),
        "decisions_with_logs": sum(bool(record.log_types) for record in records),
        "decisions_with_logs_rate": round(
            _safe_rate(sum(bool(record.log_types) for record in records), len(records)),
            6,
        ),
        "log_events": log_events,
        "mean_logs_per_decision": round(_safe_rate(log_events, len(records)), 6),
        "observed_game_win_rate": round(
            _safe_rate(
                sum(outcome > 0.0 for outcome in game_outcomes.values()),
                len(game_outcomes),
            ),
            6,
        ),
    }


def _group_summaries(records: list[CoverageRecord], key) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[CoverageRecord]] = defaultdict(list)
    for record in records:
        grouped[str(key(record))].append(record)
    ordered = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    return {name: summarize_coverage(group) for name, group in ordered}


class CoverageAgent:
    name = "coverage-policy"

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        card_catalog: dict[int, object],
        attack_catalog: dict[int, object],
        seed: int,
        device: str,
        deterministic: bool,
        context_names: dict[int, str],
        select_type_names: dict[int, str],
    ) -> None:
        self._policy = BCPolicyAgent(
            checkpoint,
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            seed=seed,
            device=device,
            deterministic=deterministic,
        )
        self._context_names = context_names
        self._select_type_names = select_type_names
        self._game = 0
        self._game_start = 0
        self.records: list[CoverageRecord] = []

    def start_game(self, game: int) -> None:
        self._game = game
        self._game_start = len(self.records)

    def finish_game(self, outcome: float) -> None:
        for record in self.records[self._game_start :]:
            record.outcome = outcome

    def choose_action(self, observation: dict) -> list[int]:
        selection = observation.get("select")
        state = observation.get("current")
        if selection is None or state is None:
            raise ValueError("CoverageAgent requires a decision observation")
        shape = classify_selection(selection)
        context = int(selection["context"])
        select_type = int(selection["type"])
        self.records.append(
            CoverageRecord(
                game=self._game,
                player=int(state["yourIndex"]),
                context=context,
                context_name=self._context_names.get(context, f"CONTEXT_{context}"),
                select_type=select_type,
                select_type_name=self._select_type_names.get(
                    select_type, f"SELECT_TYPE_{select_type}"
                ),
                classification=shape.classification,
                option_count=shape.option_count,
                minimum=shape.minimum,
                maximum=shape.maximum,
                valid_action_count=shape.valid_action_count,
                log_types=[int(log["type"]) for log in observation.get("logs") or []],
            )
        )
        return self._policy.choose_action(observation)


def _enum_names(enum_type) -> dict[int, str]:
    return {int(item): item.name for item in enum_type}


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


def _catalog_audit(
    checkpoint: str | Path,
    deck: list[int],
    card_catalog: dict[int, object],
    attack_catalog: dict[int, object],
) -> dict[str, Any]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["model_config"]
    card_vocab_size = int(config.get("card_vocab_size", 0))
    attack_vocab_size = int(config.get("attack_vocab_size", 0))
    card_ids = sorted(card_catalog)
    attack_ids = sorted(attack_catalog)
    deck_counts = Counter(deck)
    cards_with_skill_text = sum(
        any(bool(getattr(skill, "text", "")) for skill in getattr(card, "skills", []))
        for card in card_catalog.values()
    )
    attacks_with_text = sum(
        bool(getattr(attack, "text", "")) for attack in attack_catalog.values()
    )
    return {
        "model_type": config.get("model_type", "mlp_v1"),
        "card_catalog": {
            "count": len(card_ids),
            "minimum_id": min(card_ids),
            "maximum_id": max(card_ids),
            "embedding_vocab_size": card_vocab_size,
            "ids_outside_embedding": sum(
                card_id < 0 or card_id >= card_vocab_size for card_id in card_ids
            )
            if card_vocab_size
            else None,
            "cards_with_skill_text": cards_with_skill_text,
        },
        "attack_catalog": {
            "count": len(attack_ids),
            "minimum_id": min(attack_ids),
            "maximum_id": max(attack_ids),
            "embedding_vocab_size": attack_vocab_size,
            "ids_outside_embedding": sum(
                attack_id < 0 or attack_id >= attack_vocab_size for attack_id in attack_ids
            )
            if attack_vocab_size
            else None,
            "attacks_with_text": attacks_with_text,
        },
        "deck": {
            "cards": len(deck),
            "unique_cards": len(deck_counts),
            "card_counts": {str(card_id): count for card_id, count in sorted(deck_counts.items())},
        },
    }


def run_coverage_diagnostics(
    checkpoint: str | Path,
    *,
    opponent: str,
    games_per_seat: int,
    seed: int,
    official_dir: str | Path | None = None,
    deck_path: str | Path | None = None,
    device_name: str = "cpu",
    stochastic: bool = False,
) -> tuple[dict[str, Any], list[CoverageRecord]]:
    if games_per_seat < 1:
        raise ValueError("games_per_seat must be positive")
    if opponent not in {"rule", "random"}:
        raise ValueError("opponent must be 'rule' or 'random'")
    device = resolve_device(device_name)
    engine = OfficialEngine(official_dir)
    api = importlib.import_module("cg.api")
    context_names = _enum_names(api.SelectContext)
    select_type_names = _enum_names(api.SelectType)
    log_type_names = _enum_names(api.LogType)
    deck = engine.load_deck(deck_path or engine.sample_deck_path)
    card_catalog = engine.card_catalog()
    attack_catalog = engine.attack_catalog()
    records: list[CoverageRecord] = []
    outcomes: list[float] = []

    for model_player in (0, 1):
        pairing_seed = seed + model_player * 10_000
        model_agent = CoverageAgent(
            checkpoint,
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            seed=pairing_seed,
            device=str(device),
            deterministic=not stochastic,
            context_names=context_names,
            select_type_names=select_type_names,
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

    log_type_counts = Counter(log_type for record in records for log_type in record.log_types)
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
        "win_rate": round(_safe_rate(sum(outcome > 0.0 for outcome in outcomes), len(outcomes)), 6),
        "overall": summarize_coverage(records),
        "by_player": _group_summaries(records, lambda record: record.player),
        "by_context": _group_summaries(
            records, lambda record: f"{record.context}:{record.context_name}"
        ),
        "by_select_type": _group_summaries(
            records, lambda record: f"{record.select_type}:{record.select_type_name}"
        ),
        "history": {
            "log_type_counts": {
                f"{log_type}:{log_type_names.get(log_type, f'LOG_{log_type}')}": count
                for log_type, count in sorted(log_type_counts.items())
            }
        },
        "catalog": _catalog_audit(checkpoint, deck, card_catalog, attack_catalog),
    }
    return result, records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit neural decision coverage, public logs, IDs, and the active deck."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opponent", choices=("rule", "random"), default="rule")
    parser.add_argument("--games-per-seat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20_260_719)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--deck", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--records-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, records = run_coverage_diagnostics(
        args.checkpoint,
        opponent=args.opponent,
        games_per_seat=args.games_per_seat,
        seed=args.seed,
        official_dir=args.official_dir,
        deck_path=args.deck,
        device_name=args.device,
        stochastic=args.stochastic,
    )
    if args.records_output:
        records_path = args.records_output
    elif args.output:
        records_path = args.output.with_name(f"{args.output.stem}_records.jsonl")
    else:
        records_path = None
    if records_path:
        records_path.parent.mkdir(parents=True, exist_ok=True)
        with records_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")
        result["records_output"] = str(records_path.expanduser().resolve())
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
