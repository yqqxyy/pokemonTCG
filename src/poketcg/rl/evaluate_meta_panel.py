"""Multi-deck, multi-agent fixed panel for policy and MCTS comparisons."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from poketcg.agents import BCPolicyAgent, ExternalPythonAgent, RandomAgent, RuleAgent
from poketcg.engine import OfficialEngine
from poketcg.match import MatchResult, play_match
from poketcg.mcts import (
    DeckDeterminizer,
    DeckHypothesis,
    MCTSConfig,
    OpponentDeckBelief,
    PolicyValueMCTSAgent,
)
from poketcg.paths import PROJECT_ROOT, resolve_official_dir
from poketcg.rl.evaluate_panel import wilson_interval


@dataclass(frozen=True, slots=True)
class DeckSpec:
    """Named deck used by one panel axis or by the MCTS belief."""

    name: str
    path: str


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Serializable agent definition for a panel worker."""

    name: str
    kind: str
    checkpoint: str | None = None
    source: str | None = None
    deck_path: str | None = None


@dataclass(frozen=True, slots=True)
class PanelTask:
    """One candidate/opponent/deck cell, evaluated in both seats."""

    candidate: AgentSpec
    opponent: AgentSpec
    model_deck: DeckSpec
    opponent_deck: DeckSpec
    belief_decks: tuple[DeckSpec, ...]
    games_per_seat: int
    seed: int
    official_dir: str
    stochastic: bool
    torch_threads: int
    mcts_prior: str
    mcts_config: MCTSConfig


def _named_path(specification: str, option: str) -> tuple[str, Path]:
    try:
        name, raw_path = specification.split("=", 1)
    except ValueError as error:
        raise ValueError(f"{option} must use NAME=PATH") from error
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise ValueError(f"{option} must use non-empty NAME=PATH")
    return name, Path(raw_path).expanduser().resolve()


def _external_opponent_spec(specification: str) -> AgentSpec:
    try:
        name, raw_paths = specification.split("=", 1)
        raw_source, raw_deck = raw_paths.rsplit(",", 1)
    except ValueError as error:
        raise ValueError(
            "--external-opponent must use NAME=SOURCE,DECK"
        ) from error
    name = name.strip()
    source = Path(raw_source.strip()).expanduser().resolve()
    deck = Path(raw_deck.strip()).expanduser().resolve()
    if not name or not raw_source.strip() or not raw_deck.strip():
        raise ValueError("--external-opponent must use non-empty NAME=SOURCE,DECK")
    if not source.is_file():
        raise FileNotFoundError(f"External agent source not found: {source}")
    if not deck.is_file():
        raise FileNotFoundError(f"External agent deck not found: {deck}")
    return AgentSpec(name, "external", source=str(source), deck_path=str(deck))


def _summary(results: list[tuple[MatchResult, int]]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot summarize an empty result set")
    wins = sum(result.winner == player for result, player in results)
    draws = sum(result.winner == 2 for result, _ in results)
    games = len(results)
    total_turns = sum(result.turns for result, _ in results)
    total_decisions = sum(result.decisions for result, _ in results)
    total_elapsed_ms = sum(result.elapsed_ms for result, _ in results)
    low, high = wilson_interval(wins, games)
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": games - wins - draws,
        "win_rate": round(wins / games, 6),
        "win_rate_ci95": [round(low, 6), round(high, 6)],
        "total_turns": total_turns,
        "mean_turns": round(total_turns / games, 3),
        "total_decisions": total_decisions,
        "mean_decisions": round(total_decisions / games, 3),
        "total_elapsed_ms": round(total_elapsed_ms, 3),
        "mean_elapsed_ms": round(total_elapsed_ms / games, 3),
    }


def _combine_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("Cannot combine an empty summary list")
    games = sum(int(item["games"]) for item in summaries)
    wins = sum(int(item["wins"]) for item in summaries)
    draws = sum(int(item["draws"]) for item in summaries)
    turns = sum(int(item["total_turns"]) for item in summaries)
    decisions = sum(int(item["total_decisions"]) for item in summaries)
    elapsed = sum(float(item["total_elapsed_ms"]) for item in summaries)
    low, high = wilson_interval(wins, games)
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": games - wins - draws,
        "win_rate": round(wins / games, 6),
        "win_rate_ci95": [round(low, 6), round(high, 6)],
        "total_turns": turns,
        "mean_turns": round(turns / games, 3),
        "total_decisions": decisions,
        "mean_decisions": round(decisions / games, 3),
        "total_elapsed_ms": round(elapsed, 3),
        "mean_elapsed_ms": round(elapsed / games, 3),
    }


def _reset_episode(agent: Any) -> None:
    reset = getattr(agent, "reset_episode", None)
    if reset is not None:
        reset()


def _search_metrics(agent: Any) -> dict[str, Any] | None:
    if isinstance(agent, PolicyValueMCTSAgent):
        return agent.metrics()
    return None


def _build_agent(
    specification: AgentSpec,
    *,
    player: int,
    seed: int,
    decks: tuple[list[int], list[int]],
    belief_hypotheses: tuple[DeckHypothesis, ...],
    basic_card_ids: set[int],
    card_catalog: dict[int, object],
    attack_catalog: dict[int, object],
    stochastic: bool,
    mcts_prior: str,
    mcts_config: MCTSConfig,
) -> Any:
    if specification.kind == "random":
        return RandomAgent(seed)
    if specification.kind == "rule":
        return RuleAgent(
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            seed=seed,
        )
    if specification.kind == "external":
        if specification.source is None or specification.deck_path is None:
            raise ValueError(
                f"External agent {specification.name!r} requires source and deck paths"
            )
        return ExternalPythonAgent(
            specification.source,
            specification.deck_path,
            name=specification.name,
            expected_deck=decks[player],
        )
    if specification.checkpoint is None:
        raise ValueError(f"Agent {specification.name!r} requires a checkpoint")
    policy = BCPolicyAgent(
        specification.checkpoint,
        card_catalog=card_catalog,
        attack_catalog=attack_catalog,
        seed=seed,
        deterministic=not stochastic,
    )
    if specification.kind == "policy":
        return policy
    if specification.kind != "mcts":
        raise ValueError(f"Unknown agent kind: {specification.kind}")

    opponent = 1 - player
    prior_decks = [list(decks[0]), list(decks[1])]
    belief = None
    if mcts_prior == "fixed-model":
        prior_decks[opponent] = list(decks[player])
    elif mcts_prior == "belief":
        belief = OpponentDeckBelief(list(belief_hypotheses))
    elif mcts_prior != "oracle":
        raise ValueError(f"Unknown MCTS prior mode: {mcts_prior}")

    determinizer = DeckDeterminizer(
        prior_decks[0],
        prior_decks[1],
        basic_card_ids=basic_card_ids,
        seed=seed + 300_000,
        opponent_belief=belief,
    )
    return PolicyValueMCTSAgent(
        policy,
        determinizer,
        config=mcts_config,
        seed=seed + 400_000,
    )


def _run_cell(task: PanelTask) -> dict[str, Any]:
    torch.set_num_threads(task.torch_threads)
    engine = OfficialEngine(task.official_dir)
    model_deck = engine.load_deck(task.model_deck.path)
    opponent_deck = engine.load_deck(task.opponent_deck.path)
    belief_hypotheses = tuple(
        DeckHypothesis(item.name, tuple(engine.load_deck(item.path)))
        for item in task.belief_decks
    )
    card_catalog = engine.card_catalog()
    attack_catalog = engine.attack_catalog()
    basic_card_ids = {
        card_id
        for card_id, card in card_catalog.items()
        if bool(getattr(card, "basic", False))
    }
    seats: dict[str, dict[str, Any]] = {}
    all_results: list[tuple[MatchResult, int]] = []

    for candidate_player in (0, 1):
        pairing_seed = task.seed + candidate_player * 10_000
        decks = (
            (model_deck, opponent_deck)
            if candidate_player == 0
            else (opponent_deck, model_deck)
        )
        candidate = _build_agent(
            task.candidate,
            player=candidate_player,
            seed=pairing_seed,
            decks=decks,
            belief_hypotheses=belief_hypotheses,
            basic_card_ids=basic_card_ids,
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            stochastic=task.stochastic,
            mcts_prior=task.mcts_prior,
            mcts_config=task.mcts_config,
        )
        opponent_player = 1 - candidate_player
        opponent = _build_agent(
            task.opponent,
            player=opponent_player,
            seed=pairing_seed + 1,
            decks=decks,
            belief_hypotheses=belief_hypotheses,
            basic_card_ids=basic_card_ids,
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            stochastic=task.stochastic,
            mcts_prior=task.mcts_prior,
            mcts_config=task.mcts_config,
        )
        results: list[tuple[MatchResult, int]] = []
        for game in range(task.games_per_seat):
            _reset_episode(candidate)
            _reset_episode(opponent)
            agents = (
                (candidate, opponent)
                if candidate_player == 0
                else (opponent, candidate)
            )
            result = play_match(
                engine,
                decks[0],
                decks[1],
                agents[0],
                agents[1],
                game=game,
                agent_seed0=pairing_seed,
                agent_seed1=pairing_seed + 1,
            )
            results.append((result, candidate_player))
            all_results.append((result, candidate_player))
        seat = _summary(results)
        candidate_search = _search_metrics(candidate)
        opponent_search = _search_metrics(opponent)
        if candidate_search is not None:
            seat["candidate_search"] = candidate_search
        if opponent_search is not None:
            seat["opponent_search"] = opponent_search
        seats[f"as_player{candidate_player}"] = seat

    return {
        "key": (
            f"{task.candidate.name}__vs_{task.opponent.name}"
            f"__deck_{task.opponent_deck.name}"
        ),
        "candidate": task.candidate.name,
        "candidate_kind": task.candidate.kind,
        "opponent": task.opponent.name,
        "opponent_kind": task.opponent.kind,
        "opponent_deck": task.opponent_deck.name,
        "seed": task.seed,
        "overall": _summary(all_results),
        "seats": seats,
    }


def _report_views(
    cells: list[dict[str, Any]], candidate_names: list[str]
) -> dict[str, Any]:
    by_candidate = {
        name: _combine_summaries(
            [cell["overall"] for cell in cells if cell["candidate"] == name]
        )
        for name in candidate_names
    }
    by_candidate_opponent: dict[str, dict[str, Any]] = {}
    by_candidate_deck: dict[str, dict[str, Any]] = {}
    for candidate in candidate_names:
        candidate_cells = [cell for cell in cells if cell["candidate"] == candidate]
        by_candidate_opponent[candidate] = {
            opponent: _combine_summaries(
                [
                    cell["overall"]
                    for cell in candidate_cells
                    if cell["opponent"] == opponent
                ]
            )
            for opponent in sorted({cell["opponent"] for cell in candidate_cells})
        }
        by_candidate_deck[candidate] = {
            deck: _combine_summaries(
                [
                    cell["overall"]
                    for cell in candidate_cells
                    if cell["opponent_deck"] == deck
                ]
            )
            for deck in sorted({cell["opponent_deck"] for cell in candidate_cells})
        }

    comparisons = {}
    if len(candidate_names) > 1:
        baseline = "policy" if "policy" in candidate_names else candidate_names[0]
        baseline_cells = {
            (cell["opponent"], cell["opponent_deck"]): cell
            for cell in cells
            if cell["candidate"] == baseline
        }
        for challenger in candidate_names:
            if challenger == baseline:
                continue
            challenger_cells = {
                (cell["opponent"], cell["opponent_deck"]): cell
                for cell in cells
                if cell["candidate"] == challenger
            }
            cell_deltas = []
            for key in sorted(baseline_cells.keys() & challenger_cells.keys()):
                reference = baseline_cells[key]
                candidate = challenger_cells[key]
                cell_deltas.append(
                    {
                        "opponent": key[0],
                        "opponent_deck": key[1],
                        "win_rate_delta": round(
                            candidate["overall"]["win_rate"]
                            - reference["overall"]["win_rate"],
                            6,
                        ),
                        "player0_delta": round(
                            candidate["seats"]["as_player0"]["win_rate"]
                            - reference["seats"]["as_player0"]["win_rate"],
                            6,
                        ),
                        "player1_delta": round(
                            candidate["seats"]["as_player1"]["win_rate"]
                            - reference["seats"]["as_player1"]["win_rate"],
                            6,
                        ),
                    }
                )
            comparisons[f"{challenger}_minus_{baseline}"] = {
                "baseline": baseline,
                "challenger": challenger,
                "overall_win_rate_delta": round(
                    by_candidate[challenger]["win_rate"]
                    - by_candidate[baseline]["win_rate"],
                    6,
                ),
                "cells_improved": sum(
                    item["win_rate_delta"] > 0 for item in cell_deltas
                ),
                "cells_tied": sum(item["win_rate_delta"] == 0 for item in cell_deltas),
                "cells_worsened": sum(
                    item["win_rate_delta"] < 0 for item in cell_deltas
                ),
                "cells": cell_deltas,
            }
    return {
        "by_candidate": by_candidate,
        "by_candidate_opponent": by_candidate_opponent,
        "by_candidate_deck": by_candidate_deck,
        "comparisons": comparisons,
    }


def _default_decks(official_dir: Path) -> list[DeckSpec]:
    candidates = [
        DeckSpec("sample", str((official_dir / "deck.csv").resolve())),
        DeckSpec(
            "meta_a",
            str((PROJECT_ROOT / "configs/opponent_decks/meta_a.csv").resolve()),
        ),
        DeckSpec(
            "fishcat_v8",
            str((PROJECT_ROOT / "configs/opponent_decks/fishcat_v8.csv").resolve()),
        ),
        DeckSpec(
            "mcts_sample",
            str((PROJECT_ROOT / "configs/opponent_decks/mcts_sample.csv").resolve()),
        ),
    ]
    return [item for item in candidates if Path(item.path).is_file()]


def evaluate_meta_panel(
    checkpoint: str | Path,
    *,
    games_per_seat: int,
    seed: int,
    candidates: list[str],
    opponents: list[AgentSpec],
    opponent_decks: list[DeckSpec],
    official_dir: str | Path | None = None,
    model_deck_path: str | Path | None = None,
    stochastic: bool = True,
    workers: int = 1,
    torch_threads: int = 1,
    mcts_prior: str = "fixed-model",
    simulations: int = 16,
    determinizations: int = 1,
    c_puct: float = 1.25,
    max_depth: int = 12,
    max_actions: int = 16,
    progress: bool = False,
) -> dict[str, Any]:
    if games_per_seat <= 0:
        raise ValueError("games_per_seat must be positive")
    if workers <= 0 or torch_threads <= 0:
        raise ValueError("workers and torch_threads must be positive")
    if not candidates or not opponents or not opponent_decks:
        raise ValueError("candidates, opponents, and opponent_decks cannot be empty")
    if len(candidates) != len(set(candidates)):
        raise ValueError("candidate names must be unique")
    if len({item.name for item in opponents}) != len(opponents):
        raise ValueError("opponent names must be unique")
    if len({item.name for item in opponent_decks}) != len(opponent_decks):
        raise ValueError("opponent deck names must be unique")

    resolved_checkpoint = str(Path(checkpoint).expanduser().resolve())
    resolved_official_dir = resolve_official_dir(official_dir)
    model_deck = DeckSpec(
        "model",
        str(
            Path(model_deck_path or resolved_official_dir / "deck.csv")
            .expanduser()
            .resolve()
        ),
    )
    OfficialEngine.load_deck(model_deck.path)
    for item in opponent_decks:
        OfficialEngine.load_deck(item.path)

    candidate_specs = [
        AgentSpec(name, name, resolved_checkpoint) for name in candidates
    ]
    belief_decks = list(opponent_decks)
    if Path(model_deck.path) not in {Path(item.path) for item in belief_decks}:
        belief_decks.append(model_deck)
    mcts_config = MCTSConfig(
        simulations=simulations,
        determinizations=determinizations,
        c_puct=c_puct,
        max_depth=max_depth,
        max_actions=max_actions,
    )
    tasks = []
    for opponent_index, opponent in enumerate(opponents):
        for deck_index, opponent_deck in enumerate(opponent_decks):
            if (
                opponent.deck_path is not None
                and Path(opponent.deck_path) != Path(opponent_deck.path)
            ):
                continue
            cell_seed = seed + opponent_index * 1_000_000 + deck_index * 100_000
            for candidate in candidate_specs:
                tasks.append(
                    PanelTask(
                        candidate=candidate,
                        opponent=opponent,
                        model_deck=model_deck,
                        opponent_deck=opponent_deck,
                        belief_decks=tuple(belief_decks),
                        games_per_seat=games_per_seat,
                        seed=cell_seed,
                        official_dir=str(resolved_official_dir),
                        stochastic=stochastic,
                        torch_threads=torch_threads,
                        mcts_prior=mcts_prior,
                        mcts_config=mcts_config,
                    )
                )

    started = perf_counter()
    cells = []

    def completed(cell: dict[str, Any]) -> None:
        cells.append(cell)
        if progress:
            rate = cell["overall"]["win_rate"]
            print(
                f"[{len(cells)}/{len(tasks)}] {cell['key']} win_rate={rate:.3f}",
                file=sys.stderr,
                flush=True,
            )

    if workers == 1:
        for task in tasks:
            completed(_run_cell(task))
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = {executor.submit(_run_cell, task): task for task in tasks}
            for future in as_completed(futures):
                completed(future.result())

    cells.sort(key=lambda item: item["key"])
    candidate_names = [item.name for item in candidate_specs]
    return {
        "format": "poketcg-meta-panel-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "checkpoint": resolved_checkpoint,
        "model_deck": asdict(model_deck),
        "games_per_seat": games_per_seat,
        "seed": seed,
        "action_selection": "stochastic" if stochastic else "deterministic",
        "workers": workers,
        "torch_threads_per_worker": torch_threads,
        "mcts_prior": mcts_prior,
        "mcts_config": asdict(mcts_config),
        "candidates": [asdict(item) for item in candidate_specs],
        "opponents": [asdict(item) for item in opponents],
        "opponent_decks": [asdict(item) for item in opponent_decks],
        "wall_seconds": round(perf_counter() - started, 3),
        "views": _report_views(cells, candidate_names),
        "cells": cells,
        "notes": [
            "The CLI seed controls agent and determinization RNG, not native battle RNG.",
            (
                "Candidate deltas compare independent runs and are diagnostics, "
                "not paired causal estimates."
            ),
            (
                "fixed-model makes each MCTS agent assume the opponent uses its own "
                "deck, matching the first submission."
            ),
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games-per-seat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20_260_720)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--model-deck", type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        choices=("policy", "mcts"),
        help="Repeat to select candidates; defaults to policy and mcts.",
    )
    parser.add_argument(
        "--opponent",
        action="append",
        choices=("random", "rule", "policy", "mcts"),
        help="Repeat to select built-ins; defaults to rule and policy.",
    )
    parser.add_argument(
        "--policy-opponent",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT",
        help="Add a fixed policy checkpoint opponent.",
    )
    parser.add_argument(
        "--mcts-opponent",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT",
        help="Add a fixed MCTS checkpoint opponent.",
    )
    parser.add_argument(
        "--external-opponent",
        action="append",
        default=[],
        metavar="NAME=SOURCE,DECK",
        help=(
            "Add an inspected public .py/.ipynb agent and bind it to its own deck. "
            "The source is executed locally."
        ),
    )
    parser.add_argument(
        "--opponent-deck",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeat to define the deck axis; defaults to the four bundled snapshots.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--mcts-prior",
        choices=("fixed-model", "oracle", "belief"),
        default="fixed-model",
    )
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--determinizations", type=int, default=1)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    official_dir = resolve_official_dir(args.official_dir)
    checkpoint = args.checkpoint.expanduser().resolve()
    builtin_opponents = (
        args.opponent
        if args.opponent is not None
        else ([] if args.external_opponent else ["rule", "policy"])
    )
    opponents = [
        AgentSpec(
            name=name,
            kind=name,
            checkpoint=str(checkpoint) if name in {"policy", "mcts"} else None,
        )
        for name in builtin_opponents
    ]
    try:
        for specification in args.policy_opponent:
            name, path = _named_path(specification, "--policy-opponent")
            opponents.append(AgentSpec(name, "policy", str(path)))
        for specification in args.mcts_opponent:
            name, path = _named_path(specification, "--mcts-opponent")
            opponents.append(AgentSpec(name, "mcts", str(path)))
        external_opponents = [
            _external_opponent_spec(specification)
            for specification in args.external_opponent
        ]
        opponents.extend(external_opponents)
        if args.opponent_deck:
            decks = [
                DeckSpec(name, str(path))
                for name, path in (
                    _named_path(item, "--opponent-deck")
                    for item in args.opponent_deck
                )
            ]
        else:
            decks = _default_decks(official_dir)
        known_deck_paths = {Path(item.path) for item in decks}
        for opponent in external_opponents:
            if opponent.deck_path is None:
                continue
            deck_path = Path(opponent.deck_path)
            if deck_path not in known_deck_paths:
                decks.append(DeckSpec(opponent.name, str(deck_path)))
                known_deck_paths.add(deck_path)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    result = evaluate_meta_panel(
        checkpoint,
        games_per_seat=args.games_per_seat,
        seed=args.seed,
        candidates=args.candidate or ["policy", "mcts"],
        opponents=opponents,
        opponent_decks=decks,
        official_dir=official_dir,
        model_deck_path=args.model_deck,
        stochastic=not args.deterministic,
        workers=args.workers,
        torch_threads=args.torch_threads,
        mcts_prior=args.mcts_prior,
        simulations=args.simulations,
        determinizations=args.determinizations,
        c_puct=args.c_puct,
        max_depth=args.max_depth,
        max_actions=args.max_actions,
        progress=True,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "wall_seconds": result["wall_seconds"],
                "views": result["views"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
