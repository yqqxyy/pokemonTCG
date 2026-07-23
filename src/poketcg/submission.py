"""Build a self-contained Kaggle agent archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .engine import OfficialEngine
from .paths import resolve_official_dir

REQUIRED_ARCHIVE_FILES = {
    "agent_config.json",
    "main.py",
    "deck.csv",
    "model.pt",
    "poketcg/__init__.py",
    "cg/api.py",
    "cg/game.py",
    "cg/libcg.so",
}


def _load_inference_checkpoint(checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict):
        raise TypeError("Checkpoint must contain a dictionary")
    missing = {"model_config", "model_state_dict"} - set(saved)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
    if not isinstance(saved["model_config"], dict):
        raise TypeError("Checkpoint model_config must be a dictionary")
    if not isinstance(saved["model_state_dict"], dict):
        raise TypeError("Checkpoint model_state_dict must be a dictionary")
    inference_checkpoint = {
        "model_config": saved["model_config"],
        "model_state_dict": saved["model_state_dict"],
    }
    if "advantage_config" in saved:
        inference_checkpoint["advantage_config"] = saved["advantage_config"]
    return inference_checkpoint, saved["model_config"]


def _copy_runtime_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store"),
    )


def _archive_members(archive_path: Path) -> set[str]:
    with tarfile.open(archive_path, "r:gz") as archive:
        return {member.name for member in archive.getmembers() if member.isfile()}


def _belief_hypotheses(
    specifications: Sequence[tuple[str, str | Path]] | None,
) -> list[dict[str, Any]]:
    """Load named opponent decks into the small JSON runtime configuration."""
    hypotheses: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw_name, raw_path in specifications or ():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Belief deck names cannot be empty")
        if name in names:
            raise ValueError(f"Duplicate belief deck name: {name!r}")
        names.add(name)
        path = Path(raw_path).expanduser().resolve()
        hypotheses.append(
            {
                "name": name,
                "deck": OfficialEngine.load_deck(path),
                "prior": 1.0,
            }
        )
    return hypotheses


def _named_path(value: str, option: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise ValueError(f"{option} expects NAME=PATH; received {value!r}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{option} path not found: {path}")
    return name.strip(), path


def build_submission(
    checkpoint: str | Path,
    output: str | Path,
    *,
    official_dir: str | Path | None = None,
    deck: str | Path | None = None,
    mcts_simulations: int = 0,
    mcts_determinizations: int = 1,
    mcts_c_puct: float = 1.25,
    mcts_max_depth: int = 12,
    mcts_max_actions: int = 16,
    plan_mcts: bool = False,
    plan_determinizations: int = 4,
    plan_max_steps: int = 32,
    plan_mcts_prior: str = "fixed-model",
    opponent_belief_decks: Sequence[tuple[str, str | Path]] | None = None,
    mega_expert: bool = False,
    tactical_planner: bool = False,
    planner_only: bool = False,
    planner_threshold: float = 0.8,
    planner_weight: float = 4.0,
    planner_confidence_routing: bool = True,
    planner_turn_ownership: bool = False,
    planner_commitment_ownership: bool = False,
    advantage_baseline_source: str | Path | None = None,
    advantage_round0_checkpoint: str | Path | None = None,
    advantage_minimum_turn: int = 4,
    advantage_gate_threshold: float = 0.05,
    advantage_allowed_transitions: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Create and validate a root-layout ``submission.tar.gz`` archive."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    official_path = resolve_official_dir(official_dir)
    deck_path = Path(deck).expanduser().resolve() if deck else official_path / "deck.csv"
    deck_values = OfficialEngine.load_deck(deck_path)
    inference_checkpoint, model_config = _load_inference_checkpoint(checkpoint_path)
    advantage_enabled = (
        advantage_baseline_source is not None
        or advantage_round0_checkpoint is not None
        or bool(advantage_allowed_transitions)
    )
    if advantage_enabled and (
        advantage_baseline_source is None or advantage_round0_checkpoint is None
    ):
        raise ValueError(
            "Advantage mode requires baseline source and Round 0 checkpoint"
        )
    advantage_source_path = (
        Path(advantage_baseline_source).expanduser().resolve()
        if advantage_baseline_source is not None
        else None
    )
    advantage_round0_path = (
        Path(advantage_round0_checkpoint).expanduser().resolve()
        if advantage_round0_checkpoint is not None
        else None
    )
    if advantage_source_path is not None and not advantage_source_path.is_file():
        raise FileNotFoundError(
            f"Advantage baseline source not found: {advantage_source_path}"
        )
    if advantage_round0_path is not None and not advantage_round0_path.is_file():
        raise FileNotFoundError(
            f"Advantage Round 0 checkpoint not found: {advantage_round0_path}"
        )
    if advantage_enabled and "advantage_config" not in inference_checkpoint:
        raise ValueError("Advantage mode requires an advantage checkpoint")
    if advantage_minimum_turn <= 0:
        raise ValueError("advantage_minimum_turn must be positive")
    if advantage_gate_threshold < 0:
        raise ValueError("advantage_gate_threshold must be non-negative")
    transitions = sorted(set(advantage_allowed_transitions or ()))
    if any(source < 0 or target < 0 for source, target in transitions):
        raise ValueError("Advantage option-type transitions must be non-negative")
    if mcts_simulations < 0:
        raise ValueError("mcts_simulations must be non-negative")
    if mcts_determinizations <= 0:
        raise ValueError("mcts_determinizations must be positive")
    if mcts_simulations and mcts_determinizations > mcts_simulations:
        raise ValueError("mcts_determinizations cannot exceed mcts_simulations")
    if mcts_c_puct < 0:
        raise ValueError("mcts_c_puct must be non-negative")
    if mcts_max_depth <= 0 or mcts_max_actions <= 0:
        raise ValueError("MCTS depth and action limits must be positive")
    if plan_determinizations <= 0 or plan_max_steps <= 0:
        raise ValueError("Plan-level MCTS budgets must be positive")
    if plan_mcts_prior not in {"fixed-model", "belief"}:
        raise ValueError("plan_mcts_prior must be 'fixed-model' or 'belief'")
    belief_hypotheses = _belief_hypotheses(opponent_belief_decks)
    if not plan_mcts and (plan_mcts_prior != "fixed-model" or belief_hypotheses):
        raise ValueError("Opponent belief decks require plan_mcts")
    if plan_mcts_prior == "belief" and not belief_hypotheses:
        raise ValueError("belief prior requires at least one opponent belief deck")
    if plan_mcts_prior == "fixed-model" and belief_hypotheses:
        raise ValueError("Opponent belief decks require plan_mcts_prior='belief'")
    if plan_mcts and mcts_simulations:
        raise ValueError("plan_mcts cannot be combined with atomic MCTS")
    if mega_expert and (
        mcts_simulations
        or plan_mcts
        or tactical_planner
        or planner_only
        or planner_turn_ownership
        or planner_commitment_ownership
    ):
        raise ValueError("mega_expert standalone mode cannot be combined with other modes")
    if not 0.0 <= planner_threshold <= 1.0:
        raise ValueError("planner_threshold must be in [0, 1]")
    if planner_weight < 0:
        raise ValueError("planner_weight must be non-negative")
    if planner_turn_ownership and planner_commitment_ownership:
        raise ValueError(
            "planner_turn_ownership and planner_commitment_ownership are mutually exclusive"
        )
    planner_enabled = (
        tactical_planner
        or planner_only
        or plan_mcts
        or planner_turn_ownership
        or planner_commitment_ownership
    )
    if advantage_enabled and (
        mcts_simulations
        or plan_mcts
        or mega_expert
        or planner_enabled
    ):
        raise ValueError("Advantage mode cannot be combined with other agent modes")
    if planner_only and mcts_simulations:
        raise ValueError("planner_only cannot be combined with MCTS")
    if planner_turn_ownership and mcts_simulations:
        raise ValueError("planner_turn_ownership cannot be combined with atomic MCTS")
    if planner_commitment_ownership and mcts_simulations:
        raise ValueError(
            "planner_commitment_ownership cannot be combined with atomic MCTS"
        )
    if plan_mcts and (
        planner_only or planner_turn_ownership or planner_commitment_ownership
    ):
        raise ValueError("plan_mcts owns executor selection and cannot use ownership flags")
    if advantage_enabled:
        mode = "advantage"
    elif mega_expert:
        mode = "mega-expert"
    elif plan_mcts:
        mode = "plan-mcts"
    elif planner_only:
        mode = "planner"
    elif planner_enabled and mcts_simulations:
        mode = "planner-mcts"
    elif planner_enabled:
        mode = "planner-policy"
    elif mcts_simulations:
        mode = "mcts"
    else:
        mode = "policy"
    agent_config = {
        "mode": mode,
        "planner": {
            "enabled": planner_enabled,
            "threshold": planner_threshold,
            "weight": planner_weight,
            "confidence_routing": planner_confidence_routing,
            "turn_ownership": planner_turn_ownership,
            "commitment_ownership": planner_commitment_ownership,
            "profile": "mega-lucario-ex",
        },
        "mcts": {
            "simulations": mcts_simulations,
            "determinizations": mcts_determinizations,
            "c_puct": mcts_c_puct,
            "max_depth": mcts_max_depth,
            "max_actions": mcts_max_actions,
            "root_contexts": [0],
        },
        "plan_mcts": {
            "enabled": plan_mcts,
            "determinizations": plan_determinizations,
            "max_macro_steps": plan_max_steps,
            "root_contexts": [0],
            "prior": plan_mcts_prior,
            "belief_hypotheses": belief_hypotheses,
        },
        "mega_expert": {"enabled": mega_expert},
        "advantage": {
            "enabled": advantage_enabled,
            "minimum_turn": advantage_minimum_turn,
            "gate_threshold": advantage_gate_threshold,
            "uncertainty_multiplier": 0.0,
            "allowed_transitions": [list(item) for item in transitions],
        },
    }

    runtime_source = Path(__file__).with_name("submission_runtime.py")
    package_source = Path(__file__).resolve().parent
    temporary_archive = output_path.parent / f".{output_path.name}.tmp"

    with tempfile.TemporaryDirectory(prefix="poketcg-submission-") as temporary:
        root = Path(temporary)
        shutil.copy2(runtime_source, root / "main.py")
        (root / "deck.csv").write_text(
            "".join(f"{card_id}\n" for card_id in deck_values),
            encoding="utf-8",
        )
        (root / "agent_config.json").write_text(
            json.dumps(agent_config, indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(inference_checkpoint, root / "model.pt")
        if advantage_enabled:
            assert advantage_source_path is not None
            assert advantage_round0_path is not None
            round0_checkpoint, _ = _load_inference_checkpoint(advantage_round0_path)
            torch.save(round0_checkpoint, root / "round0_model.pt")
            shutil.copy2(advantage_source_path, root / "libraryout_baseline.py")
        _copy_runtime_tree(package_source, root / "poketcg")
        _copy_runtime_tree(official_path / "cg", root / "cg")

        if temporary_archive.exists():
            temporary_archive.unlink()
        with tarfile.open(temporary_archive, "w:gz") as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=path.relative_to(root).as_posix())

    members = _archive_members(temporary_archive)
    missing = REQUIRED_ARCHIVE_FILES - members
    if advantage_enabled:
        missing |= {"round0_model.pt", "libraryout_baseline.py"} - members
    if missing:
        temporary_archive.unlink(missing_ok=True)
        raise RuntimeError(f"Submission archive is missing files: {sorted(missing)}")
    temporary_archive.replace(output_path)

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return {
        "archive": str(output_path),
        "size_mib": round(output_path.stat().st_size / 1024 / 1024, 3),
        "sha256": digest,
        "file_count": len(members),
        "model_config": model_config,
        "agent_config": agent_config,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--deck", type=Path)
    parser.add_argument("--advantage-baseline-source", type=Path)
    parser.add_argument("--advantage-round0-checkpoint", type=Path)
    parser.add_argument("--advantage-minimum-turn", type=int, default=4)
    parser.add_argument("--advantage-gate-threshold", type=float, default=0.05)
    parser.add_argument(
        "--advantage-allowed-transition",
        action="append",
        default=[],
        metavar="FROM->TO",
    )
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=0,
        help="Enable MCTS with this total simulation budget; zero keeps direct policy.",
    )
    parser.add_argument("--mcts-determinizations", type=int, default=1)
    parser.add_argument("--mcts-c-puct", type=float, default=1.25)
    parser.add_argument("--mcts-max-depth", type=int, default=12)
    parser.add_argument("--mcts-max-actions", type=int, default=16)
    parser.add_argument(
        "--plan-mcts",
        action="store_true",
        help="Search between local-router and full-turn Planner macro executors.",
    )
    parser.add_argument("--plan-determinizations", type=int, default=4)
    parser.add_argument("--plan-max-steps", type=int, default=32)
    parser.add_argument(
        "--mega-expert",
        action="store_true",
        help="Build the native public Mega Lucario expert as a standalone agent.",
    )
    parser.add_argument(
        "--plan-mcts-prior",
        choices=("fixed-model", "belief"),
        default="fixed-model",
        help="Opponent-deck prior used by Plan MCTS determinization.",
    )
    parser.add_argument(
        "--belief-deck",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeat to package a candidate opponent deck for the belief prior.",
    )
    parser.add_argument(
        "--tactical-planner",
        action="store_true",
        help="Blend the Mega Lucario tactical planner with policy/MCTS decisions.",
    )
    parser.add_argument(
        "--planner-only",
        action="store_true",
        help="Build the planner-only ablation instead of using the neural policy.",
    )
    parser.add_argument("--planner-threshold", type=float, default=0.8)
    parser.add_argument("--planner-weight", type=float, default=4.0)
    confidence_group = parser.add_mutually_exclusive_group()
    confidence_group.add_argument(
        "--planner-confidence-routing",
        dest="planner_confidence_routing",
        action="store_true",
        default=True,
    )
    confidence_group.add_argument(
        "--no-planner-confidence-routing",
        dest="planner_confidence_routing",
        action="store_false",
    )
    parser.add_argument(
        "--planner-turn-ownership",
        action="store_true",
        help="Keep the turn-start Planner/Policy owner for every decision in that turn.",
    )
    parser.add_argument(
        "--planner-commitment-ownership",
        action="store_true",
        help=(
            "Own ordinary resolver chains, but claim the full turn only for an "
            "explicit TacticalPlan commitment."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        belief_decks = [
            _named_path(value, "--belief-deck") for value in args.belief_deck
        ]
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    try:
        advantage_transitions = []
        for value in args.advantage_allowed_transition:
            source, separator, target = value.partition("->")
            if not separator:
                raise ValueError(
                    "--advantage-allowed-transition expects FROM->TO"
                )
            advantage_transitions.append((int(source), int(target)))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    result = build_submission(
        args.checkpoint,
        args.output,
        official_dir=args.official_dir,
        deck=args.deck,
        mcts_simulations=args.mcts_simulations,
        mcts_determinizations=args.mcts_determinizations,
        mcts_c_puct=args.mcts_c_puct,
        mcts_max_depth=args.mcts_max_depth,
        mcts_max_actions=args.mcts_max_actions,
        plan_mcts=args.plan_mcts,
        plan_determinizations=args.plan_determinizations,
        plan_max_steps=args.plan_max_steps,
        plan_mcts_prior=args.plan_mcts_prior,
        opponent_belief_decks=belief_decks,
        mega_expert=args.mega_expert,
        tactical_planner=args.tactical_planner,
        planner_only=args.planner_only,
        planner_threshold=args.planner_threshold,
        planner_weight=args.planner_weight,
        planner_confidence_routing=args.planner_confidence_routing,
        planner_turn_ownership=args.planner_turn_ownership,
        planner_commitment_ownership=args.planner_commitment_ownership,
        advantage_baseline_source=args.advantage_baseline_source,
        advantage_round0_checkpoint=args.advantage_round0_checkpoint,
        advantage_minimum_turn=args.advantage_minimum_turn,
        advantage_gate_threshold=args.advantage_gate_threshold,
        advantage_allowed_transitions=advantage_transitions,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
