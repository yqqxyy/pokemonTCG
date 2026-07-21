"""Build a self-contained Kaggle agent archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
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
    tactical_planner: bool = False,
    planner_only: bool = False,
    planner_threshold: float = 0.8,
    planner_weight: float = 4.0,
    planner_confidence_routing: bool = True,
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
    if not 0.0 <= planner_threshold <= 1.0:
        raise ValueError("planner_threshold must be in [0, 1]")
    if planner_weight < 0:
        raise ValueError("planner_weight must be non-negative")
    planner_enabled = tactical_planner or planner_only
    if planner_only and mcts_simulations:
        raise ValueError("planner_only cannot be combined with MCTS")
    if planner_only:
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        tactical_planner=args.tactical_planner,
        planner_only=args.planner_only,
        planner_threshold=args.planner_threshold,
        planner_weight=args.planner_weight,
        planner_confidence_routing=args.planner_confidence_routing,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
