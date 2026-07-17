"""Project path discovery."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_DIR = (
    PROJECT_ROOT / "data" / "official" / "sample_submission" / "sample_submission"
)


def resolve_official_dir(value: str | Path | None = None) -> Path:
    """Resolve and validate the directory containing the official ``cg`` package."""
    configured = value or os.environ.get("POKETCG_OFFICIAL_DIR") or DEFAULT_OFFICIAL_DIR
    path = Path(configured).expanduser().resolve()
    required = (path / "cg" / "game.py", path / "cg" / "api.py", path / "deck.csv")
    missing = [item for item in required if not item.is_file()]
    if missing:
        rendered = ", ".join(str(item) for item in missing)
        raise FileNotFoundError(
            f"Official simulator files were not found under {path}. Missing: {rendered}"
        )
    return path
