"""Package an audited external Python agent as a Kaggle CABT submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from .engine import OfficialEngine
from .paths import resolve_official_dir

REQUIRED_FILES = {
    "main.py",
    "deck.csv",
    "cg/api.py",
    "cg/game.py",
    "cg/libcg.so",
}


def build_external_submission(
    source: str | Path,
    deck: str | Path,
    output: str | Path,
    *,
    official_dir: str | Path | None = None,
    requirements: str | Path | None = None,
) -> dict[str, Any]:
    """Build a root-layout archive from an already inspected external agent."""
    source_path = Path(source).expanduser().resolve()
    deck_path = Path(deck).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Agent source not found: {source_path}")
    if source_path.suffix != ".py":
        raise ValueError("External submission source must be a .py file")
    if not deck_path.is_file():
        raise FileNotFoundError(f"Deck not found: {deck_path}")

    source_text = source_path.read_text()
    compile(source_text, str(source_path), "exec")
    deck_values = OfficialEngine.load_deck(deck_path)
    official_path = resolve_official_dir(official_dir)
    cg_path = official_path / "cg"
    if not cg_path.is_dir():
        raise FileNotFoundError(f"Official cg directory not found: {cg_path}")

    requirements_path: Path | None
    if requirements is not None:
        requirements_path = Path(requirements).expanduser().resolve()
        if not requirements_path.is_file():
            raise FileNotFoundError(f"Requirements file not found: {requirements_path}")
    else:
        adjacent = source_path.with_name("requirements.txt")
        requirements_path = adjacent if adjacent.is_file() else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = output_path.parent / f".{output_path.name}.tmp"
    with tempfile.TemporaryDirectory(prefix="poketcg-external-submission-") as temp:
        root = Path(temp)
        shutil.copy2(source_path, root / "main.py")
        (root / "deck.csv").write_text(
            "".join(f"{card_id}\n" for card_id in deck_values), encoding="utf-8"
        )
        shutil.copytree(
            cg_path,
            root / "cg",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store"),
        )
        if requirements_path is not None:
            shutil.copy2(requirements_path, root / "requirements.txt")

        temporary_archive.unlink(missing_ok=True)
        with tarfile.open(temporary_archive, "w:gz") as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=path.relative_to(root).as_posix())

    with tarfile.open(temporary_archive, "r:gz") as archive:
        members = {member.name for member in archive.getmembers() if member.isfile()}
    missing = REQUIRED_FILES - members
    if missing:
        temporary_archive.unlink(missing_ok=True)
        raise RuntimeError(f"Submission archive is missing files: {sorted(missing)}")
    temporary_archive.replace(output_path)

    return {
        "archive": str(output_path),
        "size_mib": round(output_path.stat().st_size / 1024 / 1024, 3),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "deck_sha256": hashlib.sha256(deck_path.read_bytes()).hexdigest(),
        "deck_length": len(deck_values),
        "file_count": len(members),
        "source": str(source_path),
        "deck": str(deck_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--requirements", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_external_submission(
        args.source,
        args.deck,
        args.output,
        official_dir=args.official_dir,
        requirements=args.requirements,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
