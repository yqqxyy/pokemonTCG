"""Adapter for inspected public Kaggle agents kept outside the source tree."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _notebook_agent_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        raw_source = cell.get("source", "")
        source = "".join(raw_source) if isinstance(raw_source, list) else str(raw_source)
        if "def agent(" not in source:
            continue
        lines = source.splitlines()
        if lines and lines[0].lstrip().startswith("%%writefile"):
            lines = lines[1:]
        if any(line.lstrip().startswith(("%", "!")) for line in lines):
            raise ValueError(f"Unsupported notebook magic in external agent: {path}")
        return "\n".join(lines) + "\n"
    raise ValueError(f"No code cell defining agent(...) was found in {path}")


def _agent_source(path: Path) -> str:
    if path.suffix.lower() == ".ipynb":
        return _notebook_agent_source(path)
    if path.suffix.lower() == ".py":
        return path.read_text(encoding="utf-8")
    raise ValueError("External agent source must be a .py or .ipynb file")


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class ExternalPythonAgent:
    """Load a trusted public ``agent(obs)`` implementation from a local artifact.

    The source is executed once in the directory containing its deck, matching
    Kaggle submission behavior for agents that open ``deck.csv`` at import time.
    Public notebook code is untrusted input and must be inspected before use.
    """

    def __init__(
        self,
        source: str | Path,
        deck_path: str | Path,
        *,
        name: str,
        expected_deck: list[int] | None = None,
    ) -> None:
        source_path = Path(source).expanduser().resolve()
        resolved_deck = Path(deck_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"External agent source not found: {source_path}")
        if not resolved_deck.is_file():
            raise FileNotFoundError(f"External agent deck not found: {resolved_deck}")
        namespace: dict[str, Any] = {
            "__file__": str(source_path),
            "__name__": f"poketcg_external_{name}",
        }
        with _working_directory(resolved_deck.parent):
            exec(compile(_agent_source(source_path), str(source_path), "exec"), namespace)
        choose_action = namespace.get("agent")
        if not callable(choose_action):
            raise TypeError(f"External source does not define callable agent(...): {source_path}")
        declared_deck = namespace.get("my_deck")
        if (
            expected_deck is not None
            and declared_deck is not None
            and [int(value) for value in declared_deck] != list(expected_deck)
        ):
            raise ValueError(
                f"External agent {name!r} loaded a deck different from {resolved_deck}"
            )
        self.name = f"external-{name}"
        self._choose_action = choose_action
        self._reset = namespace.get("reset_episode")

    def reset_episode(self) -> None:
        if callable(self._reset):
            self._reset()

    def choose_action(self, observation: dict) -> list[int]:
        action = self._choose_action(observation)
        if not isinstance(action, list) or not all(isinstance(index, int) for index in action):
            raise TypeError(f"{self.name} returned an action other than list[int]")
        return action
