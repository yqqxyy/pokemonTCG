"""Adapter for inspected public Kaggle agents kept outside the source tree."""

from __future__ import annotations

import json
import os
import sys
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
        self._source_path = Path(source).expanduser().resolve()
        self._resolved_deck = Path(deck_path).expanduser().resolve()
        if not self._source_path.is_file():
            raise FileNotFoundError(
                f"External agent source not found: {self._source_path}"
            )
        if not self._resolved_deck.is_file():
            raise FileNotFoundError(
                f"External agent deck not found: {self._resolved_deck}"
            )
        self._external_name = name
        self._expected_deck = (
            list(expected_deck) if expected_deck is not None else None
        )
        self.name = f"external-{name}"
        self._used = False
        self._load()

    def _load(self) -> None:
        namespace: dict[str, Any] = {
            "__file__": str(self._source_path),
            "__name__": f"poketcg_external_{self._external_name}",
        }
        with _working_directory(self._resolved_deck.parent):
            exec(
                compile(
                    _agent_source(self._source_path),
                    str(self._source_path),
                    "exec",
                ),
                namespace,
            )
        choose_action = namespace.get("agent")
        if not callable(choose_action):
            raise TypeError(
                "External source does not define callable agent(...): "
                f"{self._source_path}"
            )
        declared_deck = namespace.get("my_deck")
        if (
            self._expected_deck is not None
            and declared_deck is not None
            and [int(value) for value in declared_deck] != self._expected_deck
        ):
            raise ValueError(
                f"External agent {self._external_name!r} loaded a deck different "
                f"from {self._resolved_deck}"
            )
        self._choose_action = choose_action
        self._reset = namespace.get("reset_episode")
        scored_function = namespace.get("_agent")
        self._scored_function_code = (
            scored_function.__code__ if callable(scored_function) else None
        )

    def reset_episode(self) -> None:
        if callable(self._reset):
            self._reset()
        elif self._used:
            self._load()
        self._used = False

    def choose_action(self, observation: dict) -> list[int]:
        self._used = True
        action = self._choose_action(observation)
        if not isinstance(action, list) or not all(isinstance(index, int) for index in action):
            raise TypeError(f"{self.name} returned an action other than list[int]")
        return action

    def choose_action_with_scores(self, observation: dict) -> tuple[list[int], list[float]]:
        """Return an external scorer's action and its per-option score vector.

        The audited Library-Out agent builds a local ``scores`` list inside
        ``_agent``.  Capturing it at function return avoids maintaining a fork of
        the public strategy solely for trajectory collection.  Agents without
        that explicit score vector fail closed instead of fabricating priors.
        """
        if self._scored_function_code is None:
            raise RuntimeError(f"{self.name} does not expose an auditable _agent scorer")
        captured: list[float] | None = None
        previous_trace = sys.gettrace()

        def trace(frame, event, _argument):
            nonlocal captured
            if event == "return" and frame.f_code is self._scored_function_code:
                raw_scores = frame.f_locals.get("scores")
                if isinstance(raw_scores, list) and all(
                    isinstance(value, int | float) for value in raw_scores
                ):
                    captured = [float(value) for value in raw_scores]
            return trace

        sys.settrace(trace)
        try:
            action = self.choose_action(observation)
        finally:
            sys.settrace(previous_trace)
        option_count = len(observation.get("select", {}).get("option", []))
        if captured is None or len(captured) != option_count:
            raise RuntimeError(
                f"{self.name} did not expose one score for each legal option"
            )
        return action, captured
