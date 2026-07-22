from __future__ import annotations

import json
from pathlib import Path

import pytest

from poketcg.agents import ExternalPythonAgent


def _deck(path: Path, value: int = 7) -> list[int]:
    cards = [value] * 60
    path.write_text("".join(f"{card}\n" for card in cards), encoding="utf-8")
    return cards


def test_external_python_agent_loads_relative_deck_without_changing_cwd(
    tmp_path: Path,
) -> None:
    expected = _deck(tmp_path / "deck.csv")
    source = tmp_path / "main.py"
    source.write_text(
        "my_deck = [int(x) for x in open('deck.csv').read().split()]\n"
        "def agent(obs):\n"
        "    return my_deck if obs.get('select') is None else [1]\n",
        encoding="utf-8",
    )
    previous = Path.cwd()

    agent = ExternalPythonAgent(
        source,
        tmp_path / "deck.csv",
        name="public",
        expected_deck=expected,
    )

    assert Path.cwd() == previous
    assert agent.choose_action({"select": {}}) == [1]
    assert agent.name == "external-public"


def test_external_python_agent_extracts_writefile_notebook_cell(tmp_path: Path) -> None:
    expected = _deck(tmp_path / "deck.csv")
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": "Explanation"},
            {
                "cell_type": "code",
                "source": (
                    "%%writefile main.py\n"
                    "my_deck = [int(x) for x in open('deck.csv').read().split()]\n"
                    "def agent(obs):\n"
                    "    return [0]\n"
                ),
            },
        ]
    }
    path = tmp_path / "agent.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")

    agent = ExternalPythonAgent(
        path,
        tmp_path / "deck.csv",
        name="notebook",
        expected_deck=expected,
    )

    assert agent.choose_action({"select": {}}) == [0]


def test_external_python_agent_captures_audited_option_scores(tmp_path: Path) -> None:
    expected = _deck(tmp_path / "deck.csv")
    source = tmp_path / "main.py"
    source.write_text(
        "def _agent(obs):\n"
        "    scores = [float(option['score']) for option in obs['select']['option']]\n"
        "    return [max(range(len(scores)), key=scores.__getitem__)]\n"
        "def agent(obs, configuration=None):\n"
        "    return _agent(obs)\n",
        encoding="utf-8",
    )
    agent = ExternalPythonAgent(
        source,
        tmp_path / "deck.csv",
        name="scored",
        expected_deck=expected,
    )

    action, scores = agent.choose_action_with_scores(
        {"select": {"option": [{"score": 3}, {"score": 9}, {"score": -1}]}}
    )

    assert action == [1]
    assert scores == [3.0, 9.0, -1.0]


def test_external_python_agent_rejects_mismatched_deck(tmp_path: Path) -> None:
    _deck(tmp_path / "deck.csv")
    source = tmp_path / "main.py"
    source.write_text(
        "my_deck = [int(x) for x in open('deck.csv').read().split()]\n"
        "def agent(obs): return [0]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different"):
        ExternalPythonAgent(
            source,
            tmp_path / "deck.csv",
            name="wrong-deck",
            expected_deck=[8] * 60,
        )


def test_external_python_agent_reloads_state_when_no_reset_hook(
    tmp_path: Path,
) -> None:
    expected = _deck(tmp_path / "deck.csv")
    source = tmp_path / "main.py"
    source.write_text(
        "counter = 0\n"
        "def agent(obs):\n"
        "    global counter\n"
        "    counter += 1\n"
        "    return [counter]\n",
        encoding="utf-8",
    )
    agent = ExternalPythonAgent(
        source,
        tmp_path / "deck.csv",
        name="stateful",
        expected_deck=expected,
    )

    assert agent.choose_action({"select": {}}) == [1]
    assert agent.choose_action({"select": {}}) == [2]
    agent.reset_episode()
    assert agent.choose_action({"select": {}}) == [1]
