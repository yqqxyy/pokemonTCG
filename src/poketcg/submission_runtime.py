"""Kaggle runtime entry point copied to ``main.py`` in submission archives."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any


def _find_agent_root() -> Path:
    """Locate bundled files when Kaggle executes source without defining ``__file__``."""
    candidates = (Path("/kaggle_simulations/agent"), Path.cwd())
    for candidate in candidates:
        if (candidate / "deck.csv").is_file() and (candidate / "poketcg").is_dir():
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not locate the agent bundle under: {rendered}")


_ROOT = _find_agent_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_POLICY: Any | None = None
_MCTS: Any | None = None
_RULE_AGENT: Any | None = None
_POLICY_DISABLED = False
_MCTS_DISABLED = False
_POLICY_ERROR_REPORTED = False
_MCTS_ERROR_REPORTED = False


def _read_config() -> dict[str, Any]:
    path = _ROOT / "agent_config.json"
    if not path.is_file():
        return {"mode": "policy", "mcts": {}}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("agent_config.json must contain an object")
    return loaded


_CONFIG = _read_config()


def _catalogs() -> tuple[dict[int, object], dict[int, object]]:
    from cg.api import all_attack, all_card_data

    cards = {int(card.cardId): card for card in all_card_data()}
    attacks = {int(attack.attackId): attack for attack in all_attack()}
    return cards, attacks


def _read_deck() -> list[int]:
    values = [line.strip() for line in (_ROOT / "deck.csv").read_text().splitlines()]
    deck = [int(value) for value in values if value]
    if len(deck) != 60:
        raise ValueError(f"deck.csv must contain exactly 60 card IDs; found {len(deck)}")
    return deck


def _get_policy() -> Any | None:
    global _POLICY, _POLICY_DISABLED, _POLICY_ERROR_REPORTED
    if _POLICY is not None:
        return _POLICY
    if _POLICY_DISABLED:
        return None

    try:
        from poketcg.agents import BCPolicyAgent

        cards, attacks = _catalogs()
        _POLICY = BCPolicyAgent(
            _ROOT / "model.pt",
            card_catalog=cards,
            attack_catalog=attacks,
            seed=20260720,
            device="cpu",
            deterministic=False,
        )
    except Exception:
        _POLICY_DISABLED = True
        if not _POLICY_ERROR_REPORTED:
            traceback.print_exc()
            _POLICY_ERROR_REPORTED = True
    return _POLICY


def _get_rule_agent() -> Any | None:
    global _RULE_AGENT
    if _RULE_AGENT is not None:
        return _RULE_AGENT
    try:
        from poketcg.agents import RuleAgent

        cards, attacks = _catalogs()
        _RULE_AGENT = RuleAgent(card_catalog=cards, attack_catalog=attacks, seed=20260720)
    except Exception:
        traceback.print_exc()
    return _RULE_AGENT


def _get_mcts() -> Any | None:
    global _MCTS, _MCTS_DISABLED, _MCTS_ERROR_REPORTED
    if _CONFIG.get("mode") != "mcts":
        return None
    if _MCTS is not None:
        return _MCTS
    if _MCTS_DISABLED:
        return None

    try:
        from poketcg.mcts import DeckDeterminizer, MCTSConfig, PolicyValueMCTSAgent

        policy = _get_policy()
        if policy is None:
            raise RuntimeError("MCTS policy failed to initialize")
        cards, _ = _catalogs()
        deck = _read_deck()
        basic_card_ids = {
            card_id
            for card_id, card in cards.items()
            if bool(getattr(card, "basic", False))
        }
        settings = _CONFIG.get("mcts") or {}
        config = MCTSConfig(
            simulations=int(settings.get("simulations", 16)),
            determinizations=int(settings.get("determinizations", 1)),
            c_puct=float(settings.get("c_puct", 1.25)),
            max_depth=int(settings.get("max_depth", 12)),
            max_actions=int(settings.get("max_actions", 16)),
            root_contexts=tuple(int(value) for value in settings.get("root_contexts", [0])),
        )
        determinizer = DeckDeterminizer(
            deck,
            deck,
            basic_card_ids=basic_card_ids,
            seed=20260721,
        )
        _MCTS = PolicyValueMCTSAgent(
            policy,
            determinizer,
            config=config,
            seed=20260722,
        )
    except Exception:
        _MCTS_DISABLED = True
        if not _MCTS_ERROR_REPORTED:
            traceback.print_exc()
            _MCTS_ERROR_REPORTED = True
    return _MCTS


def _minimum_legal_action(observation: dict) -> list[int]:
    selection = observation["select"]
    minimum = int(selection["minCount"])
    option_count = len(selection["option"])
    if minimum > option_count:
        raise ValueError("Selection minimum exceeds the number of available options")
    return list(range(minimum))


def agent(obs_dict: dict) -> list[int]:
    """Return the deck initially, then select legal option indices during battle."""
    if obs_dict.get("select") is None:
        if _MCTS is not None:
            _MCTS.reset_episode()
        return _read_deck()

    mcts = _get_mcts()
    if mcts is not None:
        try:
            return mcts.choose_action(obs_dict)
        except Exception:
            traceback.print_exc()

    policy = _get_policy()
    if policy is not None:
        try:
            return policy.choose_action(obs_dict)
        except Exception:
            traceback.print_exc()

    rule_agent = _get_rule_agent()
    if rule_agent is not None:
        try:
            return rule_agent.choose_action(obs_dict)
        except Exception:
            traceback.print_exc()

    return _minimum_legal_action(obs_dict)
