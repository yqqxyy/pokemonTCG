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
_PLANNER: Any | None = None
_HYBRID: Any | None = None
_MCTS: Any | None = None
_PLAN_MCTS: Any | None = None
_MEGA_EXPERT: Any | None = None
_ADVANTAGE: Any | None = None
_LIBRARYOUT_BASELINE: Any | None = None
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


def _get_libraryout_baseline() -> Any | None:
    global _LIBRARYOUT_BASELINE
    if _CONFIG.get("mode") != "advantage":
        return None
    if _LIBRARYOUT_BASELINE is not None:
        return _LIBRARYOUT_BASELINE
    try:
        from poketcg.agents import ExternalPythonAgent

        _LIBRARYOUT_BASELINE = ExternalPythonAgent(
            _ROOT / "libraryout_baseline.py",
            _ROOT / "deck.csv",
            name="libraryout-baseline-fallback",
            expected_deck=_read_deck(),
        )
    except Exception:
        traceback.print_exc()
    return _LIBRARYOUT_BASELINE


def _get_advantage() -> Any | None:
    global _ADVANTAGE
    if _CONFIG.get("mode") != "advantage":
        return None
    if _ADVANTAGE is not None:
        return _ADVANTAGE
    try:
        from poketcg.agents import AdvantageRerankerAgent

        cards, attacks = _catalogs()
        settings = _CONFIG.get("advantage") or {}
        transitions = {
            (int(item[0]), int(item[1]))
            for item in settings.get("allowed_transitions") or []
        }
        _ADVANTAGE = AdvantageRerankerAgent(
            [_ROOT / "model.pt"],
            _ROOT / "round0_model.pt",
            _ROOT / "libraryout_baseline.py",
            _ROOT / "deck.csv",
            card_catalog=cards,
            attack_catalog=attacks,
            expected_deck=_read_deck(),
            device="cpu",
            minimum_turn=int(settings.get("minimum_turn", 4)),
            gate_threshold=float(settings.get("gate_threshold", 0.05)),
            uncertainty_multiplier=float(
                settings.get("uncertainty_multiplier", 0.0)
            ),
            allowed_transitions=transitions,
        )
    except Exception:
        traceback.print_exc()
    return _ADVANTAGE


def _get_mega_expert() -> Any | None:
    global _MEGA_EXPERT
    if _CONFIG.get("mode") != "mega-expert":
        return None
    if _MEGA_EXPERT is not None:
        return _MEGA_EXPERT
    try:
        from poketcg.agents import MegaLucarioExpertAgent

        cards, _ = _catalogs()
        _MEGA_EXPERT = MegaLucarioExpertAgent(
            card_catalog=cards,
            deck=_read_deck(),
        )
    except Exception:
        traceback.print_exc()
    return _MEGA_EXPERT


def _get_planner() -> Any | None:
    global _PLANNER
    if _PLANNER is not None:
        return _PLANNER
    try:
        from poketcg.agents import TacticalPlannerAgent

        cards, attacks = _catalogs()
        _PLANNER = TacticalPlannerAgent(
            card_catalog=cards,
            attack_catalog=attacks,
            seed=20260720,
        )
    except Exception:
        traceback.print_exc()
    return _PLANNER


def _get_hybrid() -> Any | None:
    global _HYBRID
    if _HYBRID is not None:
        return _HYBRID
    try:
        from poketcg.agents import PlannerPolicyAgent

        policy = _get_policy()
        planner = _get_planner()
        if policy is None or planner is None:
            raise RuntimeError("Planner-policy components failed to initialize")
        settings = _CONFIG.get("planner") or {}
        _HYBRID = PlannerPolicyAgent(
            policy,
            planner,
            planner_threshold=float(settings.get("threshold", 0.8)),
            planner_weight=float(settings.get("weight", 4.0)),
            confidence_routing=bool(settings.get("confidence_routing", True)),
            turn_ownership=bool(settings.get("turn_ownership", False)),
            commitment_ownership=bool(
                settings.get("commitment_ownership", False)
            ),
            deterministic=False,
            seed=20260721,
        )
    except Exception:
        traceback.print_exc()
    return _HYBRID


def _get_mcts() -> Any | None:
    global _MCTS, _MCTS_DISABLED, _MCTS_ERROR_REPORTED
    mode = _CONFIG.get("mode")
    if mode not in {"mcts", "planner-mcts"}:
        return None
    if _MCTS is not None:
        return _MCTS
    if _MCTS_DISABLED:
        return None

    try:
        from poketcg.mcts import DeckDeterminizer, MCTSConfig, PolicyValueMCTSAgent

        policy = _get_hybrid() if mode == "planner-mcts" else _get_policy()
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


def _get_plan_mcts() -> Any | None:
    global _PLAN_MCTS
    if _CONFIG.get("mode") != "plan-mcts":
        return None
    if _PLAN_MCTS is not None:
        return _PLAN_MCTS
    try:
        from poketcg.agents import PlannerPolicyAgent, TacticalPlannerAgent
        from poketcg.mcts import (
            DeckDeterminizer,
            DeckHypothesis,
            OpponentDeckBelief,
            PlanLevelMCTSAgent,
            PlanMCTSConfig,
        )

        policy = _get_policy()
        if policy is None:
            raise RuntimeError("Plan-level MCTS policy failed to initialize")
        cards, attacks = _catalogs()
        deck = _read_deck()
        basic_card_ids = {
            card_id
            for card_id, card in cards.items()
            if bool(getattr(card, "basic", False))
        }
        planner_settings = _CONFIG.get("planner") or {}
        local_executor = PlannerPolicyAgent(
            policy,
            TacticalPlannerAgent(
                card_catalog=cards,
                attack_catalog=attacks,
                seed=20260723,
            ),
            planner_threshold=float(planner_settings.get("threshold", 0.8)),
            planner_weight=float(planner_settings.get("weight", 4.0)),
            confidence_routing=bool(
                planner_settings.get("confidence_routing", True)
            ),
            deterministic=True,
            seed=20260724,
        )
        planner_executor = TacticalPlannerAgent(
            card_catalog=cards,
            attack_catalog=attacks,
            seed=20260725,
        )
        settings = _CONFIG.get("plan_mcts") or {}
        config = PlanMCTSConfig(
            determinizations=int(settings.get("determinizations", 4)),
            max_macro_steps=int(settings.get("max_macro_steps", 32)),
            root_contexts=tuple(
                int(value) for value in settings.get("root_contexts", [0])
            ),
        )
        prior = str(settings.get("prior", "fixed-model"))
        opponent_belief = None
        if prior == "belief":
            hypotheses = []
            names = set()
            for item in settings.get("belief_hypotheses") or []:
                if not isinstance(item, dict):
                    raise TypeError("Plan MCTS belief hypotheses must be objects")
                name = str(item.get("name", "")).strip()
                values = tuple(int(value) for value in item.get("deck") or [])
                if not name or name in names:
                    raise ValueError(f"Invalid Plan MCTS belief deck name: {name!r}")
                names.add(name)
                hypotheses.append(
                    DeckHypothesis(
                        name,
                        values,
                        prior=float(item.get("prior", 1.0)),
                    )
                )
            if "model" not in names:
                hypotheses.append(DeckHypothesis("model", tuple(deck)))
            if not hypotheses:
                raise ValueError("Plan MCTS belief prior has no deck hypotheses")
            opponent_belief = OpponentDeckBelief(hypotheses)
        elif prior != "fixed-model":
            raise ValueError(f"Unknown Plan MCTS prior: {prior!r}")
        determinizer = DeckDeterminizer(
            deck,
            deck,
            basic_card_ids=basic_card_ids,
            seed=20260726,
            opponent_belief=opponent_belief,
        )
        _PLAN_MCTS = PlanLevelMCTSAgent(
            local_executor,
            planner_executor,
            policy,
            determinizer,
            config=config,
        )
    except Exception:
        traceback.print_exc()
    return _PLAN_MCTS


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
        if _ADVANTAGE is not None:
            _ADVANTAGE.reset_episode()
        if _LIBRARYOUT_BASELINE is not None:
            _LIBRARYOUT_BASELINE.reset_episode()
        if _MEGA_EXPERT is not None:
            _MEGA_EXPERT.reset_episode()
        if _PLAN_MCTS is not None:
            _PLAN_MCTS.reset_episode()
        if _MCTS is not None:
            _MCTS.reset_episode()
        if _HYBRID is not None:
            _HYBRID.reset_episode()
        elif _PLANNER is not None:
            _PLANNER.reset_episode()
        return _read_deck()

    mega_expert = _get_mega_expert()
    if mega_expert is not None:
        try:
            return mega_expert.choose_action(obs_dict)
        except Exception:
            traceback.print_exc()

    if _CONFIG.get("mode") == "advantage":
        advantage = _get_advantage()
        if advantage is not None:
            try:
                return advantage.choose_action(obs_dict)
            except Exception:
                traceback.print_exc()
        baseline = _get_libraryout_baseline()
        if baseline is not None:
            try:
                return baseline.choose_action(obs_dict)
            except Exception:
                traceback.print_exc()
        rule_agent = _get_rule_agent()
        if rule_agent is not None:
            try:
                return rule_agent.choose_action(obs_dict)
            except Exception:
                traceback.print_exc()
        return _minimum_legal_action(obs_dict)

    plan_mcts = _get_plan_mcts()
    if plan_mcts is not None:
        try:
            return plan_mcts.choose_action(obs_dict)
        except Exception:
            traceback.print_exc()

    mcts = _get_mcts()
    if mcts is not None:
        try:
            return mcts.choose_action(obs_dict)
        except Exception:
            traceback.print_exc()

    mode = _CONFIG.get("mode")
    if mode == "planner":
        planner = _get_planner()
        if planner is not None:
            try:
                return planner.choose_action(obs_dict)
            except Exception:
                traceback.print_exc()
    elif mode in {"planner-policy", "planner-mcts", "plan-mcts"}:
        hybrid = _get_hybrid()
        if hybrid is not None:
            try:
                return hybrid.choose_action(obs_dict)
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
