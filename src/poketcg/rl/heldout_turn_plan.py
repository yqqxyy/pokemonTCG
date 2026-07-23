"""Belief-consistent proposal/held-out evaluation of semantic turn plans."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from poketcg.mcts import HiddenStateGuess, SearchPosition

from .paired_rollout import paired_summary
from .semantic_plan import (
    SemanticTurnPlan,
    resolve_semantic_action,
    semantic_action,
)
from .turn_synergy import PlanLeaf, TurnSynergyEvaluator


def _gain_lcb(summary: dict, risk_multiplier: float) -> float:
    gain = summary.get("paired_advantage")
    stderr = summary.get("paired_stderr")
    if gain is None or stderr is None:
        return -float("inf")
    return float(gain) - risk_multiplier * float(stderr)


class HeldoutTurnPlanEvaluator(TurnSynergyEvaluator):
    """Select one semantic plan without looking at held-out hidden worlds."""

    def __init__(
        self,
        *args,
        proposal_determinizations: int = 4,
        heldout_determinizations: int = 4,
        plan_pool_size: int = 16,
        selection_risk_multiplier: float = 1.0,
        **kwargs,
    ) -> None:
        if proposal_determinizations <= 0 or heldout_determinizations <= 0:
            raise ValueError("Proposal and held-out determinizations must be positive")
        if plan_pool_size <= 0:
            raise ValueError("plan_pool_size must be positive")
        if selection_risk_multiplier < 0:
            raise ValueError("selection_risk_multiplier must be non-negative")
        super().__init__(
            *args,
            determinizations=proposal_determinizations
            + heldout_determinizations,
            **kwargs,
        )
        self._proposal_determinizations = proposal_determinizations
        self._heldout_determinizations = heldout_determinizations
        self._plan_pool_size = plan_pool_size
        self._selection_risk_multiplier = selection_risk_multiplier

    def _proposal_plans(
        self,
        observation: dict,
        hidden_states: list[HiddenStateGuess],
        *,
        root_player: int,
        root_turn: int,
    ) -> tuple[list[SemanticTurnPlan], list[dict], dict[str, int]]:
        proposals: dict[tuple, dict[str, Any]] = {}
        worlds = []
        errors: Counter[str] = Counter()
        for world_id, hidden in enumerate(hidden_states):
            began = False
            try:
                root = self._backend.begin(observation, hidden)
                began = True
                baseline_policy = self._root_policy_factory()
                baseline_action = self._choose(baseline_policy, root.observation)
                baseline_position = self._backend.step(
                    root.search_id, baseline_action
                )
                baseline_leaf = self._complete_turn_with_v1(
                    baseline_position,
                    [baseline_action],
                    [semantic_action(root.observation, baseline_action)],
                    root_player=root_player,
                    root_turn=root_turn,
                )
                leaves = [
                    baseline_leaf,
                    *self._beam_plans(
                        root,
                        root_player=root_player,
                        root_turn=root_turn,
                    ),
                ]
                accepted = 0
                for leaf in leaves:
                    if leaf.boundary not in {"terminal", "turn_changed"}:
                        continue
                    plan = SemanticTurnPlan(leaf.semantic_sequence)
                    key = plan.semantic_key()
                    item = proposals.setdefault(
                        key,
                        {
                            "plan": plan,
                            "proposal_worlds": set(),
                            "heuristic_sum": 0.0,
                        },
                    )
                    item["proposal_worlds"].add(world_id)
                    item["heuristic_sum"] += float(leaf.heuristic)
                    accepted += 1
                worlds.append(
                    {
                        "world_id": world_id,
                        "opponent_deck_name": hidden.opponent_deck_name,
                        "plans": accepted,
                    }
                )
            except Exception as error:
                key = f"{type(error).__name__}: {error}"
                errors[key] += 1
                worlds.append(
                    {
                        "world_id": world_id,
                        "opponent_deck_name": hidden.opponent_deck_name,
                        "error": type(error).__name__,
                        "error_message": str(error),
                    }
                )
            finally:
                if began:
                    self._backend.end()
        ranked = sorted(
            proposals.values(),
            key=lambda item: (
                len(item["proposal_worlds"]),
                item["heuristic_sum"]
                / max(1, len(item["proposal_worlds"])),
                -len(item["plan"].actions),
                repr(item["plan"].semantic_key()),
            ),
            reverse=True,
        )[: self._plan_pool_size]
        return [item["plan"] for item in ranked], worlds, dict(sorted(errors.items()))

    def _baseline_from_root(
        self, root: SearchPosition, *, root_player: int, root_turn: int
    ) -> tuple[PlanLeaf, dict[str, Any]]:
        policy = self._root_policy_factory()
        action = self._choose(policy, root.observation)
        position = self._backend.step(root.search_id, action)
        leaf = self._complete_turn_with_v1(
            position,
            [action],
            [semantic_action(root.observation, action)],
            root_player=root_player,
            root_turn=root_turn,
        )
        return leaf, self._rollout(leaf, root_player=root_player)

    def _replay_from_root(
        self,
        root: SearchPosition,
        plan: SemanticTurnPlan,
        *,
        root_player: int,
        root_turn: int,
    ) -> tuple[PlanLeaf, dict[str, Any]]:
        current = root
        raw_sequence: list[tuple[int, ...]] = []
        semantic_sequence = []
        resolved = 0
        fallback_reason: str | None = None
        for directive in plan.actions:
            if not self._same_turn(
                current.observation, root_player, root_turn
            ):
                break
            action = resolve_semantic_action(current.observation, directive)
            if action is None:
                fallback_reason = "unresolved_directive"
                break
            semantic_sequence.append(semantic_action(current.observation, action))
            current = self._backend.step(current.search_id, action)
            raw_sequence.append(tuple(action))
            resolved += 1
        if self._same_turn(current.observation, root_player, root_turn):
            if fallback_reason is None:
                fallback_reason = "plan_exhausted"
            leaf = self._complete_turn_with_v1(
                current,
                raw_sequence,
                semantic_sequence,
                root_player=root_player,
                root_turn=root_turn,
            )
        else:
            leaf = self._leaf(
                current,
                raw_sequence,
                semantic_sequence,
                root_player=root_player,
                root_turn=root_turn,
            )
        replay = {
            "resolved_directives": resolved,
            "plan_directives": len(plan.actions),
            "resolved_fraction": round(resolved / max(1, len(plan.actions)), 6),
            "fallback": fallback_reason is not None,
            "fallback_reason": fallback_reason,
            "plan_boundary": leaf.boundary,
            "executed_sequence": [list(action) for action in leaf.sequence],
            **self._rollout(leaf, root_player=root_player),
        }
        return leaf, replay

    def _evaluate_plan_set(
        self,
        observation: dict,
        hidden_states: list[HiddenStateGuess],
        plans: list[SemanticTurnPlan],
        *,
        root_player: int,
        root_turn: int,
    ) -> tuple[list[dict], list[dict], dict[str, int]]:
        plan_returns: list[list[float]] = [[] for _ in plans]
        baseline_returns: list[list[float]] = [[] for _ in plans]
        replay_successes = [0] * len(plans)
        replay_fallbacks = [0] * len(plans)
        resolved_fractions: list[list[float]] = [[] for _ in plans]
        samples = []
        errors: Counter[str] = Counter()
        for world_id, hidden in enumerate(hidden_states):
            began = False
            try:
                root = self._backend.begin(observation, hidden)
                began = True
                _, baseline = self._baseline_from_root(
                    root, root_player=root_player, root_turn=root_turn
                )
                branches = []
                for index, plan in enumerate(plans):
                    _, replay = self._replay_from_root(
                        root,
                        plan,
                        root_player=root_player,
                        root_turn=root_turn,
                    )
                    value = float(replay["return"])
                    base = float(baseline["return"])
                    plan_returns[index].append(value)
                    baseline_returns[index].append(base)
                    replay_successes[index] += int(not replay["fallback"])
                    replay_fallbacks[index] += int(replay["fallback"])
                    resolved_fractions[index].append(
                        float(replay["resolved_fraction"])
                    )
                    branches.append({"plan_index": index, **replay})
                samples.append(
                    {
                        "world_id": world_id,
                        "opponent_deck_name": hidden.opponent_deck_name,
                        "baseline": baseline,
                        "branches": branches,
                    }
                )
            except Exception as error:
                key = f"{type(error).__name__}: {error}"
                errors[key] += 1
                samples.append(
                    {
                        "world_id": world_id,
                        "opponent_deck_name": hidden.opponent_deck_name,
                        "error": type(error).__name__,
                        "error_message": str(error),
                    }
                )
            finally:
                if began:
                    self._backend.end()
        summaries = []
        for index, plan in enumerate(plans):
            summary = paired_summary(plan_returns[index], baseline_returns[index])
            count = int(summary["effective_pairs"])
            summaries.append(
                {
                    "plan_index": index,
                    "plan": plan.to_dict(),
                    **summary,
                    "gain_lcb": round(
                        _gain_lcb(summary, self._selection_risk_multiplier), 6
                    ),
                    "replay_success_rate": round(
                        replay_successes[index] / max(1, count), 6
                    ),
                    "fallback_rate": round(
                        replay_fallbacks[index] / max(1, count), 6
                    ),
                    "mean_resolved_fraction": round(
                        statistics.mean(resolved_fractions[index])
                        if resolved_fractions[index]
                        else 0.0,
                        6,
                    ),
                }
            )
        return summaries, samples, dict(sorted(errors.items()))

    def evaluate(self, observation: dict) -> dict[str, Any]:
        """Propose on one hidden-world split and report gain only on the other."""
        self._branch_errors = {}
        root_player = int(observation["current"]["yourIndex"])
        root_turn = int(observation["current"]["turn"])
        hidden_states = [
            self._determinizer.sample(observation)
            for _ in range(
                self._proposal_determinizations
                + self._heldout_determinizations
            )
        ]
        proposal_hidden = hidden_states[: self._proposal_determinizations]
        heldout_hidden = hidden_states[self._proposal_determinizations :]
        plans, proposal_search, proposal_search_errors = self._proposal_plans(
            observation,
            proposal_hidden,
            root_player=root_player,
            root_turn=root_turn,
        )
        if not plans:
            raise RuntimeError("Semantic beam search proposed no complete plans")
        proposal_summaries, proposal_samples, proposal_eval_errors = (
            self._evaluate_plan_set(
                observation,
                proposal_hidden,
                plans,
                root_player=root_player,
                root_turn=root_turn,
            )
        )
        eligible = [
            summary
            for summary in proposal_summaries
            if int(summary["effective_pairs"]) > 0
        ]
        if not eligible:
            raise RuntimeError("No semantic plan completed proposal evaluation")
        selected = max(
            eligible,
            key=lambda item: (
                float(item["gain_lcb"]),
                float(item["replay_success_rate"]),
                float(item["paired_advantage"]),
                -int(item["plan_index"]),
            ),
        )
        selected_index = int(selected["plan_index"])
        selected_plan = plans[selected_index]
        heldout_summaries, heldout_samples, heldout_errors = (
            self._evaluate_plan_set(
                observation,
                heldout_hidden,
                [selected_plan],
                root_player=root_player,
                root_turn=root_turn,
            )
        )
        heldout = heldout_summaries[0]
        heldout["source_plan_index"] = selected_index
        heldout_gain = heldout.get("paired_advantage")
        heldout_stderr = heldout.get("paired_stderr")
        accepted = (
            heldout_gain is not None
            and heldout_stderr is not None
            and float(heldout_gain)
            - self._selection_risk_multiplier * float(heldout_stderr)
            > 0.0
        )
        errors = Counter()
        errors.update(proposal_search_errors)
        errors.update(proposal_eval_errors)
        errors.update(heldout_errors)
        return {
            "diagnostic_kind": "heldout_semantic_turn_plan",
            "root_player": root_player,
            "root_turn": root_turn,
            "proposal_determinizations": self._proposal_determinizations,
            "heldout_determinizations": self._heldout_determinizations,
            "plan_pool_size": self._plan_pool_size,
            "candidate_plans": len(plans),
            "selection_risk_multiplier": self._selection_risk_multiplier,
            "selected_plan_index": selected_index,
            "selected_plan": selected_plan.to_dict(),
            "proposal_selected": selected,
            "heldout_selected": heldout,
            "heldout_accepted": accepted,
            "proposal_search": proposal_search,
            "proposal_samples": proposal_samples,
            "heldout_samples": heldout_samples,
            "errors": dict(sorted(errors.items())),
            "branch_errors": dict(sorted(self._branch_errors.items())),
        }
