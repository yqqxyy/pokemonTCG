"""Oracle diagnostic for improvements that require several decisions in one turn."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from poketcg.mcts import (
    DeckDeterminizer,
    OfficialSearchBackend,
    PolicyValueMCTSAgent,
    SearchBackend,
    SearchPosition,
)

from .action_space import deterministic_subset, legal_action_set_count
from .advantage_candidates import root_candidates
from .paired_rollout import RootCandidate
from .semantic_plan import SemanticAction, SemanticTurnPlan, semantic_action


class TurnPolicy(Protocol):
    def choose_action(self, observation: dict) -> list[int]: ...


CandidateFactory = Callable[[dict], Sequence[RootCandidate]]


def turn_candidates(
    baseline_action: list[int],
    rule_scores: list[float],
    model_logits: list[float],
    selection: dict,
    *,
    maximum: int,
) -> list[RootCandidate]:
    """Build a small legal candidate set for any selection cardinality."""
    options = selection["option"]
    option_count = len(options)
    minimum = int(selection["minCount"])
    upper = int(selection["maxCount"])
    legal_action_set_count(option_count, minimum, upper)
    if len(rule_scores) != option_count or len(model_logits) != option_count:
        raise ValueError("Candidate score vectors must match the option count")
    if not minimum <= len(baseline_action) <= upper:
        raise ValueError("Baseline action violates selection cardinality")
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    if option_count > 1 and minimum == upper == 1:
        return root_candidates(
            baseline_action,
            rule_scores,
            model_logits,
            [int(option["type"]) for option in options],
            maximum=max(2, maximum),
        )[:maximum]

    actions: dict[tuple[int, ...], set[str]] = {
        tuple(sorted(baseline_action)): {"v1_choice"}
    }
    model_tensor = torch.tensor(model_logits)
    rule_tensor = torch.tensor(rule_scores)
    model_choice = tuple(
        deterministic_subset(model_tensor, minimum, upper)
    )
    rule_choice = tuple(
        deterministic_subset(rule_tensor, minimum, upper)
    )
    actions.setdefault(model_choice, set()).add("round0_subset")
    actions.setdefault(rule_choice, set()).add("rule_subset")

    model_order = sorted(range(option_count), key=model_logits.__getitem__, reverse=True)
    rule_order = sorted(range(option_count), key=rule_scores.__getitem__, reverse=True)
    combined_order = sorted(
        range(option_count),
        key=lambda index: (
            model_order.index(index) + rule_order.index(index),
            model_order.index(index),
        ),
    )
    for count in range(minimum, upper + 1):
        for name, order in (
            ("round0_cardinality", model_order),
            ("rule_cardinality", rule_order),
            ("combined_cardinality", combined_order),
        ):
            action = tuple(sorted(order[:count]))
            actions.setdefault(action, set()).add(name)

    return [
        RootCandidate(action, tuple(sorted(sources)))
        for action, sources in list(actions.items())[:maximum]
    ]


@dataclass(frozen=True, slots=True)
class PlanLeaf:
    """One cloned engine position reached by a within-turn action sequence."""

    position: SearchPosition
    sequence: tuple[tuple[int, ...], ...]
    semantic_sequence: tuple[SemanticAction, ...]
    boundary: str
    heuristic: float


def _sequence_distance(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]
) -> int:
    """Count changed decision steps, including a length mismatch."""
    shared = min(len(left), len(right))
    changed = sum(tuple(left[index]) != tuple(right[index]) for index in range(shared))
    return changed + abs(len(left) - len(right))


class TurnSynergyEvaluator:
    """Compare V1, one-step deviations, and full-turn beam search.

    The best full-turn branch is selected separately inside each sampled hidden
    world.  It is therefore an oracle diagnostic, not a deployable policy label.
    """

    def __init__(
        self,
        determinizer: DeckDeterminizer,
        root_policy_factory: Callable[[], TurnPolicy],
        opponent_policy_factory: Callable[[], TurnPolicy],
        candidate_factory: CandidateFactory,
        *,
        determinizations: int = 8,
        beam_width: int = 8,
        branch_width: int = 4,
        max_plan_steps: int = 32,
        max_rollout_steps: int = 1_000,
        value_policy: Any | None = None,
        backend: SearchBackend | None = None,
    ) -> None:
        positive = {
            "determinizations": determinizations,
            "beam_width": beam_width,
            "branch_width": branch_width,
            "max_plan_steps": max_plan_steps,
            "max_rollout_steps": max_rollout_steps,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self._determinizer = determinizer
        self._root_policy_factory = root_policy_factory
        self._opponent_policy_factory = opponent_policy_factory
        self._candidate_factory = candidate_factory
        self._determinizations = determinizations
        self._beam_width = beam_width
        self._branch_width = branch_width
        self._max_plan_steps = max_plan_steps
        self._max_rollout_steps = max_rollout_steps
        self._value_policy = value_policy
        self._backend = backend or OfficialSearchBackend()
        self._branch_errors: dict[str, int] = {}

    def _record_branch_error(self, error: Exception) -> None:
        key = f"{type(error).__name__}: {error}"
        self._branch_errors[key] = self._branch_errors.get(key, 0) + 1

    @staticmethod
    def _choose(policy: TurnPolicy, observation: dict) -> list[int]:
        deterministic = getattr(policy, "choose_deterministic_action", None)
        if callable(deterministic):
            return list(deterministic(observation))
        return list(policy.choose_action(observation))

    @staticmethod
    def _terminal_value(observation: dict, root_player: int) -> float | None:
        return PolicyValueMCTSAgent._terminal_value(observation, root_player)

    @staticmethod
    def _same_turn(observation: dict, root_player: int, root_turn: int) -> bool:
        state = observation["current"]
        return (
            int(state["result"]) == -1
            and int(state["yourIndex"]) == root_player
            and int(state["turn"]) == root_turn
        )

    def _heuristic(self, observation: dict, root_player: int) -> float:
        terminal = self._terminal_value(observation, root_player)
        if terminal is not None:
            return terminal
        if self._value_policy is None:
            return 0.0
        evaluation = self._value_policy.evaluate(observation)
        acting_player = int(observation["current"]["yourIndex"])
        value = float(evaluation.value)
        return value if acting_player == root_player else -value

    def _leaf(
        self,
        position: SearchPosition,
        sequence: Sequence[Sequence[int]],
        semantic_sequence: Sequence[SemanticAction],
        *,
        root_player: int,
        root_turn: int,
        capped: bool = False,
    ) -> PlanLeaf:
        observation = position.observation
        terminal = self._terminal_value(observation, root_player)
        if terminal is not None:
            boundary = "terminal"
        elif not self._same_turn(observation, root_player, root_turn):
            boundary = "turn_changed"
        elif capped:
            boundary = "max_plan_steps"
        else:
            boundary = "active"
        return PlanLeaf(
            position=position,
            sequence=tuple(tuple(action) for action in sequence),
            semantic_sequence=tuple(semantic_sequence),
            boundary=boundary,
            heuristic=self._heuristic(observation, root_player),
        )

    def _complete_turn_with_v1(
        self,
        position: SearchPosition,
        sequence: Sequence[Sequence[int]],
        semantic_sequence: Sequence[SemanticAction],
        *,
        root_player: int,
        root_turn: int,
    ) -> PlanLeaf:
        policy = self._root_policy_factory()
        actions = [tuple(action) for action in sequence]
        directives = list(semantic_sequence)
        current = position
        while len(actions) < self._max_plan_steps and self._same_turn(
            current.observation, root_player, root_turn
        ):
            action = self._choose(policy, current.observation)
            directives.append(semantic_action(current.observation, action))
            current = self._backend.step(current.search_id, action)
            actions.append(tuple(action))
        return self._leaf(
            current,
            actions,
            directives,
            root_player=root_player,
            root_turn=root_turn,
            capped=self._same_turn(current.observation, root_player, root_turn),
        )

    def _rollout(self, leaf: PlanLeaf, *, root_player: int) -> dict[str, Any]:
        terminal = self._terminal_value(leaf.position.observation, root_player)
        if terminal is not None:
            return {"return": terminal, "rollout_steps": 0, "rollout_boundary": "terminal"}
        policies = {
            root_player: self._root_policy_factory(),
            1 - root_player: self._opponent_policy_factory(),
        }
        current = leaf.position
        steps = 0
        while steps < self._max_rollout_steps:
            observation = current.observation
            terminal = self._terminal_value(observation, root_player)
            if terminal is not None:
                return {
                    "return": terminal,
                    "rollout_steps": steps,
                    "rollout_boundary": "terminal",
                }
            player = int(observation["current"]["yourIndex"])
            action = self._choose(policies[player], observation)
            current = self._backend.step(current.search_id, action)
            steps += 1
        return {
            "return": round(self._heuristic(current.observation, root_player), 6),
            "rollout_steps": steps,
            "rollout_boundary": "value_bootstrap",
        }

    def _beam_plans(
        self, root: SearchPosition, *, root_player: int, root_turn: int
    ) -> list[PlanLeaf]:
        active = [
            self._leaf(
                root,
                (),
                (),
                root_player=root_player,
                root_turn=root_turn,
            )
        ]
        completed: list[PlanLeaf] = []
        for _ in range(self._max_plan_steps):
            children: list[PlanLeaf] = []
            for leaf in active:
                candidates = list(self._candidate_factory(leaf.position.observation))
                unique: list[RootCandidate] = []
                seen: set[tuple[int, ...]] = set()
                for candidate in candidates:
                    if candidate.action not in seen:
                        unique.append(candidate)
                        seen.add(candidate.action)
                    if len(unique) == self._branch_width:
                        break
                if not unique:
                    raise RuntimeError("Turn candidate factory returned no legal actions")
                for candidate in unique:
                    try:
                        position = self._backend.step(
                            leaf.position.search_id, list(candidate.action)
                        )
                    except Exception as error:
                        self._record_branch_error(error)
                        continue
                    child = self._leaf(
                        position,
                        (*leaf.sequence, candidate.action),
                        (
                            *leaf.semantic_sequence,
                            semantic_action(
                                leaf.position.observation, candidate.action
                            ),
                        ),
                        root_player=root_player,
                        root_turn=root_turn,
                    )
                    if child.boundary == "active":
                        children.append(child)
                    else:
                        completed.append(child)
            if not children:
                active = []
                break
            active = sorted(
                children,
                key=lambda item: (item.heuristic, -len(item.sequence), item.sequence),
                reverse=True,
            )[: self._beam_width]
            completed = sorted(
                completed,
                key=lambda item: (item.heuristic, -len(item.sequence), item.sequence),
                reverse=True,
            )[: self._beam_width]
        completed.extend(
            self._leaf(
                leaf.position,
                leaf.sequence,
                leaf.semantic_sequence,
                root_player=root_player,
                root_turn=root_turn,
                capped=True,
            )
            for leaf in active
        )
        unique_leaves: dict[tuple[tuple[int, ...], ...], PlanLeaf] = {}
        for leaf in completed:
            unique_leaves.setdefault(leaf.sequence, leaf)
        return sorted(
            unique_leaves.values(),
            key=lambda item: (item.heuristic, -len(item.sequence), item.sequence),
            reverse=True,
        )[: self._beam_width]

    def _evaluate_world(
        self, root: SearchPosition, *, root_player: int, root_turn: int
    ) -> dict[str, Any]:
        baseline_policy = self._root_policy_factory()
        baseline_action = self._choose(baseline_policy, root.observation)
        baseline_directive = semantic_action(root.observation, baseline_action)
        baseline_position = self._backend.step(root.search_id, baseline_action)
        baseline_leaf = self._complete_turn_with_v1(
            baseline_position,
            [baseline_action],
            [baseline_directive],
            root_player=root_player,
            root_turn=root_turn,
        )
        baseline_rollout = self._rollout(baseline_leaf, root_player=root_player)
        baseline_result = {
            "root_action": baseline_action,
            "sequence": [list(action) for action in baseline_leaf.sequence],
            "semantic_plan": SemanticTurnPlan(
                baseline_leaf.semantic_sequence
            ).to_dict(),
            "plan_boundary": baseline_leaf.boundary,
            **baseline_rollout,
        }

        root_candidates = list(self._candidate_factory(root.observation))
        if tuple(baseline_action) not in {candidate.action for candidate in root_candidates}:
            root_candidates.insert(
                0, RootCandidate(tuple(baseline_action), ("v1_fallback",))
            )
        one_step = []
        one_step_by_root: dict[tuple[int, ...], dict[str, Any]] = {}
        for candidate in root_candidates[: self._branch_width]:
            try:
                position = self._backend.step(
                    root.search_id, list(candidate.action)
                )
            except Exception as error:
                self._record_branch_error(error)
                continue
            leaf = self._complete_turn_with_v1(
                position,
                [candidate.action],
                [semantic_action(root.observation, candidate.action)],
                root_player=root_player,
                root_turn=root_turn,
            )
            result = {
                "root_action": list(candidate.action),
                "sources": list(candidate.sources),
                "sequence": [list(action) for action in leaf.sequence],
                "semantic_plan": SemanticTurnPlan(
                    leaf.semantic_sequence
                ).to_dict(),
                "plan_boundary": leaf.boundary,
                **self._rollout(leaf, root_player=root_player),
            }
            one_step.append(result)
            one_step_by_root[candidate.action] = result
        best_one_step = max(
            [baseline_result, *one_step],
            key=lambda item: (float(item["return"]), -len(item["sequence"])),
        )

        searched = []
        for leaf in self._beam_plans(
            root, root_player=root_player, root_turn=root_turn
        ):
            result = {
                "root_action": list(leaf.sequence[0]) if leaf.sequence else [],
                "sequence": [list(action) for action in leaf.sequence],
                "semantic_plan": SemanticTurnPlan(
                    leaf.semantic_sequence
                ).to_dict(),
                "plan_boundary": leaf.boundary,
                "heuristic": round(leaf.heuristic, 6),
                **self._rollout(leaf, root_player=root_player),
            }
            searched.append(result)
        best_full_turn = max(
            [baseline_result, *one_step, *searched],
            key=lambda item: (float(item["return"]), -len(item["sequence"])),
        )
        reference = one_step_by_root.get(
            tuple(best_full_turn["root_action"]), baseline_result
        )
        joint_deviation_count = _sequence_distance(
            best_full_turn["sequence"], reference["sequence"]
        )
        baseline_return = float(baseline_result["return"])
        one_step_return = float(best_one_step["return"])
        full_return = float(best_full_turn["return"])
        return {
            "baseline": baseline_result,
            "one_step_candidates": one_step,
            "searched_plans": searched,
            "best_one_step": best_one_step,
            "best_full_turn": best_full_turn,
            "baseline_return": baseline_return,
            "best_one_step_return": one_step_return,
            "best_full_turn_return": full_return,
            "one_step_gain": round(one_step_return - baseline_return, 6),
            "full_turn_gain": round(full_return - baseline_return, 6),
            "synergy_gain": round(full_return - one_step_return, 6),
            "joint_deviation_count": joint_deviation_count,
            "joint_rescue": (
                one_step_return <= baseline_return
                and full_return > baseline_return
                and joint_deviation_count > 0
            ),
        }

    def evaluate(self, observation: dict) -> dict[str, Any]:
        """Run hidden-world oracle comparisons and summarize synergy frequency."""
        self._branch_errors = {}
        root_player = int(observation["current"]["yourIndex"])
        root_turn = int(observation["current"]["turn"])
        samples = []
        errors: dict[str, int] = {}
        for determinization_id in range(self._determinizations):
            hidden = self._determinizer.sample(observation)
            began = False
            try:
                root = self._backend.begin(observation, hidden)
                began = True
                result = self._evaluate_world(
                    root, root_player=root_player, root_turn=root_turn
                )
            except Exception as error:
                name = type(error).__name__
                errors[name] = errors.get(name, 0) + 1
                samples.append(
                    {
                        "determinization_id": determinization_id,
                        "opponent_deck_name": hidden.opponent_deck_name,
                        "error": name,
                        "error_message": str(error),
                    }
                )
            else:
                samples.append(
                    {
                        "determinization_id": determinization_id,
                        "opponent_deck_name": hidden.opponent_deck_name,
                        **result,
                    }
                )
            finally:
                if began:
                    self._backend.end()

        valid = [sample for sample in samples if "error" not in sample]
        if not valid:
            raise RuntimeError("Every turn-synergy determinization failed")

        def mean(name: str) -> float:
            return round(
                sum(float(sample[name]) for sample in valid) / len(valid), 6
            )

        return {
            "root_player": root_player,
            "root_turn": root_turn,
            "determinizations": self._determinizations,
            "effective_determinizations": len(valid),
            "beam_width": self._beam_width,
            "branch_width": self._branch_width,
            "max_plan_steps": self._max_plan_steps,
            "mean_baseline_return": mean("baseline_return"),
            "mean_best_one_step_return": mean("best_one_step_return"),
            "mean_best_full_turn_return": mean("best_full_turn_return"),
            "mean_one_step_gain": mean("one_step_gain"),
            "mean_full_turn_gain": mean("full_turn_gain"),
            "mean_synergy_gain": mean("synergy_gain"),
            "hidden_synergy_rate": round(
                sum(float(sample["synergy_gain"]) > 0 for sample in valid) / len(valid),
                6,
            ),
            "joint_rescue_rate": round(
                sum(bool(sample["joint_rescue"]) for sample in valid) / len(valid),
                6,
            ),
            "different_continuation_rate": round(
                sum(int(sample["joint_deviation_count"]) > 0 for sample in valid)
                / len(valid),
                6,
            ),
            "errors": dict(sorted(errors.items())),
            "branch_errors": dict(sorted(self._branch_errors.items())),
            "samples": samples,
        }
