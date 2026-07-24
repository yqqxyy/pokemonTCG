"""Full-turn macro-plan oracle and executor-teacher trajectory collector."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from poketcg.mcts import SearchPosition

from .macro_plan import (
    HeuristicPlanExecutor,
    MacroPlanGenerator,
    MacroPlanType,
    PlanOption,
    PlanProgress,
)
from .paired_rollout import RootCandidate, paired_summary
from .semantic_plan import SemanticAction, resolve_semantic_action, semantic_action
from .turn_synergy import PlanLeaf, TurnSynergyEvaluator, _sequence_distance


@dataclass(frozen=True, slots=True)
class MacroStep:
    """One public decision made while a macro plan owns the turn."""

    decision: dict[str, Any]
    action: tuple[int, ...]
    semantic_action: SemanticAction
    progress_before: PlanProgress
    progress_after: PlanProgress

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "action": list(self.action),
            "semantic_action": self.semantic_action.to_dict(),
            "progress_before": self.progress_before.to_dict(),
            "progress_after": self.progress_after.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class MacroLeaf:
    """One plan-owned search branch ending at a turn boundary or search cap."""

    position: SearchPosition
    steps: tuple[MacroStep, ...]
    progress: PlanProgress
    boundary: str
    heuristic: float

    @property
    def sequence(self) -> tuple[tuple[int, ...], ...]:
        return tuple(step.action for step in self.steps)


class MacroPlanOracleEvaluator(TurnSynergyEvaluator):
    """Allocate a complete beam to each public macro candidate.

    The best continuation is still selected separately in every hidden world,
    so this evaluator measures an upper bound and creates search-teacher data.
    It is not a deployable plan selector.
    """

    def __init__(
        self,
        *args,
        plan_generator: MacroPlanGenerator,
        decision_encoder: Callable[[dict], Any],
        plan_pool_size: int = 8,
        alignment_weight: float = 0.05,
        executor: HeuristicPlanExecutor | None = None,
        **kwargs,
    ) -> None:
        if plan_pool_size <= 0:
            raise ValueError("plan_pool_size must be positive")
        if alignment_weight < 0:
            raise ValueError("alignment_weight must be non-negative")
        super().__init__(*args, **kwargs)
        self._plan_generator = plan_generator
        self._decision_encoder = decision_encoder
        self._plan_pool_size = plan_pool_size
        self._alignment_weight = alignment_weight
        self._executor = executor or HeuristicPlanExecutor(plan_generator)

    def _encode(self, observation: dict) -> dict[str, Any]:
        encoded = self._decision_encoder(observation)
        to_dict = getattr(encoded, "to_dict", None)
        return to_dict() if callable(to_dict) else dict(encoded)

    def _macro_boundary(
        self,
        observation: dict,
        progress: PlanProgress,
        plan: PlanOption,
        *,
        capped: bool = False,
    ) -> str:
        terminal = self._terminal_value(observation, progress.owner_player)
        if terminal is not None:
            return "terminal"
        if not progress.active(observation, plan):
            current = observation["current"]
            if (
                int(current["yourIndex"]) != progress.owner_player
                or int(current["turn"]) != progress.start_turn
            ):
                return "turn_changed"
            return "plan_limit"
        return "max_plan_steps" if capped else "active"

    def _macro_leaf(
        self,
        position: SearchPosition,
        steps: Sequence[MacroStep],
        progress: PlanProgress,
        plan: PlanOption,
        *,
        capped: bool = False,
    ) -> MacroLeaf:
        boundary = self._macro_boundary(
            position.observation, progress, plan, capped=capped
        )
        base = self._heuristic(position.observation, progress.owner_player)
        alignment = progress.plan_hits / max(1, progress.decisions)
        return MacroLeaf(
            position=position,
            steps=tuple(steps),
            progress=progress,
            boundary=boundary,
            heuristic=base + self._alignment_weight * alignment,
        )

    def _advance(
        self,
        position: SearchPosition,
        plan: PlanOption,
        progress: PlanProgress,
        action: Sequence[int],
    ) -> tuple[SearchPosition, MacroStep, PlanProgress]:
        raw = tuple(int(index) for index in action)
        directive = semantic_action(position.observation, raw)
        updated = progress.advance(plan, directive)
        step = MacroStep(
            decision=self._encode(position.observation),
            action=raw,
            semantic_action=directive,
            progress_before=progress,
            progress_after=updated,
        )
        return self._backend.step(position.search_id, list(raw)), step, updated

    def _search_plan(
        self,
        root: SearchPosition,
        plan: PlanOption,
    ) -> list[MacroLeaf]:
        action = resolve_semantic_action(root.observation, plan.root_action)
        if action is None:
            raise RuntimeError("Unable to resolve macro root action")
        progress = PlanProgress.start(root.observation)
        position, step, progress = self._advance(
            root, plan, progress, action
        )
        initial = self._macro_leaf(position, [step], progress, plan)
        if initial.boundary != "active":
            return [initial]
        return self._search_continuation(
            position,
            plan,
            progress,
            prefix_steps=(step,),
        )

    def _search_continuation(
        self,
        position: SearchPosition,
        plan: PlanOption,
        progress: PlanProgress,
        *,
        prefix_steps: Sequence[MacroStep] = (),
    ) -> list[MacroLeaf]:
        """Search again from one state actually visited by an Executor.

        Unlike :meth:`_search_plan`, this method does not replay the root
        action.  That distinction is what makes it usable as a closed-loop
        DAgger oracle after a student deviation.
        """
        initial = self._macro_leaf(
            position,
            prefix_steps,
            progress,
            plan,
        )
        if initial.boundary != "active":
            return [initial]
        if plan.plan_type is MacroPlanType.BASELINE_V1:
            policy = self._root_policy_factory()
            current = initial
            steps = list(initial.steps)
            while current.boundary == "active":
                action = self._choose(policy, current.position.observation)
                position, step, progress = self._advance(
                    current.position,
                    plan,
                    current.progress,
                    action,
                )
                steps.append(step)
                current = self._macro_leaf(
                    position, steps, progress, plan
                )
            return [current]
        active = [initial]
        completed: list[MacroLeaf] = []
        remaining = max(0, plan.maximum_steps - progress.decisions)
        for _ in range(remaining):
            children: list[MacroLeaf] = []
            for leaf in active:
                candidates = list(
                    self._candidate_factory(leaf.position.observation)
                )
                ranked = self._executor.rank(
                    leaf.position.observation,
                    plan,
                    leaf.progress,
                    candidates,
                )
                unique: list[RootCandidate] = []
                seen: set[tuple[int, ...]] = set()
                for candidate in ranked:
                    if candidate.action not in seen:
                        unique.append(candidate)
                        seen.add(candidate.action)
                    if len(unique) == self._branch_width:
                        break
                if not unique:
                    raise RuntimeError(
                        "Plan executor returned no legal continuation"
                    )
                for candidate in unique:
                    try:
                        position, step, progress = self._advance(
                            leaf.position,
                            plan,
                            leaf.progress,
                            candidate.action,
                        )
                    except Exception as error:
                        self._record_branch_error(error)
                        continue
                    child = self._macro_leaf(
                        position,
                        (*leaf.steps, step),
                        progress,
                        plan,
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
                key=lambda item: (
                    item.heuristic,
                    item.progress.plan_hits,
                    -len(item.steps),
                    item.sequence,
                ),
                reverse=True,
            )[: self._beam_width]
            completed = sorted(
                completed,
                key=lambda item: (
                    item.heuristic,
                    item.progress.plan_hits,
                    -len(item.steps),
                    item.sequence,
                ),
                reverse=True,
            )[: self._beam_width]
        completed.extend(
            self._macro_leaf(
                leaf.position,
                leaf.steps,
                leaf.progress,
                plan,
                capped=True,
            )
            for leaf in active
        )
        unique_leaves: dict[tuple[tuple[int, ...], ...], MacroLeaf] = {}
        for leaf in completed:
            unique_leaves.setdefault(leaf.sequence, leaf)
        return sorted(
            unique_leaves.values(),
            key=lambda item: (
                item.heuristic,
                item.progress.plan_hits,
                -len(item.steps),
                item.sequence,
            ),
            reverse=True,
        )[: self._beam_width]

    def _root_only(
        self,
        root: SearchPosition,
        plan: PlanOption,
        *,
        root_player: int,
        root_turn: int,
    ) -> tuple[PlanLeaf, dict[str, Any]]:
        action = resolve_semantic_action(root.observation, plan.root_action)
        if action is None:
            raise RuntimeError("Unable to resolve root-only macro action")
        position = self._backend.step(root.search_id, action)
        leaf = self._complete_turn_with_v1(
            position,
            [action],
            [semantic_action(root.observation, action)],
            root_player=root_player,
            root_turn=root_turn,
        )
        return leaf, {
            "sequence": [list(item) for item in leaf.sequence],
            "plan_boundary": leaf.boundary,
            **self._rollout(leaf, root_player=root_player),
        }

    def _rollout_macro_leaf(
        self, leaf: MacroLeaf, *, root_player: int
    ) -> dict[str, Any]:
        return self._rollout(
            PlanLeaf(
                position=leaf.position,
                sequence=leaf.sequence,
                semantic_sequence=tuple(
                    step.semantic_action for step in leaf.steps
                ),
                boundary=leaf.boundary,
                heuristic=leaf.heuristic,
            ),
            root_player=root_player,
        )

    def _trajectory(
        self,
        plan: PlanOption,
        leaf: MacroLeaf,
        rollout: dict[str, Any],
        *,
        baseline_return: float,
        root_only_return: float,
    ) -> dict[str, Any]:
        value = float(rollout["return"])
        return {
            "schema_version": 2,
            "plan_id": plan.plan_id,
            "plan_type": plan.plan_type.value,
            "plan": plan.to_dict(),
            "steps": [step.to_dict() for step in leaf.steps],
            "decision_count": len(leaf.steps),
            "plan_hits": leaf.progress.plan_hits,
            "plan_boundary": leaf.boundary,
            "return": value,
            "baseline_return": baseline_return,
            "root_only_return": root_only_return,
            "paired_advantage": round(value - baseline_return, 6),
            "macro_synergy": round(value - root_only_return, 6),
            **{
                key: rollout[key]
                for key in ("rollout_steps", "rollout_boundary")
            },
        }

    def _evaluate_world(
        self,
        root: SearchPosition,
        plans: Sequence[PlanOption],
        *,
        root_player: int,
        root_turn: int,
    ) -> dict[str, Any]:
        baseline_plan = plans[0]
        if baseline_plan.plan_type is not MacroPlanType.BASELINE_V1:
            raise AssertionError("The first macro candidate must be baseline_v1")
        baseline_leaf = self._search_plan(root, baseline_plan)[0]
        baseline_rollout = self._rollout_macro_leaf(
            baseline_leaf, root_player=root_player
        )
        baseline = {
            "root_action": list(baseline_leaf.steps[0].action),
            "sequence": [list(item) for item in baseline_leaf.sequence],
            "plan_boundary": baseline_leaf.boundary,
            **baseline_rollout,
        }
        baseline_return = float(baseline_rollout["return"])
        plan_results = []
        for plan_index, plan in enumerate(plans):
            if plan.plan_type is MacroPlanType.BASELINE_V1:
                root_leaf = PlanLeaf(
                    position=baseline_leaf.position,
                    sequence=baseline_leaf.sequence,
                    semantic_sequence=tuple(
                        step.semantic_action
                        for step in baseline_leaf.steps
                    ),
                    boundary=baseline_leaf.boundary,
                    heuristic=baseline_leaf.heuristic,
                )
                root_only = dict(baseline)
                root_return = baseline_return
                leaves = [baseline_leaf]
            else:
                root_leaf, root_only = self._root_only(
                    root,
                    plan,
                    root_player=root_player,
                    root_turn=root_turn,
                )
                root_return = float(root_only["return"])
                leaves = self._search_plan(root, plan)
            trajectories = []
            for leaf in leaves:
                rollout = (
                    baseline_rollout
                    if plan.plan_type is MacroPlanType.BASELINE_V1
                    else self._rollout_macro_leaf(
                        leaf, root_player=root_player
                    )
                )
                trajectories.append(
                    self._trajectory(
                        plan,
                        leaf,
                        rollout,
                        baseline_return=baseline_return,
                        root_only_return=root_return,
                    )
                )
            if not trajectories:
                raise RuntimeError("Macro search produced no trajectories")
            best = max(
                trajectories,
                key=lambda item: (
                    float(item["return"]),
                    float(item["macro_synergy"]),
                    -int(item["decision_count"]),
                ),
            )
            plan_results.append(
                {
                    "plan_index": plan_index,
                    "plan_id": plan.plan_id,
                    "plan_type": plan.plan_type.value,
                    "plan": plan.to_dict(),
                    "root_only": root_only,
                    "root_only_return": root_return,
                    "root_only_gain": round(
                        root_return - baseline_return, 6
                    ),
                    "root_only_sequence": [
                        list(item) for item in root_leaf.sequence
                    ],
                    "trajectories": trajectories,
                    "best_trajectory": best,
                    "best_macro_return": float(best["return"]),
                    "best_macro_gain": round(
                        float(best["return"]) - baseline_return, 6
                    ),
                    "macro_synergy": round(
                        float(best["return"]) - root_return, 6
                    ),
                }
            )
        best_root = max(
            plan_results,
            key=lambda item: (
                float(item["root_only_return"]),
                -int(item["plan_index"]),
            ),
        )
        best_macro = max(
            plan_results,
            key=lambda item: (
                float(item["best_macro_return"]),
                float(item["macro_synergy"]),
                -int(item["plan_index"]),
            ),
        )
        best_sequence = best_macro["best_trajectory"]["steps"]
        macro_raw_sequence = [
            tuple(int(index) for index in step["action"])
            for step in best_sequence
        ]
        root_raw_sequence = [
            tuple(int(index) for index in item)
            for item in best_macro["root_only_sequence"]
        ]
        deviation_count = _sequence_distance(
            macro_raw_sequence, root_raw_sequence
        )
        root_gain = float(best_root["root_only_return"]) - baseline_return
        macro_gain = float(best_macro["best_macro_return"]) - baseline_return
        synergy = float(best_macro["best_macro_return"]) - float(
            best_root["root_only_return"]
        )
        return {
            "baseline": baseline,
            "baseline_sequence": [
                list(item) for item in baseline_leaf.sequence
            ],
            "plans": plan_results,
            "best_root_only": best_root,
            "best_macro": best_macro,
            "baseline_return": baseline_return,
            "best_one_step_return": float(best_root["root_only_return"]),
            "best_full_turn_return": float(best_macro["best_macro_return"]),
            "one_step_gain": round(root_gain, 6),
            "full_turn_gain": round(macro_gain, 6),
            "synergy_gain": round(synergy, 6),
            "joint_deviation_count": deviation_count,
            "joint_rescue": (
                root_gain <= 0.0 and macro_gain > 0.0 and deviation_count > 0
            ),
        }

    def evaluate(self, observation: dict) -> dict[str, Any]:
        self._branch_errors = {}
        root_player = int(observation["current"]["yourIndex"])
        root_turn = int(observation["current"]["turn"])
        baseline_policy = self._root_policy_factory()
        baseline_action = self._choose(baseline_policy, observation)
        root_candidates = list(self._candidate_factory(observation))
        plans = self._plan_generator.generate(
            observation,
            root_candidates,
            baseline_action=baseline_action,
            maximum=self._plan_pool_size,
        )
        if not plans:
            raise RuntimeError("Macro plan generator returned no candidates")

        samples = []
        errors: Counter[str] = Counter()
        macro_returns: list[list[float]] = [[] for _ in plans]
        root_returns: list[list[float]] = [[] for _ in plans]
        baseline_returns: list[list[float]] = [[] for _ in plans]
        best_types: Counter[str] = Counter()
        for world_id in range(self._determinizations):
            hidden = self._determinizer.sample(observation)
            began = False
            try:
                root = self._backend.begin(observation, hidden)
                began = True
                result = self._evaluate_world(
                    root,
                    plans,
                    root_player=root_player,
                    root_turn=root_turn,
                )
                for item in result["plans"]:
                    index = int(item["plan_index"])
                    macro_returns[index].append(
                        float(item["best_macro_return"])
                    )
                    root_returns[index].append(
                        float(item["root_only_return"])
                    )
                    baseline_returns[index].append(
                        float(result["baseline_return"])
                    )
                best_types[result["best_macro"]["plan_type"]] += 1
                samples.append(
                    {
                        "determinization_id": world_id,
                        "opponent_deck_name": hidden.opponent_deck_name,
                        **result,
                    }
                )
            except Exception as error:
                key = f"{type(error).__name__}: {error}"
                errors[key] += 1
                samples.append(
                    {
                        "determinization_id": world_id,
                        "opponent_deck_name": hidden.opponent_deck_name,
                        "error": type(error).__name__,
                        "error_message": str(error),
                    }
                )
            finally:
                if began:
                    self._backend.end()
        valid = [sample for sample in samples if "error" not in sample]
        if not valid:
            raise RuntimeError("Every macro-plan determinization failed")

        plan_summaries = []
        for index, plan in enumerate(plans):
            macro = paired_summary(
                macro_returns[index], baseline_returns[index]
            )
            root_only = paired_summary(
                root_returns[index], baseline_returns[index]
            )
            plan_summaries.append(
                {
                    "plan_index": index,
                    "plan": plan.to_dict(),
                    "macro": macro,
                    "root_only": root_only,
                    "mean_macro_synergy": round(
                        statistics.mean(
                            macro_value - root_value
                            for macro_value, root_value in zip(
                                macro_returns[index],
                                root_returns[index],
                                strict=True,
                            )
                        ),
                        6,
                    ),
                }
            )

        def mean(name: str) -> float:
            return round(
                statistics.mean(float(sample[name]) for sample in valid), 6
            )

        return {
            "diagnostic_kind": "macro_plan_oracle_v2_libraryout",
            "root_player": root_player,
            "root_turn": root_turn,
            "determinizations": self._determinizations,
            "effective_determinizations": len(valid),
            "plan_pool_size": self._plan_pool_size,
            "candidate_plans": len(plans),
            "plans": [plan.to_dict() for plan in plans],
            "plan_summaries": plan_summaries,
            "mean_baseline_return": mean("baseline_return"),
            "mean_best_one_step_return": mean("best_one_step_return"),
            "mean_best_full_turn_return": mean("best_full_turn_return"),
            "mean_one_step_gain": mean("one_step_gain"),
            "mean_full_turn_gain": mean("full_turn_gain"),
            "mean_synergy_gain": mean("synergy_gain"),
            "hidden_synergy_rate": round(
                sum(float(sample["synergy_gain"]) > 0 for sample in valid)
                / len(valid),
                6,
            ),
            "joint_rescue_rate": round(
                sum(bool(sample["joint_rescue"]) for sample in valid)
                / len(valid),
                6,
            ),
            "different_continuation_rate": round(
                sum(
                    int(sample["joint_deviation_count"]) > 0
                    for sample in valid
                )
                / len(valid),
                6,
            ),
            "best_plan_type_counts": dict(sorted(best_types.items())),
            "errors": dict(sorted(errors.items())),
            "branch_errors": dict(sorted(self._branch_errors.items())),
            "samples": samples,
        }
