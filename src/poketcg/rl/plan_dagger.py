"""Closed-loop DAgger for the plan-conditioned Library-Out Executor."""

from __future__ import annotations

import random
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from poketcg.mcts import SearchPosition

from .action_space import deterministic_subset
from .data import BCExample, collate_bc
from .executor_data import EXECUTOR_CONDITION_SIZE, encode_executor_condition
from .features import EncodedDecision, build_feature_encoder
from .macro_oracle import MacroLeaf, MacroPlanOracleEvaluator
from .macro_plan import MacroPlanType, PlanOption, PlanProgress
from .model import action_space_version, build_model, encoder_version
from .semantic_plan import resolve_semantic_action, semantic_action


class PlanExecutorPolicy:
    """Inference wrapper for a V2 plan-conditioned Executor checkpoint."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        card_catalog: dict[int, object],
        attack_catalog: dict[int, object],
        device: str = "cpu",
    ) -> None:
        self._device = torch.device(device)
        saved = torch.load(
            checkpoint,
            map_location=self._device,
            weights_only=False,
        )
        if saved.get("checkpoint_kind") != "plan_conditioned_executor_v2":
            raise ValueError(
                "closed-loop Plan DAgger requires an Executor V2 checkpoint"
            )
        model_config = saved.get("model_config")
        if not isinstance(model_config, dict):
            raise TypeError("Executor checkpoint is missing model_config")
        if int(model_config.get("condition_feature_size", 0)) != (
            EXECUTOR_CONDITION_SIZE
        ):
            raise ValueError("Executor checkpoint uses another condition schema")
        if action_space_version(model_config) < 2:
            raise ValueError("Executor checkpoint must use Action Space V2")
        self._model = build_model(model_config).to(self._device)
        self._model.load_state_dict(saved["model_state_dict"])
        self._model.eval()
        self._encoder = build_feature_encoder(
            encoder_version(model_config),
            card_catalog,
            attack_catalog,
        )

    def encode(self, observation: dict) -> EncodedDecision:
        return self._encoder.encode(observation)

    def choose_action(
        self,
        observation: dict,
        plan: PlanOption,
        progress: PlanProgress,
    ) -> list[int]:
        """Choose a cardinality-valid action under the persistent plan."""
        decision = self.encode(observation)
        example = BCExample(
            decision=decision,
            action=list(range(decision.minimum)),
            value_target=0.0,
            player=int(observation["current"]["yourIndex"]),
            game=0,
        )
        batch = collate_bc([example])
        batch["condition"] = torch.tensor(
            [encode_executor_condition(plan.to_dict(), progress.to_dict())],
            dtype=torch.float32,
        )
        moved = {
            key: value.to(self._device)
            for key, value in batch.items()
        }
        with torch.no_grad():
            logits, _ = self._model(moved)
        return deterministic_subset(
            logits[0, : len(decision.options)].detach().cpu(),
            decision.minimum,
            decision.maximum,
        )


class ClosedLoopPlanDAggerEvaluator(MacroPlanOracleEvaluator):
    """Roll in with Student/Oracle mixtures and relabel every visited state.

    Plans are public and fixed at the root.  After each executed action, the
    teacher searches again from the resulting state under the same plan.  This
    avoids treating a trajectory generated before a student deviation as a
    valid label for the state reached after that deviation.
    """

    def __init__(
        self,
        *args,
        student_policy: Any,
        beta: float = 0.5,
        dagger_plan_limit: int = 4,
        rng: random.Random | None = None,
        **kwargs,
    ) -> None:
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if dagger_plan_limit <= 0:
            raise ValueError("dagger_plan_limit must be positive")
        super().__init__(*args, **kwargs)
        self._student_policy = student_policy
        self._beta = beta
        self._dagger_plan_limit = dagger_plan_limit
        self._rng = rng or random.Random()

    @staticmethod
    def _semantic_equal(
        observation: dict,
        left: Sequence[int],
        right: Sequence[int],
    ) -> bool:
        return (
            semantic_action(observation, left).semantic_key()
            == semantic_action(observation, right).semantic_key()
        )

    def _best_continuation(
        self,
        position: SearchPosition,
        plan: PlanOption,
        progress: PlanProgress,
        *,
        root_player: int,
    ) -> tuple[MacroLeaf, dict[str, Any]]:
        leaves = self._search_continuation(position, plan, progress)
        if not leaves or not leaves[0].steps:
            raise RuntimeError("closed-loop oracle produced no continuation")
        scored = [
            (leaf, self._rollout_macro_leaf(leaf, root_player=root_player))
            for leaf in leaves
        ]
        return max(
            scored,
            key=lambda item: (
                float(item[1]["return"]),
                item[0].heuristic,
                item[0].progress.plan_hits,
                -len(item[0].steps),
                item[0].sequence,
            ),
        )

    def _best_open_loop(
        self,
        root: SearchPosition,
        plan: PlanOption,
        *,
        root_player: int,
    ) -> tuple[MacroLeaf, dict[str, Any]]:
        scored = [
            (leaf, self._rollout_macro_leaf(leaf, root_player=root_player))
            for leaf in self._search_plan(root, plan)
        ]
        return max(
            scored,
            key=lambda item: (
                float(item[1]["return"]),
                item[0].heuristic,
                -len(item[0].steps),
                item[0].sequence,
            ),
        )

    def _roll_in_plan(
        self,
        root: SearchPosition,
        plan: PlanOption,
        *,
        root_player: int,
        baseline_return: float,
    ) -> dict[str, Any]:
        root_action = resolve_semantic_action(
            root.observation,
            plan.root_action,
        )
        if root_action is None:
            raise RuntimeError("Unable to resolve DAgger plan root action")
        progress = PlanProgress.start(root.observation)
        position, root_step, progress = self._advance(
            root,
            plan,
            progress,
            root_action,
        )
        executed_steps = [root_step]
        labels: list[dict[str, Any]] = []
        teacher_steps = 0
        student_steps = 0
        student_failures = 0
        disagreements = 0
        root_teacher_leaf: MacroLeaf | None = None
        root_teacher_rollout: dict[str, Any] | None = None

        while self._macro_boundary(
            position.observation,
            progress,
            plan,
        ) == "active":
            teacher_leaf, teacher_rollout = self._best_continuation(
                position,
                plan,
                progress,
                root_player=root_player,
            )
            if root_teacher_leaf is None:
                root_teacher_leaf = teacher_leaf
                root_teacher_rollout = teacher_rollout
            teacher_step = teacher_leaf.steps[0]
            teacher_action = list(teacher_step.action)
            student_action = list(
                self._student_policy.choose_action(
                    position.observation,
                    plan,
                    progress,
                )
            )
            disagreed = not self._semantic_equal(
                position.observation,
                student_action,
                teacher_action,
            )
            disagreements += int(disagreed)
            use_teacher = self._rng.random() < self._beta
            proposed = teacher_action if use_teacher else student_action
            source = "oracle" if use_teacher else "student"
            try:
                next_position, executed_step, next_progress = self._advance(
                    position,
                    plan,
                    progress,
                    proposed,
                )
            except Exception:
                if use_teacher:
                    raise
                student_failures += 1
                source = "oracle_fallback"
                proposed = teacher_action
                next_position, executed_step, next_progress = self._advance(
                    position,
                    plan,
                    progress,
                    proposed,
                )

            labels.append(
                {
                    "decision": self._encode(position.observation),
                    "progress_before": progress.to_dict(),
                    "target_action": teacher_action,
                    "target_semantic_action": (
                        teacher_step.semantic_action.to_dict()
                    ),
                    "student_action": student_action,
                    "student_semantic_action": semantic_action(
                        position.observation,
                        student_action,
                    ).to_dict(),
                    "executed_action": list(proposed),
                    "roll_in_source": source,
                    "semantic_disagreement": disagreed,
                    "teacher_continuation_return": float(
                        teacher_rollout["return"]
                    ),
                }
            )
            teacher_steps += int(source != "student")
            student_steps += int(source == "student")
            executed_steps.append(executed_step)
            position = next_position
            progress = next_progress

        mixed_leaf = self._macro_leaf(
            position,
            executed_steps,
            progress,
            plan,
        )
        mixed_rollout = self._rollout_macro_leaf(
            mixed_leaf,
            root_player=root_player,
        )
        if root_teacher_leaf is None or root_teacher_rollout is None:
            raise RuntimeError(
                "Plan DAgger reached no post-root Executor decisions"
            )
        mixed_return = float(mixed_rollout["return"])
        oracle_return = float(root_teacher_rollout["return"])
        return {
            "plan_id": plan.plan_id,
            "plan_type": plan.plan_type.value,
            "plan": plan.to_dict(),
            "labels": labels,
            "visited_states": len(labels),
            "student_steps": student_steps,
            "teacher_steps": teacher_steps,
            "student_step_failures": student_failures,
            "semantic_disagreements": disagreements,
            "semantic_disagreement_rate": (
                round(disagreements / len(labels), 6) if labels else 0.0
            ),
            "realized_beta": (
                round(teacher_steps / len(labels), 6) if labels else 0.0
            ),
            "mixed_sequence": [
                list(step.action) for step in executed_steps
            ],
            "oracle_sequence": [
                list(root_action),
                *[
                    list(step.action)
                    for step in root_teacher_leaf.steps
                ],
            ],
            "plan_boundary": mixed_leaf.boundary,
            "mixed_return": mixed_return,
            "oracle_return": oracle_return,
            "baseline_return": baseline_return,
            "mixed_advantage": round(mixed_return - baseline_return, 6),
            "oracle_advantage": round(oracle_return - baseline_return, 6),
            "oracle_gap": round(oracle_return - mixed_return, 6),
            "rollout_boundary": mixed_rollout["rollout_boundary"],
        }

    def evaluate(self, observation: dict) -> dict[str, Any]:
        self._branch_errors = {}
        root_player = int(observation["current"]["yourIndex"])
        root_turn = int(observation["current"]["turn"])
        baseline_policy = self._root_policy_factory()
        baseline_action = self._choose(baseline_policy, observation)
        plans = self._plan_generator.generate(
            observation,
            list(self._candidate_factory(observation)),
            baseline_action=baseline_action,
            maximum=max(self._plan_pool_size, self._dagger_plan_limit + 1),
        )
        selected_plans = [
            plan
            for plan in plans
            if plan.plan_type is not MacroPlanType.BASELINE_V1
        ][: self._dagger_plan_limit]
        if not selected_plans:
            raise RuntimeError("Plan DAgger found no non-baseline plan")

        samples: list[dict[str, Any]] = []
        errors: Counter[str] = Counter()
        for world_id in range(self._determinizations):
            hidden = self._determinizer.sample(observation)
            began = False
            try:
                root = self._backend.begin(observation, hidden)
                began = True
                baseline_plan = next(
                    plan
                    for plan in plans
                    if plan.plan_type is MacroPlanType.BASELINE_V1
                )
                baseline_leaf, baseline_rollout = self._best_open_loop(
                    root,
                    baseline_plan,
                    root_player=root_player,
                )
                plan_results = []
                for plan in selected_plans:
                    try:
                        plan_results.append(
                            self._roll_in_plan(
                                root,
                                plan,
                                root_player=root_player,
                                baseline_return=float(
                                    baseline_rollout["return"]
                                ),
                            )
                        )
                    except Exception as error:
                        self._record_branch_error(error)
                        plan_results.append(
                            {
                                "plan_id": plan.plan_id,
                                "plan_type": plan.plan_type.value,
                                "plan": plan.to_dict(),
                                "error": type(error).__name__,
                                "error_message": str(error),
                            }
                        )
                samples.append(
                    {
                        "determinization_id": world_id,
                        "opponent_deck_name": hidden.opponent_deck_name,
                        "baseline_return": float(
                            baseline_rollout["return"]
                        ),
                        "baseline_sequence": [
                            list(step.action)
                            for step in baseline_leaf.steps
                        ],
                        "plans": plan_results,
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

        valid_plans = [
            plan
            for sample in samples
            if "error" not in sample
            for plan in sample["plans"]
            if "error" not in plan
        ]
        if not valid_plans:
            raise RuntimeError("Every closed-loop Plan DAgger branch failed")
        visited = sum(int(plan["visited_states"]) for plan in valid_plans)
        if not visited:
            raise RuntimeError(
                "Plan DAgger reached no post-root Executor decisions"
            )
        teacher_steps = sum(int(plan["teacher_steps"]) for plan in valid_plans)
        disagreements = sum(
            int(plan["semantic_disagreements"]) for plan in valid_plans
        )

        def mean(name: str) -> float:
            return round(
                statistics.mean(float(plan[name]) for plan in valid_plans),
                6,
            )

        return {
            "diagnostic_kind": "closed_loop_plan_dagger_v1",
            "root_player": root_player,
            "root_turn": root_turn,
            "determinizations": self._determinizations,
            "effective_determinizations": sum(
                "error" not in sample for sample in samples
            ),
            "candidate_plans": len(selected_plans),
            "plans": [plan.to_dict() for plan in selected_plans],
            "configured_beta": self._beta,
            "visited_states": visited,
            "teacher_steps": teacher_steps,
            "student_steps": visited - teacher_steps,
            "semantic_disagreements": disagreements,
            "semantic_disagreement_rate": round(
                disagreements / visited, 6
            ),
            "realized_beta": round(teacher_steps / visited, 6),
            "mean_mixed_return": mean("mixed_return"),
            "mean_oracle_return": mean("oracle_return"),
            "mean_baseline_return": mean("baseline_return"),
            "mean_mixed_advantage": mean("mixed_advantage"),
            "mean_oracle_gap": mean("oracle_gap"),
            "errors": dict(sorted(errors.items())),
            "branch_errors": dict(sorted(self._branch_errors.items())),
            "samples": samples,
        }
