"""Paired one-step-deviation rollouts from one official search root."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import stdev
from typing import Any, Protocol

from poketcg.mcts import (
    DeckDeterminizer,
    OfficialSearchBackend,
    PolicyValueMCTSAgent,
    SearchBackend,
    SearchPosition,
)


class RolloutPolicy(Protocol):
    def choose_action(self, observation: dict) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class RootCandidate:
    action: tuple[int, ...]
    sources: tuple[str, ...]


def paired_summary(candidate_returns: list[float], baseline_returns: list[float]) -> dict:
    """Summarize paired differences, keeping invalid branches out of both arms."""
    if len(candidate_returns) != len(baseline_returns):
        raise ValueError("paired return vectors must have equal length")
    differences = [
        candidate - baseline
        for candidate, baseline in zip(
            candidate_returns, baseline_returns, strict=True
        )
    ]
    count = len(differences)
    if not count:
        return {
            "effective_pairs": 0,
            "mean_return": None,
            "paired_advantage": None,
            "paired_stderr": None,
            "paired_ci95": None,
            "positive_pair_rate": None,
        }
    mean_return = sum(candidate_returns) / count
    advantage = sum(differences) / count
    stderr = stdev(differences) / math.sqrt(count) if count > 1 else 0.0
    return {
        "effective_pairs": count,
        "mean_return": round(mean_return, 6),
        "paired_advantage": round(advantage, 6),
        "paired_stderr": round(stderr, 6),
        "paired_ci95": [
            round(advantage - 1.96 * stderr, 6),
            round(advantage + 1.96 * stderr, 6),
        ],
        "positive_pair_rate": round(
            sum(difference > 0 for difference in differences) / count, 6
        ),
    }


class PairedRolloutEvaluator:
    """Compare root candidates under shared hidden-state determinizations."""

    def __init__(
        self,
        determinizer: DeckDeterminizer,
        root_policy_factory: Callable[[], RolloutPolicy],
        opponent_policy_factory: Callable[[], RolloutPolicy],
        *,
        determinizations: int = 16,
        max_rollout_steps: int = 1_000,
        value_policy: Any | None = None,
        backend: SearchBackend | None = None,
    ) -> None:
        if determinizations <= 0:
            raise ValueError("determinizations must be positive")
        if max_rollout_steps <= 0:
            raise ValueError("max_rollout_steps must be positive")
        self._determinizer = determinizer
        self._root_policy_factory = root_policy_factory
        self._opponent_policy_factory = opponent_policy_factory
        self._determinizations = determinizations
        self._max_rollout_steps = max_rollout_steps
        self._value_policy = value_policy
        self._backend = backend or OfficialSearchBackend()

    @staticmethod
    def _choose(policy: RolloutPolicy, observation: dict) -> list[int]:
        deterministic = getattr(policy, "choose_deterministic_action", None)
        if callable(deterministic):
            return list(deterministic(observation))
        return list(policy.choose_action(observation))

    def _endpoint_value(self, observation: dict, root_player: int) -> float:
        terminal = PolicyValueMCTSAgent._terminal_value(observation, root_player)
        if terminal is not None:
            return terminal
        if self._value_policy is None:
            raise RuntimeError("rollout reached its step cap without a value policy")
        evaluation = self._value_policy.evaluate(observation)
        acting_player = int(observation["current"]["yourIndex"])
        return evaluation.value if acting_player == root_player else -evaluation.value

    def _rollout(
        self,
        root: SearchPosition,
        candidate: RootCandidate,
        *,
        root_player: int,
        root_turn: int,
    ) -> dict[str, Any]:
        policies = {
            root_player: self._root_policy_factory(),
            1 - root_player: self._opponent_policy_factory(),
        }
        position = self._backend.step(root.search_id, list(candidate.action))
        option_sequence = [list(candidate.action)]
        option_boundary: str | None = None
        steps = 1
        while steps < self._max_rollout_steps:
            observation = position.observation
            terminal = PolicyValueMCTSAgent._terminal_value(
                observation, root_player
            )
            if terminal is not None:
                if option_boundary is None:
                    option_boundary = "terminal"
                return {
                    "return": terminal,
                    "steps": steps,
                    "boundary": "terminal",
                    "option_boundary": option_boundary,
                    "option_sequence": option_sequence,
                }
            state = observation["current"]
            acting_player = int(state["yourIndex"])
            turn = int(state["turn"])
            context = int(observation["select"]["context"])
            if option_boundary is None:
                if acting_player != root_player or turn != root_turn:
                    option_boundary = "turn_changed"
                elif context == 0:
                    option_boundary = "return_main"
            action = self._choose(policies[acting_player], observation)
            if option_boundary is None:
                option_sequence.append(list(action))
            position = self._backend.step(position.search_id, action)
            steps += 1

        observation = position.observation
        if option_boundary is None:
            option_boundary = "max_steps"
        return {
            "return": round(self._endpoint_value(observation, root_player), 6),
            "steps": steps,
            "boundary": "value_bootstrap",
            "option_boundary": option_boundary,
            "option_sequence": option_sequence,
        }

    def evaluate(
        self, observation: dict, candidates: Sequence[RootCandidate]
    ) -> dict[str, Any]:
        if len(candidates) < 2:
            raise ValueError("paired evaluation requires a baseline and a candidate")
        if len({candidate.action for candidate in candidates}) != len(candidates):
            raise ValueError("root candidate actions must be unique")
        root_player = int(observation["current"]["yourIndex"])
        root_turn = int(observation["current"]["turn"])
        samples = []
        returns: list[list[float | None]] = [list() for _ in candidates]
        error_counts: dict[str, int] = {}
        for determinization_id in range(self._determinizations):
            hidden = self._determinizer.sample(observation)
            began = False
            branches = []
            try:
                root = self._backend.begin(observation, hidden)
                began = True
                for index, candidate in enumerate(candidates):
                    try:
                        result = self._rollout(
                            root,
                            candidate,
                            root_player=root_player,
                            root_turn=root_turn,
                        )
                    except Exception as error:
                        name = type(error).__name__
                        error_counts[name] = error_counts.get(name, 0) + 1
                        returns[index].append(None)
                        branches.append(
                            {
                                "action": list(candidate.action),
                                "sources": list(candidate.sources),
                                "error": name,
                            }
                        )
                        continue
                    value = float(result["return"])
                    returns[index].append(value)
                    branches.append(
                        {
                            "action": list(candidate.action),
                            "sources": list(candidate.sources),
                            **result,
                        }
                    )
            finally:
                if began:
                    self._backend.end()
            samples.append(
                {
                    "determinization_id": determinization_id,
                    "opponent_deck_name": hidden.opponent_deck_name,
                    "branches": branches,
                }
            )

        candidate_summaries = []
        baseline = returns[0]
        for index, candidate in enumerate(candidates):
            paired_candidate = []
            paired_baseline = []
            for value, base in zip(returns[index], baseline, strict=True):
                if value is not None and base is not None:
                    paired_candidate.append(value)
                    paired_baseline.append(base)
            candidate_summaries.append(
                {
                    "action": list(candidate.action),
                    "sources": list(candidate.sources),
                    **paired_summary(paired_candidate, paired_baseline),
                }
            )
        return {
            "root_player": root_player,
            "root_turn": root_turn,
            "determinizations": self._determinizations,
            "candidates": candidate_summaries,
            "samples": samples,
            "errors": dict(sorted(error_counts.items())),
        }

