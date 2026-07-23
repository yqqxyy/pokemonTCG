"""Turn-gated advantage reranking over the Library-Out V1 action."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch

from poketcg.rl.advantage_candidates import root_candidates
from poketcg.rl.residual_data import normalize_rule_scores

from .bc_agent import BCPolicyAgent
from .external_agent import ExternalPythonAgent


def conservative_candidate_scores(
    member_logits: list[torch.Tensor],
    baseline_index: int,
    candidate_indices: list[int],
    *,
    uncertainty_multiplier: float,
) -> dict[int, float]:
    """Return ensemble mean-minus-std advantages relative to the V1 option."""
    if not member_logits:
        raise ValueError("At least one advantage member is required")
    if uncertainty_multiplier < 0:
        raise ValueError("uncertainty_multiplier must be non-negative")
    stacked = torch.stack(
        [
            logits[candidate_indices] - logits[baseline_index]
            for logits in member_logits
        ]
    )
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=False)
    conservative = mean - uncertainty_multiplier * std
    return {
        index: float(score)
        for index, score in zip(candidate_indices, conservative, strict=True)
    }


class AdvantageRerankerAgent:
    """Keep V1 except for a gated exact-one MAIN decision at a late turn."""

    name = "libraryout-advantage-reranker"

    def __init__(
        self,
        advantage_checkpoints: list[str | Path],
        round0_checkpoint: str | Path,
        baseline_source: str | Path,
        baseline_deck: str | Path,
        *,
        card_catalog: dict[int, object],
        attack_catalog: dict[int, object],
        expected_deck: list[int] | None = None,
        device: str = "cpu",
        minimum_turn: int = 4,
        gate_threshold: float = 0.05,
        uncertainty_multiplier: float = 0.0,
        allowed_transitions: set[tuple[int, int]] | None = None,
        shadow: bool = False,
    ) -> None:
        if not advantage_checkpoints:
            raise ValueError("At least one advantage checkpoint is required")
        if minimum_turn <= 0:
            raise ValueError("minimum_turn must be positive")
        self._baseline = ExternalPythonAgent(
            baseline_source,
            baseline_deck,
            name="libraryout-baseline",
            expected_deck=expected_deck,
        )
        common = {
            "card_catalog": card_catalog,
            "attack_catalog": attack_catalog,
            "device": device,
            "deterministic": True,
        }
        self._round0 = BCPolicyAgent(round0_checkpoint, **common)
        self._members = []
        for checkpoint in advantage_checkpoints:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            config = saved.get("advantage_config") or {}
            if not bool(config.get("baseline_relative", False)):
                raise ValueError(f"Not a baseline-relative advantage checkpoint: {checkpoint}")
            self._members.append(BCPolicyAgent(checkpoint, **common))
        self._minimum_turn = minimum_turn
        self._gate_threshold = gate_threshold
        self._uncertainty_multiplier = uncertainty_multiplier
        self._allowed_transitions = (
            set(allowed_transitions) if allowed_transitions is not None else None
        )
        self._shadow = shadow
        self._metrics: Counter[str] = Counter()
        self._transitions: Counter[str] = Counter()

    def reset_episode(self) -> None:
        self._baseline.reset_episode()

    def metrics(self) -> dict:
        eligible = self._metrics["eligible"]
        return {
            **dict(self._metrics),
            "transitions": dict(sorted(self._transitions.items())),
            "override_rate": (
                round(self._metrics["executed_overrides"] / eligible, 6)
                if eligible
                else 0.0
            ),
            "proposed_override_rate": (
                round(self._metrics["proposed_overrides"] / eligible, 6)
                if eligible
                else 0.0
            ),
            "minimum_turn": self._minimum_turn,
            "gate_threshold": self._gate_threshold,
            "uncertainty_multiplier": self._uncertainty_multiplier,
            "ensemble_size": len(self._members),
            "allowed_transitions": (
                [
                    f"{source}->{target}"
                    for source, target in sorted(self._allowed_transitions)
                ]
                if self._allowed_transitions is not None
                else None
            ),
            "shadow": self._shadow,
        }

    def choose_action(self, observation: dict) -> list[int]:
        baseline_action, raw_scores = self._baseline.choose_action_with_scores(observation)
        self._metrics["decisions"] += 1
        selection = observation.get("select") or {}
        exact_main = (
            int(selection.get("context", -1)) == 0
            and int(selection.get("minCount", -1)) == 1
            and int(selection.get("maxCount", -1)) == 1
            and len(selection.get("option") or []) >= 2
        )
        turn = int(observation["current"]["turn"])
        if not exact_main or turn < self._minimum_turn or len(baseline_action) != 1:
            return baseline_action
        self._metrics["eligible"] += 1
        try:
            normalized_scores = normalize_rule_scores(raw_scores)
            round0_logits = self._round0.evaluate(observation).logits
            option_count = len(selection["option"])
            candidates = root_candidates(
                baseline_action,
                normalized_scores,
                round0_logits[:option_count].tolist(),
                [int(option["type"]) for option in selection["option"]],
            )
            baseline_index = baseline_action[0]
            candidate_indices = [
                candidate.action[0]
                for candidate in candidates
                if candidate.action[0] != baseline_index
            ]
            if not candidate_indices:
                return baseline_action
            member_logits = [
                member.evaluate(observation).logits[:option_count]
                for member in self._members
            ]
            scores = conservative_candidate_scores(
                member_logits,
                baseline_index,
                candidate_indices,
                uncertainty_multiplier=self._uncertainty_multiplier,
            )
        except Exception:
            self._metrics["inference_errors"] += 1
            return baseline_action
        proposed_index = max(scores, key=scores.__getitem__)
        proposed_score = scores[proposed_index]
        if proposed_score <= self._gate_threshold:
            self._metrics["gate_rejections"] += 1
            return baseline_action
        self._metrics["proposed_overrides"] += 1
        option_types = [int(option["type"]) for option in selection["option"]]
        transition = (option_types[baseline_index], option_types[proposed_index])
        self._transitions[f"{transition[0]}->{transition[1]}"] += 1
        if (
            self._allowed_transitions is not None
            and transition not in self._allowed_transitions
        ):
            self._metrics["semantic_rejections"] += 1
            return baseline_action
        if self._shadow:
            return baseline_action
        self._metrics["executed_overrides"] += 1
        return [proposed_index]
