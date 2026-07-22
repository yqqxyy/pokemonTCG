"""Confidence-gated neural reranking over a scored external rule agent."""

from __future__ import annotations

from pathlib import Path

import torch

from poketcg.rl.action_space import deterministic_subset, neural_selection
from poketcg.rl.data import BCExample, collate_bc
from poketcg.rl.features import build_feature_encoder
from poketcg.rl.model import build_model, encoder_version
from poketcg.rl.residual_data import normalize_rule_scores

from .external_agent import ExternalPythonAgent


class ResidualRerankerAgent:
    """Keep the rule action unless a residual model clears a strict gate."""

    name = "libraryout-residual-reranker"

    def __init__(
        self,
        checkpoint: str | Path,
        baseline_source: str | Path,
        baseline_deck: str | Path,
        *,
        card_catalog: dict[int, object],
        attack_catalog: dict[int, object],
        expected_deck: list[int] | None = None,
        device: str = "cpu",
        shadow: bool = False,
        override_margin: float | None = None,
        minimum_confidence: float | None = None,
    ) -> None:
        self._device = torch.device(device)
        saved = torch.load(checkpoint, map_location=self._device, weights_only=False)
        self._model = build_model(saved["model_config"]).to(self._device)
        self._model.load_state_dict(saved["model_state_dict"])
        self._model.eval()
        residual_config = dict(saved.get("residual_config") or {})
        self._prior_strength = float(residual_config.get("prior_strength", 2.0))
        self._override_margin = float(
            residual_config.get("override_margin", 0.5)
            if override_margin is None
            else override_margin
        )
        self._minimum_confidence = float(
            residual_config.get("minimum_confidence", 0.65)
            if minimum_confidence is None
            else minimum_confidence
        )
        self._exact_one_only = bool(residual_config.get("exact_one_only", True))
        self._shadow = shadow
        self._encoder = build_feature_encoder(
            encoder_version(saved["model_config"]), card_catalog, attack_catalog
        )
        self._baseline = ExternalPythonAgent(
            baseline_source,
            baseline_deck,
            name="libraryout-baseline",
            expected_deck=expected_deck,
        )
        self._metrics = {
            "decisions": 0,
            "eligible": 0,
            "proposed_overrides": 0,
            "gate_accepts": 0,
            "executed_overrides": 0,
            "gate_rejections": 0,
        }

    def reset_episode(self) -> None:
        self._baseline.reset_episode()

    def metrics(self) -> dict[str, int | float | bool]:
        eligible = self._metrics["eligible"]
        return {
            **self._metrics,
            "override_rate": round(
                self._metrics["executed_overrides"] / eligible, 6
            )
            if eligible
            else 0.0,
            "shadow": self._shadow,
            "prior_strength": self._prior_strength,
            "override_margin": self._override_margin,
            "minimum_confidence": self._minimum_confidence,
        }

    def choose_action(self, observation: dict) -> list[int]:
        baseline_action, raw_scores = self._baseline.choose_action_with_scores(observation)
        self._metrics["decisions"] += 1
        selection = observation["select"]
        exact_one = int(selection["minCount"]) == int(selection["maxCount"]) == 1
        if not neural_selection(selection, action_space_version=2) or (
            self._exact_one_only and not exact_one
        ):
            return baseline_action
        self._metrics["eligible"] += 1
        decision = self._encoder.encode(observation)
        example = BCExample(
            decision=decision,
            action=list(baseline_action),
            value_target=0.0,
            player=int(observation["current"]["yourIndex"]),
            game=0,
        )
        batch = {
            key: value.to(self._device)
            for key, value in collate_bc([example]).items()
        }
        with torch.no_grad():
            residual_logits, _ = self._model(batch)
        residual_logits = residual_logits.squeeze(0).detach().cpu()
        rule_scores = torch.tensor(normalize_rule_scores(raw_scores))
        combined = self._prior_strength * rule_scores + residual_logits
        proposed = deterministic_subset(combined, 1, 1)
        if proposed == baseline_action:
            return baseline_action
        self._metrics["proposed_overrides"] += 1
        probabilities = combined.softmax(dim=0)
        proposed_index = proposed[0]
        baseline_index = baseline_action[0]
        margin = float(combined[proposed_index] - combined[baseline_index])
        confidence = float(probabilities[proposed_index])
        if margin < self._override_margin or confidence < self._minimum_confidence:
            self._metrics["gate_rejections"] += 1
            return baseline_action
        self._metrics["gate_accepts"] += 1
        if self._shadow:
            return baseline_action
        self._metrics["executed_overrides"] += 1
        return proposed
