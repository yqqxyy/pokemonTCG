"""Behavior-cloned policy with a RuleAgent resolver for unsupported selections."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from poketcg.rl.action_space import deterministic_subset, neural_selection, sample_subset
from poketcg.rl.data import BCExample, collate_bc
from poketcg.rl.features import EncodedDecision, build_feature_encoder
from poketcg.rl.model import action_space_version, build_model, encoder_version

from .rule_agent import RuleAgent


@dataclass(frozen=True, slots=True)
class PolicyValueEvaluation:
    """Neural option scores and value from the acting player's perspective."""

    logits: Tensor
    value: float
    minimum: int
    maximum: int


class BCPolicyAgent:
    """Use the neural policy for single-choice decisions and rules elsewhere."""

    name = "bc-policy"

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        card_catalog: dict[int, object],
        attack_catalog: dict[int, object],
        seed: int | None = None,
        device: str = "cpu",
        deterministic: bool = True,
    ) -> None:
        self._device = torch.device(device)
        # Checkpoints are produced locally by this project and may contain Path objects.
        saved = torch.load(checkpoint, map_location=self._device, weights_only=False)
        self._model = build_model(saved["model_config"]).to(self._device)
        self._model.load_state_dict(saved["model_state_dict"])
        self._model.eval()
        self._deterministic = deterministic
        self._rng = random.Random(seed)
        self._action_space_version = action_space_version(saved["model_config"])
        self._encoder = build_feature_encoder(
            encoder_version(saved["model_config"]),
            card_catalog,
            attack_catalog,
        )
        self._fallback = RuleAgent(
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            seed=seed,
        )

    @property
    def action_space_version(self) -> int:
        return self._action_space_version

    def encode(self, observation: dict) -> EncodedDecision:
        """Expose the acting-player-relative representation for diagnostics."""
        return self._encoder.encode(observation)

    def evaluate(self, observation: dict) -> PolicyValueEvaluation:
        """Evaluate any non-initial selection, including resolver-owned decisions."""
        selection = observation.get("select")
        if selection is None:
            raise ValueError("BCPolicyAgent received the initial deck-selection observation.")

        decision = self.encode(observation)
        example = BCExample(
            decision=decision,
            action=list(range(decision.minimum)),
            value_target=0.0,
            player=int(observation["current"]["yourIndex"]),
            game=0,
        )
        batch = {
            key: value.to(self._device) for key, value in collate_bc([example]).items()
        }
        with torch.no_grad():
            logits, value_logits = self._model(batch)
            value = self._model.expected_value(value_logits)
        return PolicyValueEvaluation(
            logits=logits.squeeze(0).detach().cpu(),
            value=float(value.item()),
            minimum=int(selection["minCount"]),
            maximum=int(selection["maxCount"]),
        )

    def choose_action(self, observation: dict) -> list[int]:
        selection = observation.get("select")
        if selection is None:
            raise ValueError("BCPolicyAgent received the initial deck-selection observation.")
        if not neural_selection(selection, self._action_space_version):
            return self._fallback.choose_action(observation)

        evaluation = self.evaluate(observation)
        valid_logits = evaluation.logits
        minimum = int(selection["minCount"])
        maximum = int(selection["maxCount"])
        if self._deterministic:
            return deterministic_subset(valid_logits, minimum, maximum)
        return sample_subset(valid_logits, minimum, maximum, rng=self._rng)

    def choose_deterministic_action(self, observation: dict) -> list[int]:
        """Choose the policy argmax without changing the configured online mode."""
        selection = observation.get("select")
        if selection is None:
            raise ValueError("BCPolicyAgent received the initial deck-selection observation.")
        if not neural_selection(selection, self._action_space_version):
            return self._fallback.choose_action(observation)
        evaluation = self.evaluate(observation)
        return deterministic_subset(
            evaluation.logits,
            evaluation.minimum,
            evaluation.maximum,
        )


class HybridPolicyAgent:
    """Route exact-one decisions to one policy and set decisions to another."""

    name = "hybrid-policy"

    def __init__(
        self,
        single_checkpoint: str | Path,
        multiselect_checkpoint: str | Path,
        *,
        card_catalog: dict[int, object],
        attack_catalog: dict[int, object],
        seed: int | None = None,
        device: str = "cpu",
        deterministic: bool = True,
    ) -> None:
        common = {
            "card_catalog": card_catalog,
            "attack_catalog": attack_catalog,
            "seed": seed,
            "device": device,
            "deterministic": deterministic,
        }
        self._single_policy = BCPolicyAgent(single_checkpoint, **common)
        self._multiselect_policy = BCPolicyAgent(multiselect_checkpoint, **common)
        if self._multiselect_policy.action_space_version < 2:
            raise ValueError("The multiselect checkpoint must use Action Space V2 or newer")

    def choose_action(self, observation: dict) -> list[int]:
        selection = observation.get("select")
        if selection is None:
            raise ValueError("HybridPolicyAgent received the initial deck-selection observation.")
        exact_one = int(selection["minCount"]) == int(selection["maxCount"]) == 1
        policy = self._single_policy if exact_one else self._multiselect_policy
        return policy.choose_action(observation)
