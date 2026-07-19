"""Behavior-cloned policy with a RuleAgent resolver for unsupported selections."""

from __future__ import annotations

import random
from pathlib import Path

import torch

from poketcg.rl.data import BCExample, collate_bc
from poketcg.rl.features import build_feature_encoder
from poketcg.rl.model import build_model, encoder_version

from .rule_agent import RuleAgent


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

    def choose_action(self, observation: dict) -> list[int]:
        selection = observation.get("select")
        if selection is None:
            raise ValueError("BCPolicyAgent received the initial deck-selection observation.")
        learnable = (
            len(selection["option"]) > 1
            and int(selection["minCount"]) == 1
            and int(selection["maxCount"]) == 1
        )
        if not learnable:
            return self._fallback.choose_action(observation)

        decision = self._encoder.encode(observation)
        example = BCExample(
            decision=decision,
            action=0,
            value_target=0.0,
            player=int(observation["current"]["yourIndex"]),
            game=0,
        )
        batch = {
            key: value.to(self._device) for key, value in collate_bc([example]).items()
        }
        with torch.no_grad():
            logits, _ = self._model(batch)
        if self._deterministic:
            return [int(logits.argmax(dim=-1).item())]
        probabilities = logits.softmax(dim=-1).squeeze(0).cpu().tolist()
        return [self._rng.choices(range(len(probabilities)), weights=probabilities, k=1)[0]]
