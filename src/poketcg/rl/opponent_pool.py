"""Weighted fixed-opponent pool for population and historical self-play."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from poketcg.agents import Agent, RandomAgent, RuleAgent

from .action_space import neural_selection, sample_subset
from .data import BCExample, collate_bc
from .features import FeatureEncoder, build_feature_encoder
from .model import PolicyValueModel, action_space_version, build_model, encoder_version


class FrozenPolicyAgent:
    """A stochastic agent backed by a shared, immutable policy model."""

    def __init__(
        self,
        model: PolicyValueModel,
        encoder: FeatureEncoder,
        *,
        card_catalog: dict[int, object],
        attack_catalog: dict[int, object],
        seed: int,
        action_version: int = 1,
    ) -> None:
        self.name = "frozen-policy"
        self._model = model
        self._encoder = encoder
        self._rng = random.Random(seed)
        self._action_space_version = action_version
        self._fallback = RuleAgent(
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            seed=seed,
        )

    def choose_action(self, observation: dict) -> list[int]:
        selection = observation.get("select")
        if selection is None:
            raise ValueError("FrozenPolicyAgent received the initial deck-selection observation.")
        if not neural_selection(selection, self._action_space_version):
            return self._fallback.choose_action(observation)

        decision = self._encoder.encode(observation)
        example = BCExample(
            decision=decision,
            action=list(range(decision.minimum)),
            value_target=0.0,
            player=int(observation["current"]["yourIndex"]),
            game=0,
        )
        batch = collate_bc([example])
        with torch.no_grad():
            policy_logits, _ = self._model(batch)
        logits = policy_logits.squeeze(0)
        return sample_subset(
            logits,
            int(selection["minCount"]),
            int(selection["maxCount"]),
            rng=self._rng,
        )


@dataclass(slots=True)
class _PoolEntry:
    name: str
    kind: str
    weight: float
    model: PolicyValueModel | None = None
    model_config: dict[str, Any] | None = None
    encoder_version: int = 1
    action_space_version: int = 1
    games: int = 0
    score_sum: float = 0.0
    ema_score: float = 0.5


class OpponentPool:
    """Sample fixed baselines and frozen neural policies with explicit weights."""

    def __init__(
        self,
        *,
        card_catalog: dict[int, object],
        attack_catalog: dict[int, object],
        seed: int,
        snapshot_weight: float = 0.0,
        max_snapshots: int = 4,
        adaptive_sampling: str = "none",
        adaptive_alpha: float = 1.0,
        adaptive_min_multiplier: float = 0.1,
        adaptive_ema_decay: float = 0.95,
        adaptive_warmup_games: int = 32,
    ) -> None:
        if snapshot_weight < 0.0:
            raise ValueError("snapshot_weight must be non-negative")
        if max_snapshots < 1:
            raise ValueError("max_snapshots must be at least one")
        if adaptive_sampling not in {"none", "win_rate"}:
            raise ValueError("adaptive_sampling must be 'none' or 'win_rate'")
        if adaptive_alpha < 0.0:
            raise ValueError("adaptive_alpha must be non-negative")
        if not 0.0 <= adaptive_min_multiplier <= 1.0:
            raise ValueError("adaptive_min_multiplier must be between zero and one")
        if not 0.0 <= adaptive_ema_decay < 1.0:
            raise ValueError("adaptive_ema_decay must be in [0, 1)")
        if adaptive_warmup_games < 0:
            raise ValueError("adaptive_warmup_games must be non-negative")
        self._cards = card_catalog
        self._attacks = attack_catalog
        self._encoders = {
            version: build_feature_encoder(version, card_catalog, attack_catalog)
            for version in (1, 2, 3)
        }
        self._rng = random.Random(seed)
        self._entries: list[_PoolEntry] = []
        self._snapshots: list[_PoolEntry] = []
        self._snapshot_weight = snapshot_weight
        self._max_snapshots = max_snapshots
        self._adaptive_sampling = adaptive_sampling
        self._adaptive_alpha = adaptive_alpha
        self._adaptive_min_multiplier = adaptive_min_multiplier
        self._adaptive_ema_decay = adaptive_ema_decay
        self._adaptive_warmup_games = adaptive_warmup_games

    @staticmethod
    def _validate_weight(weight: float) -> None:
        if weight <= 0.0:
            raise ValueError("opponent weights must be positive")

    def add_random(self, name: str, weight: float) -> None:
        self._validate_weight(weight)
        self._entries.append(_PoolEntry(name=name, kind="random", weight=weight))

    def add_rule(self, name: str, weight: float) -> None:
        self._validate_weight(weight)
        self._entries.append(_PoolEntry(name=name, kind="rule", weight=weight))

    def add_checkpoint(self, name: str, checkpoint: str | Path, weight: float) -> None:
        self._validate_weight(weight)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_model(saved["model_config"])
        model.load_state_dict(saved["model_state_dict"])
        model.eval()
        self._entries.append(
            _PoolEntry(
                name=name,
                kind="policy",
                weight=weight,
                model=model,
                model_config=dict(saved["model_config"]),
                encoder_version=encoder_version(saved["model_config"]),
                action_space_version=action_space_version(saved["model_config"]),
            )
        )

    def add_snapshot(
        self,
        name: str,
        model: PolicyValueModel,
        model_config: dict[str, Any],
    ) -> None:
        frozen = build_model(model_config)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        frozen.load_state_dict(state)
        frozen.eval()
        self._snapshots.append(
            _PoolEntry(
                name=name,
                kind="snapshot",
                weight=0.0,
                model=frozen,
                model_config=dict(model_config),
                encoder_version=encoder_version(model_config),
                action_space_version=action_space_version(model_config),
            )
        )
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(0)

    def _base_weighted_entries(self) -> list[tuple[_PoolEntry, float]]:
        entries = [(item, item.weight) for item in self._entries]
        if self._snapshots and self._snapshot_weight > 0.0:
            per_snapshot = self._snapshot_weight / len(self._snapshots)
            entries.extend((item, per_snapshot) for item in self._snapshots)
        return entries

    def _effective_weight(self, entry: _PoolEntry, base_weight: float) -> float:
        if (
            self._adaptive_sampling == "none"
            or entry.games < self._adaptive_warmup_games
        ):
            return base_weight
        # PFSP prioritizes opponents whose EMA score is closest to 50%.
        competitiveness = max(0.0, 4.0 * entry.ema_score * (1.0 - entry.ema_score))
        multiplier = max(
            self._adaptive_min_multiplier,
            competitiveness**self._adaptive_alpha,
        )
        return base_weight * multiplier

    def _weighted_entries(self) -> list[tuple[_PoolEntry, float, float]]:
        return [
            (entry, base_weight, self._effective_weight(entry, base_weight))
            for entry, base_weight in self._base_weighted_entries()
        ]

    def record_results(self, opponent_names: list[str], outcomes: list[float]) -> None:
        """Update per-opponent learner scores after a rollout batch."""
        if len(opponent_names) != len(outcomes):
            raise ValueError("opponent_names and outcomes must have the same length")
        by_name = {item.name: item for item in [*self._entries, *self._snapshots]}
        for name, outcome in zip(opponent_names, outcomes, strict=True):
            if name not in by_name:
                raise KeyError(f"Unknown opponent pool member: {name}")
            entry = by_name[name]
            score = 1.0 if outcome > 0.0 else (0.5 if outcome == 0.0 else 0.0)
            entry.games += 1
            entry.score_sum += score
            entry.ema_score = (
                self._adaptive_ema_decay * entry.ema_score
                + (1.0 - self._adaptive_ema_decay) * score
            )

    def sample_name(self) -> str:
        entries = self._weighted_entries()
        if not entries:
            raise RuntimeError("Opponent pool has no positive-weight entries.")
        selected = self._rng.choices(entries, weights=[item[2] for item in entries], k=1)[0]
        return selected[0].name

    def sample(self, *, seed: int) -> tuple[str, Agent]:
        entries = self._weighted_entries()
        if not entries:
            raise RuntimeError("Opponent pool has no positive-weight entries.")
        selected = self._rng.choices(entries, weights=[item[2] for item in entries], k=1)[0]
        entry = selected[0]
        if entry.kind == "random":
            agent: Agent = RandomAgent(seed)
        elif entry.kind == "rule":
            agent = RuleAgent(
                card_catalog=self._cards,
                attack_catalog=self._attacks,
                seed=seed,
            )
        else:
            if entry.model is None:
                raise RuntimeError(f"Policy opponent {entry.name} has no model.")
            agent = FrozenPolicyAgent(
                entry.model,
                self._encoders[entry.encoder_version],
                card_catalog=self._cards,
                attack_catalog=self._attacks,
                seed=seed,
                action_version=entry.action_space_version,
            )
        return entry.name, agent

    def manifest(self) -> list[dict[str, str | int | float]]:
        return [
            {
                "name": item.name,
                "kind": item.kind,
                "encoder_version": item.encoder_version,
                "base_weight": round(base_weight, 6),
                "effective_weight": round(effective_weight, 6),
                "games": item.games,
                "win_rate": round(item.score_sum / item.games, 6) if item.games else 0.5,
                "ema_win_rate": round(item.ema_score, 6),
            }
            for item, base_weight, effective_weight in self._weighted_entries()
        ]

    def effective_weights(self) -> dict[str, float]:
        return {
            item.name: round(effective_weight, 6)
            for item, _, effective_weight in self._weighted_entries()
        }

    def worker_state(self) -> list[dict[str, Any]]:
        """Export immutable opponent definitions for spawned rollout workers."""
        state: list[dict[str, Any]] = []
        for entry, _, _ in self._weighted_entries():
            item: dict[str, Any] = {
                "name": entry.name,
                "kind": entry.kind,
                "encoder_version": entry.encoder_version,
                "action_space_version": entry.action_space_version,
            }
            if entry.model is not None:
                if entry.model_config is None:
                    raise RuntimeError(f"Opponent {entry.name} has no model configuration.")
                item["model_config"] = dict(entry.model_config)
                item["model_state_dict"] = {
                    key: value.detach().cpu().clone()
                    for key, value in entry.model.state_dict().items()
                }
            state.append(item)
        return state
