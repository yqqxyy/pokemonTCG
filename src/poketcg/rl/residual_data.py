"""Library-Out trajectories with rule priors for residual policy learning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .data import BCExample, collate_bc
from .features import EncodedDecision


def normalize_rule_scores(scores: list[float], clip: float = 4.0) -> list[float]:
    """Standardize a context-local rule score vector while preserving its order."""
    if not scores:
        raise ValueError("rule score vector cannot be empty")
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("rule scores must be finite")
    mean = sum(scores) / len(scores)
    variance = sum((value - mean) ** 2 for value in scores) / len(scores)
    scale = math.sqrt(variance)
    if scale < 1e-8:
        return [0.0] * len(scores)
    return [max(-clip, min(clip, (value - mean) / scale)) for value in scores]


@dataclass(slots=True)
class ResidualExample:
    """One expert decision plus the rule prior and terminal outcome."""

    decision: EncodedDecision
    baseline_action: list[int]
    target_action: list[int]
    rule_scores: list[float]
    value_target: float
    player: int
    game: int
    decision_index: int
    opponent: str
    target_source: str = "libraryout_v1"

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.to_dict(),
            "baseline_action": self.baseline_action,
            "target_action": self.target_action,
            "rule_scores": self.rule_scores,
            "value_target": self.value_target,
            "player": self.player,
            "game": self.game,
            "decision_index": self.decision_index,
            "opponent": self.opponent,
            "target_source": self.target_source,
        }

    @classmethod
    def from_dict(cls, value: dict) -> ResidualExample:
        return cls(
            decision=EncodedDecision.from_dict(value["decision"]),
            baseline_action=[int(index) for index in value["baseline_action"]],
            target_action=[int(index) for index in value["target_action"]],
            rule_scores=[float(score) for score in value["rule_scores"]],
            value_target=float(value["value_target"]),
            player=int(value["player"]),
            game=int(value["game"]),
            decision_index=int(value["decision_index"]),
            opponent=str(value["opponent"]),
            target_source=str(value.get("target_source", "libraryout_v1")),
        )

    def validate(self) -> None:
        option_count = len(self.decision.options)
        if len(self.rule_scores) != option_count:
            raise ValueError("rule score count must match the legal option count")
        if not all(math.isfinite(value) for value in self.rule_scores):
            raise ValueError("rule scores must be finite")
        for name, action in (
            ("baseline_action", self.baseline_action),
            ("target_action", self.target_action),
        ):
            if len(action) != len(set(action)):
                raise ValueError(f"{name} contains duplicate indices")
            if any(index < 0 or index >= option_count for index in action):
                raise ValueError(f"{name} references an invalid option")
            if not self.decision.minimum <= len(action) <= self.decision.maximum:
                raise ValueError(f"{name} violates selection cardinality")

    def as_bc_example(self) -> BCExample:
        self.validate()
        return BCExample(
            decision=self.decision,
            action=list(self.target_action),
            value_target=self.value_target,
            player=self.player,
            game=self.game,
        )


class ResidualDataset(Dataset[ResidualExample]):
    def __init__(self, examples: list[ResidualExample]) -> None:
        for example in examples:
            example.validate()
        self.examples = examples

    @classmethod
    def from_jsonl(cls, path: str | Path) -> ResidualDataset:
        with Path(path).expanduser().open(encoding="utf-8") as stream:
            return cls(
                [
                    ResidualExample.from_dict(json.loads(line))
                    for line in stream
                    if line.strip()
                ]
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ResidualExample:
        return self.examples[index]


def collate_residual(examples: list[ResidualExample]) -> dict[str, Tensor]:
    """Pad encoded decisions and attach rule priors and baseline actions."""
    batch = collate_bc([example.as_bc_example() for example in examples])
    max_options = batch["option_mask"].shape[1]
    rule_scores = torch.zeros(len(examples), max_options, dtype=torch.float32)
    baseline_action_mask = torch.zeros(len(examples), max_options, dtype=torch.bool)
    for row, example in enumerate(examples):
        option_count = len(example.rule_scores)
        rule_scores[row, :option_count] = torch.tensor(
            example.rule_scores, dtype=torch.float32
        )
        baseline_action_mask[row, example.baseline_action] = True
    batch["rule_scores"] = rule_scores
    batch["baseline_action_mask"] = baseline_action_mask
    return batch

