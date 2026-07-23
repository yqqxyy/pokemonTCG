"""Paired one-step-deviation examples for baseline-relative advantage learning."""

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


@dataclass(slots=True)
class AdvantageExample:
    """One information state with rollout labels relative to the V1 action."""

    decision: EncodedDecision
    baseline_index: int
    advantage_targets: list[float]
    paired_stderr: list[float]
    effective_pairs: list[int]
    target_mask: list[bool]
    state_id: str
    opponent: str
    player: int
    game: int
    selection_reason: str

    @classmethod
    def from_paired_row(cls, row: dict) -> AdvantageExample:
        decision = EncodedDecision.from_dict(row["decision"])
        if decision.minimum != 1 or decision.maximum != 1:
            raise ValueError("Advantage learning currently requires exact-one decisions")
        rule_action = [int(index) for index in row["rule_action"]]
        if len(rule_action) != 1:
            raise ValueError("Paired rows must contain exactly one baseline action")
        option_count = len(decision.options)
        baseline_index = rule_action[0]
        if not 0 <= baseline_index < option_count:
            raise ValueError("Baseline action references an invalid option")

        targets = [0.0] * option_count
        stderrs = [0.0] * option_count
        effective_pairs = [0] * option_count
        target_mask = [False] * option_count
        for candidate in row["rollout"]["candidates"]:
            action = [int(index) for index in candidate["action"]]
            if len(action) != 1:
                raise ValueError("Candidate rollout actions must select exactly one option")
            index = action[0]
            if not 0 <= index < option_count:
                raise ValueError("Candidate rollout references an invalid option")
            if index == baseline_index or candidate.get("paired_advantage") is None:
                continue
            target = float(candidate["paired_advantage"])
            stderr = float(candidate["paired_stderr"])
            pairs = int(candidate["effective_pairs"])
            if not math.isfinite(target) or not math.isfinite(stderr) or stderr < 0:
                raise ValueError("Paired rollout labels must be finite")
            if pairs < 2:
                continue
            targets[index] = target
            stderrs[index] = stderr
            effective_pairs[index] = pairs
            target_mask[index] = True
        if not any(target_mask):
            raise ValueError("Paired row contains no usable non-baseline candidate")
        return cls(
            decision=decision,
            baseline_index=baseline_index,
            advantage_targets=targets,
            paired_stderr=stderrs,
            effective_pairs=effective_pairs,
            target_mask=target_mask,
            state_id=str(row["state_id"]),
            opponent=str(row["opponent"]),
            player=int(row["player"]),
            game=int(row["game"]),
            selection_reason=str(row["selection_reason"]),
        )

    def validate(self) -> None:
        option_count = len(self.decision.options)
        fields = (
            self.advantage_targets,
            self.paired_stderr,
            self.effective_pairs,
            self.target_mask,
        )
        if any(len(field) != option_count for field in fields):
            raise ValueError("Advantage vectors must match the legal option count")
        if not 0 <= self.baseline_index < option_count:
            raise ValueError("Invalid baseline index")
        if self.target_mask[self.baseline_index]:
            raise ValueError("The baseline must not be a supervised deviation target")
        if not any(self.target_mask):
            raise ValueError("At least one deviation target is required")

    def as_bc_example(self) -> BCExample:
        self.validate()
        return BCExample(
            decision=self.decision,
            action=[self.baseline_index],
            value_target=0.0,
            player=self.player,
            game=self.game,
        )


class AdvantageDataset(Dataset[AdvantageExample]):
    def __init__(self, examples: list[AdvantageExample]) -> None:
        for example in examples:
            example.validate()
        self.examples = examples

    @classmethod
    def from_jsonl(cls, path: str | Path) -> AdvantageDataset:
        examples: list[AdvantageExample] = []
        with Path(path).expanduser().open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    examples.append(AdvantageExample.from_paired_row(json.loads(line)))
        return cls(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> AdvantageExample:
        return self.examples[index]


def collate_advantage(examples: list[AdvantageExample]) -> dict[str, Tensor]:
    """Pad decisions and attach sparse paired-advantage regression targets."""
    batch = collate_bc([example.as_bc_example() for example in examples])
    batch_size, max_options = batch["option_mask"].shape
    targets = torch.zeros(batch_size, max_options, dtype=torch.float32)
    stderrs = torch.zeros(batch_size, max_options, dtype=torch.float32)
    pairs = torch.zeros(batch_size, max_options, dtype=torch.long)
    target_mask = torch.zeros(batch_size, max_options, dtype=torch.bool)
    for row, example in enumerate(examples):
        count = len(example.decision.options)
        targets[row, :count] = torch.tensor(example.advantage_targets)
        stderrs[row, :count] = torch.tensor(example.paired_stderr)
        pairs[row, :count] = torch.tensor(example.effective_pairs)
        target_mask[row, :count] = torch.tensor(example.target_mask)
    batch.update(
        {
            "baseline_index": torch.tensor(
                [example.baseline_index for example in examples], dtype=torch.long
            ),
            "advantage_target": targets,
            "paired_stderr": stderrs,
            "effective_pairs": pairs,
            "advantage_mask": target_mask,
        }
    )
    return batch
