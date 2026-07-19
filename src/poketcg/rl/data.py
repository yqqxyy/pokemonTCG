"""Behavior-cloning examples, JSONL storage, and padded batches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .features import (
    HISTORY_FEATURE_SIZE,
    OPTION_FEATURE_SIZE,
    SEMANTIC_FEATURE_SIZE,
    TOKEN_FEATURE_SIZE,
    EncodedDecision,
    expand_semantic_features,
)


@dataclass(slots=True)
class BCExample:
    decision: EncodedDecision
    action: int
    value_target: float
    player: int
    game: int
    policy_target: list[float] | None = None

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.to_dict(),
            "action": self.action,
            "value_target": self.value_target,
            "player": self.player,
            "game": self.game,
            "policy_target": self.policy_target,
        }

    @classmethod
    def from_dict(cls, value: dict) -> BCExample:
        return cls(
            decision=EncodedDecision.from_dict(value["decision"]),
            action=int(value["action"]),
            value_target=float(value["value_target"]),
            player=int(value["player"]),
            game=int(value["game"]),
            policy_target=value.get("policy_target"),
        )


class BCDataset(Dataset[BCExample]):
    def __init__(self, examples: list[BCExample]) -> None:
        self.examples = examples

    @classmethod
    def from_jsonl(cls, path: str | Path) -> BCDataset:
        with Path(path).open(encoding="utf-8") as stream:
            return cls([BCExample.from_dict(json.loads(line)) for line in stream if line.strip()])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> BCExample:
        return self.examples[index]


def write_jsonl(path: str | Path, examples: list[BCExample]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(example.to_dict(), separators=(",", ":")) + "\n")


def collate_bc(examples: list[BCExample]) -> dict[str, Tensor]:
    versions = {example.decision.version for example in examples}
    if len(versions) != 1:
        raise ValueError("A batch cannot mix encoded decision versions")
    version = versions.pop()
    batch_size = len(examples)
    max_options = max(len(example.decision.options) for example in examples)
    options = torch.zeros(batch_size, max_options, OPTION_FEATURE_SIZE, dtype=torch.float32)
    option_types = torch.zeros(batch_size, max_options, dtype=torch.long)
    areas = torch.zeros(batch_size, max_options, dtype=torch.long)
    in_play_areas = torch.zeros(batch_size, max_options, dtype=torch.long)
    option_mask = torch.zeros(batch_size, max_options, dtype=torch.bool)
    policy_target = torch.zeros(batch_size, max_options, dtype=torch.float32)

    for row, example in enumerate(examples):
        count = len(example.decision.options)
        options[row, :count] = torch.tensor(example.decision.options, dtype=torch.float32)
        option_types[row, :count] = torch.tensor(example.decision.option_types)
        areas[row, :count] = torch.tensor(example.decision.areas)
        in_play_areas[row, :count] = torch.tensor(example.decision.in_play_areas)
        option_mask[row, :count] = True
        if example.policy_target is None:
            policy_target[row, example.action] = 1.0
        else:
            if len(example.policy_target) != count:
                raise ValueError("policy_target length must match the number of options")
            target = torch.tensor(example.policy_target, dtype=torch.float32)
            if not torch.isclose(target.sum(), torch.tensor(1.0), atol=1e-5):
                raise ValueError("policy_target probabilities must sum to one")
            policy_target[row, :count] = target

    batch = {
        "state": torch.tensor([example.decision.state for example in examples]),
        "select_type": torch.tensor([example.decision.select_type for example in examples]),
        "context": torch.tensor([example.decision.context for example in examples]),
        "options": options,
        "option_types": option_types,
        "areas": areas,
        "in_play_areas": in_play_areas,
        "option_mask": option_mask,
        "policy_target": policy_target,
        "action": torch.tensor([example.action for example in examples]),
        "value_target": torch.tensor([example.value_target for example in examples]),
    }
    if version == 1:
        return batch
    if version not in {2, 3}:
        raise ValueError(f"Unsupported encoded decision version: {version}")

    token_counts = [len(example.decision.tokens or []) for example in examples]
    max_tokens = max(1, max(token_counts))
    tokens = torch.zeros(batch_size, max_tokens, TOKEN_FEATURE_SIZE, dtype=torch.float32)
    token_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    token_fields = {
        "token_card_ids": torch.zeros(batch_size, max_tokens, dtype=torch.long),
        "token_kinds": torch.zeros(batch_size, max_tokens, dtype=torch.long),
        "token_zones": torch.zeros(batch_size, max_tokens, dtype=torch.long),
        "token_owners": torch.zeros(batch_size, max_tokens, dtype=torch.long),
        "token_slots": torch.zeros(batch_size, max_tokens, dtype=torch.long),
        "token_card_types": torch.zeros(batch_size, max_tokens, dtype=torch.long),
        "token_energy_types": torch.zeros(batch_size, max_tokens, dtype=torch.long),
        "token_weaknesses": torch.zeros(batch_size, max_tokens, dtype=torch.long),
        "token_resistances": torch.zeros(batch_size, max_tokens, dtype=torch.long),
    }
    option_card_ids = torch.zeros(batch_size, max_options, dtype=torch.long)
    option_attack_ids = torch.zeros(batch_size, max_options, dtype=torch.long)
    option_special_conditions = torch.zeros(batch_size, max_options, dtype=torch.long)

    for row, example in enumerate(examples):
        decision = example.decision
        count = token_counts[row]
        if count:
            tokens[row, :count] = torch.tensor(decision.tokens, dtype=torch.float32)
        token_mask[row, :count] = True
        for name, tensor in token_fields.items():
            values = getattr(decision, name)
            if values is None or len(values) != count:
                raise ValueError(f"{name} must match the token count")
            tensor[row, :count] = torch.tensor(values, dtype=torch.long)
        option_count = len(decision.options)
        for name, tensor in (
            ("option_card_ids", option_card_ids),
            ("option_attack_ids", option_attack_ids),
            ("option_special_conditions", option_special_conditions),
        ):
            values = getattr(decision, name)
            if values is None or len(values) != option_count:
                raise ValueError(f"{name} must match the option count")
            tensor[row, :option_count] = torch.tensor(values, dtype=torch.long)

    batch.update(token_fields)
    batch.update(
        {
            "tokens": tokens,
            "token_mask": token_mask,
            "option_card_ids": option_card_ids,
            "option_attack_ids": option_attack_ids,
            "option_special_conditions": option_special_conditions,
        }
    )
    if version == 2:
        return batch

    token_semantics = torch.zeros(
        batch_size, max_tokens, SEMANTIC_FEATURE_SIZE, dtype=torch.float32
    )
    option_semantics = torch.zeros(
        batch_size, max_options, SEMANTIC_FEATURE_SIZE, dtype=torch.float32
    )
    history_counts = [len(example.decision.history_features or []) for example in examples]
    max_history = max(1, max(history_counts))
    history_features = torch.zeros(
        batch_size, max_history, HISTORY_FEATURE_SIZE, dtype=torch.float32
    )
    history_mask = torch.zeros(batch_size, max_history, dtype=torch.bool)
    history_fields = {
        "history_types": torch.zeros(batch_size, max_history, dtype=torch.long),
        "history_owners": torch.zeros(batch_size, max_history, dtype=torch.long),
        "history_card_ids": torch.zeros(batch_size, max_history, dtype=torch.long),
        "history_target_card_ids": torch.zeros(batch_size, max_history, dtype=torch.long),
        "history_attack_ids": torch.zeros(batch_size, max_history, dtype=torch.long),
        "history_from_zones": torch.zeros(batch_size, max_history, dtype=torch.long),
        "history_to_zones": torch.zeros(batch_size, max_history, dtype=torch.long),
    }
    for row, example in enumerate(examples):
        decision = example.decision
        token_count = token_counts[row]
        option_count = len(decision.options)
        history_count = history_counts[row]
        if decision.token_semantics is None or len(decision.token_semantics) != token_count:
            raise ValueError("token_semantics must match the V3 token count")
        if decision.option_semantics is None or len(decision.option_semantics) != option_count:
            raise ValueError("option_semantics must match the V3 option count")
        if token_count:
            token_semantics[row, :token_count] = torch.tensor(
                [expand_semantic_features(item) for item in decision.token_semantics],
                dtype=torch.float32,
            )
        option_semantics[row, :option_count] = torch.tensor(
            [expand_semantic_features(item) for item in decision.option_semantics],
            dtype=torch.float32,
        )
        if history_count:
            history_features[row, :history_count] = torch.tensor(
                decision.history_features, dtype=torch.float32
            )
        history_mask[row, :history_count] = True
        for name, tensor in history_fields.items():
            values = getattr(decision, name)
            if values is None or len(values) != history_count:
                raise ValueError(f"{name} must match the V3 history count")
            tensor[row, :history_count] = torch.tensor(values, dtype=torch.long)

    batch.update(history_fields)
    batch.update(
        {
            "token_semantics": token_semantics,
            "option_semantics": option_semantics,
            "history_features": history_features,
            "history_mask": history_mask,
        }
    )
    return batch
