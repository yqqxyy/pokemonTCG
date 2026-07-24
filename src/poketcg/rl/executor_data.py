"""Public-information targets for the plan-conditioned macro Executor."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .data import BCExample, collate_bc
from .features import SEMANTIC_TAGS, EncodedDecision
from .macro_plan import LibraryOutStrategyProfile, MacroPlanType

EXECUTOR_DATA_VERSION = 1

# Only deployable Library-Out v2 plans belong in the new executor. Legacy generic
# macro types remain readable in macro_plan.py, but mixing both ontologies would
# make the condition vector ambiguous.
EXECUTOR_PLAN_TYPES = (
    MacroPlanType.BASELINE_V1.value,
    MacroPlanType.MILL_FOUR_NOW.value,
    MacroPlanType.FIND_ANCIENT_SUPPORTER.value,
    MacroPlanType.PREPARE_NEXT_GREAT_TUSK.value,
    MacroPlanType.BUILD_CRUSTLE_WALL.value,
    MacroPlanType.ENABLE_NEUTRALIZATION_WALL.value,
    MacroPlanType.GUST_STALL_TARGET.value,
    MacroPlanType.HAND_DISRUPTION_STALL.value,
    MacroPlanType.HEAL_OR_ROTATE_WALL.value,
    MacroPlanType.PRIZE_RACE_PIVOT.value,
    MacroPlanType.PRESERVE_DECK_AND_CHAIN.value,
)

_PROFILE = LibraryOutStrategyProfile()
LIBRARYOUT_CARD_IDS = tuple(
    sorted(
        {
            _PROFILE.great_tusk,
            _PROFILE.dwebble,
            _PROFILE.crustle,
            _PROFILE.terrakion,
            _PROFILE.buddy_buddy_poffin,
            _PROFILE.ultra_ball,
            _PROFILE.pokegear,
            _PROFILE.switch,
            _PROFILE.fighting_gong,
            _PROFILE.jumbo_ice_cream,
            _PROFILE.poke_pad,
            _PROFILE.boss_orders,
            _PROFILE.explorers_guidance,
            _PROFILE.colress_tenacity,
            _PROFILE.xerosic_machinations,
            _PROFILE.lisia_appeal,
            _PROFILE.neutralization_zone,
            _PROFILE.mist_energy,
            _PROFILE.rock_fighting_energy,
        }
    )
)
LIBRARYOUT_ATTACK_IDS = (_PROFILE.land_collapse, _PROFILE.giant_tusk)

_CARD_BUCKETS = len(LIBRARYOUT_CARD_IDS) + 1
_ATTACK_BUCKETS = len(LIBRARYOUT_ATTACK_IDS) + 1
_PLAN_SCALARS = 8
_PROGRESS_SCALARS = 8
_CONTEXT_BUCKETS = 64
_OPTION_TYPE_BUCKETS = 32

EXECUTOR_CONDITION_SIZE = (
    len(EXECUTOR_PLAN_TYPES)
    + len(SEMANTIC_TAGS)
    + _CARD_BUCKETS * 4  # primary, target, preferred, preserved
    + _ATTACK_BUCKETS * 2  # requested and preferred attacks
    + _PLAN_SCALARS
    + _PROGRESS_SCALARS
    + _CONTEXT_BUCKETS
    + _OPTION_TYPE_BUCKETS
    + _CARD_BUCKETS  # cards already played while executing the plan
    + _ATTACK_BUCKETS  # attacks already used
)


def _one_hot(value: Any, vocabulary: tuple[Any, ...]) -> list[float]:
    result = [0.0] * len(vocabulary)
    if value in vocabulary:
        result[vocabulary.index(value)] = 1.0
    return result


def _id_bag(values: list[int] | tuple[int, ...], vocabulary: tuple[int, ...]) -> list[float]:
    """Encode known IDs plus one explicit out-of-vocabulary bucket."""
    result = [0.0] * (len(vocabulary) + 1)
    lookup = {value: index for index, value in enumerate(vocabulary)}
    for raw in values:
        value = int(raw)
        result[lookup.get(value, len(vocabulary))] = 1.0
    return result


def encode_executor_condition(plan: dict[str, Any], progress: dict[str, Any]) -> list[float]:
    """Encode only information available to the deployed Executor."""
    plan_type = str(plan["plan_type"])
    if plan_type not in EXECUTOR_PLAN_TYPES:
        raise ValueError(f"Unsupported Executor plan type: {plan_type}")

    features: list[float] = []
    features.extend(_one_hot(plan_type, EXECUTOR_PLAN_TYPES))
    desired_tags = {str(value) for value in plan.get("desired_tags") or ()}
    features.extend(float(tag in desired_tags) for tag in SEMANTIC_TAGS)

    primary = plan.get("primary_card_id")
    target = plan.get("target_card_id")
    attack = plan.get("attack_id")
    features.extend(
        _id_bag([] if primary is None else [int(primary)], LIBRARYOUT_CARD_IDS)
    )
    features.extend(
        _id_bag([] if target is None else [int(target)], LIBRARYOUT_CARD_IDS)
    )
    features.extend(
        _id_bag(
            [int(value) for value in plan.get("preferred_card_ids") or ()],
            LIBRARYOUT_CARD_IDS,
        )
    )
    features.extend(
        _id_bag(
            [int(value) for value in plan.get("preserve_card_ids") or ()],
            LIBRARYOUT_CARD_IDS,
        )
    )
    features.extend(
        _id_bag([] if attack is None else [int(attack)], LIBRARYOUT_ATTACK_IDS)
    )
    features.extend(
        _id_bag(
            [int(value) for value in plan.get("preferred_attack_ids") or ()],
            LIBRARYOUT_ATTACK_IDS,
        )
    )

    root_options = ((plan.get("root_action") or {}).get("options") or ())
    maximum_steps = max(1, int(plan.get("maximum_steps", 32)))
    features.extend(
        (
            float(plan.get("feasibility_score", 1.0)),
            float(bool(plan.get("require_attack", False))),
            min(maximum_steps / 32.0, 2.0),
            min(len(root_options) / 4.0, 1.0),
            min(len(plan.get("preconditions") or ()) / 8.0, 1.0),
            min(len(plan.get("success_conditions") or ()) / 8.0, 1.0),
            min(len(plan.get("public_signals") or ()) / 16.0, 1.0),
            min(len(plan.get("sources") or ()) / 4.0, 1.0),
        )
    )

    decisions = int(progress.get("decisions", 0))
    plan_hits = int(progress.get("plan_hits", 0))
    contexts = [int(value) for value in progress.get("contexts") or ()]
    option_types = [int(value) for value in progress.get("option_types") or ()]
    played_cards = [int(value) for value in progress.get("played_card_ids") or ()]
    attack_ids = [int(value) for value in progress.get("attack_ids") or ()]
    features.extend(
        (
            float(int(progress.get("owner_player", 0))),
            min(int(progress.get("start_turn", 0)) / 100.0, 1.0),
            min(decisions / maximum_steps, 1.0),
            plan_hits / max(1, decisions),
            float(bool(attack_ids)),
            min(len(contexts) / maximum_steps, 1.0),
            min(len(played_cards) / 16.0, 1.0),
            min(len(attack_ids) / 4.0, 1.0),
        )
    )
    features.extend(
        _one_hot(contexts[-1] if contexts else None, tuple(range(_CONTEXT_BUCKETS)))
    )
    features.extend(
        _one_hot(
            option_types[-1] if option_types else None,
            tuple(range(_OPTION_TYPE_BUCKETS)),
        )
    )
    features.extend(_id_bag(played_cards, LIBRARYOUT_CARD_IDS))
    features.extend(_id_bag(attack_ids, LIBRARYOUT_ATTACK_IDS))
    if len(features) != EXECUTOR_CONDITION_SIZE:
        raise AssertionError(
            f"Expected {EXECUTOR_CONDITION_SIZE} Executor features, got {len(features)}"
        )
    return features


def executor_turn_phase(turn: int) -> str:
    if turn <= 4:
        return "early"
    if turn <= 7:
        return "mid"
    return "late"


@dataclass(slots=True)
class ExecutorExample:
    decision: EncodedDecision
    condition: list[float]
    modal_action: list[int]
    inclusion_target: list[float]
    action_distribution: list[tuple[list[int], float]]
    consensus_rate: float
    normalized_entropy: float
    example_weight: float
    observation_count: int
    world_count: int
    split_group: str
    state_id: str
    input_fingerprint: str
    plan_type: str
    phase: str
    context: int
    opponent: str

    def as_bc_example(self) -> BCExample:
        return BCExample(
            decision=self.decision,
            action=self.modal_action,
            value_target=0.0,
            player=0,
            game=0,
        )


class ExecutorDataset(Dataset[ExecutorExample]):
    def __init__(self, examples: list[ExecutorExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ExecutorExample:
        return self.examples[index]


def _canonical_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


def sanitize_public_decision(value: dict[str, Any]) -> EncodedDecision:
    """Remove determinized prize identities leaked by Search observations.

    Older oracle files were encoded before FeatureEncoder V3 stopped emitting
    zone-6 tokens. Keeping this compatibility pass lets those expensive oracle
    runs be trained without granting the model information unavailable online.
    """
    decision = EncodedDecision.from_dict(value)
    token_zones = decision.token_zones or []
    keep = [index for index, zone in enumerate(token_zones) if int(zone) != 6]
    token_fields = (
        "tokens",
        "token_card_ids",
        "token_kinds",
        "token_zones",
        "token_owners",
        "token_slots",
        "token_card_types",
        "token_energy_types",
        "token_weaknesses",
        "token_resistances",
        "token_semantics",
    )
    if len(keep) != len(token_zones):
        for name in token_fields:
            values = getattr(decision, name)
            if values is None:
                continue
            if len(values) != len(token_zones):
                raise ValueError(f"{name} does not match token_zones")
            setattr(decision, name, [values[index] for index in keep])

    for index, area in enumerate(decision.areas):
        if int(area) != 6:
            continue
        # Preserve generic player/index/count fields, but erase card-derived
        # features and identities for a face-down prize choice.
        decision.options[index][6:] = [0.0] * (len(decision.options[index]) - 6)
        if decision.option_card_ids is not None:
            decision.option_card_ids[index] = 0
        if decision.option_attack_ids is not None:
            decision.option_attack_ids[index] = 0
        if decision.option_special_conditions is not None:
            decision.option_special_conditions[index] = 0
        if decision.option_semantics is not None:
            decision.option_semantics[index] = [
                0.0
            ] * len(decision.option_semantics[index])
    return decision


def _legal_action(action: tuple[int, ...], decision: EncodedDecision) -> bool:
    return (
        len(action) == len(set(action))
        and decision.minimum <= len(action) <= decision.maximum
        and all(0 <= index < len(decision.options) for index in action)
    )


def _aggregate_group(
    rows: list[dict[str, Any]],
    *,
    minimum_consensus_weight: float,
) -> ExecutorExample:
    first = rows[0]
    executor_input = first["_public_executor_input"]
    decision = EncodedDecision.from_dict(executor_input["decision"])
    if decision.version != 3:
        raise ValueError("Plan-conditioned Executor training requires FeatureEncoder V3")
    plan = executor_input["plan"]
    progress = executor_input["progress"]

    votes: Counter[tuple[int, ...]] = Counter()
    seen_worlds: dict[int, tuple[int, ...]] = {}
    for row in rows:
        if row["_public_executor_input"] != executor_input:
            raise ValueError("Executor aggregation group contains different public inputs")
        action = tuple(sorted(int(index) for index in row["target_action"]))
        if not _legal_action(action, decision):
            raise ValueError(
                f"Invalid Executor target action {action} for state {first['state_id']}"
            )
        world = int(row["determinization_id"])
        previous = seen_worlds.setdefault(world, action)
        if previous != action:
            raise ValueError("One hidden world assigns two actions to one public input")
        votes[action] += 1

    observation_count = sum(votes.values())
    modal_action, modal_count = min(
        votes.items(),
        key=lambda item: (-item[1], item[0]),
    )
    inclusion = [0.0] * len(decision.options)
    for action, count in votes.items():
        for index in action:
            inclusion[index] += count / observation_count

    probabilities = [count / observation_count for count in votes.values()]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    legal_count = sum(
        math.comb(len(decision.options), count)
        for count in range(decision.minimum, decision.maximum + 1)
    )
    normalized_entropy = entropy / math.log(max(2, legal_count))
    normalized_entropy = min(max(normalized_entropy, 0.0), 1.0)
    consensus_rate = modal_count / observation_count
    example_weight = minimum_consensus_weight + (
        1.0 - minimum_consensus_weight
    ) * (1.0 - normalized_entropy)

    return ExecutorExample(
        decision=decision,
        condition=encode_executor_condition(plan, progress),
        modal_action=list(modal_action),
        inclusion_target=inclusion,
        action_distribution=[
            (list(action), count / observation_count)
            for action, count in sorted(votes.items())
        ],
        consensus_rate=consensus_rate,
        normalized_entropy=normalized_entropy,
        example_weight=example_weight,
        observation_count=observation_count,
        world_count=len(seen_worlds),
        split_group=str(first["split_group"]),
        state_id=str(first["state_id"]),
        input_fingerprint=_canonical_hash(executor_input),
        plan_type=str(first["plan_type"]),
        phase=executor_turn_phase(int(first["turn"])),
        context=int(decision.context),
        opponent=str(first["opponent"]),
    )


def load_executor_dataset(
    path: str | Path,
    *,
    minimum_consensus_weight: float = 0.25,
) -> tuple[ExecutorDataset, dict[str, Any]]:
    """Aggregate hidden-world actions for identical deployable public inputs."""
    if not 0.0 <= minimum_consensus_weight <= 1.0:
        raise ValueError("minimum_consensus_weight must be in [0, 1]")
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    raw_rows = 0
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != 3:
                raise ValueError("Executor input must use macro dataset schema 3")
            if row.get("example_type") != "macro_executor_action":
                raise ValueError("Input contains a non-Executor example")
            executor_input = row.get("executor_input")
            if not isinstance(executor_input, dict):
                raise ValueError("Executor row is missing executor_input")
            public_input = {
                "decision": sanitize_public_decision(
                    executor_input["decision"]
                ).to_dict(),
                "plan": executor_input["plan"],
                "progress": executor_input["progress"],
            }
            row["_public_executor_input"] = public_input
            fingerprint = _canonical_hash(public_input)
            key = (
                str(row["split_group"]),
                str(row["state_id"]),
                str(row["plan_id"]),
                fingerprint,
            )
            groups[key].append(row)
            raw_rows += 1
    if not groups:
        raise ValueError("Executor dataset is empty")

    examples = [
        _aggregate_group(rows, minimum_consensus_weight=minimum_consensus_weight)
        for _, rows in sorted(groups.items())
    ]
    summary = {
        "executor_data_version": EXECUTOR_DATA_VERSION,
        "public_sanitization": "hidden_prize_identity_v1",
        "raw_rows": raw_rows,
        "public_inputs": len(examples),
        "split_groups": len({example.split_group for example in examples}),
        "states": len({example.state_id for example in examples}),
        "mean_world_count": sum(example.world_count for example in examples) / len(examples),
        "mean_consensus_rate": sum(example.consensus_rate for example in examples)
        / len(examples),
        "mean_normalized_entropy": sum(
            example.normalized_entropy for example in examples
        )
        / len(examples),
        "ambiguous_inputs": sum(example.consensus_rate < 1.0 for example in examples),
        "plan_types": dict(Counter(example.plan_type for example in examples)),
        "phases": dict(Counter(example.phase for example in examples)),
        "contexts": {
            str(key): value
            for key, value in sorted(Counter(example.context for example in examples).items())
        },
    }
    return ExecutorDataset(examples), summary


def split_executor_dataset(
    dataset: ExecutorDataset,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[ExecutorDataset, ExecutorDataset, ExecutorDataset, dict[str, list[str]]]:
    """Split whole games, approximately stratified by opponent and turn phase."""
    if validation_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("validation_fraction and test_fraction must be positive")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than one")

    by_group: dict[str, list[ExecutorExample]] = defaultdict(list)
    for example in dataset.examples:
        by_group[example.split_group].append(example)
    if len(by_group) < 3:
        raise ValueError("Executor data needs at least three split groups")

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group, examples in by_group.items():
        opponent = Counter(example.opponent for example in examples).most_common(1)[0][0]
        phase = Counter(example.phase for example in examples).most_common(1)[0][0]
        strata[(opponent, phase)].append(group)

    generator = random.Random(seed)
    assignments = {"train": [], "validation": [], "test": []}
    for groups in strata.values():
        groups = sorted(groups)
        generator.shuffle(groups)
        if len(groups) >= 3:
            validation_count = max(1, round(len(groups) * validation_fraction))
            test_count = max(1, round(len(groups) * test_fraction))
            while validation_count + test_count >= len(groups):
                if validation_count >= test_count and validation_count > 1:
                    validation_count -= 1
                elif test_count > 1:
                    test_count -= 1
                else:
                    break
        else:
            validation_count = 0
            test_count = 0
        assignments["validation"].extend(groups[:validation_count])
        assignments["test"].extend(
            groups[validation_count : validation_count + test_count]
        )
        assignments["train"].extend(groups[validation_count + test_count :])

    # Very small pilot datasets may have sparse strata. Preserve whole groups while
    # ensuring that smoke tests still have both held-out partitions.
    for destination in ("validation", "test"):
        if not assignments[destination]:
            if len(assignments["train"]) <= 1:
                raise ValueError("Not enough split groups to construct held-out sets")
            assignments[destination].append(assignments["train"].pop())

    group_sets = {name: set(groups) for name, groups in assignments.items()}
    if any(
        group_sets[left].intersection(group_sets[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise AssertionError("Executor split groups overlap")

    def subset(name: str) -> ExecutorDataset:
        groups = group_sets[name]
        return ExecutorDataset(
            [example for example in dataset.examples if example.split_group in groups]
        )

    return subset("train"), subset("validation"), subset("test"), {
        name: sorted(groups) for name, groups in assignments.items()
    }


def collate_executor(examples: list[ExecutorExample]) -> dict[str, Tensor | list[dict[str, Any]]]:
    batch: dict[str, Tensor | list[dict[str, Any]]] = collate_bc(
        [example.as_bc_example() for example in examples]
    )
    max_options = int(batch["option_mask"].shape[1])  # type: ignore[union-attr]
    inclusion_target = torch.zeros(len(examples), max_options, dtype=torch.float32)
    for row, example in enumerate(examples):
        inclusion_target[row, : len(example.inclusion_target)] = torch.tensor(
            example.inclusion_target, dtype=torch.float32
        )
    batch.update(
        {
            "condition": torch.tensor(
                [example.condition for example in examples], dtype=torch.float32
            ),
            "inclusion_target": inclusion_target,
            "example_weight": torch.tensor(
                [example.example_weight for example in examples], dtype=torch.float32
            ),
            "consensus_rate": torch.tensor(
                [example.consensus_rate for example in examples], dtype=torch.float32
            ),
            "normalized_entropy": torch.tensor(
                [example.normalized_entropy for example in examples], dtype=torch.float32
            ),
            "observation_count": torch.tensor(
                [example.observation_count for example in examples], dtype=torch.long
            ),
            "metadata": [
                {
                    "plan_type": example.plan_type,
                    "phase": example.phase,
                    "context": example.context,
                    "opponent": example.opponent,
                    "split_group": example.split_group,
                    "state_id": example.state_id,
                    "fingerprint": example.input_fingerprint,
                    "action_distribution": example.action_distribution,
                }
                for example in examples
            ],
        }
    )
    return batch
