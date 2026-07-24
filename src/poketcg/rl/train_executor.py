"""Train the closed-loop, plan-conditioned Library-Out Executor."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .action_space import (
    constrained_log_partition,
    deterministic_subset,
    semantic_subset_log_probability,
)
from .executor_data import (
    EXECUTOR_CONDITION_SIZE,
    ExecutorDataset,
    collate_executor,
    load_executor_dataset,
    split_executor_dataset,
)
from .model import PolicyValueModel, action_space_version, build_model
from .train_bc import TokenBucketBatchSampler, resolve_device

PHYSICAL_TARGET_KIND = "physical_subset_v1"
SEMANTIC_TARGET_KIND = "semantic_equivalence_v2"
CHECKPOINT_KINDS = {
    PHYSICAL_TARGET_KIND: "plan_conditioned_executor_v1",
    SEMANTIC_TARGET_KIND: "plan_conditioned_executor_v2",
}


@dataclass(slots=True)
class ExecutorEpochMetrics:
    epoch: int
    train_weighted_nll: float
    train_target_accuracy: float
    train_physical_accuracy: float
    train_semantic_accuracy: float
    validation_weighted_nll: float
    validation_nll: float
    validation_target_accuracy: float
    validation_physical_accuracy: float
    validation_semantic_accuracy: float


def move_executor_batch(
    batch: dict[str, Tensor | list[dict[str, Any]]],
    device: torch.device,
) -> dict[str, Tensor | list[dict[str, Any]]]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def expected_subset_nll(
    policy_logits: Tensor,
    batch: dict[str, Tensor | list[dict[str, Any]]],
) -> Tensor:
    """Exact E_target[-log P(S)] for additive, cardinality-constrained sets.

    Hidden-world votes are represented by marginal inclusion probabilities.
    Since log P(S) = sum(i in S) logit_i - log Z, those marginals are sufficient
    for the exact expectation, including empty and multi-select actions.
    """
    option_mask = batch["option_mask"]
    inclusion_target = batch["inclusion_target"]
    minimum = batch["minimum"]
    maximum = batch["maximum"]
    assert isinstance(option_mask, Tensor)
    assert isinstance(inclusion_target, Tensor)
    assert isinstance(minimum, Tensor)
    assert isinstance(maximum, Tensor)
    values: list[Tensor] = []
    for row in range(policy_logits.shape[0]):
        count = int(option_mask[row].sum())
        logits = policy_logits[row, :count]
        expected_score = (
            inclusion_target[row, :count] * logits
        ).sum()
        values.append(
            constrained_log_partition(
                logits,
                int(minimum[row]),
                int(maximum[row]),
            )
            - expected_score
        )
    return torch.stack(values)


def expected_semantic_subset_nll(
    policy_logits: Tensor,
    batch: dict[str, Tensor | list[dict[str, Any]]],
) -> Tensor:
    """Exact target NLL after summing all equivalent physical implementations."""
    option_mask = batch["option_mask"]
    equivalence_class_ids = batch["equivalence_class_ids"]
    minimum = batch["minimum"]
    maximum = batch["maximum"]
    metadata = batch["metadata"]
    assert isinstance(option_mask, Tensor)
    assert isinstance(equivalence_class_ids, Tensor)
    assert isinstance(minimum, Tensor)
    assert isinstance(maximum, Tensor)
    assert isinstance(metadata, list)
    values: list[Tensor] = []
    for row in range(policy_logits.shape[0]):
        count = int(option_mask[row].sum())
        logits = policy_logits[row, :count]
        class_ids = equivalence_class_ids[row, :count]
        expected_log_probability = logits.new_zeros(())
        distribution = metadata[row]["semantic_action_distribution"]
        for semantic_action, probability in distribution:
            action_tensor = torch.tensor(
                semantic_action,
                dtype=torch.long,
                device=logits.device,
            )
            expected_log_probability = expected_log_probability + float(
                probability
            ) * semantic_subset_log_probability(
                logits,
                class_ids,
                action_tensor,
                int(minimum[row]),
                int(maximum[row]),
            )
        values.append(-expected_log_probability)
    return torch.stack(values)


def _target_nll(
    policy_logits: Tensor,
    batch: dict[str, Tensor | list[dict[str, Any]]],
    target_kind: str,
) -> Tensor:
    if target_kind == PHYSICAL_TARGET_KIND:
        return expected_subset_nll(policy_logits, batch)
    if target_kind == SEMANTIC_TARGET_KIND:
        return expected_semantic_subset_nll(policy_logits, batch)
    raise ValueError(f"Unsupported Executor target kind: {target_kind}")


def _target_weights(
    batch: dict[str, Tensor | list[dict[str, Any]]],
    target_kind: str,
) -> Tensor:
    name = (
        "semantic_example_weight"
        if target_kind == SEMANTIC_TARGET_KIND
        else "example_weight"
    )
    weights = batch[name]
    assert isinstance(weights, Tensor)
    return weights


def executor_loss(
    model: PolicyValueModel,
    batch: dict[str, Tensor | list[dict[str, Any]]],
    target_kind: str = PHYSICAL_TARGET_KIND,
) -> tuple[Tensor, Tensor, Tensor]:
    policy_logits, _ = model(batch)  # type: ignore[arg-type]
    nll = _target_nll(policy_logits, batch, target_kind)
    weights = _target_weights(batch, target_kind)
    loss = (weights * nll).sum() / weights.sum().clamp_min(1e-8)
    return loss, policy_logits, nll


def _predicted_actions(
    policy_logits: Tensor,
    batch: dict[str, Tensor | list[dict[str, Any]]],
) -> list[list[int]]:
    option_mask = batch["option_mask"]
    minimum = batch["minimum"]
    maximum = batch["maximum"]
    assert isinstance(option_mask, Tensor)
    assert isinstance(minimum, Tensor)
    assert isinstance(maximum, Tensor)
    return [
        deterministic_subset(
            policy_logits[row, : int(option_mask[row].sum())],
            int(minimum[row]),
            int(maximum[row]),
        )
        for row in range(policy_logits.shape[0])
    ]


def _batch_exact_count(
    predictions: list[list[int]],
    batch: dict[str, Tensor | list[dict[str, Any]]],
) -> int:
    target = batch["action_mask"]
    option_mask = batch["option_mask"]
    assert isinstance(target, Tensor)
    assert isinstance(option_mask, Tensor)
    correct = 0
    for row, prediction in enumerate(predictions):
        count = int(option_mask[row].sum())
        predicted_mask = torch.zeros(count, dtype=torch.bool, device=target.device)
        predicted_mask[prediction] = True
        correct += int(torch.equal(predicted_mask, target[row, :count]))
    return correct


def _semantic_prediction(
    prediction: list[int],
    equivalence_class_ids: Tensor,
) -> tuple[int, ...]:
    return tuple(sorted(int(equivalence_class_ids[index]) for index in prediction))


def _batch_semantic_count(
    predictions: list[list[int]],
    batch: dict[str, Tensor | list[dict[str, Any]]],
) -> int:
    class_ids = batch["equivalence_class_ids"]
    metadata = batch["metadata"]
    assert isinstance(class_ids, Tensor)
    assert isinstance(metadata, list)
    return sum(
        _semantic_prediction(prediction, class_ids[row])
        == tuple(metadata[row]["modal_semantic_action"])
        for row, prediction in enumerate(predictions)
    )


def _empirical_action_probability(
    prediction: list[int],
    distribution: list[tuple[list[int], float]],
) -> float:
    key = tuple(prediction)
    return sum(
        probability
        for action, probability in distribution
        if tuple(action) == key
    )


def _summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        return {
            "examples": 0,
            "weighted_nll": 0.0,
            "nll": 0.0,
            "exact_accuracy": 0.0,
            "empirical_action_probability": 0.0,
            "mean_consensus_rate": 0.0,
            "physical_exact_accuracy": 0.0,
            "semantic_exact_accuracy": 0.0,
        }
    total_weight = sum(record["target_weight"] for record in records)
    physical_weight = sum(record["physical_weight"] for record in records)
    semantic_weight = sum(record["semantic_weight"] for record in records)
    return {
        "examples": len(records),
        "weighted_nll": sum(
            record["target_weight"] * record["target_nll"] for record in records
        )
        / max(total_weight, 1e-8),
        "nll": sum(record["target_nll"] for record in records) / len(records),
        "exact_accuracy": sum(record["target_exact"] for record in records)
        / len(records),
        "empirical_action_probability": sum(
            record["target_empirical_action_probability"] for record in records
        )
        / len(records),
        "mean_consensus_rate": sum(
            record["target_consensus_rate"] for record in records
        )
        / len(records),
        "physical_weighted_nll": sum(
            record["physical_weight"] * record["physical_nll"]
            for record in records
        )
        / max(physical_weight, 1e-8),
        "physical_nll": sum(record["physical_nll"] for record in records)
        / len(records),
        "physical_exact_accuracy": sum(
            record["physical_exact"] for record in records
        )
        / len(records),
        "physical_empirical_action_probability": sum(
            record["physical_empirical_action_probability"]
            for record in records
        )
        / len(records),
        "semantic_weighted_nll": sum(
            record["semantic_weight"] * record["semantic_nll"]
            for record in records
        )
        / max(semantic_weight, 1e-8),
        "semantic_nll": sum(record["semantic_nll"] for record in records)
        / len(records),
        "semantic_exact_accuracy": sum(
            record["semantic_exact"] for record in records
        )
        / len(records),
        "semantic_empirical_action_probability": sum(
            record["semantic_empirical_action_probability"]
            for record in records
        )
        / len(records),
        "equivalence_accuracy_credit": (
            sum(record["semantic_exact"] for record in records)
            - sum(record["physical_exact"] for record in records)
        )
        / len(records),
        "mean_physical_consensus_rate": sum(
            record["physical_consensus_rate"] for record in records
        )
        / len(records),
        "mean_semantic_consensus_rate": sum(
            record["semantic_consensus_rate"] for record in records
        )
        / len(records),
    }


def _group_report(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[field])].append(record)
    return {key: _summary(groups[key]) for key in sorted(groups)}


def evaluation_report(
    model: PolicyValueModel,
    loader: DataLoader,
    device: torch.device,
    target_kind: str = PHYSICAL_TARGET_KIND,
) -> dict[str, Any]:
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for raw_batch in loader:
            metadata = raw_batch["metadata"]
            assert isinstance(metadata, list)
            batch = move_executor_batch(raw_batch, device)
            policy_logits, _ = model(batch)  # type: ignore[arg-type]
            physical_nll = expected_subset_nll(policy_logits, batch)
            semantic_nll = expected_semantic_subset_nll(policy_logits, batch)
            predictions = _predicted_actions(policy_logits, batch)
            action_mask = batch["action_mask"]
            option_mask = batch["option_mask"]
            physical_weights = batch["example_weight"]
            semantic_weights = batch["semantic_example_weight"]
            physical_consensus = batch["consensus_rate"]
            semantic_consensus = batch["semantic_consensus_rate"]
            class_ids = batch["equivalence_class_ids"]
            assert isinstance(action_mask, Tensor)
            assert isinstance(option_mask, Tensor)
            assert isinstance(physical_weights, Tensor)
            assert isinstance(semantic_weights, Tensor)
            assert isinstance(physical_consensus, Tensor)
            assert isinstance(semantic_consensus, Tensor)
            assert isinstance(class_ids, Tensor)
            for row, prediction in enumerate(predictions):
                count = int(option_mask[row].sum())
                predicted_mask = torch.zeros(
                    count, dtype=torch.bool, device=action_mask.device
                )
                predicted_mask[prediction] = True
                item = metadata[row]
                physical_exact = int(
                    torch.equal(predicted_mask, action_mask[row, :count])
                )
                semantic_prediction = _semantic_prediction(
                    prediction, class_ids[row]
                )
                semantic_exact = int(
                    semantic_prediction
                    == tuple(item["modal_semantic_action"])
                )
                physical_probability = _empirical_action_probability(
                    prediction, item["action_distribution"]
                )
                semantic_probability = _empirical_action_probability(
                    list(semantic_prediction),
                    item["semantic_action_distribution"],
                )
                if target_kind == SEMANTIC_TARGET_KIND:
                    target_nll = semantic_nll[row]
                    target_weight = semantic_weights[row]
                    target_exact = semantic_exact
                    target_probability = semantic_probability
                    target_consensus = semantic_consensus[row]
                else:
                    target_nll = physical_nll[row]
                    target_weight = physical_weights[row]
                    target_exact = physical_exact
                    target_probability = physical_probability
                    target_consensus = physical_consensus[row]
                records.append(
                    {
                        **item,
                        "target_nll": float(target_nll),
                        "target_weight": float(target_weight),
                        "target_exact": target_exact,
                        "target_empirical_action_probability": target_probability,
                        "target_consensus_rate": float(target_consensus),
                        "physical_nll": float(physical_nll[row]),
                        "physical_weight": float(physical_weights[row]),
                        "physical_exact": physical_exact,
                        "physical_empirical_action_probability": physical_probability,
                        "physical_consensus_rate": float(physical_consensus[row]),
                        "semantic_nll": float(semantic_nll[row]),
                        "semantic_weight": float(semantic_weights[row]),
                        "semantic_exact": semantic_exact,
                        "semantic_empirical_action_probability": semantic_probability,
                        "semantic_consensus_rate": float(semantic_consensus[row]),
                    }
                )
    return {
        "target_kind": target_kind,
        "overall": _summary(records),
        "by_plan_type": _group_report(records, "plan_type"),
        "by_phase": _group_report(records, "phase"),
        "by_context": _group_report(records, "context"),
        "by_opponent": _group_report(records, "opponent"),
    }


def _dataset_breakdown(dataset: ExecutorDataset) -> dict[str, Any]:
    def counts(field: str) -> dict[str, int]:
        result: dict[str, int] = defaultdict(int)
        for example in dataset.examples:
            result[str(getattr(example, field))] += 1
        return dict(sorted(result.items()))

    return {
        "examples": len(dataset),
        "split_groups": len({example.split_group for example in dataset.examples}),
        "states": len({example.state_id for example in dataset.examples}),
        "plan_types": counts("plan_type"),
        "phases": counts("phase"),
        "contexts": counts("context"),
        "opponents": counts("opponent"),
    }


def _make_loader(
    dataset: ExecutorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=TokenBucketBatchSampler(
            dataset, batch_size, shuffle=shuffle, seed=seed  # type: ignore[arg-type]
        ),
        collate_fn=collate_executor,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def _initial_model(
    args: argparse.Namespace,
) -> tuple[PolicyValueModel, dict[str, Any], dict[str, Any]]:
    initialization: dict[str, Any] = {
        "source": None,
        "loaded_tensors": 0,
        "new_tensors": [],
    }
    initial_state: dict[str, Tensor] | None = None
    if args.initialize_from is not None:
        source = args.initialize_from.expanduser().resolve()
        saved = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(saved, dict) or not isinstance(saved.get("model_config"), dict):
            raise TypeError("Initialization checkpoint must contain model_config")
        if not isinstance(saved.get("model_state_dict"), dict):
            raise TypeError("Initialization checkpoint must contain model_state_dict")
        model_config = dict(saved["model_config"])
        if model_config.get("model_type") != "transformer_v3":
            raise ValueError("Executor initialization requires a transformer_v3 checkpoint")
        if action_space_version(model_config) < 2:
            raise ValueError("Executor initialization must support action-space v2")
        existing_size = int(model_config.get("condition_feature_size", 0))
        if existing_size not in {0, EXECUTOR_CONDITION_SIZE}:
            raise ValueError("Initialization checkpoint uses another condition schema")
        model_config["condition_feature_size"] = EXECUTOR_CONDITION_SIZE
        initial_state = saved["model_state_dict"]
        initialization["source"] = str(source)
    else:
        model_config = {
            "model_type": "transformer_v3",
            "hidden_size": args.hidden_size,
            "value_bins": args.value_bins,
            "action_space_version": 2,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "card_vocab_size": args.card_vocab_size,
            "attack_vocab_size": args.attack_vocab_size,
            "use_card_semantics": True,
            "use_history": True,
            "condition_feature_size": EXECUTOR_CONDITION_SIZE,
        }

    model = build_model(model_config)
    if initial_state is not None:
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in initial_state.items()
            if key in current and current[key].shape == value.shape
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        initialization.update(
            {
                "loaded_tensors": len(compatible),
                "new_tensors": sorted(missing),
                "ignored_source_tensors": sorted(
                    set(initial_state).difference(compatible)
                ),
                "unexpected_tensors": sorted(unexpected),
            }
        )
    return model, model_config, initialization


def train(args: argparse.Namespace) -> tuple[PolicyValueModel, dict[str, Any]]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset, dataset_summary = load_executor_dataset(
        args.input,
        minimum_consensus_weight=args.minimum_consensus_weight,
    )
    train_data, validation_data, test_data, split_manifest = split_executor_dataset(
        dataset,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    train_loader = _make_loader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    validation_loader = _make_loader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.split_seed,
        num_workers=args.num_workers,
    )
    test_loader = _make_loader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.split_seed,
        num_workers=args.num_workers,
    )

    device = resolve_device(args.device)
    model, model_config, initialization = _initial_model(args)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[ExecutorEpochMetrics] = []
    best_validation_nll = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_weighted_nll = 0.0
        total_weight = 0.0
        total_examples = 0
        total_physical_correct = 0
        total_semantic_correct = 0
        for raw_batch in train_loader:
            batch = move_executor_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, policy_logits, nll = executor_loss(
                model, batch, args.target_kind
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            size = int(policy_logits.shape[0])
            weights = _target_weights(batch, args.target_kind)
            total_weighted_nll += float((weights * nll.detach()).sum())
            total_weight += float(weights.sum())
            predictions = _predicted_actions(policy_logits, batch)
            total_physical_correct += _batch_exact_count(predictions, batch)
            total_semantic_correct += _batch_semantic_count(
                predictions, batch
            )
            total_examples += size

        validation = evaluation_report(
            model, validation_loader, device, args.target_kind
        )
        overall = validation["overall"]
        train_target_correct = (
            total_semantic_correct
            if args.target_kind == SEMANTIC_TARGET_KIND
            else total_physical_correct
        )
        metrics = ExecutorEpochMetrics(
            epoch=epoch,
            train_weighted_nll=round(
                total_weighted_nll / max(total_weight, 1e-8), 6
            ),
            train_target_accuracy=round(
                train_target_correct / total_examples, 6
            ),
            train_physical_accuracy=round(
                total_physical_correct / total_examples, 6
            ),
            train_semantic_accuracy=round(
                total_semantic_correct / total_examples, 6
            ),
            validation_weighted_nll=round(float(overall["weighted_nll"]), 6),
            validation_nll=round(float(overall["nll"]), 6),
            validation_target_accuracy=round(
                float(overall["exact_accuracy"]), 6
            ),
            validation_physical_accuracy=round(
                float(overall["physical_exact_accuracy"]), 6
            ),
            validation_semantic_accuracy=round(
                float(overall["semantic_exact_accuracy"]), 6
            ),
        )
        history.append(metrics)
        print(json.dumps(asdict(metrics), separators=(",", ":")))
        if metrics.validation_weighted_nll < best_validation_nll:
            best_validation_nll = metrics.validation_weighted_nll
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Executor training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_report = evaluation_report(
        model, validation_loader, device, args.target_kind
    )
    test_report = evaluation_report(
        model, test_loader, device, args.target_kind
    )
    split_summary = {
        "train": _dataset_breakdown(train_data),
        "validation": _dataset_breakdown(validation_data),
        "test": _dataset_breakdown(test_data),
    }
    checkpoint_kind = CHECKPOINT_KINDS[args.target_kind]
    selection_metric = (
        "validation_semantic_weighted_nll"
        if args.target_kind == SEMANTIC_TARGET_KIND
        else "validation_physical_weighted_nll"
    )
    report = {
        "checkpoint_kind": checkpoint_kind,
        "target_kind": args.target_kind,
        "selected_epoch": best_epoch,
        "selection_metric": selection_metric,
        "dataset": dataset_summary,
        "splits": split_summary,
        "validation": validation_report,
        "test": test_report,
        "initialization": initialization,
    }

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_kind": checkpoint_kind,
            "model_state_dict": best_state,
            "model_config": model_config,
            "executor_config": {
                "executor_data_version": 2,
                "condition_feature_size": EXECUTOR_CONDITION_SIZE,
                "target_kind": args.target_kind,
                "semantic_equivalence": (
                    "public_option_signature_without_physical_indices"
                    if args.target_kind == SEMANTIC_TARGET_KIND
                    else None
                ),
                "minimum_consensus_weight": args.minimum_consensus_weight,
            },
            "training_config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "history": [asdict(item) for item in history],
            "selected_epoch": best_epoch,
            "selection_metric": selection_metric,
            "dataset_summary": dataset_summary,
            "split_summary": split_summary,
            "split_manifest": split_manifest,
            "validation_report": validation_report,
            "test_report": test_report,
            "initialization": initialization,
        },
        output,
    )
    report_path = (
        args.report_output.expanduser()
        if args.report_output is not None
        else output.with_suffix(".report.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "checkpoint": str(output.resolve()),
                "report": str(report_path.resolve()),
                "selected_epoch": best_epoch,
                "validation": validation_report["overall"],
                "test": test_report["overall"],
            },
            separators=(",", ":"),
        )
    )
    return model, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a V3 policy conditioned on a persistent Library-Out turn plan."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="V3 action-space-v2 checkpoint used to initialize all shared tensors.",
    )
    parser.add_argument(
        "--target-kind",
        choices=(PHYSICAL_TARGET_KIND, SEMANTIC_TARGET_KIND),
        default=SEMANTIC_TARGET_KIND,
        help="Executor V2 defaults to semantic-equivalence probability mass.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--value-bins", type=int, default=101)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--card-vocab-size", type=int, default=2048)
    parser.add_argument("--attack-vocab-size", type=int, default=2048)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-consensus-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=20260818)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
