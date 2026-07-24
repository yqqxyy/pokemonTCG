"""Counterfactual condition ablations for a trained macro Executor."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .action_space import deterministic_subset
from .executor_data import (
    EXECUTOR_CONDITION_SIZE,
    EXECUTOR_PLAN_CONDITION_SLICE,
    EXECUTOR_PROGRESS_CONDITION_SLICE,
    ExecutorDataset,
    ExecutorExample,
    collate_executor,
    load_executor_dataset,
)
from .model import build_model
from .train_bc import TokenBucketBatchSampler, resolve_device
from .train_executor import (
    PHYSICAL_TARGET_KIND,
    SEMANTIC_TARGET_KIND,
    evaluation_report,
    move_executor_batch,
)

Prediction = tuple[tuple[int, ...], tuple[int, ...]]


def _loader(dataset: ExecutorDataset, batch_size: int, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=TokenBucketBatchSampler(
            dataset, batch_size, shuffle=False, seed=seed  # type: ignore[arg-type]
        ),
        collate_fn=collate_executor,
    )


def _example_key(metadata: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(metadata["split_group"]),
        str(metadata["state_id"]),
        str(metadata["fingerprint"]),
    )


def prediction_map(
    model,
    loader: DataLoader,
    device: torch.device,
) -> dict[tuple[str, str, str], Prediction]:
    model.eval()
    predictions: dict[tuple[str, str, str], Prediction] = {}
    with torch.no_grad():
        for raw_batch in loader:
            metadata = raw_batch["metadata"]
            if not isinstance(metadata, list):
                raise TypeError("Executor metadata must remain a Python list")
            batch = move_executor_batch(raw_batch, device)
            logits, _ = model(batch)
            option_mask = batch["option_mask"]
            minimum = batch["minimum"]
            maximum = batch["maximum"]
            class_ids = batch["equivalence_class_ids"]
            assert isinstance(option_mask, Tensor)
            assert isinstance(minimum, Tensor)
            assert isinstance(maximum, Tensor)
            assert isinstance(class_ids, Tensor)
            for row, item in enumerate(metadata):
                option_count = int(option_mask[row].sum())
                action = deterministic_subset(
                    logits[row, :option_count],
                    int(minimum[row]),
                    int(maximum[row]),
                )
                key = _example_key(item)
                if key in predictions:
                    raise ValueError(f"Duplicate Executor diagnostic key: {key}")
                semantic_action = tuple(
                    sorted(int(class_ids[row, index]) for index in action)
                )
                predictions[key] = (tuple(action), semantic_action)
    return predictions


def _replace_segment(
    target: ExecutorExample,
    source: ExecutorExample,
    segment: slice,
) -> ExecutorExample:
    condition = list(target.condition)
    condition[segment] = source.condition[segment]
    return replace(target, condition=condition)


def _shuffle_conditions(
    dataset: ExecutorDataset,
    *,
    segment: slice,
    matching_fields: tuple[str, ...],
    require_different_plan: bool,
    seed: int,
) -> tuple[ExecutorDataset, dict[str, Any]]:
    generator = random.Random(seed)
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, example in enumerate(dataset.examples):
        key = tuple(getattr(example, field) for field in matching_fields)
        groups.setdefault(key, []).append(index)

    changed = 0
    eligible = 0
    sources: list[int] = []
    for index, example in enumerate(dataset.examples):
        key = tuple(getattr(example, field) for field in matching_fields)
        candidates = [
            candidate
            for candidate in groups[key]
            if candidate != index
            and (
                not require_different_plan
                or dataset.examples[candidate].plan_type != example.plan_type
            )
            and dataset.examples[candidate].condition[segment]
            != example.condition[segment]
        ]
        if candidates:
            eligible += 1
            source = generator.choice(candidates)
            changed += 1
        else:
            source = index
        sources.append(source)

    examples = [
        _replace_segment(example, dataset.examples[source], segment)
        for example, source in zip(dataset.examples, sources, strict=True)
    ]
    return ExecutorDataset(examples), {
        "examples": len(examples),
        "eligible_examples": eligible,
        "changed_examples": changed,
        "changed_fraction": changed / max(1, len(examples)),
        "matching_fields": list(matching_fields),
        "require_different_plan": require_different_plan,
        "seed": seed,
    }


def condition_variants(
    dataset: ExecutorDataset,
    *,
    seed: int,
) -> dict[str, tuple[ExecutorDataset, dict[str, Any]]]:
    zero = ExecutorDataset(
        [
            replace(example, condition=[0.0] * EXECUTOR_CONDITION_SIZE)
            for example in dataset.examples
        ]
    )
    shuffled_plan = _shuffle_conditions(
        dataset,
        segment=EXECUTOR_PLAN_CONDITION_SLICE,
        matching_fields=("phase", "context"),
        require_different_plan=True,
        seed=seed,
    )
    shuffled_progress = _shuffle_conditions(
        dataset,
        segment=EXECUTOR_PROGRESS_CONDITION_SLICE,
        matching_fields=("phase", "context", "plan_type"),
        require_different_plan=False,
        seed=seed + 1,
    )
    return {
        "correct": (
            dataset,
            {
                "examples": len(dataset),
                "changed_examples": 0,
                "changed_fraction": 0.0,
            },
        ),
        "zero_condition": (
            zero,
            {
                "examples": len(dataset),
                "changed_examples": len(dataset),
                "changed_fraction": 1.0,
            },
        ),
        "shuffled_plan": shuffled_plan,
        "shuffled_progress": shuffled_progress,
    }


def _action_change_rate(
    reference: dict[tuple[str, str, str], Prediction],
    candidate: dict[tuple[str, str, str], Prediction],
    *,
    semantic: bool,
) -> float:
    if reference.keys() != candidate.keys():
        raise ValueError("Condition variants produced different diagnostic examples")
    index = 1 if semantic else 0
    return sum(
        reference[key][index] != candidate[key][index] for key in reference
    ) / max(1, len(reference))


def _overall_delta(
    reference: dict[str, float | int],
    candidate: dict[str, float | int],
) -> dict[str, float]:
    return {
        "weighted_nll": float(candidate["weighted_nll"])
        - float(reference["weighted_nll"]),
        "nll": float(candidate["nll"]) - float(reference["nll"]),
        "exact_accuracy": float(candidate["exact_accuracy"])
        - float(reference["exact_accuracy"]),
        "empirical_action_probability": float(
            candidate["empirical_action_probability"]
        )
        - float(reference["empirical_action_probability"]),
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if saved.get("checkpoint_kind") not in {
        "plan_conditioned_executor_v1",
        "plan_conditioned_executor_v2",
    }:
        raise ValueError("Checkpoint is not a supported plan-conditioned Executor")
    model_config = saved.get("model_config")
    if not isinstance(model_config, dict):
        raise TypeError("Executor checkpoint is missing model_config")
    if int(model_config.get("condition_feature_size", 0)) != EXECUTOR_CONDITION_SIZE:
        raise ValueError("Executor condition schema does not match this code")

    minimum_weight = float(
        (saved.get("executor_config") or {}).get("minimum_consensus_weight", 0.25)
    )
    target_kind = str(
        (saved.get("executor_config") or {}).get(
            "target_kind", PHYSICAL_TARGET_KIND
        )
    )
    # Compatibility with the descriptive V1 label written before target kinds
    # became explicit.
    if target_kind == "hidden_world_expected_subset_nll":
        target_kind = PHYSICAL_TARGET_KIND
    if target_kind not in {PHYSICAL_TARGET_KIND, SEMANTIC_TARGET_KIND}:
        raise ValueError(f"Unsupported Executor target kind: {target_kind}")
    dataset, dataset_summary = load_executor_dataset(
        args.input,
        minimum_consensus_weight=minimum_weight,
    )
    if args.split == "all":
        selected = dataset
        selected_groups = sorted(
            {example.split_group for example in dataset.examples}
        )
    else:
        manifest = saved.get("split_manifest")
        if not isinstance(manifest, dict) or not isinstance(
            manifest.get(args.split), list
        ):
            raise TypeError("Checkpoint is missing the requested split manifest")
        selected_groups = [str(value) for value in manifest[args.split]]
        group_set = set(selected_groups)
        selected = ExecutorDataset(
            [
                example
                for example in dataset.examples
                if example.split_group in group_set
            ]
        )
        found = {example.split_group for example in selected.examples}
        missing = sorted(group_set.difference(found))
        if missing:
            raise ValueError(
                "Executor input does not contain checkpoint split groups: "
                + ", ".join(missing)
            )
    if not selected.examples:
        raise ValueError("Selected Executor diagnostic split is empty")

    device = resolve_device(args.device)
    model = build_model(model_config).to(device)
    model.load_state_dict(saved["model_state_dict"])
    variants = condition_variants(selected, seed=args.seed)
    reports: dict[str, dict[str, Any]] = {}
    predictions: dict[
        str, dict[tuple[str, str, str], Prediction]
    ] = {}
    for name, (variant, manipulation) in variants.items():
        loader = _loader(variant, args.batch_size, args.seed)
        reports[name] = {
            "manipulation": manipulation,
            "metrics": evaluation_report(
                model, loader, device, target_kind
            ),
        }
        predictions[name] = prediction_map(model, loader, device)

    reference_overall = reports["correct"]["metrics"]["overall"]
    reference_predictions = predictions["correct"]
    comparison: dict[str, dict[str, Any]] = {}
    for name in ("zero_condition", "shuffled_plan", "shuffled_progress"):
        physical_change = _action_change_rate(
            reference_predictions,
            predictions[name],
            semantic=False,
        )
        semantic_change = _action_change_rate(
            reference_predictions,
            predictions[name],
            semantic=True,
        )
        comparison[name] = {
            "delta_from_correct": _overall_delta(
                reference_overall,
                reports[name]["metrics"]["overall"],
            ),
            "action_change_rate": (
                semantic_change
                if target_kind == SEMANTIC_TARGET_KIND
                else physical_change
            ),
            "physical_action_change_rate": physical_change,
            "semantic_action_change_rate": semantic_change,
        }

    result = {
        "diagnostic_kind": "executor_condition_counterfactual_v1",
        "checkpoint": str(checkpoint_path),
        "input": str(args.input.expanduser().resolve()),
        "split": args.split,
        "target_kind": target_kind,
        "split_groups": selected_groups,
        "examples": len(selected),
        "dataset_summary": dataset_summary,
        "condition_schema": {
            "total_size": EXECUTOR_CONDITION_SIZE,
            "plan_slice": [
                EXECUTOR_PLAN_CONDITION_SLICE.start,
                EXECUTOR_PLAN_CONDITION_SLICE.stop,
            ],
            "progress_slice": [
                EXECUTOR_PROGRESS_CONDITION_SLICE.start,
                EXECUTOR_PROGRESS_CONDITION_SLICE.stop,
            ],
        },
        "variants": reports,
        "comparison": comparison,
    }
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "split": args.split,
                "examples": len(selected),
                "correct": reference_overall,
                "comparison": comparison,
            },
            separators=(",", ":"),
        )
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure whether a macro Executor actually uses Plan and Progress."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("validation", "test", "train", "all"),
        default="test",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    diagnose(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
