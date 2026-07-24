"""Build separate Executor and Plan-Value datasets from macro-oracle output."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3
EXECUTOR_SELECTION = "best_per_plan_per_hidden_world"
PLAN_LABEL_KIND = "macro_oracle_upper_bound"


def _validate_record(record: dict[str, Any]) -> None:
    diagnostic = record.get("diagnostic") or {}
    kind = diagnostic.get("diagnostic_kind")
    if kind != "macro_plan_oracle_v2_libraryout":
        raise ValueError(
            "macro_data accepts only macro_plan_oracle_v2_libraryout; "
            f"received {kind!r}"
        )
    invalid = [
        plan.get("plan_id")
        for plan in diagnostic.get("plans") or ()
        if plan.get("strategy_version") != "libraryout_v2"
    ]
    if invalid:
        raise ValueError(
            "non-libraryout_v2 plans found: "
            + ", ".join(str(item) for item in invalid)
        )


def _split_group(record: dict[str, Any]) -> str:
    return f"seed{record['collector_seed']}-game{record['game']}"


def _best_trajectory_index(plan_result: dict[str, Any]) -> int:
    """Locate the retained trajectory in the original beam for auditing."""
    if "trajectories" not in plan_result:
        if "best_trajectory_index" not in plan_result:
            raise ValueError(
                "compact plan result is missing best_trajectory_index"
            )
        return int(plan_result["best_trajectory_index"])
    best = plan_result["best_trajectory"]
    for index, trajectory in enumerate(plan_result["trajectories"]):
        if trajectory == best:
            return index
    raise ValueError(
        f"best_trajectory missing from beam for plan {plan_result['plan_id']}"
    )


def iter_executor_examples(record: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield actions from only the best trajectory for each plan and world.

    The plan's quality is deliberately not used to filter Executor examples:
    Executor learns how to carry out a requested plan. Whether that plan should
    be selected in the root state is a separate Plan-Value problem.
    """
    _validate_record(record)
    diagnostic = record["diagnostic"]
    state_id = str(record["state_id"])
    split_group = _split_group(record)
    for sample in diagnostic["samples"]:
        if "error" in sample:
            continue
        world_id = int(sample["determinization_id"])
        for plan_result in sample["plans"]:
            trajectory = plan_result["best_trajectory"]
            trajectory_index = _best_trajectory_index(plan_result)
            common = {
                "schema_version": SCHEMA_VERSION,
                "example_type": "macro_executor_action",
                "trajectory_selection": EXECUTOR_SELECTION,
                "state_id": state_id,
                "split_group": split_group,
                "collector_seed": int(record["collector_seed"]),
                "game": int(record["game"]),
                "decision_index": int(record["decision_index"]),
                "opponent": str(record["opponent"]),
                "player": int(record["player"]),
                "turn": int(record["turn"]),
                "selection_reason": str(record["selection_reason"]),
                "plan_id": str(trajectory["plan_id"]),
                "plan_type": str(trajectory["plan_type"]),
                "plan": trajectory["plan"],
                "determinization_id": world_id,
                "beam_trajectory_index": trajectory_index,
                "trajectory_return": float(trajectory["return"]),
                "baseline_return": float(trajectory["baseline_return"]),
                "root_only_return": float(trajectory["root_only_return"]),
                "paired_advantage": float(trajectory["paired_advantage"]),
                "macro_synergy": float(trajectory["macro_synergy"]),
                "plan_boundary": str(trajectory["plan_boundary"]),
                "rollout_boundary": str(trajectory["rollout_boundary"]),
                "trajectory_decisions": int(trajectory["decision_count"]),
            }
            for step_index, step in enumerate(trajectory["steps"]):
                yield {
                    **common,
                    "step_index": step_index,
                    # This is the complete deployable Executor input contract.
                    "executor_input": {
                        "decision": step["decision"],
                        "plan": trajectory["plan"],
                        "progress": step["progress_before"],
                    },
                    "target_action": step["action"],
                    "target_semantic_action": step["semantic_action"],
                    "progress_after": step["progress_after"],
                }


def _plan_summary_by_id(diagnostic: dict[str, Any]) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for summary in diagnostic["plan_summaries"]:
        plan = summary["plan"]
        plan_id = str(plan["plan_id"])
        if plan_id in summaries:
            raise ValueError(f"duplicate plan summary: {plan_id}")
        summaries[plan_id] = summary
    return summaries


def iter_plan_value_examples(
    record: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Yield one public root/plan row with labels aggregated across worlds."""
    _validate_record(record)
    diagnostic = record["diagnostic"]
    summaries = _plan_summary_by_id(diagnostic)
    plans = diagnostic["plans"]
    if len(plans) != len(summaries):
        raise ValueError("plan and plan-summary counts differ")
    advantages = {
        plan_id: float(summary["macro"]["paired_advantage"])
        for plan_id, summary in summaries.items()
    }
    best_advantage = max(advantages.values())
    state_id = str(record["state_id"])
    split_group = _split_group(record)
    for plan in plans:
        plan_id = str(plan["plan_id"])
        summary = summaries[plan_id]
        macro = summary["macro"]
        root_only = summary["root_only"]
        advantage = advantages[plan_id]
        yield {
            "schema_version": SCHEMA_VERSION,
            "example_type": "macro_plan_value",
            "label_kind": PLAN_LABEL_KIND,
            "state_id": state_id,
            "split_group": split_group,
            "collector_seed": int(record["collector_seed"]),
            "game": int(record["game"]),
            "decision_index": int(record["decision_index"]),
            "opponent": str(record["opponent"]),
            "player": int(record["player"]),
            "turn": int(record["turn"]),
            "selection_reason": str(record["selection_reason"]),
            "plan_id": plan_id,
            "plan_type": str(plan["plan_type"]),
            # Hidden worlds, returns, and formal outcomes never enter this map.
            "selector_input": {
                "decision": record["decision"],
                "plan": plan,
            },
            "labels": {
                "effective_pairs": int(macro["effective_pairs"]),
                "oracle_mean_return": float(macro["mean_return"]),
                "oracle_paired_advantage": advantage,
                "oracle_paired_stderr": float(macro["paired_stderr"]),
                "oracle_paired_ci95": [
                    float(item) for item in macro["paired_ci95"]
                ],
                "oracle_positive_pair_rate": float(
                    macro["positive_pair_rate"]
                ),
                "root_only_mean_return": float(root_only["mean_return"]),
                "root_only_paired_advantage": float(
                    root_only["paired_advantage"]
                ),
                "mean_macro_synergy": float(
                    summary["mean_macro_synergy"]
                ),
                "is_best_mean_plan": advantage == best_advantage,
            },
        }


def prepare_macro_executor_data(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write one JSONL row per best-trajectory Executor decision."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    roots: set[str] = set()
    groups: set[str] = set()
    trajectories: set[tuple[str, int, str]] = set()
    plan_types: Counter[str] = Counter()
    positive_trajectories: set[tuple[str, int, str]] = set()
    synergistic_trajectories: set[tuple[str, int, str]] = set()
    with (
        input_path.open(encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as target,
    ):
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            for example in iter_executor_examples(record):
                target.write(
                    json.dumps(example, separators=(",", ":")) + "\n"
                )
                rows += 1
                roots.add(str(example["state_id"]))
                groups.add(str(example["split_group"]))
                trajectory_key = (
                    str(example["state_id"]),
                    int(example["determinization_id"]),
                    str(example["plan_id"]),
                )
                trajectories.add(trajectory_key)
                plan_types[str(example["plan_type"])] += 1
                if float(example["paired_advantage"]) > 0.0:
                    positive_trajectories.add(trajectory_key)
                if float(example["macro_synergy"]) > 0.0:
                    synergistic_trajectories.add(trajectory_key)
    count = len(trajectories)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_kind": "macro_executor_best_per_plan",
        "trajectory_selection": EXECUTOR_SELECTION,
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "roots": len(roots),
        "split_groups": len(groups),
        "trajectories": count,
        "executor_rows": rows,
        "positive_trajectory_rate": (
            round(len(positive_trajectories) / count, 6) if count else 0.0
        ),
        "synergistic_trajectory_rate": (
            round(len(synergistic_trajectories) / count, 6)
            if count
            else 0.0
        ),
        "plan_type_rows": dict(sorted(plan_types.items())),
        "input_contract": (
            "Training code may read executor_input only. Hidden-world IDs, "
            "returns, and advantages are audit metadata, never model inputs."
        ),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def prepare_macro_plan_value_data(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write one public root/plan row with cross-world oracle labels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    roots: set[str] = set()
    groups: set[str] = set()
    plan_types: Counter[str] = Counter()
    positive = 0
    best = 0
    with (
        input_path.open(encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as target,
    ):
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            for example in iter_plan_value_examples(record):
                target.write(
                    json.dumps(example, separators=(",", ":")) + "\n"
                )
                rows += 1
                roots.add(str(example["state_id"]))
                groups.add(str(example["split_group"]))
                plan_types[str(example["plan_type"])] += 1
                labels = example["labels"]
                positive += int(
                    float(labels["oracle_paired_advantage"]) > 0.0
                )
                best += int(bool(labels["is_best_mean_plan"]))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_kind": "macro_plan_value_oracle",
        "label_kind": PLAN_LABEL_KIND,
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "roots": len(roots),
        "split_groups": len(groups),
        "plan_rows": rows,
        "positive_plan_rate": round(positive / rows, 6) if rows else 0.0,
        "best_mean_plan_rate": round(best / rows, 6) if rows else 0.0,
        "plan_type_rows": dict(sorted(plan_types.items())),
        "input_contract": (
            "Training code may read selector_input only. labels are oracle "
            "targets aggregated across hidden worlds and are not deployable "
            "fixed-Executor returns."
        ),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plan-output",
        type=Path,
        help="Optional public root/plan oracle-label JSONL output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    result = {
        "executor": prepare_macro_executor_data(input_path, output_path)
    }
    if args.plan_output is not None:
        result["plan_value"] = prepare_macro_plan_value_data(
            input_path,
            args.plan_output.expanduser().resolve(),
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
