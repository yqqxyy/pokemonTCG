"""Convert closed-loop Plan DAgger diagnostics into Executor schema 3."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .macro_data import SCHEMA_VERSION


def _split_group(record: dict[str, Any]) -> str:
    return f"seed{record['collector_seed']}-game{record['game']}"


def iter_plan_dagger_examples(
    record: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    diagnostic = record.get("diagnostic") or {}
    if diagnostic.get("diagnostic_kind") != "closed_loop_plan_dagger_v1":
        raise ValueError(
            "plan_dagger_data accepts only closed_loop_plan_dagger_v1"
        )
    common = {
        "schema_version": SCHEMA_VERSION,
        "example_type": "macro_executor_action",
        "trajectory_selection": "closed_loop_dagger_visited_state",
        "split_group": _split_group(record),
        "collector_seed": int(record["collector_seed"]),
        "game": int(record["game"]),
        "decision_index": int(record["decision_index"]),
        "opponent": str(record["opponent"]),
        "player": int(record["player"]),
        "turn": int(record["turn"]),
        "selection_reason": str(record["selection_reason"]),
        "configured_beta": float(diagnostic["configured_beta"]),
    }
    root_state_id = str(record["state_id"])
    for sample in diagnostic["samples"]:
        if "error" in sample:
            continue
        world_id = int(sample["determinization_id"])
        for plan_result in sample["plans"]:
            if "error" in plan_result:
                continue
            plan = plan_result["plan"]
            for step_index, label in enumerate(plan_result["labels"]):
                yield {
                    **common,
                    "state_id": root_state_id,
                    "plan_id": str(plan_result["plan_id"]),
                    "plan_type": str(plan_result["plan_type"]),
                    "plan": plan,
                    "determinization_id": world_id,
                    "dagger_step": step_index,
                    "roll_in_source": str(label["roll_in_source"]),
                    "semantic_disagreement": bool(
                        label["semantic_disagreement"]
                    ),
                    "student_action": label["student_action"],
                    "student_semantic_action": label[
                        "student_semantic_action"
                    ],
                    "executed_action": label["executed_action"],
                    "teacher_continuation_return": float(
                        label["teacher_continuation_return"]
                    ),
                    "mixed_return": float(plan_result["mixed_return"]),
                    "oracle_return": float(plan_result["oracle_return"]),
                    "baseline_return": float(plan_result["baseline_return"]),
                    "oracle_gap": float(plan_result["oracle_gap"]),
                    "executor_input": {
                        "decision": label["decision"],
                        "plan": plan,
                        "progress": label["progress_before"],
                    },
                    "target_action": label["target_action"],
                    "target_semantic_action": label[
                        "target_semantic_action"
                    ],
                }


def prepare_plan_dagger_data(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    roots: set[str] = set()
    groups: set[str] = set()
    roll_in_sources: Counter[str] = Counter()
    plan_types: Counter[str] = Counter()
    disagreements = 0
    with (
        input_path.open(encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as target,
    ):
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            for example in iter_plan_dagger_examples(record):
                target.write(
                    json.dumps(example, separators=(",", ":")) + "\n"
                )
                rows += 1
                roots.add(str(example["state_id"]))
                groups.add(str(example["split_group"]))
                roll_in_sources[str(example["roll_in_source"])] += 1
                plan_types[str(example["plan_type"])] += 1
                disagreements += int(example["semantic_disagreement"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_kind": "closed_loop_plan_dagger",
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "roots": len(roots),
        "split_groups": len(groups),
        "executor_rows": rows,
        "semantic_disagreement_rate": (
            round(disagreements / rows, 6) if rows else 0.0
        ),
        "roll_in_sources": dict(sorted(roll_in_sources.items())),
        "plan_type_rows": dict(sorted(plan_types.items())),
        "input_contract": (
            "Only executor_input is available to the model. Student actions, "
            "hidden-world returns, and roll-in sources are audit metadata."
        ),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    print(
        json.dumps(
            prepare_plan_dagger_data(input_path, output_path),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
