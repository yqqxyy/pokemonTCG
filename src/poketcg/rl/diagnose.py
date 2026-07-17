"""Offline policy/value diagnostics, including per-context metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from poketcg.agents.rule_agent import SelectContext

from .data import BCDataset, collate_bc
from .model import build_model
from .train_bc import move_batch, resolve_device


def _context_name(context: int) -> str:
    try:
        return SelectContext(context).name
    except ValueError:
        return f"CONTEXT_{context}"


def diagnose(
    checkpoint: str | Path,
    dataset_path: str | Path,
    *,
    batch_size: int,
    device_name: str,
) -> dict:
    device = resolve_device(device_name)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(saved["model_config"]).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    dataset = BCDataset.from_jsonl(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_bc)
    totals: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "correct": 0.0,
            "target_probability": 0.0,
            "entropy": 0.0,
            "value_error": 0.0,
        }
    )

    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            policy_logits, value_logits = model(batch)
            prediction = policy_logits.argmax(dim=-1)
            entropy = Categorical(logits=policy_logits).entropy()
            value_error = (model.expected_value(value_logits) - batch["value_target"]).abs()
            for row in range(batch["action"].shape[0]):
                context = int(batch["context"][row])
                item = totals[context]
                item["count"] += 1
                predicted_target = batch["policy_target"][row, prediction[row]]
                item["correct"] += float(predicted_target > 0)
                item["target_probability"] += float(predicted_target)
                item["entropy"] += float(entropy[row])
                item["value_error"] += float(value_error[row])

    contexts = {}
    overall = {
        "count": 0.0,
        "correct": 0.0,
        "target_probability": 0.0,
        "entropy": 0.0,
        "value_error": 0.0,
    }
    for context, values in sorted(totals.items()):
        count = values["count"]
        contexts[str(context)] = {
            "name": _context_name(context),
            "count": int(count),
            "policy_accuracy": round(values["correct"] / count, 6),
            "mean_teacher_probability": round(values["target_probability"] / count, 6),
            "mean_entropy": round(values["entropy"] / count, 6),
            "value_mae": round(values["value_error"] / count, 6),
        }
        for key in overall:
            overall[key] += values[key]

    count = overall["count"]
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "dataset": str(Path(dataset_path).resolve()),
        "device": str(device),
        "examples": int(count),
        "policy_accuracy": round(overall["correct"] / count, 6),
        "mean_teacher_probability": round(overall["target_probability"] / count, 6),
        "mean_entropy": round(overall["entropy"] / count, 6),
        "value_mae": round(overall["value_error"] / count, 6),
        "contexts": contexts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose a policy/value checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = diagnose(
        args.checkpoint,
        args.dataset,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
