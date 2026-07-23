"""Train a bootstrap ensemble to predict paired advantage over Library-Out V1."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .advantage_data import AdvantageDataset, AdvantageExample, collate_advantage
from .model import build_model
from .train_bc import TokenBucketBatchSampler, move_batch, resolve_device


@dataclass(slots=True)
class AdvantageEpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_mae: float
    validation_weighted_mae: float
    validation_sign_accuracy: float
    validation_override_rate: float
    validation_selected_gain: float
    validation_selected_gain_stderr: float
    validation_selected_gain_lcb: float
    validation_harmful_override_rate: float
    validation_oracle_gain: float


def split_stratified(
    dataset: AdvantageDataset,
    validation_fraction: float,
    seed: int,
) -> tuple[AdvantageDataset, AdvantageDataset]:
    """Keep every opponent represented in a fixed, state-level validation split."""
    by_opponent: dict[str, list[AdvantageExample]] = {}
    for example in dataset.examples:
        by_opponent.setdefault(example.opponent, []).append(example)
    generator = random.Random(seed)
    train: list[AdvantageExample] = []
    validation: list[AdvantageExample] = []
    for examples in by_opponent.values():
        examples = list(examples)
        generator.shuffle(examples)
        count = max(1, round(len(examples) * validation_fraction))
        if count >= len(examples):
            count = len(examples) - 1
        validation.extend(examples[:count])
        train.extend(examples[count:])
    if not train or not validation:
        raise ValueError("Advantage data needs at least two states per opponent")
    return AdvantageDataset(train), AdvantageDataset(validation)


def uncertainty_weights(
    stderr: Tensor,
    mask: Tensor,
    *,
    noise_floor: float,
    maximum: float,
) -> Tensor:
    """Inverse-variance weights with a floor and cap, normalized over valid labels."""
    weights = (stderr.square() + noise_floor**2).reciprocal().clamp_max(maximum)
    weights = weights * mask
    return weights / weights.masked_select(mask).mean().clamp_min(1e-8)


def predicted_advantage(policy_logits: Tensor, baseline_index: Tensor) -> Tensor:
    baseline = policy_logits.gather(1, baseline_index.unsqueeze(1))
    return policy_logits - baseline


def advantage_loss(
    model,
    batch: dict[str, Tensor],
    *,
    noise_floor: float,
    maximum_weight: float,
    huber_delta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    policy_logits, _ = model(batch)
    prediction = predicted_advantage(policy_logits, batch["baseline_index"])
    mask = batch["advantage_mask"]
    weights = uncertainty_weights(
        batch["paired_stderr"],
        mask,
        noise_floor=noise_floor,
        maximum=maximum_weight,
    )
    valid_prediction = prediction.masked_select(mask)
    valid_target = batch["advantage_target"].masked_select(mask)
    valid_weights = weights.masked_select(mask)
    errors = nn.functional.huber_loss(
        valid_prediction,
        valid_target,
        reduction="none",
        delta=huber_delta,
    )
    loss = (errors * valid_weights).mean()
    return loss, prediction, weights


def _prediction_metrics(
    records: list[dict[str, list[float]]],
    *,
    gate_threshold: float,
    noise_floor: float,
    maximum_weight: float,
    risk_multiplier: float,
) -> dict[str, float]:
    errors: list[float] = []
    weighted_errors: list[float] = []
    weights: list[float] = []
    sign_correct = 0
    nonzero = 0
    overrides = 0
    harmful = 0
    selected_gain = 0.0
    state_gains: list[float] = []
    oracle_gain = 0.0
    for record in records:
        prediction = record["prediction"]
        target = record["target"]
        stderr = record["stderr"]
        for predicted, observed, error in zip(prediction, target, stderr, strict=True):
            weight = min(maximum_weight, 1.0 / (error**2 + noise_floor**2))
            absolute = abs(predicted - observed)
            errors.append(absolute)
            weighted_errors.append(weight * absolute)
            weights.append(weight)
            if observed != 0.0:
                sign_correct += int((predicted > 0.0) == (observed > 0.0))
                nonzero += 1
        best = max(range(len(prediction)), key=prediction.__getitem__)
        if prediction[best] > gate_threshold:
            overrides += 1
            selected_gain += target[best]
            state_gains.append(target[best])
            harmful += int(target[best] < 0.0)
        else:
            state_gains.append(0.0)
        oracle_gain += max(0.0, *target)
    count = max(1, len(records))
    mean_selected_gain = selected_gain / count
    selected_gain_stderr = (
        statistics.stdev(state_gains) / math.sqrt(len(state_gains))
        if len(state_gains) > 1
        else 0.0
    )
    return {
        "mae": sum(errors) / max(1, len(errors)),
        "weighted_mae": sum(weighted_errors) / max(1e-8, sum(weights)),
        "sign_accuracy": sign_correct / max(1, nonzero),
        "override_rate": overrides / count,
        "selected_gain": mean_selected_gain,
        "selected_gain_stderr": selected_gain_stderr,
        "selected_gain_lcb": mean_selected_gain
        - risk_multiplier * selected_gain_stderr,
        "harmful_override_rate": harmful / max(1, overrides),
        "oracle_gain": oracle_gain / count,
    }


def predict_records(model, loader: DataLoader, device: torch.device) -> list[dict]:
    model.eval()
    records: list[dict] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            logits, _ = model(batch)
            prediction = predicted_advantage(logits, batch["baseline_index"])
            for row in range(prediction.shape[0]):
                mask = batch["advantage_mask"][row]
                records.append(
                    {
                        "prediction": prediction[row].masked_select(mask).cpu().tolist(),
                        "target": batch["advantage_target"][row]
                        .masked_select(mask)
                        .cpu()
                        .tolist(),
                        "stderr": batch["paired_stderr"][row]
                        .masked_select(mask)
                        .cpu()
                        .tolist(),
                    }
                )
    return records


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float], list[dict]]:
    model.eval()
    total_loss = 0.0
    total_labels = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            loss, _, _ = advantage_loss(
                model,
                batch,
                noise_floor=args.noise_floor,
                maximum_weight=args.maximum_weight,
                huber_delta=args.huber_delta,
            )
            labels = int(batch["advantage_mask"].sum())
            total_loss += float(loss) * labels
            total_labels += labels
    records = predict_records(model, loader, device)
    metrics = _prediction_metrics(
        records,
        gate_threshold=args.gate_threshold,
        noise_floor=args.noise_floor,
        maximum_weight=args.maximum_weight,
        risk_multiplier=args.selection_risk_multiplier,
    )
    return total_loss / max(1, total_labels), metrics, records


def _configure_trainable(model, scope: str) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = scope == "all"
    if scope == "policy_head":
        for parameter in model.policy_head.parameters():
            parameter.requires_grad = True
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters selected")
    return trainable


def _set_training_mode(model, scope: str) -> None:
    if scope == "all":
        model.train()
    else:
        model.eval()
        model.policy_head.train()


def _bootstrap(dataset: AdvantageDataset, seed: int) -> AdvantageDataset:
    generator = random.Random(seed)
    return AdvantageDataset(generator.choices(dataset.examples, k=len(dataset)))


def _train_member(
    member: int,
    train_data: AdvantageDataset,
    validation_data: AdvantageDataset,
    initialization: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Path, list[dict]]:
    seed = args.seed + member
    torch.manual_seed(seed)
    member_train = _bootstrap(train_data, seed)
    train_loader = DataLoader(
        member_train,
        batch_sampler=TokenBucketBatchSampler(
            member_train, args.batch_size, shuffle=True, seed=seed
        ),
        collate_fn=collate_advantage,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_sampler=TokenBucketBatchSampler(
            validation_data, args.batch_size, shuffle=False, seed=args.split_seed
        ),
        collate_fn=collate_advantage,
    )
    model = build_model(initialization["model_config"]).to(device)
    model.load_state_dict(initialization["model_state_dict"])
    trainable = _configure_trainable(model, args.train_scope)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        _set_training_mode(model, args.train_scope)
        total_loss = 0.0
        total_labels = 0
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = advantage_loss(
                model,
                batch,
                noise_floor=args.noise_floor,
                maximum_weight=args.maximum_weight,
                huber_delta=args.huber_delta,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            labels = int(batch["advantage_mask"].sum())
            total_loss += float(loss.detach()) * labels
            total_labels += labels
        validation_loss, validation, _ = evaluate(
            model, validation_loader, device, args
        )
        metrics = AdvantageEpochMetrics(
            epoch=epoch,
            train_loss=round(total_loss / max(1, total_labels), 6),
            validation_loss=round(validation_loss, 6),
            validation_mae=round(validation["mae"], 6),
            validation_weighted_mae=round(validation["weighted_mae"], 6),
            validation_sign_accuracy=round(validation["sign_accuracy"], 6),
            validation_override_rate=round(validation["override_rate"], 6),
            validation_selected_gain=round(validation["selected_gain"], 6),
            validation_selected_gain_stderr=round(
                validation["selected_gain_stderr"], 6
            ),
            validation_selected_gain_lcb=round(
                validation["selected_gain_lcb"], 6
            ),
            validation_harmful_override_rate=round(
                validation["harmful_override_rate"], 6
            ),
            validation_oracle_gain=round(validation["oracle_gain"], 6),
        )
        item = asdict(metrics)
        history.append(item)
        print(json.dumps({"member": member, **item}))
        selection_score = (
            -validation_loss
            if args.checkpoint_selection == "loss"
            else validation["selected_gain_lcb"]
        )
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Advantage training produced no checkpoint")
    output = args.output_dir / f"advantage_member{member:02d}.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": initialization["model_config"],
            "advantage_config": {
                "target": "paired_q_v1_difference",
                "baseline_relative": True,
                "exact_one_only": True,
                "noise_floor": args.noise_floor,
                "maximum_weight": args.maximum_weight,
                "huber_delta": args.huber_delta,
                "gate_threshold": args.gate_threshold,
                "train_scope": args.train_scope,
                "checkpoint_selection": args.checkpoint_selection,
                "selection_risk_multiplier": args.selection_risk_multiplier,
                "member": member,
                "seed": seed,
            },
            "selected_epoch": best_epoch,
            "history": history,
            "train_states": len(train_data),
            "validation_states": len(validation_data),
        },
        output,
    )
    return output, history


def _ensemble_records(member_records: list[list[dict]], multiplier: float) -> list[dict]:
    ensemble: list[dict] = []
    for rows in zip(*member_records, strict=True):
        predictions = torch.tensor([row["prediction"] for row in rows])
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0, unbiased=len(rows) > 1)
        ensemble.append(
            {
                "prediction": (mean - multiplier * std).tolist(),
                "mean_prediction": mean.tolist(),
                "target": rows[0]["target"],
                "stderr": rows[0]["stderr"],
            }
        )
    return ensemble


def train(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = AdvantageDataset.from_jsonl(args.input)
    if not dataset.examples:
        raise ValueError("Advantage dataset is empty")
    versions = {example.decision.version for example in dataset.examples}
    if versions != {3}:
        raise ValueError("Round 1 advantage training requires FeatureEncoder V3")
    if args.validation_input is None:
        train_data, validation_data = split_stratified(
            dataset, args.validation_fraction, args.split_seed
        )
    else:
        train_data = dataset
        validation_data = AdvantageDataset.from_jsonl(args.validation_input)
        validation_versions = {
            example.decision.version for example in validation_data.examples
        }
        if validation_versions != {3}:
            raise ValueError("Validation advantage data requires FeatureEncoder V3")
    initialization = torch.load(
        args.initialize_from, map_location="cpu", weights_only=False
    )
    device = resolve_device(args.device)
    checkpoints: list[Path] = []
    histories: list[list[dict]] = []
    member_records: list[list[dict]] = []
    for member in range(args.ensemble_size):
        checkpoint, history = _train_member(
            member,
            train_data,
            validation_data,
            initialization,
            args,
            device,
        )
        checkpoints.append(checkpoint)
        histories.append(history)
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model = build_model(saved["model_config"]).to(device)
        model.load_state_dict(saved["model_state_dict"])
        loader = DataLoader(
            validation_data,
            batch_sampler=TokenBucketBatchSampler(
                validation_data,
                args.batch_size,
                shuffle=False,
                seed=args.split_seed,
            ),
            collate_fn=collate_advantage,
        )
        member_records.append(predict_records(model, loader, device))

    ensemble_records = _ensemble_records(member_records, args.uncertainty_multiplier)
    lcb_metrics = _prediction_metrics(
        ensemble_records,
        gate_threshold=args.gate_threshold,
        noise_floor=args.noise_floor,
        maximum_weight=args.maximum_weight,
        risk_multiplier=args.selection_risk_multiplier,
    )
    mean_records = [
        {**record, "prediction": record["mean_prediction"]}
        for record in ensemble_records
    ]
    mean_metrics = _prediction_metrics(
        mean_records,
        gate_threshold=args.gate_threshold,
        noise_floor=args.noise_floor,
        maximum_weight=args.maximum_weight,
        risk_multiplier=args.selection_risk_multiplier,
    )
    manifest = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "validation_input": (
            str(args.validation_input.resolve()) if args.validation_input else None
        ),
        "initialize_from": str(args.initialize_from.resolve()),
        "train_states": len(train_data),
        "validation_states": len(validation_data),
        "ensemble_size": args.ensemble_size,
        "members": [
            {
                "checkpoint": str(path.resolve()),
                "selected_epoch": torch.load(
                    path, map_location="cpu", weights_only=False
                )["selected_epoch"],
                "final_logged_metrics": history[-1],
            }
            for path, history in zip(checkpoints, histories, strict=True)
        ],
        "gate": {
            "score": "ensemble_mean_minus_multiplier_std",
            "uncertainty_multiplier": args.uncertainty_multiplier,
            "threshold": args.gate_threshold,
        },
        "validation_mean_metrics": {
            key: round(value, 6) for key, value in mean_metrics.items()
        },
        "validation_lcb_metrics": {
            key: round(value, 6) for key, value in lcb_metrics.items()
        },
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    manifest_path = args.output_dir / "ensemble_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ensemble_manifest": str(manifest_path),
                **manifest["validation_lcb_metrics"],
            }
        )
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--validation-input", type=Path)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--huber-delta", type=float, default=0.25)
    parser.add_argument("--noise-floor", type=float, default=0.15)
    parser.add_argument("--maximum-weight", type=float, default=20.0)
    parser.add_argument("--gate-threshold", type=float, default=0.05)
    parser.add_argument("--uncertainty-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("gain_lcb", "loss"),
        default="gain_lcb",
    )
    parser.add_argument("--selection-risk-multiplier", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--train-scope", choices=("policy_head", "all"), default="policy_head")
    parser.add_argument("--split-seed", type=int, default=20_260_722)
    parser.add_argument("--seed", type=int, default=20_260_723)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    train(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
