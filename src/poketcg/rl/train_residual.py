"""Train a confidence-gated residual reranker over Library-Out rule scores."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .action_space import batch_subset_log_probabilities, deterministic_subset
from .model import build_model, categorical_value_targets
from .residual_data import ResidualDataset, collate_residual
from .train_bc import TokenBucketBatchSampler, move_batch, resolve_device


@dataclass(slots=True)
class ResidualEpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_policy_loss: float
    validation_value_mae: float
    validation_target_accuracy: float
    validation_override_rate: float


def outcome_weights(
    values: torch.Tensor,
    *,
    win_weight: float,
    draw_weight: float,
    loss_weight: float,
) -> torch.Tensor:
    """AWR-style weights: imitate successful trajectories most strongly."""
    return torch.where(
        values > 0.5,
        torch.full_like(values, win_weight),
        torch.where(
            values < -0.5,
            torch.full_like(values, loss_weight),
            torch.full_like(values, draw_weight),
        ),
    )


def residual_loss(
    model,
    batch: dict[str, torch.Tensor],
    *,
    prior_strength: float,
    value_coefficient: float,
    residual_coefficient: float,
    win_weight: float,
    draw_weight: float,
    loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_logits, value_logits = model(batch)
    combined_logits = prior_strength * batch["rule_scores"] + residual_logits
    per_example_policy_loss = -batch_subset_log_probabilities(
        combined_logits,
        batch["option_mask"],
        batch["action_mask"],
        batch["minimum"],
        batch["maximum"],
    )
    weights = outcome_weights(
        batch["value_target"],
        win_weight=win_weight,
        draw_weight=draw_weight,
        loss_weight=loss_weight,
    )
    policy_loss = (weights * per_example_policy_loss).sum() / weights.sum().clamp_min(1e-8)
    value_targets = categorical_value_targets(batch["value_target"], model.value_support)
    value_loss = -(value_targets * value_logits.log_softmax(dim=-1)).sum(dim=-1).mean()
    residual_penalty = residual_logits.masked_select(batch["option_mask"]).square().mean()
    total = (
        policy_loss
        + value_coefficient * value_loss
        + residual_coefficient * residual_penalty
    )
    return total, combined_logits, value_logits, policy_loss


def _routing_counts(
    logits: torch.Tensor, batch: dict[str, torch.Tensor]
) -> tuple[int, int, int]:
    correct = 0
    overrides = 0
    eligible = 0
    for row in range(logits.shape[0]):
        if int(batch["minimum"][row]) != 1 or int(batch["maximum"][row]) != 1:
            continue
        option_count = int(batch["option_mask"][row].sum())
        predicted = deterministic_subset(logits[row, :option_count], 1, 1)
        target = batch["action_mask"][row, :option_count]
        baseline = batch["baseline_action_mask"][row, :option_count]
        correct += int(bool(target[predicted[0]]))
        overrides += int(not bool(baseline[predicted[0]]))
        eligible += 1
    return correct, overrides, eligible


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_error = 0.0
    total_examples = 0
    correct = overrides = eligible = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            loss, logits, value_logits, policy_loss = residual_loss(
                model,
                batch,
                prior_strength=args.prior_strength,
                value_coefficient=args.value_coefficient,
                residual_coefficient=args.residual_coefficient,
                win_weight=args.win_weight,
                draw_weight=args.draw_weight,
                loss_weight=args.loss_weight,
            )
            size = int(batch["value_target"].shape[0])
            total_loss += float(loss) * size
            total_policy_loss += float(policy_loss) * size
            predicted_values = model.expected_value(value_logits)
            total_value_error += float(
                (predicted_values - batch["value_target"]).abs().sum()
            )
            batch_correct, batch_overrides, batch_eligible = _routing_counts(logits, batch)
            correct += batch_correct
            overrides += batch_overrides
            eligible += batch_eligible
            total_examples += size
    return (
        total_loss / total_examples,
        total_policy_loss / total_examples,
        total_value_error / total_examples,
        correct / eligible if eligible else 0.0,
        overrides / eligible if eligible else 0.0,
    )


def split_by_game(
    dataset: ResidualDataset, validation_fraction: float, seed: int
) -> tuple[ResidualDataset, ResidualDataset]:
    games = sorted({example.game for example in dataset.examples})
    random.Random(seed).shuffle(games)
    validation_count = max(1, round(len(games) * validation_fraction))
    validation_games = set(games[:validation_count])
    train_examples = [
        example for example in dataset.examples if example.game not in validation_games
    ]
    validation_examples = [
        example for example in dataset.examples if example.game in validation_games
    ]
    if not train_examples or not validation_examples:
        raise ValueError("Dataset needs at least two games for a train/validation split")
    return ResidualDataset(train_examples), ResidualDataset(validation_examples)


def train(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    dataset = ResidualDataset.from_jsonl(args.input)
    if not dataset.examples:
        raise ValueError("Residual dataset is empty")
    versions = {example.decision.version for example in dataset.examples}
    if len(versions) != 1:
        raise ValueError("Residual data cannot mix encoder versions")
    version = versions.pop()
    if version not in {2, 3}:
        raise ValueError("Residual reranking requires encoder V2 or V3")
    train_data, validation_data = split_by_game(
        dataset, args.validation_fraction, args.seed
    )
    model_config = {
        "model_type": f"transformer_v{version}",
        "hidden_size": args.hidden_size,
        "value_bins": args.value_bins,
        "action_space_version": 2,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "card_vocab_size": args.card_vocab_size,
        "attack_vocab_size": args.attack_vocab_size,
    }
    if version == 3:
        model_config.update({"use_card_semantics": True, "use_history": True})
    train_loader = DataLoader(
        train_data,
        batch_sampler=TokenBucketBatchSampler(
            train_data, args.batch_size, shuffle=True, seed=args.seed
        ),
        collate_fn=collate_residual,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_sampler=TokenBucketBatchSampler(
            validation_data, args.batch_size, shuffle=False, seed=args.seed
        ),
        collate_fn=collate_residual,
    )
    device = resolve_device(args.device)
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history: list[ResidualEpochMetrics] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = residual_loss(
                model,
                batch,
                prior_strength=args.prior_strength,
                value_coefficient=args.value_coefficient,
                residual_coefficient=args.residual_coefficient,
                win_weight=args.win_weight,
                draw_weight=args.draw_weight,
                loss_weight=args.loss_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            size = int(batch["value_target"].shape[0])
            total_loss += float(loss.detach()) * size
            total_examples += size
        validation = evaluate(model, validation_loader, device, args)
        metrics = ResidualEpochMetrics(
            epoch=epoch,
            train_loss=round(total_loss / total_examples, 6),
            validation_loss=round(validation[0], 6),
            validation_policy_loss=round(validation[1], 6),
            validation_value_mae=round(validation[2], 6),
            validation_target_accuracy=round(validation[3], 6),
            validation_override_rate=round(validation[4], 6),
        )
        history.append(metrics)
        print(json.dumps(asdict(metrics)))
        if validation[0] < best_loss:
            best_loss = validation[0]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Training finished without a checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": model_config,
            "residual_config": {
                "prior_strength": args.prior_strength,
                "override_margin": args.override_margin,
                "minimum_confidence": args.minimum_confidence,
                "exact_one_only": True,
                "normalization": "context_zscore_clip4",
            },
            "training_config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "history": [asdict(item) for item in history],
            "selected_epoch": best_epoch,
            "train_examples": len(train_data),
            "validation_examples": len(validation_data),
        },
        args.output,
    )
    return model, history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--value-bins", type=int, default=101)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--card-vocab-size", type=int, default=2048)
    parser.add_argument("--attack-vocab-size", type=int, default=2048)
    parser.add_argument("--prior-strength", type=float, default=2.0)
    parser.add_argument("--value-coefficient", type=float, default=0.2)
    parser.add_argument("--residual-coefficient", type=float, default=0.01)
    parser.add_argument("--win-weight", type=float, default=1.0)
    parser.add_argument("--draw-weight", type=float, default=0.5)
    parser.add_argument("--loss-weight", type=float, default=0.25)
    parser.add_argument("--override-margin", type=float, default=0.5)
    parser.add_argument("--minimum-confidence", type=float, default=0.65)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20_260_722)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
