"""Train the initial candidate policy and value model from RuleAgent data."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from .action_space import batch_subset_log_probabilities, deterministic_subset
from .data import BCDataset, collate_bc
from .model import (
    PolicyValueModel,
    action_space_version,
    build_model,
    categorical_value_targets,
)


@dataclass(slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_policy_accuracy: float
    validation_loss: float
    validation_policy_accuracy: float
    validation_value_mae: float


class TokenBucketBatchSampler(Sampler[list[int]]):
    """Group similarly sized token sequences to reduce attention padding."""

    def __init__(
        self,
        dataset: BCDataset,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            generator.shuffle(indices)
        indices.sort(key=lambda index: len(self.dataset[index].decision.tokens or []))
        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        if self.shuffle:
            generator.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def batch_loss(
    model: PolicyValueModel,
    batch: dict[str, torch.Tensor],
    value_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    policy_logits, value_logits = model(batch)
    soft_policy_loss = -(
        batch["policy_target"] * policy_logits.log_softmax(dim=-1)
    ).sum(dim=-1)
    hard_policy_loss = -batch_subset_log_probabilities(
        policy_logits,
        batch["option_mask"],
        batch["action_mask"],
        batch["minimum"],
        batch["maximum"],
    )
    policy_loss = torch.where(
        batch["has_soft_policy_target"], soft_policy_loss, hard_policy_loss
    ).mean()
    value_targets = categorical_value_targets(batch["value_target"], model.value_support)
    value_loss = -(value_targets * value_logits.log_softmax(dim=-1)).sum(dim=-1).mean()
    return policy_loss + value_coefficient * value_loss, policy_logits, value_logits


def policy_correct_count(policy_logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> int:
    correct = 0
    for row in range(policy_logits.shape[0]):
        option_count = int(batch["option_mask"][row].sum())
        predicted = deterministic_subset(
            policy_logits[row, :option_count],
            int(batch["minimum"][row]),
            int(batch["maximum"][row]),
        )
        if bool(batch["has_soft_policy_target"][row]):
            target = batch["policy_target"][row, :option_count]
            correct += int(
                len(predicted) == 1
                and bool(
                    torch.isclose(
                        target[predicted[0]],
                        target.max(),
                        rtol=0.0,
                        atol=1e-8,
                    )
                )
            )
        else:
            target = batch["action_mask"][row, :option_count]
            predicted_mask = torch.zeros_like(target)
            predicted_mask[predicted] = True
            correct += int(torch.equal(predicted_mask, target))
    return correct


def evaluate(
    model: PolicyValueModel,
    loader: DataLoader,
    device: torch.device,
    value_coefficient: float,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_value_error = 0.0
    total_examples = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            loss, policy_logits, value_logits = batch_loss(model, batch, value_coefficient)
            size = batch["action"].shape[0]
            total_loss += float(loss) * size
            total_correct += policy_correct_count(policy_logits, batch)
            predicted_values = model.expected_value(value_logits)
            total_value_error += float((predicted_values - batch["value_target"]).abs().sum())
            total_examples += size
    return (
        total_loss / total_examples,
        total_correct / total_examples,
        total_value_error / total_examples,
    )


def split_by_game(dataset: BCDataset, validation_fraction: float, seed: int):
    games = sorted({example.game for example in dataset.examples})
    random.Random(seed).shuffle(games)
    validation_count = max(1, round(len(games) * validation_fraction))
    validation_games = set(games[:validation_count])
    train = [example for example in dataset.examples if example.game not in validation_games]
    validation = [example for example in dataset.examples if example.game in validation_games]
    if not train or not validation:
        raise ValueError("Dataset must contain enough games for a train/validation split.")
    return BCDataset(train), BCDataset(validation)


def train(args: argparse.Namespace) -> tuple[PolicyValueModel, list[EpochMetrics]]:
    torch.manual_seed(args.seed)
    dataset = BCDataset.from_jsonl(args.input)
    data_versions = {example.decision.version for example in dataset.examples}
    if len(data_versions) != 1:
        raise ValueError("Training data cannot mix encoder versions")
    data_version = data_versions.pop()
    has_multiselect = any(
        example.decision.minimum != 1 or example.decision.maximum != 1
        for example in dataset.examples
    )
    inferred_model_type = {
        1: "mlp_v1",
        2: "transformer_v2",
        3: "transformer_v3",
    }.get(data_version)
    if inferred_model_type is None:
        raise ValueError(f"Unsupported encoder version: {data_version}")
    expected_model_type = {
        1: "mlp_v1",
        2: "transformer_v2",
        3: "transformer_v3",
    }[data_version]
    initial_state: dict[str, torch.Tensor] | None = None
    if args.initialize_from is not None:
        saved = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
        if not isinstance(saved, dict) or not isinstance(saved.get("model_config"), dict):
            raise TypeError("Initialization checkpoint must contain model_config")
        if not isinstance(saved.get("model_state_dict"), dict):
            raise TypeError("Initialization checkpoint must contain model_state_dict")
        model_config = dict(saved["model_config"])
        model_type = str(model_config.get("model_type"))
        initial_state = saved["model_state_dict"]
        if model_type != expected_model_type:
            raise ValueError(
                f"Initialization checkpoint {model_type} is incompatible with "
                f"encoder V{data_version} data"
            )
        if has_multiselect and action_space_version(model_config) < 2:
            raise ValueError(
                "Initialization checkpoint does not support multiselect training examples"
            )
    else:
        model_type = inferred_model_type if args.model_type == "auto" else args.model_type
        if model_type != expected_model_type:
            raise ValueError(f"{model_type} is incompatible with encoder V{data_version} data")
        if model_type != "transformer_v3" and (
            args.disable_card_semantics or args.disable_history
        ):
            raise ValueError("V3 ablation flags require encoder V3 training data")
        model_config = {
            "model_type": model_type,
            "hidden_size": args.hidden_size,
            "value_bins": args.value_bins,
            "action_space_version": 2 if has_multiselect else 1,
        }
        if model_type in {"transformer_v2", "transformer_v3"}:
            model_config.update(
                {
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "dropout": args.dropout,
                    "card_vocab_size": args.card_vocab_size,
                    "attack_vocab_size": args.attack_vocab_size,
                }
            )
        if model_type == "transformer_v3":
            model_config.update(
                {
                    "use_card_semantics": not args.disable_card_semantics,
                    "use_history": not args.disable_history,
                }
            )
    train_data, validation_data = split_by_game(dataset, args.validation_fraction, args.seed)
    if data_version in {2, 3}:
        train_loader = DataLoader(
            train_data,
            batch_sampler=TokenBucketBatchSampler(
                train_data, args.batch_size, shuffle=True, seed=args.seed
            ),
            collate_fn=collate_bc,
        )
        validation_loader = DataLoader(
            validation_data,
            batch_sampler=TokenBucketBatchSampler(
                validation_data, args.batch_size, shuffle=False, seed=args.seed
            ),
            collate_fn=collate_bc,
        )
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_bc,
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_bc,
        )
    device = resolve_device(args.device)
    model = build_model(model_config).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history: list[EpochMetrics] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, policy_logits, _ = batch_loss(model, batch, args.value_coefficient)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            size = batch["action"].shape[0]
            total_loss += float(loss.detach()) * size
            total_correct += policy_correct_count(policy_logits, batch)
            total_examples += size

        validation_loss, validation_accuracy, validation_value_mae = evaluate(
            model, validation_loader, device, args.value_coefficient
        )
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=round(total_loss / total_examples, 6),
            train_policy_accuracy=round(total_correct / total_examples, 6),
            validation_loss=round(validation_loss, 6),
            validation_policy_accuracy=round(validation_accuracy, 6),
            validation_value_mae=round(validation_value_mae, 6),
        )
        history.append(metrics)
        print(json.dumps(asdict(metrics)))
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint")
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": model_config,
            "training_config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "history": [asdict(item) for item in history],
            "train_examples": len(train_data),
            "validation_examples": len(validation_data),
            "selected_epoch": best_epoch,
            "selection_metric": "validation_loss",
            "initialized_from": (
                str(args.initialize_from.expanduser().resolve())
                if args.initialize_from is not None
                else None
            ),
        },
        args.output,
    )
    return model, history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train policy/value heads from RuleAgent data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Fine-tune an existing compatible policy/value checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--value-bins", type=int, default=101)
    parser.add_argument(
        "--model-type",
        choices=("auto", "mlp_v1", "transformer_v2", "transformer_v3"),
        default="auto",
    )
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--card-vocab-size", type=int, default=2048)
    parser.add_argument("--attack-vocab-size", type=int, default=2048)
    parser.add_argument("--disable-card-semantics", action="store_true")
    parser.add_argument("--disable-history", action="store_true")
    parser.add_argument("--value-coefficient", type=float, default=0.25)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
