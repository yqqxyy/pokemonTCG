"""Initial masked PPO fine-tuning against a fixed RuleAgent opponent."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from poketcg.agents import RuleAgent
from poketcg.engine import OfficialEngine

from .data import BCExample, collate_bc
from .features import EncodedDecision, FeatureEncoder, FeatureEncoderV2
from .model import (
    PolicyValueModel,
    build_model,
    categorical_value_targets,
    encoder_version,
)
from .opponent_pool import OpponentPool
from .train_bc import resolve_device


@dataclass(slots=True)
class PPOTransition:
    decision: EncodedDecision
    action: int
    old_log_probability: float
    old_value: float
    advantage: float = 0.0
    value_target: float = 0.0


@dataclass(slots=True)
class IterationMetrics:
    iteration: int
    games: int
    wins: int
    draws: int
    losses: int
    transitions: int
    mean_return: float
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    opponents: dict[str, dict[str, int | float]]
    sampling_weights: dict[str, float]


def save_checkpoint(
    path: Path,
    model: PolicyValueModel,
    model_config: dict,
    args: argparse.Namespace,
    history: list[IterationMetrics],
    opponent_pool: OpponentPool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "ppo_config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "history": [asdict(item) for item in history],
            "opponent_pool": opponent_pool.manifest(),
        },
        path,
    )


def labeled_checkpoint_path(output: Path, label: str) -> Path:
    return output.with_name(f"{output.stem}_{label}{output.suffix}")


def compute_gae(
    values: list[float],
    terminal_return: float,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """Compute GAE over one player's decisions, excluding opponent and forced actions."""
    advantages = [0.0] * len(values)
    next_value = 0.0
    next_advantage = 0.0
    for index in range(len(values) - 1, -1, -1):
        reward = terminal_return if index == len(values) - 1 else 0.0
        delta = reward + gamma * next_value - values[index]
        next_advantage = delta + gamma * gae_lambda * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    targets = [advantage + value for advantage, value in zip(advantages, values, strict=True)]
    return advantages, targets


def _learnable(selection: dict) -> bool:
    return (
        len(selection["option"]) > 1
        and int(selection["minCount"]) == 1
        and int(selection["maxCount"]) == 1
    )


def _single_batch(decision: EncodedDecision, device: torch.device) -> dict[str, torch.Tensor]:
    example = BCExample(decision, action=0, value_target=0.0, player=0, game=0)
    return {key: value.to(device) for key, value in collate_bc([example]).items()}


def collect_rollout(
    engine: OfficialEngine,
    deck: list[int],
    model: PolicyValueModel,
    encoder: FeatureEncoder,
    *,
    games: int,
    seed: int,
    device: torch.device,
    opponent_pool: OpponentPool,
    card_catalog: dict[int, object],
    attack_catalog: dict[int, object],
    gamma: float,
    gae_lambda: float,
) -> tuple[list[PPOTransition], list[float], list[str]]:
    transitions: list[PPOTransition] = []
    outcomes: list[float] = []
    opponent_names: list[str] = []
    model.eval()

    for game in range(games):
        learning_player = game % 2
        opponent_seed = seed + game
        opponent_name, opponent = opponent_pool.sample(seed=opponent_seed)
        opponent_names.append(opponent_name)
        resolver = RuleAgent(
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            seed=opponent_seed + 100_000,
        )
        observation, start_data = engine.start(deck, deck)
        if observation is None:
            raise RuntimeError(
                "Official simulator failed to start "
                f"(errorPlayer={start_data.errorPlayer}, errorType={start_data.errorType})."
            )

        episode: list[PPOTransition] = []
        try:
            while int(observation["current"]["result"]) == -1:
                player = int(observation["current"]["yourIndex"])
                if player != learning_player:
                    action = opponent.choose_action(observation)
                elif not _learnable(observation["select"]):
                    action = resolver.choose_action(observation)
                else:
                    decision = encoder.encode(observation)
                    batch = _single_batch(decision, device)
                    with torch.no_grad():
                        policy_logits, value_logits = model(batch)
                        distribution = Categorical(logits=policy_logits)
                        sampled_action = distribution.sample()
                        log_probability = distribution.log_prob(sampled_action)
                        value = model.expected_value(value_logits)
                    action = [int(sampled_action.item())]
                    episode.append(
                        PPOTransition(
                            decision=decision,
                            action=action[0],
                            old_log_probability=float(log_probability.item()),
                            old_value=float(value.item()),
                        )
                    )
                observation = engine.select(action)

            winner = int(observation["current"]["result"])
            terminal_return = 0.0 if winner == 2 else (1.0 if winner == learning_player else -1.0)
            outcomes.append(terminal_return)
            values = [item.old_value for item in episode]
            advantages, targets = compute_gae(
                values,
                terminal_return,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            for item, advantage, target in zip(episode, advantages, targets, strict=True):
                item.advantage = advantage
                item.value_target = max(-1.0, min(1.0, target))
            transitions.extend(episode)
        finally:
            engine.finish()
    return transitions, outcomes, opponent_names


def summarize_opponent_results(
    opponent_names: list[str], outcomes: list[float]
) -> dict[str, dict[str, int | float]]:
    if len(opponent_names) != len(outcomes):
        raise ValueError("opponent_names and outcomes must have the same length")
    summary: dict[str, dict[str, int | float]] = {}
    for name in sorted(set(opponent_names)):
        selected = [
            outcome
            for opponent, outcome in zip(opponent_names, outcomes, strict=True)
            if opponent == name
        ]
        summary[name] = {
            "games": len(selected),
            "wins": sum(value > 0.0 for value in selected),
            "draws": sum(value == 0.0 for value in selected),
            "losses": sum(value < 0.0 for value in selected),
            "mean_return": round(sum(selected) / len(selected), 6),
        }
    return summary


def collate_transitions(
    transitions: list[PPOTransition],
    indices: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    selected = [transitions[index] for index in indices]
    examples = [
        BCExample(item.decision, item.action, item.value_target, player=0, game=0)
        for item in selected
    ]
    batch = {key: value.to(device) for key, value in collate_bc(examples).items()}
    batch["old_log_probability"] = torch.tensor(
        [item.old_log_probability for item in selected], device=device
    )
    batch["advantage"] = torch.tensor([item.advantage for item in selected], device=device)
    return batch


def ppo_update(
    model: PolicyValueModel,
    optimizer: torch.optim.Optimizer,
    transitions: list[PPOTransition],
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    clip_ratio: float,
    value_coefficient: float,
    entropy_coefficient: float,
    max_grad_norm: float,
    target_kl: float,
    seed: int,
) -> tuple[float, float, float, float]:
    advantages = torch.tensor([item.advantage for item in transitions])
    normalized = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
    for item, advantage in zip(transitions, normalized.tolist(), strict=True):
        item.advantage = advantage

    random_generator = random.Random(seed)
    totals = [0.0, 0.0, 0.0, 0.0]
    updates = 0
    stop_early = False
    model.train()
    for _ in range(epochs):
        indices = list(range(len(transitions)))
        random_generator.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch = collate_transitions(transitions, indices[start : start + batch_size], device)
            policy_logits, value_logits = model(batch)
            distribution = Categorical(logits=policy_logits)
            log_probability = distribution.log_prob(batch["action"])
            log_ratio = log_probability - batch["old_log_probability"]
            ratio = log_ratio.exp()
            unclipped = ratio * batch["advantage"]
            clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * batch["advantage"]
            policy_loss = -torch.minimum(unclipped, clipped).mean()

            targets = categorical_value_targets(batch["value_target"], model.value_support)
            value_loss = -(targets * value_logits.log_softmax(dim=-1)).sum(dim=-1).mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            approximate_kl = float(((ratio - 1.0) - log_ratio).mean().detach())
            values = [
                float(policy_loss.detach()),
                float(value_loss.detach()),
                float(entropy.detach()),
                approximate_kl,
            ]
            totals = [total + value for total, value in zip(totals, values, strict=True)]
            updates += 1
            if approximate_kl > target_kl:
                stop_early = True
                break
        if stop_early:
            break
    return tuple(total / updates for total in totals)


def build_opponent_pool(
    args: argparse.Namespace,
    *,
    card_catalog: dict[int, object],
    attack_catalog: dict[int, object],
) -> OpponentPool:
    named_weights = {
        "random_weight": args.random_weight,
        "rule_weight": args.rule_weight,
        "initial_policy_weight": args.initial_policy_weight,
        "self_play_weight": args.self_play_weight,
        "pool_checkpoint_weight": args.pool_checkpoint_weight,
    }
    for name, weight in named_weights.items():
        if weight < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if args.snapshot_every < 0:
        raise ValueError("snapshot_every must be non-negative")
    pool = OpponentPool(
        card_catalog=card_catalog,
        attack_catalog=attack_catalog,
        seed=args.seed + 700_000,
        snapshot_weight=args.self_play_weight if args.opponent == "population" else 0.0,
        max_snapshots=args.max_snapshots,
        adaptive_sampling=(args.adaptive_sampling if args.opponent == "population" else "none"),
        adaptive_alpha=args.adaptive_alpha,
        adaptive_min_multiplier=args.adaptive_min_multiplier,
        adaptive_ema_decay=args.adaptive_ema_decay,
        adaptive_warmup_games=args.adaptive_warmup_games,
    )
    if args.opponent == "random":
        pool.add_random("random", 1.0)
        return pool
    if args.opponent == "rule":
        pool.add_rule("rule", 1.0)
        return pool

    if args.random_weight > 0.0:
        pool.add_random("random", args.random_weight)
    if args.rule_weight > 0.0:
        pool.add_rule("rule", args.rule_weight)
    if args.initial_policy_weight > 0.0:
        pool.add_checkpoint("initial_policy", args.input, args.initial_policy_weight)
    for index, checkpoint in enumerate(args.pool_checkpoint):
        pool.add_checkpoint(
            f"extra_{index}_{checkpoint.stem}", checkpoint, args.pool_checkpoint_weight
        )
    if not pool.manifest():
        raise ValueError("Population pool needs at least one positive static opponent weight.")
    return pool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a BC checkpoint with masked PPO.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--games-per-iteration", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.25)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument(
        "--opponent", choices=("random", "rule", "population"), default="rule"
    )
    parser.add_argument("--random-weight", type=float, default=0.1)
    parser.add_argument("--rule-weight", type=float, default=0.4)
    parser.add_argument("--initial-policy-weight", type=float, default=0.25)
    parser.add_argument("--self-play-weight", type=float, default=0.75)
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--max-snapshots", type=int, default=4)
    parser.add_argument("--pool-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--pool-checkpoint-weight", type=float, default=0.25)
    parser.add_argument(
        "--adaptive-sampling", choices=("none", "win_rate"), default="none"
    )
    parser.add_argument("--adaptive-alpha", type=float, default=1.0)
    parser.add_argument("--adaptive-min-multiplier", type=float, default=0.1)
    parser.add_argument("--adaptive-ema-decay", type=float, default=0.95)
    parser.add_argument("--adaptive-warmup-games", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--rollout-device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="cpu",
        help=(
            "Device used for per-decision environment inference. CPU is usually faster "
            "for this small model; --device controls batched PPO updates."
        ),
    )
    parser.add_argument("--deck", type=Path)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    learner_device = resolve_device(args.device)
    rollout_device = resolve_device(args.rollout_device)
    saved = torch.load(args.input, map_location=learner_device, weights_only=False)
    model = build_model(saved["model_config"]).to(learner_device)
    model.load_state_dict(saved["model_state_dict"])
    if rollout_device == learner_device:
        rollout_model = model
    else:
        rollout_model = build_model(saved["model_config"]).to(rollout_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    engine = OfficialEngine(args.official_dir)
    deck = engine.load_deck(args.deck or engine.sample_deck_path)
    card_catalog = engine.card_catalog()
    attack_catalog = engine.attack_catalog()
    encoder_class = (
        FeatureEncoderV2
        if encoder_version(saved["model_config"]) == 2
        else FeatureEncoder
    )
    encoder = encoder_class(card_catalog, attack_catalog)
    opponent_pool = build_opponent_pool(
        args,
        card_catalog=card_catalog,
        attack_catalog=attack_catalog,
    )
    history: list[IterationMetrics] = []
    best_rollout_return = float("-inf")

    for iteration in range(1, args.iterations + 1):
        if rollout_model is not model:
            rollout_model.load_state_dict(model.state_dict())
        transitions, outcomes, opponent_names = collect_rollout(
            engine,
            deck,
            rollout_model,
            encoder,
            games=args.games_per_iteration,
            seed=args.seed + iteration * args.games_per_iteration,
            device=rollout_device,
            opponent_pool=opponent_pool,
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        if not transitions:
            raise RuntimeError("Rollout produced no learnable transitions.")
        opponent_pool.record_results(opponent_names, outcomes)
        policy_loss, value_loss, entropy, approximate_kl = ppo_update(
            model,
            optimizer,
            transitions,
            device=learner_device,
            epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            clip_ratio=args.clip_ratio,
            value_coefficient=args.value_coefficient,
            entropy_coefficient=args.entropy_coefficient,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            seed=args.seed + iteration,
        )
        metrics = IterationMetrics(
            iteration=iteration,
            games=len(outcomes),
            wins=sum(value > 0 for value in outcomes),
            draws=sum(value == 0 for value in outcomes),
            losses=sum(value < 0 for value in outcomes),
            transitions=len(transitions),
            mean_return=round(sum(outcomes) / len(outcomes), 6),
            policy_loss=round(policy_loss, 6),
            value_loss=round(value_loss, 6),
            entropy=round(entropy, 6),
            approximate_kl=round(approximate_kl, 6),
            opponents=summarize_opponent_results(opponent_names, outcomes),
            sampling_weights=opponent_pool.effective_weights(),
        )
        history.append(metrics)
        print(json.dumps(asdict(metrics)))

        if (
            args.opponent == "population"
            and args.snapshot_every > 0
            and iteration % args.snapshot_every == 0
        ):
            opponent_pool.add_snapshot(
                f"self_iter{iteration:04d}", model, saved["model_config"]
            )

        if args.checkpoint_every > 0 and iteration % args.checkpoint_every == 0:
            periodic_path = labeled_checkpoint_path(args.output, f"iter{iteration:04d}")
            save_checkpoint(
                periodic_path,
                model,
                saved["model_config"],
                args,
                history,
                opponent_pool,
            )
        if metrics.mean_return > best_rollout_return:
            best_rollout_return = metrics.mean_return
            best_path = labeled_checkpoint_path(args.output, "best_rollout")
            save_checkpoint(
                best_path,
                model,
                saved["model_config"],
                args,
                history,
                opponent_pool,
            )

    save_checkpoint(
        args.output,
        model,
        saved["model_config"],
        args,
        history,
        opponent_pool,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
