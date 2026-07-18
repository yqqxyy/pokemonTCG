"""Initial masked PPO fine-tuning against a fixed RuleAgent opponent."""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from poketcg.agents import RandomAgent, RuleAgent
from poketcg.engine import OfficialEngine

from .data import BCExample, collate_bc
from .features import EncodedDecision, FeatureEncoder, FeatureEncoderV2
from .model import (
    PolicyValueModel,
    build_model,
    categorical_value_targets,
    encoder_version,
)
from .opponent_pool import FrozenPolicyAgent, OpponentPool
from .train_bc import resolve_device


@dataclass(slots=True)
class PPOTransition:
    decision: EncodedDecision
    action: int
    old_log_probability: float
    old_value: float
    potential: float = 0.0
    shaping_reward: float = 0.0
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
    clip_fraction: float
    explained_variance: float
    gradient_norm: float
    rollout_seconds: float
    update_seconds: float
    games_per_second: float
    mean_abs_shaping_reward: float
    mean_shaping_return: float
    opponents: dict[str, dict[str, int | float]]
    sampling_weights: dict[str, float]


@dataclass(slots=True)
class PPOUpdateMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float


@dataclass(slots=True)
class _WorkerGameResult:
    game: int
    transitions: list[PPOTransition]
    outcome: float
    opponent_name: str


@dataclass(slots=True)
class _WorkerRolloutTask:
    game_indices: list[int]
    opponent_names: list[str]
    model_state_dict: dict[str, torch.Tensor]
    opponent_state: list[dict[str, Any]]
    seed: int
    gamma: float
    gae_lambda: float
    value_gae_lambda: float
    reward_shaping: str
    reward_shaping_scale: float


@dataclass(slots=True)
class _RolloutWorkerContext:
    engine: OfficialEngine
    deck: list[int]
    model: PolicyValueModel
    encoder: FeatureEncoder
    encoders: dict[int, FeatureEncoder]
    card_catalog: dict[int, object]
    attack_catalog: dict[int, object]
    opponent_models: dict[str, PolicyValueModel]


_ROLLOUT_WORKER_CONTEXT: _RolloutWorkerContext | None = None


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
    terminal_return: float | None = None,
    *,
    rewards: list[float] | None = None,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """Compute GAE over one player's decisions, excluding opponent and forced actions."""
    if rewards is None:
        if terminal_return is None:
            raise ValueError("terminal_return is required when rewards are not provided")
        rewards = [0.0] * len(values)
        if rewards:
            rewards[-1] = terminal_return
    elif terminal_return is not None:
        raise ValueError("provide either terminal_return or rewards, not both")
    if len(rewards) != len(values):
        raise ValueError("rewards and values must have the same length")
    advantages = [0.0] * len(values)
    next_value = 0.0
    next_advantage = 0.0
    for index in range(len(values) - 1, -1, -1):
        delta = rewards[index] + gamma * next_value - values[index]
        next_advantage = delta + gamma * gae_lambda * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    targets = [advantage + value for advantage, value in zip(advantages, values, strict=True)]
    return advantages, targets


def prize_potential(observation: dict, learning_player: int) -> float:
    """Return normalized relative prize progress from the learner's perspective."""
    if learning_player not in {0, 1}:
        raise ValueError("learning_player must be zero or one")
    players = observation["current"]["players"]
    own_remaining = len(players[learning_player].get("prize") or [])
    opponent_remaining = len(players[1 - learning_player].get("prize") or [])
    return (opponent_remaining - own_remaining) / 6.0


def potential_shaping_rewards(
    potentials: list[float],
    *,
    gamma: float,
    scale: float,
) -> list[float]:
    """Redistribute reward with a terminal-zero potential difference."""
    next_potentials = [*potentials[1:], 0.0]
    return [
        scale * (gamma * next_potential - potential)
        for potential, next_potential in zip(potentials, next_potentials, strict=True)
    ]


def assign_episode_returns(
    episode: list[PPOTransition],
    terminal_return: float,
    *,
    gamma: float,
    gae_lambda: float,
    value_gae_lambda: float,
    reward_shaping: str,
    reward_shaping_scale: float,
) -> None:
    """Assign shaped policy advantages and unshaped terminal value targets."""
    values = [item.old_value for item in episode]
    base_rewards = [0.0] * len(episode)
    if base_rewards:
        base_rewards[-1] = terminal_return
    base_advantages, _ = compute_gae(
        values,
        rewards=base_rewards,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    _, base_targets = compute_gae(
        values,
        rewards=base_rewards,
        gamma=gamma,
        gae_lambda=value_gae_lambda,
    )
    shaping_rewards = [0.0] * len(episode)
    policy_advantages = base_advantages
    if reward_shaping == "prize":
        shaping_rewards = potential_shaping_rewards(
            [item.potential for item in episode],
            gamma=gamma,
            scale=reward_shaping_scale,
        )
        policy_advantages, _ = compute_gae(
            values,
            rewards=[
                reward + shaping
                for reward, shaping in zip(base_rewards, shaping_rewards, strict=True)
            ],
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
    elif reward_shaping != "none":
        raise ValueError(f"Unsupported reward shaping mode: {reward_shaping}")

    for item, advantage, target, shaping in zip(
        episode,
        policy_advantages,
        base_targets,
        shaping_rewards,
        strict=True,
    ):
        item.advantage = advantage
        item.value_target = max(-1.0, min(1.0, target))
        item.shaping_reward = shaping


def explained_variance(values: list[float], targets: list[float]) -> float:
    """Measure how much return-target variance is explained by critic predictions."""
    if len(values) != len(targets):
        raise ValueError("values and targets must have the same length")
    if not values:
        return 0.0
    value_tensor = torch.tensor(values, dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    target_variance = target_tensor.var(unbiased=False)
    if float(target_variance) < 1e-8:
        return 0.0
    residual_variance = (target_tensor - value_tensor).var(unbiased=False)
    return float(1.0 - residual_variance / target_variance)


def _learnable(selection: dict) -> bool:
    return (
        len(selection["option"]) > 1
        and int(selection["minCount"]) == 1
        and int(selection["maxCount"]) == 1
    )


def _single_batch(decision: EncodedDecision, device: torch.device) -> dict[str, torch.Tensor]:
    example = BCExample(decision, action=0, value_target=0.0, player=0, game=0)
    return {key: value.to(device) for key, value in collate_bc([example]).items()}


def partition_rollout_assignments(
    opponent_names: list[str], workers: int
) -> list[tuple[list[int], list[str]]]:
    """Split ordered game assignments into balanced, non-empty worker shards."""
    if workers < 1:
        raise ValueError("rollout workers must be at least one")
    if not opponent_names:
        return []
    shard_count = min(workers, len(opponent_names))
    base, remainder = divmod(len(opponent_names), shard_count)
    shards: list[tuple[list[int], list[str]]] = []
    cursor = 0
    for worker in range(shard_count):
        count = base + int(worker < remainder)
        indices = list(range(cursor, cursor + count))
        shards.append((indices, opponent_names[cursor : cursor + count]))
        cursor += count
    return shards


def _share_state_dict(model: PolicyValueModel) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        shared = value.detach().cpu().clone()
        shared.share_memory_()
        state[key] = shared
    return state


def _share_opponent_state(state: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for entry in state:
        model_state = entry.get("model_state_dict")
        if model_state is None:
            continue
        for tensor in model_state.values():
            tensor.share_memory_()
    return state


def _initialize_rollout_worker(
    official_dir: str,
    deck: list[int],
    model_config: dict[str, Any],
    model_encoder_version: int,
    torch_threads: int,
) -> None:
    """Initialize one process-local native simulator and frozen CPU policy."""
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    torch.set_num_threads(torch_threads)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)

    engine = OfficialEngine(official_dir)
    card_catalog = engine.card_catalog()
    attack_catalog = engine.attack_catalog()
    encoders: dict[int, FeatureEncoder] = {
        1: FeatureEncoder(card_catalog, attack_catalog),
        2: FeatureEncoderV2(card_catalog, attack_catalog),
    }
    global _ROLLOUT_WORKER_CONTEXT
    _ROLLOUT_WORKER_CONTEXT = _RolloutWorkerContext(
        engine=engine,
        deck=deck,
        model=build_model(model_config).cpu().eval(),
        encoder=encoders[model_encoder_version],
        encoders=encoders,
        card_catalog=card_catalog,
        attack_catalog=attack_catalog,
        opponent_models={},
    )


def _refresh_worker_models(
    context: _RolloutWorkerContext,
    task: _WorkerRolloutTask,
) -> dict[str, dict[str, Any]]:
    context.model.load_state_dict(task.model_state_dict)
    context.model.eval()
    by_name = {str(item["name"]): item for item in task.opponent_state}
    neural_names = {
        name
        for name, item in by_name.items()
        if item["kind"] in {"policy", "snapshot"}
    }
    for stale_name in set(context.opponent_models) - neural_names:
        del context.opponent_models[stale_name]
    for name in neural_names:
        item = by_name[name]
        model = context.opponent_models.get(name)
        if model is None:
            model = build_model(item["model_config"]).cpu()
            context.opponent_models[name] = model
        model.load_state_dict(item["model_state_dict"])
        model.eval()
    return by_name


def _worker_opponent(
    context: _RolloutWorkerContext,
    specification: dict[str, Any],
    seed: int,
):
    kind = specification["kind"]
    if kind == "random":
        return RandomAgent(seed)
    if kind == "rule":
        return RuleAgent(
            card_catalog=context.card_catalog,
            attack_catalog=context.attack_catalog,
            seed=seed,
        )
    name = str(specification["name"])
    return FrozenPolicyAgent(
        context.opponent_models[name],
        context.encoders[int(specification["encoder_version"])],
        card_catalog=context.card_catalog,
        attack_catalog=context.attack_catalog,
        seed=seed,
    )


def _collect_rollout_shard(task: _WorkerRolloutTask) -> list[_WorkerGameResult]:
    context = _ROLLOUT_WORKER_CONTEXT
    if context is None:
        raise RuntimeError("Rollout worker was not initialized.")
    if len(task.game_indices) != len(task.opponent_names):
        raise ValueError("Worker game indices and opponent names must have the same length.")
    opponent_state = _refresh_worker_models(context, task)
    results: list[_WorkerGameResult] = []

    for game, opponent_name in zip(
        task.game_indices, task.opponent_names, strict=True
    ):
        learning_player = game % 2
        opponent_seed = task.seed + game
        torch.manual_seed(opponent_seed + 200_000)
        opponent = _worker_opponent(
            context,
            opponent_state[opponent_name],
            opponent_seed,
        )
        resolver = RuleAgent(
            card_catalog=context.card_catalog,
            attack_catalog=context.attack_catalog,
            seed=opponent_seed + 100_000,
        )
        observation, start_data = context.engine.start(context.deck, context.deck)
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
                    decision = context.encoder.encode(observation)
                    batch = _single_batch(decision, torch.device("cpu"))
                    with torch.no_grad():
                        policy_logits, value_logits = context.model(batch)
                        distribution = Categorical(logits=policy_logits)
                        sampled_action = distribution.sample()
                        log_probability = distribution.log_prob(sampled_action)
                        value = context.model.expected_value(value_logits)
                    action = [int(sampled_action.item())]
                    episode.append(
                        PPOTransition(
                            decision=decision,
                            action=action[0],
                            old_log_probability=float(log_probability.item()),
                            old_value=float(value.item()),
                            potential=(
                                prize_potential(observation, learning_player)
                                if task.reward_shaping == "prize"
                                else 0.0
                            ),
                        )
                    )
                observation = context.engine.select(action)

            winner = int(observation["current"]["result"])
            terminal_return = (
                0.0 if winner == 2 else (1.0 if winner == learning_player else -1.0)
            )
            assign_episode_returns(
                episode,
                terminal_return,
                gamma=task.gamma,
                gae_lambda=task.gae_lambda,
                value_gae_lambda=task.value_gae_lambda,
                reward_shaping=task.reward_shaping,
                reward_shaping_scale=task.reward_shaping_scale,
            )
            results.append(
                _WorkerGameResult(
                    game=game,
                    transitions=episode,
                    outcome=terminal_return,
                    opponent_name=opponent_name,
                )
            )
        finally:
            context.engine.finish()
    return results


def collect_rollout_parallel(
    executor: ProcessPoolExecutor,
    model: PolicyValueModel,
    *,
    games: int,
    seed: int,
    workers: int,
    opponent_pool: OpponentPool,
    gamma: float,
    gae_lambda: float,
    value_gae_lambda: float,
    reward_shaping: str,
    reward_shaping_scale: float,
) -> tuple[list[PPOTransition], list[float], list[str]]:
    """Collect rollout shards concurrently in process-isolated simulators."""
    opponent_names = [opponent_pool.sample_name() for _ in range(games)]
    assignments = partition_rollout_assignments(opponent_names, workers)
    model_state = _share_state_dict(model)
    opponent_state = _share_opponent_state(opponent_pool.worker_state())
    futures = [
        executor.submit(
            _collect_rollout_shard,
            _WorkerRolloutTask(
                game_indices=indices,
                opponent_names=names,
                model_state_dict=model_state,
                opponent_state=opponent_state,
                seed=seed,
                gamma=gamma,
                gae_lambda=gae_lambda,
                value_gae_lambda=value_gae_lambda,
                reward_shaping=reward_shaping,
                reward_shaping_scale=reward_shaping_scale,
            ),
        )
        for indices, names in assignments
    ]
    game_results = [item for future in futures for item in future.result()]
    game_results.sort(key=lambda item: item.game)
    transitions = [
        transition for result in game_results for transition in result.transitions
    ]
    outcomes = [result.outcome for result in game_results]
    sampled_names = [result.opponent_name for result in game_results]
    return transitions, outcomes, sampled_names


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
    value_gae_lambda: float,
    reward_shaping: str,
    reward_shaping_scale: float,
) -> tuple[list[PPOTransition], list[float], list[str]]:
    transitions: list[PPOTransition] = []
    outcomes: list[float] = []
    opponent_names: list[str] = []
    model.eval()

    for game in range(games):
        learning_player = game % 2
        opponent_seed = seed + game
        torch.manual_seed(opponent_seed + 200_000)
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
                            potential=(
                                prize_potential(observation, learning_player)
                                if reward_shaping == "prize"
                                else 0.0
                            ),
                        )
                    )
                observation = engine.select(action)

            winner = int(observation["current"]["result"])
            terminal_return = 0.0 if winner == 2 else (1.0 if winner == learning_player else -1.0)
            outcomes.append(terminal_return)
            assign_episode_returns(
                episode,
                terminal_return,
                gamma=gamma,
                gae_lambda=gae_lambda,
                value_gae_lambda=value_gae_lambda,
                reward_shaping=reward_shaping,
                reward_shaping_scale=reward_shaping_scale,
            )
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


def iteration_log_values(metrics: IterationMetrics) -> dict[str, int | float]:
    """Flatten iteration metrics into stable W&B dashboard keys."""
    games = max(metrics.games, 1)
    values: dict[str, int | float] = {
        "iteration": metrics.iteration,
        "rollout/games": metrics.games,
        "rollout/transitions": metrics.transitions,
        "rollout/wins": metrics.wins,
        "rollout/draws": metrics.draws,
        "rollout/losses": metrics.losses,
        "rollout/win_rate": metrics.wins / games,
        "rollout/draw_rate": metrics.draws / games,
        "rollout/mean_return": metrics.mean_return,
        "train/policy_loss": metrics.policy_loss,
        "train/value_loss": metrics.value_loss,
        "train/entropy": metrics.entropy,
        "train/approximate_kl": metrics.approximate_kl,
        "train/clip_fraction": metrics.clip_fraction,
        "train/explained_variance": metrics.explained_variance,
        "train/gradient_norm": metrics.gradient_norm,
        "time/rollout_seconds": metrics.rollout_seconds,
        "time/update_seconds": metrics.update_seconds,
        "time/iteration_seconds": metrics.rollout_seconds + metrics.update_seconds,
        "performance/games_per_second": metrics.games_per_second,
        "reward/mean_abs_shaping_reward": metrics.mean_abs_shaping_reward,
        "reward/mean_shaping_return": metrics.mean_shaping_return,
    }
    for name, result in metrics.opponents.items():
        opponent_games = max(int(result["games"]), 1)
        values[f"opponent/{name}/games"] = int(result["games"])
        values[f"opponent/{name}/win_rate"] = int(result["wins"]) / opponent_games
        values[f"opponent/{name}/mean_return"] = float(result["mean_return"])
    for name, weight in metrics.sampling_weights.items():
        values[f"sampling_weight/{name}"] = weight
    return values


def initialize_wandb(args: argparse.Namespace):
    """Create an optional parent-process-only W&B run."""
    if args.wandb_mode == "disabled":
        return None
    if not args.wandb_project:
        raise ValueError("--wandb-project is required unless --wandb-mode=disabled")
    try:
        wandb = importlib.import_module("wandb")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "W&B tracking requested but wandb is not installed. Install poketcg-agent[tracking]."
        ) from exc
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"wandb_run_id"}
    }
    options: dict[str, Any] = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
        "mode": args.wandb_mode,
        "config": config,
    }
    if args.wandb_run_id:
        options["id"] = args.wandb_run_id
        options["resume"] = "allow"
    return wandb.init(**options)


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
) -> PPOUpdateMetrics:
    advantages = torch.tensor([item.advantage for item in transitions])
    normalized = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
    for item, advantage in zip(transitions, normalized.tolist(), strict=True):
        item.advantage = advantage

    random_generator = random.Random(seed)
    totals = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
            gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm))
            optimizer.step()

            approximate_kl = float(((ratio - 1.0) - log_ratio).mean().detach())
            clip_fraction = float(((ratio - 1.0).abs() > clip_ratio).float().mean().detach())
            values = [
                float(policy_loss.detach()),
                float(value_loss.detach()),
                float(entropy.detach()),
                approximate_kl,
                clip_fraction,
                gradient_norm,
            ]
            totals = [total + value for total, value in zip(totals, values, strict=True)]
            updates += 1
            if approximate_kl > target_kl:
                stop_early = True
                break
        if stop_early:
            break
    averaged = [total / updates for total in totals]
    return PPOUpdateMetrics(*averaged)


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
    parser.add_argument(
        "--value-gae-lambda",
        type=float,
        help=(
            "Independent lambda for critic targets. Defaults to --gae-lambda for backward "
            "compatibility; use 1.0 for Monte Carlo targets when gamma is 1.0."
        ),
    )
    parser.add_argument(
        "--reward-shaping",
        choices=("none", "prize"),
        default="none",
        help="Optional potential-based policy reward shaping.",
    )
    parser.add_argument(
        "--reward-shaping-scale",
        type=float,
        default=1.0,
        help="Scale applied to potential differences; ignored when shaping is disabled.",
    )
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
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=1,
        help="Spawned CPU simulator processes. Values above one require --rollout-device=cpu.",
    )
    parser.add_argument(
        "--worker-torch-threads",
        type=int,
        default=1,
        help="Intra-op Torch threads per rollout worker; keep at one to avoid oversubscription.",
    )
    parser.add_argument("--deck", type=Path)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-run-id")
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "online", "offline"),
        default="disabled",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.value_gae_lambda is None:
        args.value_gae_lambda = args.gae_lambda
    if args.rollout_workers < 1:
        raise ValueError("--rollout-workers must be at least one")
    if args.worker_torch_threads < 1:
        raise ValueError("--worker-torch-threads must be at least one")
    if args.reward_shaping_scale < 0.0:
        raise ValueError("--reward-shaping-scale must be non-negative")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("--gae-lambda must be in [0, 1]")
    if not 0.0 <= args.value_gae_lambda <= 1.0:
        raise ValueError("--value-gae-lambda must be in [0, 1]")
    if args.rollout_workers > 1 and args.rollout_device != "cpu":
        raise ValueError("Parallel rollout requires --rollout-device=cpu")
    torch.manual_seed(args.seed)
    learner_device = resolve_device(args.device)
    rollout_device = resolve_device(args.rollout_device)
    saved = torch.load(args.input, map_location=learner_device, weights_only=False)
    model = build_model(saved["model_config"]).to(learner_device)
    model.load_state_dict(saved["model_state_dict"])
    if args.rollout_workers > 1:
        rollout_model = None
    elif rollout_device == learner_device:
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
    wandb_run = initialize_wandb(args)
    executor: ProcessPoolExecutor | None = None
    if args.rollout_workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=args.rollout_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_rollout_worker,
            initargs=(
                str(engine.official_dir),
                deck,
                saved["model_config"],
                encoder_version(saved["model_config"]),
                args.worker_torch_threads,
            ),
        )

    try:
        for iteration in range(1, args.iterations + 1):
            iteration_seed = args.seed + iteration * args.games_per_iteration
            rollout_started = time.perf_counter()
            if executor is None:
                if rollout_model is None:
                    raise AssertionError("Serial rollout model was not initialized.")
                if rollout_model is not model:
                    rollout_model.load_state_dict(model.state_dict())
                transitions, outcomes, opponent_names = collect_rollout(
                    engine,
                    deck,
                    rollout_model,
                    encoder,
                    games=args.games_per_iteration,
                    seed=iteration_seed,
                    device=rollout_device,
                    opponent_pool=opponent_pool,
                    card_catalog=card_catalog,
                    attack_catalog=attack_catalog,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    value_gae_lambda=args.value_gae_lambda,
                    reward_shaping=args.reward_shaping,
                    reward_shaping_scale=args.reward_shaping_scale,
                )
            else:
                transitions, outcomes, opponent_names = collect_rollout_parallel(
                    executor,
                    model,
                    games=args.games_per_iteration,
                    seed=iteration_seed,
                    workers=args.rollout_workers,
                    opponent_pool=opponent_pool,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    value_gae_lambda=args.value_gae_lambda,
                    reward_shaping=args.reward_shaping,
                    reward_shaping_scale=args.reward_shaping_scale,
                )
            rollout_seconds = time.perf_counter() - rollout_started
            if not transitions:
                raise RuntimeError("Rollout produced no learnable transitions.")
            opponent_pool.record_results(opponent_names, outcomes)
            critic_explained_variance = explained_variance(
                [item.old_value for item in transitions],
                [item.value_target for item in transitions],
            )
            torch.manual_seed(args.seed + iteration)
            update_started = time.perf_counter()
            update = ppo_update(
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
            update_seconds = time.perf_counter() - update_started
            metrics = IterationMetrics(
                iteration=iteration,
                games=len(outcomes),
                wins=sum(value > 0 for value in outcomes),
                draws=sum(value == 0 for value in outcomes),
                losses=sum(value < 0 for value in outcomes),
                transitions=len(transitions),
                mean_return=round(sum(outcomes) / len(outcomes), 6),
                policy_loss=round(update.policy_loss, 6),
                value_loss=round(update.value_loss, 6),
                entropy=round(update.entropy, 6),
                approximate_kl=round(update.approximate_kl, 6),
                clip_fraction=round(update.clip_fraction, 6),
                explained_variance=round(critic_explained_variance, 6),
                gradient_norm=round(update.gradient_norm, 6),
                rollout_seconds=round(rollout_seconds, 4),
                update_seconds=round(update_seconds, 4),
                games_per_second=round(len(outcomes) / rollout_seconds, 4),
                mean_abs_shaping_reward=round(
                    sum(abs(item.shaping_reward) for item in transitions)
                    / len(transitions),
                    6,
                ),
                mean_shaping_return=round(
                    sum(item.shaping_reward for item in transitions) / len(outcomes),
                    6,
                ),
                opponents=summarize_opponent_results(opponent_names, outcomes),
                sampling_weights=opponent_pool.effective_weights(),
            )
            history.append(metrics)
            print(json.dumps(asdict(metrics)), flush=True)
            if wandb_run is not None:
                wandb_run.log(iteration_log_values(metrics), step=iteration)

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
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if wandb_run is not None:
            wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
