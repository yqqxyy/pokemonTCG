"""Held-out evaluation of observation-conditioned card-effect option policies."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from poketcg.agents.rule_agent import OptionType
from poketcg.mcts import (
    DeckDeterminizer,
    HiddenStateGuess,
    OfficialSearchBackend,
    PolicyValueMCTSAgent,
    SearchBackend,
    SearchPosition,
)

from .paired_rollout import RootCandidate, paired_summary
from .semantic_plan import (
    SemanticAction,
    resolve_semantic_action,
    semantic_action,
)


class OptionPolicy(Protocol):
    def choose_action(self, observation: dict) -> list[int]: ...


CandidateFactory = Callable[[dict], Sequence[RootCandidate]]


def _card_signature(card: dict | None) -> tuple | None:
    if not card:
        return None

    def attached(name: str) -> tuple[int, ...]:
        return tuple(sorted(int(item.get("id") or 0) for item in card.get(name) or []))

    return (
        int(card.get("id") or 0),
        int(card.get("hp") or 0),
        int(card.get("maxHp") or 0),
        int(card.get("specialCondition") or 0),
        attached("energyCards"),
        attached("tools"),
        attached("preEvolution"),
    )


def _player_signature(player: dict, *, reveal_hand: bool) -> tuple:
    hand = player.get("hand") or []
    hand_signature = (
        tuple(sorted(int(card.get("id") or 0) for card in hand))
        if reveal_hand
        else len(hand)
    )
    return (
        len(player.get("deck") or []),
        len(player.get("prize") or []),
        hand_signature,
        tuple(sorted(int(card.get("id") or 0) for card in player.get("discard") or [])),
        tuple(_card_signature(card) for card in player.get("active") or []),
        tuple(_card_signature(card) for card in player.get("bench") or []),
    )


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True)
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


def option_selection_key(observation: dict) -> str:
    """Fingerprint the currently visible semantic choice set."""
    selection = observation["select"]
    keys = [
        semantic_action(observation, [index]).semantic_key()
        for index in range(len(selection["option"]))
    ]
    payload = (
        int(selection["context"]),
        int(selection["minCount"]),
        int(selection["maxCount"]),
        tuple(sorted(keys, key=repr)),
    )
    return _stable_hash(payload)


def option_public_state_key(observation: dict) -> str:
    """Fingerprint public state without physical serials or hidden deck ordering."""
    current = observation["current"]
    your_index = int(current["yourIndex"])
    players = current.get("players") or []
    player_signatures = tuple(
        _player_signature(player, reveal_hand=index == your_index)
        for index, player in enumerate(players)
    )
    flags = tuple(
        (name, current.get(name))
        for name in (
            "energyAttached",
            "supporter",
            "retreated",
            "stadium",
        )
        if name in current and not isinstance(current.get(name), (dict, list))
    )
    payload = (
        int(current["turn"]),
        your_index,
        player_signatures,
        tuple(_card_signature(card) for card in current.get("stadium") or []),
        flags,
        option_selection_key(observation),
    )
    return _stable_hash(payload)


@dataclass(frozen=True, slots=True)
class ClosedLoopOptionPolicy:
    """A root action plus observation-conditioned continuation rules."""

    root_action: SemanticAction
    exact_rules: dict[str, SemanticAction]
    selection_rules: dict[str, SemanticAction]
    source_action: tuple[int, ...]
    sources: tuple[str, ...]

    def continuation(self, observation: dict) -> tuple[list[int] | None, str]:
        directive = self.exact_rules.get(option_public_state_key(observation))
        if directive is not None:
            action = resolve_semantic_action(observation, directive)
            if action is not None:
                return action, "exact"
        directive = self.selection_rules.get(option_selection_key(observation))
        if directive is not None:
            action = resolve_semantic_action(observation, directive)
            if action is not None:
                return action, "selection"
        return None, "fallback"

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_action": self.root_action.to_dict(),
            "source_action": list(self.source_action),
            "sources": list(self.sources),
            "exact_rule_count": len(self.exact_rules),
            "selection_rule_count": len(self.selection_rules),
            "exact_rules": {
                key: directive.to_dict()
                for key, directive in sorted(self.exact_rules.items())
            },
            "selection_rules": {
                key: directive.to_dict()
                for key, directive in sorted(self.selection_rules.items())
            },
        }


@dataclass(frozen=True, slots=True)
class _OptionLeaf:
    position: SearchPosition
    decisions: tuple[tuple[str, str, SemanticAction], ...]
    raw_sequence: tuple[tuple[int, ...], ...]
    boundary: str
    heuristic: float


@dataclass(frozen=True, slots=True)
class _Trajectory:
    world_id: int
    decisions: tuple[tuple[str, str, SemanticAction], ...]
    return_value: float
    boundary: str


def _gain_lcb(summary: dict, risk_multiplier: float) -> float:
    gain = summary.get("paired_advantage")
    stderr = summary.get("paired_stderr")
    if gain is None or stderr is None:
        return -float("inf")
    return float(gain) - risk_multiplier * float(stderr)


class HeldoutCardEffectEvaluator:
    """Build candidates, calibrate the gate, then audit on untouched worlds."""

    def __init__(
        self,
        determinizer: DeckDeterminizer,
        root_policy_factory: Callable[[], OptionPolicy],
        opponent_policy_factory: Callable[[], OptionPolicy],
        candidate_factory: CandidateFactory,
        *,
        build_determinizations: int = 8,
        calibration_determinizations: int = 8,
        heldout_determinizations: int = 8,
        root_candidate_limit: int = 8,
        beam_width: int = 8,
        branch_width: int = 4,
        max_option_steps: int = 12,
        max_rollout_steps: int = 1_000,
        selection_risk_multiplier: float = 1.96,
        minimum_calibration_pairs: int = 8,
        minimum_closed_loop_coverage: float = 0.9,
        familywise_alpha: float = 0.05,
        value_policy: Any | None = None,
        backend: SearchBackend | None = None,
    ) -> None:
        positive = {
            "build_determinizations": build_determinizations,
            "calibration_determinizations": calibration_determinizations,
            "heldout_determinizations": heldout_determinizations,
            "root_candidate_limit": root_candidate_limit,
            "beam_width": beam_width,
            "branch_width": branch_width,
            "max_option_steps": max_option_steps,
            "max_rollout_steps": max_rollout_steps,
            "minimum_calibration_pairs": minimum_calibration_pairs,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if selection_risk_multiplier < 0:
            raise ValueError("selection_risk_multiplier must be non-negative")
        if not 0.0 <= minimum_closed_loop_coverage <= 1.0:
            raise ValueError("minimum_closed_loop_coverage must be in [0, 1]")
        if not 0.0 < familywise_alpha < 1.0:
            raise ValueError("familywise_alpha must be in (0, 1)")
        self._determinizer = determinizer
        self._root_policy_factory = root_policy_factory
        self._opponent_policy_factory = opponent_policy_factory
        self._candidate_factory = candidate_factory
        self._build_determinizations = build_determinizations
        self._calibration_determinizations = calibration_determinizations
        self._heldout_determinizations = heldout_determinizations
        self._root_candidate_limit = root_candidate_limit
        self._beam_width = beam_width
        self._branch_width = branch_width
        self._max_option_steps = max_option_steps
        self._max_rollout_steps = max_rollout_steps
        self._selection_risk_multiplier = selection_risk_multiplier
        self._minimum_calibration_pairs = minimum_calibration_pairs
        self._minimum_closed_loop_coverage = minimum_closed_loop_coverage
        self._familywise_alpha = familywise_alpha
        self._value_policy = value_policy
        self._backend = backend or OfficialSearchBackend()
        self._branch_errors: Counter[str] = Counter()

    @staticmethod
    def _choose(policy: OptionPolicy, observation: dict) -> list[int]:
        deterministic = getattr(policy, "choose_deterministic_action", None)
        if callable(deterministic):
            return list(deterministic(observation))
        return list(policy.choose_action(observation))

    @staticmethod
    def _terminal_value(observation: dict, root_player: int) -> float | None:
        return PolicyValueMCTSAgent._terminal_value(observation, root_player)

    @staticmethod
    def _inside_option(observation: dict, root_player: int, root_turn: int) -> bool:
        current = observation["current"]
        return (
            int(current["result"]) == -1
            and int(current["yourIndex"]) == root_player
            and int(current["turn"]) == root_turn
            and int(observation["select"]["context"]) != 0
        )

    def _boundary(self, observation: dict, root_player: int, root_turn: int) -> str:
        if self._terminal_value(observation, root_player) is not None:
            return "terminal"
        current = observation["current"]
        if (
            int(current["yourIndex"]) != root_player
            or int(current["turn"]) != root_turn
        ):
            return "turn_changed"
        if int(observation["select"]["context"]) == 0:
            return "return_main"
        return "active"

    def _heuristic(self, observation: dict, root_player: int) -> float:
        terminal = self._terminal_value(observation, root_player)
        if terminal is not None:
            return terminal
        if self._value_policy is None:
            return 0.0
        evaluation = self._value_policy.evaluate(observation)
        acting = int(observation["current"]["yourIndex"])
        value = float(evaluation.value)
        return value if acting == root_player else -value

    def _rollout(self, position: SearchPosition, *, root_player: int) -> dict[str, Any]:
        terminal = self._terminal_value(position.observation, root_player)
        if terminal is not None:
            return {"return": terminal, "rollout_steps": 0, "rollout_boundary": "terminal"}
        policies = {
            root_player: self._root_policy_factory(),
            1 - root_player: self._opponent_policy_factory(),
        }
        current = position
        for steps in range(self._max_rollout_steps):
            observation = current.observation
            terminal = self._terminal_value(observation, root_player)
            if terminal is not None:
                return {
                    "return": terminal,
                    "rollout_steps": steps,
                    "rollout_boundary": "terminal",
                }
            player = int(observation["current"]["yourIndex"])
            action = self._choose(policies[player], observation)
            current = self._backend.step(current.search_id, action)
        return {
            "return": round(self._heuristic(current.observation, root_player), 6),
            "rollout_steps": self._max_rollout_steps,
            "rollout_boundary": "value_bootstrap",
        }

    def _complete_with_v1(
        self,
        position: SearchPosition,
        *,
        root_player: int,
        root_turn: int,
        sequence: list[tuple[int, ...]] | None = None,
    ) -> tuple[SearchPosition, list[tuple[int, ...]], int]:
        current = position
        actions = [] if sequence is None else sequence
        fallback_steps = 0
        policy = self._root_policy_factory()
        while (
            fallback_steps < self._max_option_steps
            and self._inside_option(current.observation, root_player, root_turn)
        ):
            action = self._choose(policy, current.observation)
            current = self._backend.step(current.search_id, action)
            actions.append(tuple(action))
            fallback_steps += 1
        return current, actions, fallback_steps

    def _search_root(
        self,
        root: SearchPosition,
        candidate: RootCandidate,
        *,
        world_id: int,
        root_player: int,
        root_turn: int,
    ) -> tuple[list[_Trajectory], dict[str, Any]]:
        root_position = self._backend.step(root.search_id, list(candidate.action))
        boundary = self._boundary(root_position.observation, root_player, root_turn)
        initial = _OptionLeaf(
            position=root_position,
            decisions=(),
            raw_sequence=(candidate.action,),
            boundary=boundary,
            heuristic=self._heuristic(root_position.observation, root_player),
        )
        active = [initial] if boundary == "active" else []
        completed = [] if active else [initial]
        for _ in range(self._max_option_steps):
            children: list[_OptionLeaf] = []
            for leaf in active:
                candidates = list(self._candidate_factory(leaf.position.observation))
                unique: list[RootCandidate] = []
                seen: set[tuple[int, ...]] = set()
                for item in candidates:
                    if item.action not in seen:
                        unique.append(item)
                        seen.add(item.action)
                    if len(unique) == self._branch_width:
                        break
                for item in unique:
                    try:
                        position = self._backend.step(
                            leaf.position.search_id, list(item.action)
                        )
                    except Exception as error:
                        self._branch_errors[
                            f"{type(error).__name__}: {error}"
                        ] += 1
                        continue
                    directive = semantic_action(
                        leaf.position.observation, item.action
                    )
                    child = _OptionLeaf(
                        position=position,
                        decisions=(
                            *leaf.decisions,
                            (
                                option_public_state_key(leaf.position.observation),
                                option_selection_key(leaf.position.observation),
                                directive,
                            ),
                        ),
                        raw_sequence=(*leaf.raw_sequence, item.action),
                        boundary=self._boundary(
                            position.observation, root_player, root_turn
                        ),
                        heuristic=self._heuristic(
                            position.observation, root_player
                        ),
                    )
                    if child.boundary == "active":
                        children.append(child)
                    else:
                        completed.append(child)
            if not children:
                active = []
                break
            active = sorted(
                children,
                key=lambda item: (
                    item.heuristic,
                    -len(item.raw_sequence),
                    item.raw_sequence,
                ),
                reverse=True,
            )[: self._beam_width]
        capped = 0
        for leaf in active:
            position, sequence, _ = self._complete_with_v1(
                leaf.position,
                root_player=root_player,
                root_turn=root_turn,
                sequence=list(leaf.raw_sequence),
            )
            completed.append(
                _OptionLeaf(
                    position=position,
                    decisions=leaf.decisions,
                    raw_sequence=tuple(sequence),
                    boundary=self._boundary(
                        position.observation, root_player, root_turn
                    ),
                    heuristic=self._heuristic(position.observation, root_player),
                )
            )
            capped += 1
        completed = sorted(
            completed,
            key=lambda item: (
                item.heuristic,
                -len(item.raw_sequence),
                item.raw_sequence,
            ),
            reverse=True,
        )[: self._beam_width]
        trajectories = []
        samples = []
        for leaf in completed:
            result = self._rollout(leaf.position, root_player=root_player)
            trajectories.append(
                _Trajectory(
                    world_id=world_id,
                    decisions=leaf.decisions,
                    return_value=float(result["return"]),
                    boundary=leaf.boundary,
                )
            )
            samples.append(
                {
                    "sequence": [list(action) for action in leaf.raw_sequence],
                    "continuation_decisions": len(leaf.decisions),
                    "option_boundary": leaf.boundary,
                    **result,
                }
            )
        return trajectories, {
            "root_action": list(candidate.action),
            "sources": list(candidate.sources),
            "completed_options": len(trajectories),
            "capped_options": capped,
            "samples": samples,
        }

    @staticmethod
    def _compile_rule_table(
        trajectories: list[_Trajectory],
        *,
        exact: bool,
    ) -> dict[str, SemanticAction]:
        per_world: dict[
            tuple[str, tuple], dict[int, float]
        ] = defaultdict(dict)
        directives: dict[tuple[str, tuple], SemanticAction] = {}
        for trajectory in trajectories:
            for exact_key, selection_key, directive in trajectory.decisions:
                state_key = exact_key if exact else selection_key
                action_key = directive.semantic_key()
                key = (state_key, action_key)
                previous = per_world[key].get(trajectory.world_id)
                if previous is None or trajectory.return_value > previous:
                    per_world[key][trajectory.world_id] = trajectory.return_value
                directives[key] = directive
        by_state: dict[str, list[tuple[float, int, tuple, SemanticAction]]] = (
            defaultdict(list)
        )
        for (state_key, action_key), world_values in per_world.items():
            values = list(world_values.values())
            stderr = (
                statistics.stdev(values) / math.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )
            score = statistics.mean(values) - stderr
            by_state[state_key].append(
                (score, len(values), action_key, directives[(state_key, action_key)])
            )
        return {
            state_key: max(
                choices,
                key=lambda item: (item[0], item[1], repr(item[2])),
            )[3]
            for state_key, choices in by_state.items()
        }

    def _build_policies(
        self,
        observation: dict,
        hidden_states: list[HiddenStateGuess],
        candidates: list[RootCandidate],
        *,
        root_player: int,
        root_turn: int,
        world_id_offset: int = 0,
    ) -> tuple[list[ClosedLoopOptionPolicy], list[dict], dict[str, int]]:
        trajectories = [[] for _ in candidates]
        samples = []
        errors: Counter[str] = Counter()
        for local_world_id, hidden in enumerate(hidden_states):
            world_id = world_id_offset + local_world_id
            began = False
            branches = []
            try:
                root = self._backend.begin(observation, hidden)
                began = True
                for index, candidate in enumerate(candidates):
                    found, sample = self._search_root(
                        root,
                        candidate,
                        world_id=world_id,
                        root_player=root_player,
                        root_turn=root_turn,
                    )
                    trajectories[index].extend(found)
                    branches.append(sample)
            except Exception as error:
                key = f"{type(error).__name__}: {error}"
                errors[key] += 1
                branches.append(
                    {"error": type(error).__name__, "error_message": str(error)}
                )
            finally:
                if began:
                    self._backend.end()
            samples.append(
                {
                    "world_id": world_id,
                    "opponent_deck_name": hidden.opponent_deck_name,
                    "branches": branches,
                }
            )
        policies = []
        for candidate, paths in zip(candidates, trajectories, strict=True):
            if not paths:
                continue
            policies.append(
                ClosedLoopOptionPolicy(
                    root_action=semantic_action(observation, candidate.action),
                    exact_rules=self._compile_rule_table(paths, exact=True),
                    selection_rules=self._compile_rule_table(paths, exact=False),
                    source_action=candidate.action,
                    sources=candidate.sources,
                )
            )
        return policies, samples, dict(sorted(errors.items()))

    def _execute_policy(
        self,
        root: SearchPosition,
        policy: ClosedLoopOptionPolicy,
        *,
        root_player: int,
        root_turn: int,
    ) -> dict[str, Any]:
        action = resolve_semantic_action(root.observation, policy.root_action)
        if action is None:
            raise RuntimeError("Unable to resolve option root action")
        position = self._backend.step(root.search_id, action)
        sequence = [tuple(action)]
        routes: Counter[str] = Counter()
        planned_steps = 0
        fallback_steps = 0
        steps = 0
        fallback_policy = self._root_policy_factory()
        while (
            steps < self._max_option_steps
            and self._inside_option(position.observation, root_player, root_turn)
        ):
            continuation, route = policy.continuation(position.observation)
            if continuation is None:
                continuation = self._choose(
                    fallback_policy, position.observation
                )
                fallback_steps += 1
            else:
                planned_steps += 1
            routes[route] += 1
            position = self._backend.step(position.search_id, continuation)
            sequence.append(tuple(continuation))
            steps += 1
        capped = self._inside_option(
            position.observation, root_player, root_turn
        )
        if capped:
            position, sequence, extra = self._complete_with_v1(
                position,
                root_player=root_player,
                root_turn=root_turn,
                sequence=sequence,
            )
            fallback_steps += extra
            routes["capped_fallback"] += extra
        total = planned_steps + fallback_steps
        return {
            "option_sequence": [list(item) for item in sequence],
            "option_boundary": self._boundary(
                position.observation, root_player, root_turn
            ),
            "continuation_steps": total,
            "planned_steps": planned_steps,
            "fallback_steps": fallback_steps,
            "closed_loop_coverage": (
                round(planned_steps / total, 6) if total else None
            ),
            "routes": dict(sorted(routes.items())),
            **self._rollout(position, root_player=root_player),
        }

    def _baseline(
        self,
        root: SearchPosition,
        *,
        root_player: int,
        root_turn: int,
    ) -> dict[str, Any]:
        policy = self._root_policy_factory()
        action = self._choose(policy, root.observation)
        position = self._backend.step(root.search_id, action)
        sequence = [tuple(action)]
        position, sequence, continuation_steps = self._complete_with_v1(
            position,
            root_player=root_player,
            root_turn=root_turn,
            sequence=sequence,
        )
        return {
            "option_sequence": [list(item) for item in sequence],
            "option_boundary": self._boundary(
                position.observation, root_player, root_turn
            ),
            "continuation_steps": continuation_steps,
            **self._rollout(position, root_player=root_player),
        }

    def _evaluate_policies(
        self,
        observation: dict,
        hidden_states: list[HiddenStateGuess],
        policies: list[ClosedLoopOptionPolicy],
        *,
        root_player: int,
        root_turn: int,
        world_id_offset: int = 0,
    ) -> tuple[list[dict], list[dict], dict[str, int]]:
        candidate_returns = [[] for _ in policies]
        baseline_returns = [[] for _ in policies]
        coverages = [[] for _ in policies]
        fallback_steps = [0] * len(policies)
        continuation_steps = [0] * len(policies)
        continuation_worlds = [0] * len(policies)
        samples = []
        errors: Counter[str] = Counter()
        for local_world_id, hidden in enumerate(hidden_states):
            world_id = world_id_offset + local_world_id
            began = False
            branches = []
            try:
                root = self._backend.begin(observation, hidden)
                began = True
                baseline = self._baseline(
                    root, root_player=root_player, root_turn=root_turn
                )
                for index, policy in enumerate(policies):
                    result = self._execute_policy(
                        root,
                        policy,
                        root_player=root_player,
                        root_turn=root_turn,
                    )
                    candidate_returns[index].append(float(result["return"]))
                    baseline_returns[index].append(float(baseline["return"]))
                    coverage = result["closed_loop_coverage"]
                    if coverage is not None:
                        coverages[index].append(float(coverage))
                        continuation_worlds[index] += 1
                    fallback_steps[index] += int(result["fallback_steps"])
                    continuation_steps[index] += int(
                        result["continuation_steps"]
                    )
                    branches.append({"policy_index": index, **result})
                samples.append(
                    {
                        "world_id": world_id,
                        "opponent_deck_name": hidden.opponent_deck_name,
                        "baseline": baseline,
                        "branches": branches,
                    }
                )
            except Exception as error:
                key = f"{type(error).__name__}: {error}"
                errors[key] += 1
                samples.append(
                    {
                        "world_id": world_id,
                        "opponent_deck_name": hidden.opponent_deck_name,
                        "error": type(error).__name__,
                        "error_message": str(error),
                    }
                )
            finally:
                if began:
                    self._backend.end()
        summaries = []
        for index, policy in enumerate(policies):
            summary = paired_summary(
                candidate_returns[index], baseline_returns[index]
            )
            summaries.append(
                {
                    "policy_index": index,
                    "policy": policy.to_dict(),
                    **summary,
                    "gain_lcb": round(
                        _gain_lcb(summary, self._selection_risk_multiplier), 6
                    ),
                    "mean_closed_loop_coverage": round(
                        statistics.mean(coverages[index])
                        if coverages[index]
                        else 0.0,
                        6,
                    ),
                    "fallback_steps": fallback_steps[index],
                    "continuation_steps": continuation_steps[index],
                    "continuation_worlds": continuation_worlds[index],
                    "closed_loop_step_coverage": round(
                        1.0
                        - fallback_steps[index]
                        / max(1, continuation_steps[index]),
                        6,
                    ),
                }
            )
        return summaries, samples, dict(sorted(errors.items()))

    def evaluate(self, observation: dict) -> dict[str, Any]:
        self._branch_errors.clear()
        root_player = int(observation["current"]["yourIndex"])
        root_turn = int(observation["current"]["turn"])
        candidates = list(self._candidate_factory(observation))
        baseline_policy = self._root_policy_factory()
        baseline_action = tuple(self._choose(baseline_policy, observation))
        options = observation["select"]["option"]

        def card_effect(candidate: RootCandidate) -> bool:
            return bool(candidate.action) and all(
                int(options[index]["type"]) == int(OptionType.PLAY)
                for index in candidate.action
            )

        candidates = [candidate for candidate in candidates if card_effect(candidate)]
        existing_actions = {candidate.action for candidate in candidates}
        for index, option in enumerate(options):
            action = (index,)
            if (
                int(option["type"]) == int(OptionType.PLAY)
                and action not in existing_actions
            ):
                candidates.append(
                    RootCandidate(action, ("card_effect_exhaustive",))
                )
                existing_actions.add(action)
        if (
            baseline_action not in {candidate.action for candidate in candidates}
            and card_effect(RootCandidate(baseline_action, ("v1_choice",)))
        ):
            candidates.insert(0, RootCandidate(baseline_action, ("v1_choice",)))
        candidates = candidates[: self._root_candidate_limit]
        if not candidates:
            raise RuntimeError("Card-effect evaluator received no PLAY candidates")
        hidden_states = [
            self._determinizer.sample(observation)
            for _ in range(
                self._build_determinizations
                + self._calibration_determinizations
                + self._heldout_determinizations
            )
        ]
        build_end = self._build_determinizations
        calibration_end = build_end + self._calibration_determinizations
        build_hidden = hidden_states[:build_end]
        calibration_hidden = hidden_states[build_end:calibration_end]
        heldout_hidden = hidden_states[calibration_end:]
        policies, build_search, search_errors = self._build_policies(
            observation,
            build_hidden,
            candidates,
            root_player=root_player,
            root_turn=root_turn,
            world_id_offset=0,
        )
        if not policies:
            raise RuntimeError("Build search compiled no card-effect policies")
        calibration_summaries, calibration_samples, calibration_errors = (
            self._evaluate_policies(
                observation,
                calibration_hidden,
                policies,
                root_player=root_player,
                root_turn=root_turn,
                world_id_offset=build_end,
            )
        )
        eligible = [
            summary
            for summary in calibration_summaries
            if int(summary["effective_pairs"]) > 0
        ]
        if not eligible:
            raise RuntimeError(
                "No card-effect policy completed calibration evaluation"
            )
        candidate_count = len(eligible)
        bonferroni_multiplier = statistics.NormalDist().inv_cdf(
            1.0 - self._familywise_alpha / candidate_count
        )
        adjusted_multiplier = max(
            self._selection_risk_multiplier,
            bonferroni_multiplier,
        )
        for summary in eligible:
            summary["multiple_comparison_multiplier"] = round(
                adjusted_multiplier, 6
            )
            summary["adjusted_gain_lcb"] = round(
                _gain_lcb(summary, adjusted_multiplier), 6
            )
        selected = max(
            eligible,
            key=lambda item: (
                float(item["adjusted_gain_lcb"]),
                float(item["mean_closed_loop_coverage"]),
                float(item["paired_advantage"]),
                -int(item["policy_index"]),
            ),
        )
        selected_index = int(selected["policy_index"])
        confidence_gate_passed = float(selected["adjusted_gain_lcb"]) > 0.0
        pair_gate_passed = (
            int(selected["effective_pairs"])
            >= self._minimum_calibration_pairs
        )
        coverage_gate_passed = (
            float(selected["closed_loop_step_coverage"])
            >= self._minimum_closed_loop_coverage
        )
        calibration_gate_passed = (
            confidence_gate_passed
            and pair_gate_passed
            and coverage_gate_passed
        )
        selected_policy = policies[selected_index]
        heldout_summaries, heldout_samples, heldout_errors = (
            self._evaluate_policies(
                observation,
                heldout_hidden,
                [selected_policy],
                root_player=root_player,
                root_turn=root_turn,
                world_id_offset=calibration_end,
            )
        )
        heldout = heldout_summaries[0]
        heldout["source_policy_index"] = selected_index
        heldout_gain = heldout.get("paired_advantage")
        heldout_stderr = heldout.get("paired_stderr")
        heldout_accepted = (
            calibration_gate_passed
            and heldout_gain is not None
            and heldout_stderr is not None
            and float(heldout_gain)
            - self._selection_risk_multiplier * float(heldout_stderr)
            > 0.0
        )
        deployable_heldout_gain = (
            float(heldout_gain or 0.0)
            if calibration_gate_passed
            else 0.0
        )
        errors = Counter()
        errors.update(search_errors)
        errors.update(calibration_errors)
        errors.update(heldout_errors)
        return {
            "diagnostic_kind": "three_way_closed_loop_card_effect",
            "root_player": root_player,
            "root_turn": root_turn,
            "build_determinizations": self._build_determinizations,
            "calibration_determinizations": (
                self._calibration_determinizations
            ),
            "heldout_determinizations": self._heldout_determinizations,
            "world_id_ranges": {
                "build": [0, build_end],
                "calibration": [build_end, calibration_end],
                "heldout": [calibration_end, len(hidden_states)],
            },
            "root_candidates": len(candidates),
            "compiled_policies": len(policies),
            "selection_risk_multiplier": self._selection_risk_multiplier,
            "familywise_alpha": self._familywise_alpha,
            "multiple_comparison_multiplier": round(
                adjusted_multiplier, 6
            ),
            "minimum_calibration_pairs": self._minimum_calibration_pairs,
            "minimum_closed_loop_coverage": (
                self._minimum_closed_loop_coverage
            ),
            "selected_policy_index": selected_index,
            "confidence_gate_passed": confidence_gate_passed,
            "pair_gate_passed": pair_gate_passed,
            "coverage_gate_passed": coverage_gate_passed,
            "calibration_gate_passed": calibration_gate_passed,
            "selected_policy": selected_policy.to_dict(),
            "calibration_selected": selected,
            "heldout_selected": heldout,
            "deployable_heldout_gain": round(
                deployable_heldout_gain, 6
            ),
            "heldout_accepted": heldout_accepted,
            "build_search": build_search,
            "calibration_samples": calibration_samples,
            "heldout_samples": heldout_samples,
            "errors": dict(sorted(errors.items())),
            "branch_errors": dict(sorted(self._branch_errors.items())),
        }
