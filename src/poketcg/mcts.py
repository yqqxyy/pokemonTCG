"""Policy-value Monte Carlo tree search using the official Search API."""

from __future__ import annotations

import importlib
import itertools
import math
import random
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Protocol

import torch

from .agents.bc_agent import BCPolicyAgent, PolicyValueEvaluation
from .rl.action_space import (
    deterministic_subset,
    legal_action_set_count,
    sample_subset,
)


@dataclass(frozen=True, slots=True)
class HiddenStateGuess:
    """Card IDs used to determinize zones hidden from the acting player."""

    your_deck: list[int]
    your_prize: list[int]
    opponent_deck: list[int]
    opponent_prize: list[int]
    opponent_hand: list[int]
    opponent_active: list[int]
    opponent_deck_name: str | None = None


@dataclass(frozen=True, slots=True)
class DeckHypothesis:
    """A candidate opponent deck and its prior probability mass."""

    name: str
    deck: tuple[int, ...]
    prior: float = 1.0

    def __post_init__(self) -> None:
        if len(self.deck) != 60:
            raise ValueError(f"Deck hypothesis {self.name!r} must contain 60 cards")
        if self.prior <= 0:
            raise ValueError("Deck hypothesis priors must be positive")


@dataclass(frozen=True, slots=True)
class SearchPosition:
    """A simulator-owned immutable state and its JSON-like observation."""

    search_id: int
    observation: dict


class SearchBackend(Protocol):
    def begin(self, observation: dict, hidden: HiddenStateGuess) -> SearchPosition: ...

    def step(self, search_id: int, action: list[int]) -> SearchPosition: ...

    def end(self) -> None: ...


class MacroExecutor(Protocol):
    """Stateful policy used to execute one complete root-player turn."""

    def choose_action(self, observation: dict) -> list[int]: ...

    def reset_episode(self) -> None: ...


class OfficialSearchBackend:
    """Thin adapter around ``cg.api`` that keeps the rest of MCTS testable."""

    def __init__(self) -> None:
        self._api = importlib.import_module("cg.api")

    @staticmethod
    def _position(state: Any) -> SearchPosition:
        return SearchPosition(
            search_id=int(state.searchId),
            observation=asdict(state.observation),
        )

    def begin(self, observation: dict, hidden: HiddenStateGuess) -> SearchPosition:
        typed = self._api.to_observation_class(observation)
        state = self._api.search_begin(
            typed,
            your_deck=hidden.your_deck,
            your_prize=hidden.your_prize,
            opponent_deck=hidden.opponent_deck,
            opponent_prize=hidden.opponent_prize,
            opponent_hand=hidden.opponent_hand,
            opponent_active=hidden.opponent_active,
        )
        return self._position(state)

    def step(self, search_id: int, action: list[int]) -> SearchPosition:
        return self._position(self._api.search_step(search_id, action))

    def end(self) -> None:
        self._api.search_end()


def _card_id(card: dict | None) -> int | None:
    if not card:
        return None
    value = card.get("id")
    return int(value) if value else None


def _visible_card_counts(observation: dict) -> tuple[Counter[int], Counter[int]]:
    """Count unique currently visible physical cards for each absolute player."""
    counts = (Counter(), Counter())
    seen: set[tuple[int, int]] = set()
    anonymous = 0

    def add(card: dict | None, owner: int) -> None:
        nonlocal anonymous
        card_id = _card_id(card)
        if card_id is None:
            return
        serial = card.get("serial")
        if serial is None:
            anonymous += 1
            key = (owner, -anonymous)
        else:
            key = (owner, int(serial))
        if key in seen:
            return
        seen.add(key)
        counts[owner][card_id] += 1
        for name in ("energyCards", "tools", "preEvolution"):
            for attached in card.get(name) or []:
                add(attached, owner)

    state = observation["current"]
    for owner, player in enumerate(state["players"]):
        for name in ("active", "bench", "hand", "discard", "prize"):
            for card in player.get(name) or []:
                add(card, owner)
    for card in state.get("stadium") or []:
        add(card, int(card.get("playerIndex", 0)))
    for card in state.get("looking") or []:
        if card:
            add(card, int(card.get("playerIndex", state["yourIndex"])))

    selection = observation.get("select") or {}
    for card in selection.get("deck") or []:
        add(card, int(card.get("playerIndex", state["yourIndex"])))
    for name in ("contextCard", "effect"):
        card = selection.get(name)
        if card:
            add(card, int(card.get("playerIndex", state["yourIndex"])))
    return counts


class OpponentDeckBelief:
    """Bayesian posterior over deck hypotheses from currently visible card counts."""

    def __init__(self, hypotheses: list[DeckHypothesis]) -> None:
        if not hypotheses:
            raise ValueError("At least one opponent deck hypothesis is required")
        names = [item.name for item in hypotheses]
        if len(names) != len(set(names)):
            raise ValueError("Opponent deck hypothesis names must be unique")
        self._hypotheses = list(hypotheses)
        self._revealed_by_serial: dict[int, int] = {}
        self._cached_state: str | None = None
        self._cached_posterior: dict[str, float] | None = None

    def reset(self) -> None:
        """Forget evidence from the previous game."""
        self._revealed_by_serial.clear()
        self._cached_state = None
        self._cached_posterior = None

    def _observed_opponent_counts(self, observation: dict) -> Counter[int]:
        """Combine current public zones with cards revealed in earlier event logs."""
        opponent = 1 - int(observation["current"]["yourIndex"])
        for event in observation.get("logs") or []:
            if int(event.get("playerIndex", -1)) != opponent:
                continue
            for card_key, serial_key in (
                ("cardId", "serial"),
                ("cardIdActive", "serialActive"),
                ("cardIdBench", "serialBench"),
                ("cardIdBefore", "serialBefore"),
                ("cardIdAfter", "serialAfter"),
            ):
                card_id = int(event.get(card_key) or 0)
                serial = event.get(serial_key)
                if card_id > 0 and serial is not None:
                    self._revealed_by_serial[int(serial)] = card_id

        historical = Counter(self._revealed_by_serial.values())
        current = _visible_card_counts(observation)[opponent]
        # A physical card can occur in both sources. Per-ID maxima avoid
        # double-counting when a public-zone object omits its serial number.
        return Counter(
            {
                card_id: max(historical[card_id], current[card_id])
                for card_id in historical.keys() | current.keys()
            }
        )

    def posterior(self, observation: dict) -> dict[str, float]:
        state_key = observation.get("search_begin_input")
        if (
            isinstance(state_key, str)
            and state_key == self._cached_state
            and self._cached_posterior is not None
        ):
            return dict(self._cached_posterior)
        observed = self._observed_opponent_counts(observation)
        log_weights: list[float] = []
        for hypothesis in self._hypotheses:
            deck_counts = Counter(hypothesis.deck)
            if any(deck_counts[card_id] < count for card_id, count in observed.items()):
                log_weights.append(float("-inf"))
                continue
            log_likelihood = math.log(hypothesis.prior)
            for card_id, count in observed.items():
                available = deck_counts[card_id]
                log_likelihood += (
                    math.lgamma(available + 1)
                    - math.lgamma(count + 1)
                    - math.lgamma(available - count + 1)
                )
            log_weights.append(log_likelihood)

        finite = [value for value in log_weights if math.isfinite(value)]
        if not finite:
            total = sum(item.prior for item in self._hypotheses)
            posterior = {
                item.name: item.prior / total for item in self._hypotheses
            }
            if isinstance(state_key, str):
                self._cached_state = state_key
                self._cached_posterior = posterior
            return dict(posterior)
        maximum = max(finite)
        weights = [
            math.exp(value - maximum) if math.isfinite(value) else 0.0
            for value in log_weights
        ]
        total = sum(weights)
        posterior = {
            item.name: weight / total
            for item, weight in zip(self._hypotheses, weights, strict=True)
        }
        if isinstance(state_key, str):
            self._cached_state = state_key
            self._cached_posterior = posterior
        return dict(posterior)

    def sample(self, observation: dict, rng: random.Random) -> DeckHypothesis:
        posterior = self.posterior(observation)
        return rng.choices(
            self._hypotheses,
            weights=[posterior[item.name] for item in self._hypotheses],
            k=1,
        )[0]


class DeckDeterminizer:
    """Sample hidden zones from fixed full-deck priors without peeking at engine state."""

    def __init__(
        self,
        deck0: list[int],
        deck1: list[int],
        *,
        basic_card_ids: set[int],
        seed: int | None = None,
        opponent_belief: OpponentDeckBelief | None = None,
    ) -> None:
        if len(deck0) != 60 or len(deck1) != 60:
            raise ValueError("MCTS deck priors must contain exactly 60 cards")
        self._decks = (list(deck0), list(deck1))
        self._basic_card_ids = set(basic_card_ids)
        self._rng = random.Random(seed)
        self._opponent_belief = opponent_belief

    def reset(self) -> None:
        """Reset episode-scoped hidden-information evidence."""
        if self._opponent_belief is not None:
            self._opponent_belief.reset()

    def _remaining_bag(self, deck: list[int], visible: Counter[int]) -> list[int]:
        remaining = Counter(deck)
        for card_id, count in visible.items():
            remaining[card_id] = max(0, remaining[card_id] - count)
        bag = list(remaining.elements())
        self._rng.shuffle(bag)
        return bag

    def _take(self, bag: list[int], count: int, deck: list[int]) -> list[int]:
        result = [bag.pop() for _ in range(min(count, len(bag)))]
        result.extend(self._rng.choice(deck) for _ in range(count - len(result)))
        return result

    def _take_basic(self, bag: list[int], deck: list[int]) -> int:
        for index in range(len(bag) - 1, -1, -1):
            if bag[index] in self._basic_card_ids:
                return bag.pop(index)
        candidates = [card_id for card_id in deck if card_id in self._basic_card_ids]
        if not candidates:
            raise ValueError("Opponent deck prior has no Basic Pokemon for a face-down Active")
        return self._rng.choice(candidates)

    def _prize_guess(
        self, cards: list[dict | None], bag: list[int], deck: list[int]
    ) -> list[int]:
        unknown = self._take(bag, sum(card is None for card in cards), deck)
        iterator = iter(unknown)
        return [int(card["id"]) if card is not None else next(iterator) for card in cards]

    def sample(self, observation: dict) -> HiddenStateGuess:
        state = observation["current"]
        your_index = int(state["yourIndex"])
        opponent_index = 1 - your_index
        your_deck_prior = self._decks[your_index]
        opponent_hypothesis = (
            self._opponent_belief.sample(observation, self._rng)
            if self._opponent_belief is not None
            else DeckHypothesis(
                name=f"fixed_player{opponent_index}",
                deck=tuple(self._decks[opponent_index]),
            )
        )
        opponent_deck_prior = list(opponent_hypothesis.deck)
        visible = _visible_card_counts(observation)
        your_bag = self._remaining_bag(your_deck_prior, visible[your_index])
        opponent_bag = self._remaining_bag(
            opponent_deck_prior, visible[opponent_index]
        )
        your_state = state["players"][your_index]
        opponent_state = state["players"][opponent_index]

        opponent_active: list[int] = []
        active = opponent_state.get("active") or []
        if active and active[0] is None:
            opponent_active = [self._take_basic(opponent_bag, opponent_deck_prior)]

        your_prize = self._prize_guess(
            your_state.get("prize") or [], your_bag, your_deck_prior
        )
        opponent_prize = self._prize_guess(
            opponent_state.get("prize") or [], opponent_bag, opponent_deck_prior
        )
        opponent_hand = self._take(
            opponent_bag, int(opponent_state["handCount"]), opponent_deck_prior
        )
        selection = observation.get("select") or {}
        your_deck = (
            []
            if selection.get("deck") is not None
            else self._take(your_bag, int(your_state["deckCount"]), your_deck_prior)
        )
        opponent_deck = self._take(
            opponent_bag, int(opponent_state["deckCount"]), opponent_deck_prior
        )
        return HiddenStateGuess(
            your_deck=your_deck,
            your_prize=your_prize,
            opponent_deck=opponent_deck,
            opponent_prize=opponent_prize,
            opponent_hand=opponent_hand,
            opponent_active=opponent_active,
            opponent_deck_name=opponent_hypothesis.name,
        )


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    simulations: int = 16
    determinizations: int = 1
    c_puct: float = 1.25
    max_depth: int = 12
    max_actions: int = 16
    root_contexts: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise ValueError("simulations must be positive")
        if self.determinizations <= 0 or self.determinizations > self.simulations:
            raise ValueError("determinizations must be in [1, simulations]")
        if self.c_puct < 0:
            raise ValueError("c_puct must be non-negative")
        if self.max_depth <= 0 or self.max_actions <= 0:
            raise ValueError("max_depth and max_actions must be positive")


@dataclass(frozen=True, slots=True)
class PlanMCTSConfig:
    """Budget for root-level Monte Carlo comparison of complete turn plans."""

    determinizations: int = 4
    max_macro_steps: int = 32
    root_contexts: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if self.determinizations <= 0:
            raise ValueError("determinizations must be positive")
        if self.max_macro_steps <= 0:
            raise ValueError("max_macro_steps must be positive")


@dataclass(slots=True)
class _Edge:
    action: tuple[int, ...]
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    child: _Node | None = None

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(slots=True)
class _Node:
    position: SearchPosition
    player: int
    visits: int = 0
    value_sum: float = 0.0
    value_estimate: float = 0.0
    expanded: bool = False
    edges: list[_Edge] = field(default_factory=list)


def _candidate_actions(
    selection: dict,
    evaluation: PolicyValueEvaluation,
    policy: BCPolicyAgent,
    observation: dict,
    *,
    maximum: int,
    rng: random.Random,
) -> list[tuple[tuple[int, ...], float]]:
    option_count = len(selection["option"])
    minimum_count = int(selection["minCount"])
    maximum_count = int(selection["maxCount"])
    legal_count = legal_action_set_count(option_count, minimum_count, maximum_count)
    logits = evaluation.logits[:option_count]

    if legal_count == 1:
        action = tuple(deterministic_subset(logits, minimum_count, maximum_count))
        return [(action, 1.0)]

    actions: list[tuple[int, ...]] = []
    if minimum_count == maximum_count == 1:
        ranked = sorted(range(option_count), key=lambda index: float(logits[index]), reverse=True)
        actions = [(index,) for index in ranked[:maximum]]
    elif policy.action_space_version < 2:
        actions = [tuple(policy.choose_action(observation))]
    elif legal_count <= maximum:
        for count in range(minimum_count, maximum_count + 1):
            actions.extend(itertools.combinations(range(option_count), count))
    else:
        actions.append(tuple(deterministic_subset(logits, minimum_count, maximum_count)))
        actions.append(tuple(policy.choose_action(observation)))
        attempts = 0
        while len(set(actions)) < maximum and attempts < maximum * 20:
            actions.append(
                tuple(sample_subset(logits, minimum_count, maximum_count, rng=rng))
            )
            attempts += 1
        actions = list(dict.fromkeys(actions))[:maximum]

    scores = torch.tensor(
        [sum(float(logits[index]) for index in action) for action in actions],
        dtype=torch.float64,
    )
    priors = scores.softmax(dim=0).tolist()
    return list(zip(actions, priors, strict=True))


class PolicyValueMCTSAgent:
    """PUCT search over official cloned states, guided by an existing checkpoint."""

    name = "policy-value-mcts"

    def __init__(
        self,
        policy: BCPolicyAgent,
        determinizer: DeckDeterminizer,
        *,
        config: MCTSConfig | None = None,
        seed: int | None = None,
        backend: SearchBackend | None = None,
    ) -> None:
        self._policy = policy
        self._determinizer = determinizer
        self._config = config or MCTSConfig()
        self._rng = random.Random(seed)
        self._backend = backend or OfficialSearchBackend()
        self.last_search: dict[str, Any] | None = None
        self._search_count = 0
        self._total_elapsed_ms = 0.0
        self._total_nodes = 0
        self._total_simulations = 0
        self._total_determinizations = 0
        self._deepest = 0
        self._opponent_deck_samples: Counter[str] = Counter()
        # Keep enough samples for panel percentiles without growing forever in
        # long-running submission processes.
        self._elapsed_samples_ms: deque[float] = deque(maxlen=50_000)

    def reset_episode(self) -> None:
        """Clear stateful opponent evidence before a new battle."""
        self._determinizer.reset()

    def metrics(self) -> dict[str, Any]:
        searches = max(self._search_count, 1)

        def percentile(fraction: float) -> float:
            if not self._elapsed_samples_ms:
                return 0.0
            ordered = sorted(self._elapsed_samples_ms)
            position = fraction * (len(ordered) - 1)
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            weight = position - lower
            return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

        return {
            "searches": self._search_count,
            "simulations": self._total_simulations,
            "determinizations": self._total_determinizations,
            "nodes": self._total_nodes,
            "mean_nodes_per_search": round(self._total_nodes / searches, 3),
            "mean_elapsed_ms": round(self._total_elapsed_ms / searches, 3),
            "p50_elapsed_ms": round(percentile(0.50), 3),
            "p95_elapsed_ms": round(percentile(0.95), 3),
            "p99_elapsed_ms": round(percentile(0.99), 3),
            "max_elapsed_ms": round(max(self._elapsed_samples_ms, default=0.0), 3),
            "max_depth_reached": self._deepest,
            "opponent_deck_samples": dict(sorted(self._opponent_deck_samples.items())),
        }

    @staticmethod
    def _terminal_value(observation: dict, root_player: int) -> float | None:
        result = int(observation["current"]["result"])
        if result < 0:
            return None
        if result == 2:
            return 0.0
        return 1.0 if result == root_player else -1.0

    def _expand(self, node: _Node, root_player: int) -> float:
        terminal = self._terminal_value(node.position.observation, root_player)
        if terminal is not None:
            node.value_estimate = terminal
            node.expanded = True
            return terminal

        observation = node.position.observation
        evaluation = self._policy.evaluate(observation)
        node.value_estimate = (
            evaluation.value if node.player == root_player else -evaluation.value
        )
        candidates = _candidate_actions(
            observation["select"],
            evaluation,
            self._policy,
            observation,
            maximum=self._config.max_actions,
            rng=self._rng,
        )
        node.edges = [_Edge(action=action, prior=prior) for action, prior in candidates]
        node.expanded = True
        return node.value_estimate

    def _select_edge(self, node: _Node, root_player: int) -> _Edge:
        exploration_scale = self._config.c_puct * math.sqrt(max(node.visits, 1))

        def score(edge: _Edge) -> tuple[float, float, tuple[int, ...]]:
            exploitation = edge.q if node.player == root_player else -edge.q
            exploration = exploration_scale * edge.prior / (1 + edge.visits)
            return exploitation + exploration, edge.prior, tuple(-index for index in edge.action)

        return max(node.edges, key=score)

    def _search(
        self, root: _Node, root_player: int, simulations: int
    ) -> tuple[list[int], dict[str, Any]]:
        root_value = self._expand(root, root_player)
        node_count = 1
        deepest = 0

        for _ in range(simulations):
            node = root
            nodes = [root]
            edges: list[_Edge] = []
            depth = 0

            while node.edges and depth < self._config.max_depth:
                edge = self._select_edge(node, root_player)
                edges.append(edge)
                if edge.child is None:
                    position = self._backend.step(node.position.search_id, list(edge.action))
                    player = int(position.observation["current"]["yourIndex"])
                    edge.child = _Node(position=position, player=player)
                    node_count += 1
                node = edge.child
                nodes.append(node)
                depth += 1
                if not node.expanded:
                    value = self._expand(node, root_player)
                    break
                terminal = self._terminal_value(node.position.observation, root_player)
                if terminal is not None:
                    value = terminal
                    break
            else:
                value = node.value_estimate

            deepest = max(deepest, depth)
            for visited in nodes:
                visited.visits += 1
                visited.value_sum += value
            for edge in edges:
                edge.visits += 1
                edge.value_sum += value

        selected = max(root.edges, key=lambda edge: (edge.visits, edge.q, edge.prior))
        children = [
            {
                "action": list(edge.action),
                "visits": edge.visits,
                "q": round(edge.q, 6),
                "prior": round(edge.prior, 6),
            }
            for edge in sorted(root.edges, key=lambda item: item.visits, reverse=True)
        ]
        return list(selected.action), {
            "simulations": simulations,
            "nodes": node_count,
            "max_depth_reached": deepest,
            "root_value": round(root_value, 6),
            "selected_action": list(selected.action),
            "children": children,
        }

    def choose_action(self, observation: dict) -> list[int]:
        selection = observation.get("select")
        if selection is None:
            raise ValueError("MCTS received the initial deck-selection observation")
        if legal_action_set_count(
            len(selection["option"]),
            int(selection["minCount"]),
            int(selection["maxCount"]),
        ) <= 1 or int(selection["context"]) not in self._config.root_contexts:
            self.last_search = None
            return self._policy.choose_action(observation)

        started = perf_counter()
        root_player = int(observation["current"]["yourIndex"])
        base_budget, extra = divmod(
            self._config.simulations, self._config.determinizations
        )
        budgets = [
            base_budget + int(index < extra)
            for index in range(self._config.determinizations)
        ]
        aggregate: dict[tuple[int, ...], dict[str, float | int]] = {}
        tree_stats = []
        hypotheses: Counter[str] = Counter()

        for budget in budgets:
            hidden = self._determinizer.sample(observation)
            if hidden.opponent_deck_name is not None:
                hypotheses[hidden.opponent_deck_name] += 1
            began = False
            try:
                position = self._backend.begin(observation, hidden)
                began = True
                root = _Node(position=position, player=root_player)
                _, stats = self._search(root, root_player, budget)
                tree_stats.append(stats)
                for child in stats["children"]:
                    action = tuple(child["action"])
                    item = aggregate.setdefault(
                        action,
                        {"visits": 0, "value_sum": 0.0, "prior_sum": 0.0},
                    )
                    visits = int(child["visits"])
                    item["visits"] = int(item["visits"]) + visits
                    item["value_sum"] = float(item["value_sum"]) + float(child["q"]) * visits
                    item["prior_sum"] = float(item["prior_sum"]) + float(child["prior"])
            finally:
                if began:
                    self._backend.end()

        def aggregate_score(item: tuple[tuple[int, ...], dict[str, float | int]]):
            _, values = item
            visits = int(values["visits"])
            q = float(values["value_sum"]) / visits if visits else 0.0
            return visits, q, float(values["prior_sum"])

        selected_action, _ = max(aggregate.items(), key=aggregate_score)
        children = []
        for action, values in sorted(
            aggregate.items(), key=aggregate_score, reverse=True
        ):
            visits = int(values["visits"])
            children.append(
                {
                    "action": list(action),
                    "visits": visits,
                    "q": round(
                        float(values["value_sum"]) / visits if visits else 0.0, 6
                    ),
                    "prior": round(
                        float(values["prior_sum"]) / self._config.determinizations,
                        6,
                    ),
                }
            )
        stats = {
            "simulations": sum(int(item["simulations"]) for item in tree_stats),
            "determinizations": self._config.determinizations,
            "nodes": sum(int(item["nodes"]) for item in tree_stats),
            "max_depth_reached": max(
                int(item["max_depth_reached"]) for item in tree_stats
            ),
            "root_value": round(
                sum(float(item["root_value"]) for item in tree_stats) / len(tree_stats),
                6,
            ),
            "selected_action": list(selected_action),
            "opponent_deck_samples": dict(sorted(hypotheses.items())),
            "children": children,
            "elapsed_ms": round((perf_counter() - started) * 1_000, 3),
        }
        self.last_search = stats
        self._search_count += 1
        self._total_elapsed_ms += float(stats["elapsed_ms"])
        self._elapsed_samples_ms.append(float(stats["elapsed_ms"]))
        self._total_nodes += int(stats["nodes"])
        self._total_simulations += int(stats["simulations"])
        self._total_determinizations += int(stats["determinizations"])
        self._opponent_deck_samples.update(hypotheses)
        self._deepest = max(self._deepest, int(stats["max_depth_reached"]))
        return list(selected_action)


class PlanLevelMCTSAgent:
    """Choose a full-turn executor by evaluating macro rollouts from one root.

    V0 exhaustively compares the robust local PlannerPolicy executor against the
    specialist TacticalPlanner executor for each sampled determinization. The
    selected executor then owns the real turn, so downstream actions match the
    branch that was evaluated even when both branches share the same first action.
    """

    name = "plan-level-mcts"
    LOCAL_MODE = "local_router_turn"
    PLANNER_MODE = "planner_turn"

    def __init__(
        self,
        local_executor: MacroExecutor,
        planner_executor: MacroExecutor,
        value_policy: BCPolicyAgent,
        determinizer: DeckDeterminizer,
        *,
        config: PlanMCTSConfig | None = None,
        backend: SearchBackend | None = None,
    ) -> None:
        self._executors = {
            self.LOCAL_MODE: local_executor,
            self.PLANNER_MODE: planner_executor,
        }
        self._value_policy = value_policy
        self._determinizer = determinizer
        self._config = config or PlanMCTSConfig()
        self._backend = backend or OfficialSearchBackend()
        self._active_mode: str | None = None
        self._active_turn: int | None = None
        self._active_player: int | None = None
        self.last_search: dict[str, Any] | None = None
        self._search_count = 0
        self._total_elapsed_ms = 0.0
        self._total_rollouts = 0
        self._total_macro_steps = 0
        self._max_macro_steps_reached = 0
        self._selection_counts: Counter[str] = Counter()
        self._mode_value_sums: Counter[str] = Counter()
        self._mode_value_counts: Counter[str] = Counter()
        self._boundary_counts: dict[str, Counter[str]] = {
            mode: Counter() for mode in self._executors
        }
        self._selection_margin_sum = 0.0
        self._near_tie_searches = 0
        self._same_first_action_searches = 0
        self._opponent_deck_samples: Counter[str] = Counter()
        self._errors: Counter[str] = Counter()
        self._elapsed_samples_ms: deque[float] = deque(maxlen=50_000)

    @staticmethod
    def _reset_executor(executor: MacroExecutor) -> None:
        reset = getattr(executor, "reset_episode", None)
        if reset is not None:
            reset()

    @staticmethod
    def _choose_executor_action(
        executor: MacroExecutor, observation: dict
    ) -> list[int]:
        deterministic = getattr(executor, "choose_deterministic_action", None)
        if deterministic is not None:
            return deterministic(observation)
        return executor.choose_action(observation)

    @staticmethod
    def _terminal_value(observation: dict, root_player: int) -> float | None:
        return PolicyValueMCTSAgent._terminal_value(observation, root_player)

    def _endpoint_value(self, observation: dict, root_player: int) -> float:
        terminal = self._terminal_value(observation, root_player)
        if terminal is not None:
            return terminal
        evaluation = self._value_policy.evaluate(observation)
        player = int(observation["current"]["yourIndex"])
        return evaluation.value if player == root_player else -evaluation.value

    def _rollout(
        self,
        root: SearchPosition,
        *,
        mode: str,
        root_player: int,
        root_turn: int,
    ) -> dict[str, Any]:
        executor = self._executors[mode]
        self._reset_executor(executor)
        position = root
        first_action: list[int] | None = None
        steps = 0
        boundary = "max_steps"

        while steps < self._config.max_macro_steps:
            observation = position.observation
            terminal = self._terminal_value(observation, root_player)
            if terminal is not None:
                boundary = "terminal"
                break
            state = observation["current"]
            if (
                int(state["yourIndex"]) != root_player
                or int(state["turn"]) != root_turn
            ):
                boundary = "turn_changed"
                break
            action = self._choose_executor_action(executor, observation)
            if first_action is None:
                first_action = list(action)
            position = self._backend.step(position.search_id, action)
            steps += 1

        return {
            "value": self._endpoint_value(position.observation, root_player),
            "steps": steps,
            "boundary": boundary,
            "first_action": first_action,
        }

    def reset_episode(self) -> None:
        self._active_mode = None
        self._active_turn = None
        self._active_player = None
        self._determinizer.reset()
        for executor in self._executors.values():
            self._reset_executor(executor)

    def metrics(self) -> dict[str, Any]:
        searches = max(self._search_count, 1)

        def percentile(fraction: float) -> float:
            if not self._elapsed_samples_ms:
                return 0.0
            ordered = sorted(self._elapsed_samples_ms)
            position = fraction * (len(ordered) - 1)
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            weight = position - lower
            return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

        return {
            "searches": self._search_count,
            "macro_rollouts": self._total_rollouts,
            "mean_rollouts_per_search": round(self._total_rollouts / searches, 3),
            "macro_steps": self._total_macro_steps,
            "mean_macro_steps_per_rollout": round(
                self._total_macro_steps / max(self._total_rollouts, 1), 3
            ),
            "max_macro_steps_reached": self._max_macro_steps_reached,
            "selection_counts": dict(sorted(self._selection_counts.items())),
            "mean_value_by_mode": {
                mode: round(
                    self._mode_value_sums[mode]
                    / max(self._mode_value_counts[mode], 1),
                    6,
                )
                for mode in self._executors
            },
            "mean_selection_margin": round(
                self._selection_margin_sum / searches, 6
            ),
            "near_tie_searches": self._near_tie_searches,
            "near_tie_rate": round(self._near_tie_searches / searches, 6),
            "same_first_action_searches": self._same_first_action_searches,
            "same_first_action_rate": round(
                self._same_first_action_searches / searches, 6
            ),
            "boundaries": {
                mode: dict(sorted(counts.items()))
                for mode, counts in self._boundary_counts.items()
            },
            "mean_elapsed_ms": round(self._total_elapsed_ms / searches, 3),
            "p50_elapsed_ms": round(percentile(0.50), 3),
            "p95_elapsed_ms": round(percentile(0.95), 3),
            "p99_elapsed_ms": round(percentile(0.99), 3),
            "max_elapsed_ms": round(max(self._elapsed_samples_ms, default=0.0), 3),
            "opponent_deck_samples": dict(
                sorted(self._opponent_deck_samples.items())
            ),
            "errors": dict(sorted(self._errors.items())),
        }

    def _search_modes(self, observation: dict) -> str:
        started = perf_counter()
        root_player = int(observation["current"]["yourIndex"])
        root_turn = int(observation["current"]["turn"])
        mode_values: dict[str, list[float]] = {
            mode: [] for mode in self._executors
        }
        mode_steps: dict[str, list[int]] = {mode: [] for mode in self._executors}
        boundaries: dict[str, Counter[str]] = {
            mode: Counter() for mode in self._executors
        }
        first_actions: dict[str, Counter[tuple[int, ...]]] = {
            mode: Counter() for mode in self._executors
        }
        hypotheses: Counter[str] = Counter()

        for _ in range(self._config.determinizations):
            hidden = self._determinizer.sample(observation)
            if hidden.opponent_deck_name is not None:
                hypotheses[hidden.opponent_deck_name] += 1
            began = False
            try:
                root = self._backend.begin(observation, hidden)
                began = True
                for mode in self._executors:
                    try:
                        result = self._rollout(
                            root,
                            mode=mode,
                            root_player=root_player,
                            root_turn=root_turn,
                        )
                    except Exception as error:
                        error_name = type(error).__name__
                        self._errors[f"{mode}:{error_name}"] += 1
                        mode_values[mode].append(-1.0)
                        mode_steps[mode].append(0)
                        boundaries[mode][f"error:{error_name}"] += 1
                        continue
                    mode_values[mode].append(float(result["value"]))
                    steps = int(result["steps"])
                    mode_steps[mode].append(steps)
                    boundaries[mode][str(result["boundary"])] += 1
                    first_action = result["first_action"]
                    if first_action is not None:
                        first_actions[mode][tuple(first_action)] += 1
            finally:
                if began:
                    self._backend.end()

        mean_values = {
            mode: sum(values) / len(values) if values else -1.0
            for mode, values in mode_values.items()
        }
        # Prefer the robust local router when the estimated values tie.
        selected_mode = max(
            self._executors,
            key=lambda mode: (
                mean_values[mode],
                mode == self.LOCAL_MODE,
            ),
        )
        elapsed_ms = (perf_counter() - started) * 1_000
        self.last_search = {
            "selected_mode": selected_mode,
            "mode_values": {
                mode: {
                    "mean": round(mean_values[mode], 6),
                    "samples": [round(value, 6) for value in mode_values[mode]],
                    "mean_steps": round(
                        sum(mode_steps[mode]) / max(len(mode_steps[mode]), 1), 3
                    ),
                    "boundaries": dict(sorted(boundaries[mode].items())),
                    "first_actions": [
                        {"action": list(action), "count": count}
                        for action, count in first_actions[mode].most_common()
                    ],
                }
                for mode in self._executors
            },
            "determinizations": self._config.determinizations,
            "opponent_deck_samples": dict(sorted(hypotheses.items())),
            "elapsed_ms": round(elapsed_ms, 3),
        }
        rollouts = self._config.determinizations * len(self._executors)
        macro_steps = sum(sum(values) for values in mode_steps.values())
        self._search_count += 1
        self._total_elapsed_ms += elapsed_ms
        self._elapsed_samples_ms.append(elapsed_ms)
        self._total_rollouts += rollouts
        self._total_macro_steps += macro_steps
        self._max_macro_steps_reached = max(
            self._max_macro_steps_reached,
            max((max(values, default=0) for values in mode_steps.values()), default=0),
        )
        self._selection_counts[selected_mode] += 1
        margin = abs(
            mean_values[self.PLANNER_MODE] - mean_values[self.LOCAL_MODE]
        )
        self._selection_margin_sum += margin
        self._near_tie_searches += int(margin < 0.05)
        for mode, values in mode_values.items():
            self._mode_value_sums[mode] += sum(values)
            self._mode_value_counts[mode] += len(values)
            self._boundary_counts[mode].update(boundaries[mode])
        leading_actions = {
            mode: counts.most_common(1)[0][0] if counts else None
            for mode, counts in first_actions.items()
        }
        self._same_first_action_searches += int(
            leading_actions[self.LOCAL_MODE]
            == leading_actions[self.PLANNER_MODE]
        )
        self._opponent_deck_samples.update(hypotheses)
        return selected_mode

    def choose_action(self, observation: dict) -> list[int]:
        selection = observation.get("select")
        if selection is None:
            raise ValueError("Plan-level MCTS received the initial deck selection")
        state = observation["current"]
        turn = int(state["turn"])
        player = int(state["yourIndex"])

        if self._active_mode is not None and (
            turn != self._active_turn or player != self._active_player
        ):
            self._active_mode = None
            self._active_turn = None
            self._active_player = None

        if self._active_mode is not None:
            return self._choose_executor_action(
                self._executors[self._active_mode], observation
            )

        legal_count = legal_action_set_count(
            len(selection["option"]),
            int(selection["minCount"]),
            int(selection["maxCount"]),
        )
        if (
            legal_count <= 1
            or int(selection["context"]) not in self._config.root_contexts
        ):
            self.last_search = None
            return self._choose_executor_action(
                self._executors[self.LOCAL_MODE], observation
            )

        selected_mode = self._search_modes(observation)
        for executor in self._executors.values():
            self._reset_executor(executor)
        self._active_mode = selected_mode
        self._active_turn = turn
        self._active_player = player
        action = self._choose_executor_action(
            self._executors[selected_mode], observation
        )
        if self.last_search is not None:
            self.last_search["selected_action"] = list(action)
        return action
