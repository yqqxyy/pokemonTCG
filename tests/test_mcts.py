from __future__ import annotations

import pytest
import torch

from poketcg.agents.bc_agent import PolicyValueEvaluation
from poketcg.engine import OfficialEngine
from poketcg.mcts import (
    DeckDeterminizer,
    DeckHypothesis,
    HiddenStateGuess,
    MCTSConfig,
    OfficialSearchBackend,
    OpponentDeckBelief,
    PolicyValueMCTSAgent,
    SearchPosition,
)
from poketcg.rl.collect_expert_iteration import root_visit_policy_target


def _observation(*, player: int = 0, result: int = -1) -> dict:
    return {
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 14}, {"type": 14}],
        },
        "logs": [],
        "current": {"yourIndex": player, "result": result},
        "search_begin_input": "fake",
    }


def test_root_visit_policy_target_normalizes_visits_and_unexpanded_options() -> None:
    target = root_visit_policy_target(
        {
            "selected_action": [0],
            "children": [
                {"action": [0], "visits": 12},
                {"action": [1], "visits": 4},
            ],
        },
        3,
    )

    assert target == pytest.approx([0.75, 0.25, 0.0])


def test_root_visit_policy_target_falls_back_to_selected_action() -> None:
    target = root_visit_policy_target(
        {"selected_action": [2], "children": [{"action": [0], "visits": 0}]},
        3,
    )

    assert target == [0.0, 0.0, 1.0]


class _FakePolicy:
    action_space_version = 1

    def evaluate(self, observation: dict) -> PolicyValueEvaluation:
        del observation
        return PolicyValueEvaluation(
            logits=torch.tensor([0.0, 0.0]),
            value=0.0,
            minimum=1,
            maximum=1,
        )

    def choose_action(self, observation: dict) -> list[int]:
        del observation
        return [0]


class _StubDeterminizer:
    def sample(self, observation: dict) -> HiddenStateGuess:
        del observation
        return HiddenStateGuess([], [], [], [], [], [])


class _AdversarialBackend:
    """Action 0 lets player 1 choose win/loss; action 1 is a draw."""

    def __init__(self) -> None:
        self.ended = False
        self.end_count = 0

    def begin(self, observation: dict, hidden: HiddenStateGuess) -> SearchPosition:
        del hidden
        return SearchPosition(0, observation)

    def step(self, search_id: int, action: list[int]) -> SearchPosition:
        if search_id == 0 and action == [0]:
            return SearchPosition(1, _observation(player=1))
        if search_id == 0 and action == [1]:
            return SearchPosition(2, _observation(player=0, result=2))
        if search_id == 1 and action == [0]:
            return SearchPosition(3, _observation(player=0, result=0))
        if search_id == 1 and action == [1]:
            return SearchPosition(4, _observation(player=0, result=1))
        raise AssertionError((search_id, action))

    def end(self) -> None:
        self.ended = True
        self.end_count += 1


def test_mcts_models_opponent_as_minimizing_root_value() -> None:
    backend = _AdversarialBackend()
    agent = PolicyValueMCTSAgent(
        _FakePolicy(),  # type: ignore[arg-type]
        _StubDeterminizer(),  # type: ignore[arg-type]
        config=MCTSConfig(simulations=64, max_depth=4),
        seed=7,
        backend=backend,
    )

    action = agent.choose_action(_observation())

    assert action == [1]
    assert backend.ended
    assert agent.last_search is not None
    assert agent.last_search["nodes"] == 5


def test_mcts_aggregates_independent_determinization_trees() -> None:
    backend = _AdversarialBackend()
    agent = PolicyValueMCTSAgent(
        _FakePolicy(),  # type: ignore[arg-type]
        _StubDeterminizer(),  # type: ignore[arg-type]
        config=MCTSConfig(simulations=64, determinizations=4, max_depth=4),
        seed=7,
        backend=backend,
    )

    action = agent.choose_action(_observation())

    assert action == [1]
    assert backend.end_count == 4
    assert agent.last_search is not None
    assert agent.last_search["determinizations"] == 4
    assert agent.last_search["simulations"] == 64
    assert agent.metrics()["determinizations"] == 4
    assert agent.metrics()["p95_elapsed_ms"] >= 0.0
    assert agent.metrics()["p99_elapsed_ms"] >= agent.metrics()["p95_elapsed_ms"]


def test_deck_determinizer_fills_every_hidden_zone() -> None:
    deck0 = [10] * 60
    deck1 = [20] * 60
    observation = {
        "select": {"deck": None},
        "current": {
            "yourIndex": 0,
            "stadium": [],
            "looking": None,
            "players": [
                {
                    "active": [],
                    "bench": [],
                    "deckCount": 59,
                    "discard": [],
                    "prize": [],
                    "handCount": 1,
                    "hand": [{"id": 10, "serial": 1}],
                },
                {
                    "active": [None],
                    "bench": [],
                    "deckCount": 52,
                    "discard": [],
                    "prize": [None] * 6,
                    "handCount": 1,
                    "hand": None,
                },
            ],
        },
    }
    determinizer = DeckDeterminizer(deck0, deck1, basic_card_ids={20}, seed=3)

    hidden = determinizer.sample(observation)

    assert hidden.your_deck == [10] * 59
    assert hidden.your_prize == []
    assert hidden.opponent_active == [20]
    assert hidden.opponent_hand == [20]
    assert hidden.opponent_prize == [20] * 6
    assert hidden.opponent_deck == [20] * 52


def test_opponent_deck_belief_eliminates_incompatible_hypothesis() -> None:
    water = DeckHypothesis("water", tuple([10] * 4 + [3] * 56))
    fire = DeckHypothesis("fire", tuple([20] * 4 + [3] * 56))
    belief = OpponentDeckBelief([water, fire])
    observation = {
        "select": {"deck": None},
        "current": {
            "yourIndex": 0,
            "stadium": [],
            "looking": None,
            "players": [
                {
                    "active": [],
                    "bench": [],
                    "hand": [],
                    "discard": [],
                    "prize": [],
                },
                {
                    "active": [],
                    "bench": [{"id": 10, "serial": 1}],
                    "hand": None,
                    "discard": [],
                    "prize": [],
                },
            ],
        },
    }

    posterior = belief.posterior(observation)

    assert posterior == {"water": 1.0, "fire": 0.0}


def test_opponent_deck_belief_retains_log_evidence_until_reset() -> None:
    matching = DeckHypothesis("matching", tuple([20] + [1] * 59))
    missing = DeckHypothesis("missing", tuple([2] * 60))
    belief = OpponentDeckBelief([matching, missing])
    observation = {
        "logs": [
            {"type": 6, "playerIndex": 1, "cardId": 20, "serial": 99}
        ],
        "select": {"deck": None},
        "current": {
            "yourIndex": 0,
            "stadium": [],
            "looking": None,
            "players": [
                {"active": [], "bench": [], "hand": [], "discard": [], "prize": []},
                {"active": [], "bench": [], "hand": None, "discard": [], "prize": []},
            ],
        },
    }

    assert belief.posterior(observation) == {"matching": 1.0, "missing": 0.0}
    observation["logs"] = []
    assert belief.posterior(observation) == {"matching": 1.0, "missing": 0.0}

    belief.reset()
    assert belief.posterior(observation) == pytest.approx(
        {"matching": 0.5, "missing": 0.5}
    )


@pytest.mark.integration
def test_official_search_backend_branches_without_mutating_root() -> None:
    engine = OfficialEngine()
    backend = OfficialSearchBackend()
    deck = engine.load_deck(engine.sample_deck_path)
    basics = {card_id for card_id, card in engine.card_catalog().items() if card.basic}
    determinizer = DeckDeterminizer(deck, deck, basic_card_ids=basics, seed=11)
    observation, _ = engine.start(deck, deck)
    began = False
    try:
        root = backend.begin(observation, determinizer.sample(observation))
        began = True
        children = [backend.step(root.search_id, [index]) for index in (0, 1)]
        assert root.search_id == 0
        assert {child.search_id for child in children} == {1, 2}
        assert root.observation["current"]["turn"] == observation["current"]["turn"]
    finally:
        if began:
            backend.end()
        engine.finish()
