from __future__ import annotations

from dataclasses import dataclass

from poketcg.mcts import HiddenStateGuess, SearchPosition
from poketcg.rl.heldout_option import HeldoutCardEffectEvaluator
from poketcg.rl.paired_rollout import RootCandidate


def _root_observation() -> dict:
    return {
        "state": "root",
        "current": {
            "result": -1,
            "yourIndex": 0,
            "turn": 3,
            "players": [],
        },
        "select": {
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": 14, "cardId": 10},
                {"type": 7, "cardId": 20},
            ],
        },
    }


def _option_observation(*, heldout: bool, reversed_options: bool) -> dict:
    options = [
        {"type": 3, "cardId": 100},
        {"type": 3, "cardId": 200},
    ]
    if reversed_options:
        options.reverse()
    return {
        "state": "option",
        "current": {
            "result": -1,
            "yourIndex": 0,
            "turn": 3,
            "supporter": int(heldout),
            "players": [],
        },
        "select": {
            "context": 7,
            "minCount": 1,
            "maxCount": 1,
            "option": options,
        },
    }


def _terminal(winner: int) -> dict:
    return {
        "state": "terminal",
        "current": {
            "result": winner,
            "yourIndex": 0,
            "turn": 3,
            "players": [],
        },
        "select": {
            "context": 0,
            "minCount": 0,
            "maxCount": 0,
            "option": [],
        },
    }


class OptionBackend:
    def __init__(self) -> None:
        self._next_id = 1
        self._states: dict[int, tuple[dict, int, bool]] = {}

    def _position(
        self, observation: dict, *, heldout: int, reversed_options: bool
    ) -> SearchPosition:
        search_id = self._next_id
        self._next_id += 1
        self._states[search_id] = (observation, heldout, reversed_options)
        return SearchPosition(search_id, observation)

    def begin(self, observation: dict, hidden: HiddenStateGuess) -> SearchPosition:
        name = hidden.opponent_deck_name or ""
        phase = (
            1
            if name.startswith("calibration")
            else 2
            if name.startswith("heldout")
            else 0
        )
        return self._position(
            _root_observation(),
            heldout=phase,
            reversed_options=name.endswith("reversed"),
        )

    def step(self, search_id: int, action: list[int]) -> SearchPosition:
        observation, heldout, reversed_options = self._states[search_id]
        if observation["state"] == "root":
            if action == [0]:
                return self._position(
                    _terminal(1),
                    heldout=heldout,
                    reversed_options=reversed_options,
                )
            return self._position(
                _option_observation(
                    heldout=bool(heldout),
                    reversed_options=reversed_options,
                ),
                heldout=heldout,
                reversed_options=reversed_options,
            )
        options = observation["select"]["option"]
        selected_card = int(options[action[0]]["cardId"])
        return self._position(
            _terminal(0 if selected_card == 200 else 1),
            heldout=heldout,
            reversed_options=reversed_options,
        )

    def end(self) -> None:
        return None


class RejectingCalibrationBackend(OptionBackend):
    def step(self, search_id: int, action: list[int]) -> SearchPosition:
        observation, phase, reversed_options = self._states[search_id]
        if observation["state"] != "option":
            return super().step(search_id, action)
        options = observation["select"]["option"]
        selected_card = int(options[action[0]]["cardId"])
        candidate_wins = selected_card == 200 and phase != 1
        return self._position(
            _terminal(0 if candidate_wins else 1),
            heldout=phase,
            reversed_options=reversed_options,
        )


class SequenceDeterminizer:
    def __init__(self) -> None:
        self._names = iter(
            (
                "proposal",
                "proposal_reversed",
                "calibration",
                "calibration_reversed",
                "heldout",
                "heldout_reversed",
            )
        )

    def sample(self, observation: dict) -> HiddenStateGuess:
        return HiddenStateGuess([], [], [], [], [], [], next(self._names))


@dataclass
class BaselinePolicy:
    def choose_action(self, observation: dict) -> list[int]:
        if int(observation["select"]["context"]) == 0:
            return [0]
        for index, option in enumerate(observation["select"]["option"]):
            if int(option["cardId"]) == 100:
                return [index]
        raise AssertionError("Synthetic baseline target is missing")


def _candidates(observation: dict) -> list[RootCandidate]:
    return [
        RootCandidate((0,), ("v1_choice",)),
        RootCandidate((1,), ("card_effect",)),
    ]


def test_closed_loop_option_generalizes_by_visible_semantics() -> None:
    evaluator = HeldoutCardEffectEvaluator(
        SequenceDeterminizer(),
        BaselinePolicy,
        BaselinePolicy,
        _candidates,
        build_determinizations=2,
        calibration_determinizations=2,
        heldout_determinizations=2,
        root_candidate_limit=2,
        beam_width=4,
        branch_width=2,
        max_option_steps=3,
        minimum_calibration_pairs=2,
        backend=OptionBackend(),
    )

    result = evaluator.evaluate(_root_observation())

    assert result["calibration_gate_passed"] is True
    assert result["calibration_selected"]["paired_advantage"] == 2.0
    assert result["heldout_selected"]["paired_advantage"] == 2.0
    assert result["deployable_heldout_gain"] == 2.0
    assert result["heldout_selected"]["mean_closed_loop_coverage"] == 1.0
    assert result["heldout_accepted"] is True
    assert result["world_id_ranges"] == {
        "build": [0, 2],
        "calibration": [2, 4],
        "heldout": [4, 6],
    }
    assert [item["world_id"] for item in result["build_search"]] == [0, 1]
    assert [item["world_id"] for item in result["calibration_samples"]] == [
        2,
        3,
    ]
    assert [item["world_id"] for item in result["heldout_samples"]] == [4, 5]
    routes = [
        branch["routes"]
        for sample in result["heldout_samples"]
        for branch in sample["branches"]
    ]
    assert routes == [{"selection": 1}, {"selection": 1}]


def test_heldout_cannot_override_a_rejected_calibration_gate() -> None:
    evaluator = HeldoutCardEffectEvaluator(
        SequenceDeterminizer(),
        BaselinePolicy,
        BaselinePolicy,
        _candidates,
        build_determinizations=2,
        calibration_determinizations=2,
        heldout_determinizations=2,
        root_candidate_limit=2,
        beam_width=4,
        branch_width=2,
        max_option_steps=3,
        minimum_calibration_pairs=2,
        backend=RejectingCalibrationBackend(),
    )

    result = evaluator.evaluate(_root_observation())

    assert result["calibration_selected"]["paired_advantage"] == 0.0
    assert result["calibration_gate_passed"] is False
    assert result["heldout_selected"]["paired_advantage"] == 2.0
    assert result["deployable_heldout_gain"] == 0.0
    assert result["heldout_accepted"] is False
