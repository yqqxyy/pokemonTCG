"""A deterministic, metadata-aware rule baseline."""

from __future__ import annotations

import random
from collections.abc import Mapping
from enum import IntEnum
from typing import Any


class AreaType(IntEnum):
    """Subset of official area IDs used by RuleAgent."""

    DECK = 1
    HAND = 2
    DISCARD = 3
    ACTIVE = 4
    BENCH = 5
    PRIZE = 6


class OptionType(IntEnum):
    """Official option IDs; new competition values may be appended."""

    NUMBER = 0
    YES = 1
    NO = 2
    CARD = 3
    TOOL_CARD = 4
    ENERGY_CARD = 5
    ENERGY = 6
    PLAY = 7
    ATTACH = 8
    EVOLVE = 9
    ABILITY = 10
    DISCARD = 11
    RETREAT = 12
    ATTACK = 13
    END = 14
    SKILL = 15
    SPECIAL_CONDITION = 16


class SelectContext(IntEnum):
    """Subset of official selection contexts used by RuleAgent."""

    MAIN = 0
    SETUP_ACTIVE_POKEMON = 1
    SETUP_BENCH_POKEMON = 2
    SWITCH = 3
    TO_ACTIVE = 4
    TO_BENCH = 5
    DISCARD = 8
    DAMAGE_COUNTER = 13
    DAMAGE_COUNTER_ANY = 14
    DAMAGE = 15
    REMOVE_DAMAGE_COUNTER = 16
    HEAL = 17
    DISCARD_ENERGY_CARD = 26
    DISCARD_TOOL_CARD = 27
    DISCARD_CARD_OR_ATTACHED_CARD = 29
    DISCARD_ENERGY = 30
    ATTACK = 35
    DRAW_COUNT = 38
    IS_FIRST = 41
    MULLIGAN = 42
    ACTIVATE = 43


_MAIN_ACTION_PRIORITY = {
    OptionType.ABILITY: 1_000.0,
    OptionType.EVOLVE: 900.0,
    OptionType.PLAY: 800.0,
    OptionType.ATTACH: 700.0,
    OptionType.ATTACK: 600.0,
    OptionType.DISCARD: 200.0,
    OptionType.RETREAT: 100.0,
    OptionType.END: 0.0,
}

_RESOURCE_LOSS_CONTEXTS = {
    SelectContext.DISCARD,
    SelectContext.DISCARD_ENERGY_CARD,
    SelectContext.DISCARD_TOOL_CARD,
    SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
    SelectContext.DISCARD_ENERGY,
}


def _value(record: Any, field: str, default: Any = None) -> Any:
    if record is None:
        return default
    if isinstance(record, Mapping):
        return record.get(field, default)
    return getattr(record, field, default)


class RuleAgent:
    """RuleAgent V1: prioritize development, then choose simple tactical value."""

    name = "rule-v1"

    def __init__(
        self,
        *,
        card_catalog: Mapping[int, Any] | None = None,
        attack_catalog: Mapping[int, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self._cards = card_catalog or {}
        self._attacks = attack_catalog or {}
        self._rng = random.Random(seed)

    def choose_action(self, observation: dict) -> list[int]:
        selection = observation.get("select")
        if selection is None:
            raise ValueError("RuleAgent received the initial deck-selection observation.")

        minimum = int(selection["minCount"])
        maximum = int(selection["maxCount"])
        options = selection["option"]
        if not 0 <= minimum <= maximum <= len(options):
            raise ValueError("Simulator returned inconsistent selection bounds.")

        count = self._selection_count(selection)
        scores = self.score_options(observation)
        scored = [(score, self._rng.random(), index) for index, score in enumerate(scores)]
        scored.sort(reverse=True)
        return [index for _, _, index in scored[:count]]

    def score_options(self, observation: dict) -> list[float]:
        """Return deterministic heuristic scores before random tie-breaking."""
        selection = observation.get("select")
        if selection is None:
            raise ValueError("RuleAgent received the initial deck-selection observation.")
        return [
            self._score_option(observation, index, option)
            for index, option in enumerate(selection["option"])
        ]

    @staticmethod
    def _selection_count(selection: dict) -> int:
        context = int(selection["context"])
        minimum = int(selection["minCount"])
        maximum = int(selection["maxCount"])
        if context in {int(item) for item in _RESOURCE_LOSS_CONTEXTS}:
            return minimum
        return maximum

    def _score_option(self, observation: dict, index: int, option: dict) -> float:
        selection = observation["select"]
        context = int(selection["context"])
        option_type = int(option["type"])

        if context == SelectContext.MAIN:
            return self._main_action_score(observation, option)
        if context == SelectContext.ATTACK or option_type == OptionType.ATTACK:
            return self._attack_score(option)
        if option_type == OptionType.YES:
            return 100.0
        if option_type == OptionType.NO:
            return 0.0
        if option_type == OptionType.NUMBER:
            return float(option.get("number") or 0)

        card_score = self._card_option_score(observation, selection, option)
        target_score = self._target_score(observation, option)

        if context in {SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.TO_ACTIVE}:
            return card_score + target_score
        if context == SelectContext.SWITCH:
            return target_score
        if context in {
            SelectContext.DAMAGE_COUNTER,
            SelectContext.DAMAGE_COUNTER_ANY,
            SelectContext.DAMAGE,
        }:
            return self._damage_target_score(observation, option)
        if context in {SelectContext.REMOVE_DAMAGE_COUNTER, SelectContext.HEAL}:
            return self._heal_target_score(observation, option)
        if context in {int(item) for item in _RESOURCE_LOSS_CONTEXTS}:
            return -card_score
        return card_score + target_score

    def _main_action_score(self, observation: dict, option: dict) -> float:
        option_type = int(option["type"])
        base = _MAIN_ACTION_PRIORITY.get(option_type, 300.0)
        if option_type == OptionType.ATTACK:
            return base + self._attack_score(option)
        if option_type in {OptionType.PLAY, OptionType.EVOLVE}:
            return base + self._card_option_score(observation, observation["select"], option)
        if option_type == OptionType.ATTACH:
            return base + self._target_score(observation, option)
        return base

    def _attack_score(self, option: dict) -> float:
        attack = self._attacks.get(option.get("attackId"))
        damage = float(_value(attack, "damage", 0) or 0)
        energy_cost = len(_value(attack, "energies", []) or [])
        return damage - energy_cost * 0.01

    def _card_option_score(self, observation: dict, selection: dict, option: dict) -> float:
        card_id = self._option_card_id(observation, selection, option)
        card = self._cards.get(card_id)
        if card is None:
            return 0.0

        hp = float(_value(card, "hp", 0) or 0)
        skills = len(_value(card, "skills", []) or [])
        attacks = len(_value(card, "attacks", []) or [])
        card_type = int(_value(card, "cardType", -1))
        type_bonus = {0: 20.0, 1: 15.0, 2: 8.0, 3: 18.0, 4: 5.0}.get(card_type, 0.0)
        return hp + skills * 15.0 + attacks * 5.0 + type_bonus

    def _option_card_id(self, observation: dict, selection: dict, option: dict) -> int | None:
        if option.get("cardId") is not None:
            return int(option["cardId"])

        index = option.get("index")
        if index is None:
            return None
        index = int(index)
        area = option.get("area")
        player_index = option.get("playerIndex")
        state = observation.get("current")

        if area == AreaType.DECK and selection.get("deck"):
            return self._record_id(selection["deck"], index)
        if state is None or player_index is None:
            return None

        player = state["players"][int(player_index)]
        records = {
            AreaType.HAND: player.get("hand"),
            AreaType.DISCARD: player.get("discard"),
            AreaType.ACTIVE: player.get("active"),
            AreaType.BENCH: player.get("bench"),
            AreaType.PRIZE: player.get("prize"),
        }.get(area)
        return self._record_id(records, index)

    @staticmethod
    def _record_id(records: list | None, index: int) -> int | None:
        if records is None or not 0 <= index < len(records) or records[index] is None:
            return None
        value = _value(records[index], "id")
        return int(value) if value is not None else None

    @staticmethod
    def _target_pokemon(observation: dict, option: dict) -> dict | None:
        state = observation.get("current")
        player_index = option.get("playerIndex")
        index = option.get("inPlayIndex", option.get("index"))
        area = option.get("inPlayArea", option.get("area"))
        if state is None or player_index is None or index is None:
            return None
        player = state["players"][int(player_index)]
        records = {
            AreaType.ACTIVE: player.get("active"),
            AreaType.BENCH: player.get("bench"),
        }.get(area)
        if records is None or not 0 <= int(index) < len(records):
            return None
        return records[int(index)]

    def _target_score(self, observation: dict, option: dict) -> float:
        pokemon = self._target_pokemon(observation, option)
        if pokemon is None:
            return 0.0
        hp = float(pokemon.get("hp", 0))
        energies = len(pokemon.get("energies", []))
        target_area = option.get("inPlayArea", option.get("area"))
        active_bonus = 50.0 if target_area == AreaType.ACTIVE else 0.0
        return hp + energies * 20.0 + active_bonus

    def _damage_target_score(self, observation: dict, option: dict) -> float:
        pokemon = self._target_pokemon(observation, option)
        if pokemon is None:
            return 0.0
        your_index = int(observation["current"]["yourIndex"])
        opponent_bonus = 1_000.0 if int(option.get("playerIndex", -1)) != your_index else 0.0
        return opponent_bonus - float(pokemon.get("hp", 0))

    def _heal_target_score(self, observation: dict, option: dict) -> float:
        pokemon = self._target_pokemon(observation, option)
        if pokemon is None:
            return 0.0
        damage = float(pokemon.get("maxHp", 0)) - float(pokemon.get("hp", 0))
        your_index = int(observation["current"]["yourIndex"])
        own_bonus = 1_000.0 if int(option.get("playerIndex", -1)) == your_index else 0.0
        return own_bonus + damage
