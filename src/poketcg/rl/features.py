"""Structured observation and candidate-action features."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

STATE_FEATURE_SIZE = 30
OPTION_FEATURE_SIZE = 20
TOKEN_FEATURE_SIZE = 24
SEMANTIC_FEATURE_SIZE = 40
PACKED_SEMANTIC_FEATURE_SIZE = 9
HISTORY_FEATURE_SIZE = 5
MAX_STATE_TOKENS = 192
MAX_HISTORY_TOKENS = 32

SEMANTIC_TAGS = (
    "draw",
    "search_deck",
    "shuffle",
    "discard",
    "mill_deck",
    "recover_discard",
    "attach_energy",
    "move_energy",
    "energy_from_deck_or_discard",
    "heal",
    "damage_scaling",
    "damage_reduction_or_prevention",
    "bench_damage",
    "self_damage",
    "switch",
    "retreat",
    "special_condition",
    "coin_flip",
    "knock_out",
    "prize",
    "hand_disruption",
    "deck_manipulation",
    "evolve",
    "put_into_hand",
    "conditional",
    "once_per_turn",
    "active_target",
    "bench_target",
    "opponent_target",
    "own_target",
    "reveal",
    "look_at",
)

_SEMANTIC_PATTERNS = (
    r"\bdraw\b",
    r"search your deck|search (?:his or her|their) deck",
    r"\bshuffle\b",
    r"\bdiscard\b",
    r"discard the top .+ cards? of .+ deck|discard .+ from the top of .+ deck",
    r"from (?:your|their) discard pile|discard pile.+(?:hand|deck)",
    r"attach .+ energy",
    r"move .+ energy",
    r"attach .+ energy .+(?:deck|discard pile)|energy .+ from .+(?:deck|discard pile)",
    r"\bheal\b|remove .+ damage counter",
    r"damage for each|more damage|less damage|damage instead|additional damage",
    r"takes? .+ less damage|prevent all damage|isn.t affected by|no damage",
    r"benched pok[eé]mon|to the bench",
    r"damage to this pok[eé]mon|does .+ damage to itself",
    r"\bswitch\b",
    r"\bretreat\b",
    r"poison|burn|asleep|paraly[sz]|confus",
    r"flip .+ coin|coin flip",
    r"knock out",
    r"prize card",
    r"shuffle your hand|discard .+ hand|opponent.+hand",
    r"top .+ cards? of .+ deck|put .+ (?:top|bottom) of .+ deck",
    r"\bevolve\b|evolution pok[eé]mon",
    r"return .+ to .+ hand|put .+ into .+ hand",
    r"\bif\b|for each|during .+ turn|when(?:ever)?|unless|up to",
    r"once during your turn|once per turn",
    r"active pok[eé]mon",
    r"benched pok[eé]mon|\bbench\b",
    r"opponent(?:.s|s')|your opponent",
    r"your pok[eé]mon|this pok[eé]mon|yourself",
    r"\breveal\b",
    r"look at",
)

if len(SEMANTIC_TAGS) + 8 != SEMANTIC_FEATURE_SIZE:
    raise AssertionError("Semantic feature schema size is inconsistent")


def _value(record: Any, field: str, default: Any = None) -> Any:
    if record is None:
        return default
    if isinstance(record, Mapping):
        return record.get(field, default)
    return getattr(record, field, default)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


@dataclass(slots=True)
class EncodedDecision:
    """Fixed state features plus a variable-length set of legal options."""

    state: list[float]
    select_type: int
    context: int
    options: list[list[float]]
    option_types: list[int]
    areas: list[int]
    in_play_areas: list[int]
    version: int = 1
    tokens: list[list[float]] | None = None
    token_card_ids: list[int] | None = None
    token_kinds: list[int] | None = None
    token_zones: list[int] | None = None
    token_owners: list[int] | None = None
    token_slots: list[int] | None = None
    token_card_types: list[int] | None = None
    token_energy_types: list[int] | None = None
    token_weaknesses: list[int] | None = None
    token_resistances: list[int] | None = None
    option_card_ids: list[int] | None = None
    option_attack_ids: list[int] | None = None
    option_special_conditions: list[int] | None = None
    token_semantics: list[list[float]] | None = None
    option_semantics: list[list[float]] | None = None
    history_features: list[list[float]] | None = None
    history_types: list[int] | None = None
    history_owners: list[int] | None = None
    history_card_ids: list[int] | None = None
    history_target_card_ids: list[int] | None = None
    history_attack_ids: list[int] | None = None
    history_from_zones: list[int] | None = None
    history_to_zones: list[int] | None = None
    minimum: int = 1
    maximum: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> EncodedDecision:
        return cls(**value)


class FeatureEncoder:
    """Encode only information visible in the acting player's observation."""

    def __init__(
        self,
        card_catalog: Mapping[int, Any],
        attack_catalog: Mapping[int, Any],
    ) -> None:
        self._cards = card_catalog
        self._attacks = attack_catalog

    def encode(self, observation: dict) -> EncodedDecision:
        selection = observation.get("select")
        state = observation.get("current")
        if selection is None or state is None:
            raise ValueError("A decision observation must contain select and current state.")

        options = selection["option"]
        return EncodedDecision(
            state=self._state_features(state, selection),
            select_type=int(selection["type"]),
            context=int(selection["context"]),
            options=[self._option_features(observation, option) for option in options],
            option_types=[int(option["type"]) for option in options],
            areas=[int(option.get("area") or 0) for option in options],
            in_play_areas=[int(option.get("inPlayArea") or 0) for option in options],
            minimum=int(selection["minCount"]),
            maximum=int(selection["maxCount"]),
        )

    def _state_features(self, state: dict, selection: dict) -> list[float]:
        your_index = int(state["yourIndex"])
        first_player = int(state["firstPlayer"])
        features = [
            min(float(state["turn"]) / 100.0, 1.0),
            min(float(state["turnActionCount"]) / 20.0, 1.0),
            float(first_player == your_index) if first_player >= 0 else 0.5,
            float(state["supporterPlayed"]),
            float(state["stadiumPlayed"]),
            float(state["energyAttached"]),
            float(state["retreated"]),
            float(bool(state["stadium"])),
            min(len(selection["option"]) / 32.0, 1.0),
            min(float(selection["maxCount"]) / 10.0, 1.0),
        ]
        features.extend(self._player_features(state["players"][your_index]))
        features.extend(self._player_features(state["players"][1 - your_index]))
        if len(features) != STATE_FEATURE_SIZE:
            raise AssertionError(
                f"Expected {STATE_FEATURE_SIZE} state features, got {len(features)}"
            )
        return features

    @staticmethod
    def _player_features(player: dict) -> list[float]:
        active = player.get("active") or []
        pokemon = active[0] if active and active[0] is not None else None
        hp = float(_value(pokemon, "hp", 0) or 0)
        max_hp = float(_value(pokemon, "maxHp", 0) or 0)
        energies = _value(pokemon, "energies", []) or []
        statuses = sum(
            bool(player.get(name))
            for name in ("poisoned", "burned", "asleep", "paralyzed", "confused")
        )
        return [
            float(player["deckCount"]) / 60.0,
            float(player["handCount"]) / 60.0,
            len(player["prize"]) / 6.0,
            len(player["bench"]) / max(float(player["benchMax"]), 1.0),
            float(pokemon is not None),
            min(hp / 400.0, 1.0),
            _safe_ratio(hp, max_hp),
            min(len(energies) / 10.0, 1.0),
            statuses / 5.0,
            len(player["discard"]) / 60.0,
        ]

    def _option_features(self, observation: dict, option: dict) -> list[float]:
        state = observation["current"]
        your_index = int(state["yourIndex"])
        player_index = option.get("playerIndex")
        card_id = self._option_card_id(observation, option)
        card = self._cards.get(card_id)
        attack = self._attacks.get(option.get("attackId"))
        target = self._target_pokemon(observation, option)

        target_hp = float(_value(target, "hp", 0) or 0)
        target_max_hp = float(_value(target, "maxHp", 0) or 0)
        target_energies = _value(target, "energies", []) or []
        card_type = int(_value(card, "cardType", -1) or 0)
        features = [
            min(float(option.get("number") or 0) / 60.0, 1.0),
            min(float(option.get("count") or 0) / 10.0, 1.0),
            float(player_index is not None and int(player_index) == your_index),
            float(player_index is not None and int(player_index) != your_index),
            min(float(option.get("index") or 0) / 60.0, 1.0),
            min(float(option.get("inPlayIndex") or 0) / 5.0, 1.0),
            min(float(_value(card, "hp", 0) or 0) / 400.0, 1.0),
            max(card_type, 0) / 6.0,
            float(bool(_value(card, "basic", False))),
            float(bool(_value(card, "stage1", False))),
            float(bool(_value(card, "stage2", False))),
            float(bool(_value(card, "ex", False) or _value(card, "megaEx", False))),
            min(len(_value(card, "skills", []) or []) / 4.0, 1.0),
            min(len(_value(card, "attacks", []) or []) / 4.0, 1.0),
            min(float(_value(attack, "damage", 0) or 0) / 400.0, 1.0),
            min(len(_value(attack, "energies", []) or []) / 5.0, 1.0),
            min(target_hp / 400.0, 1.0),
            _safe_ratio(target_hp, target_max_hp),
            min(len(target_energies) / 10.0, 1.0),
            float(option.get("inPlayArea", option.get("area")) == 4),
        ]
        if len(features) != OPTION_FEATURE_SIZE:
            raise AssertionError(f"Expected {OPTION_FEATURE_SIZE} option features")
        return features

    def _option_card_id(self, observation: dict, option: dict) -> int | None:
        if option.get("cardId") is not None:
            return int(option["cardId"])
        index = option.get("index")
        area = int(option.get("area") or 0)
        player_index = option.get("playerIndex")
        if index is None:
            return None
        index = int(index)

        selection = observation["select"]
        if area == 1 and selection.get("deck"):
            return self._record_id(selection["deck"], index)
        if player_index is None:
            return None
        player = observation["current"]["players"][int(player_index)]
        records = {
            2: player.get("hand"),
            3: player.get("discard"),
            4: player.get("active"),
            5: player.get("bench"),
            6: player.get("prize"),
        }.get(area)
        return self._record_id(records, index)

    @staticmethod
    def _record_id(records: list | None, index: int) -> int | None:
        if records is None or not 0 <= index < len(records) or records[index] is None:
            return None
        card_id = _value(records[index], "id")
        return int(card_id) if card_id is not None else None

    @staticmethod
    def _target_pokemon(observation: dict, option: dict) -> dict | None:
        player_index = option.get("playerIndex")
        index = option.get("inPlayIndex", option.get("index"))
        area = int(option.get("inPlayArea", option.get("area")) or 0)
        if player_index is None or index is None or area not in {4, 5}:
            return None
        player = observation["current"]["players"][int(player_index)]
        records = player["active"] if area == 4 else player["bench"]
        if not 0 <= int(index) < len(records):
            return None
        return records[int(index)]


class FeatureEncoderV2(FeatureEncoder):
    """Encode every visible card-like object as a structured state token."""

    version = 2

    def encode(self, observation: dict) -> EncodedDecision:
        decision = super().encode(observation)
        selection = observation["select"]
        state = observation["current"]
        your_index = int(state["yourIndex"])

        token_fields: dict[str, list] = {
            "tokens": [],
            "token_card_ids": [],
            "token_kinds": [],
            "token_zones": [],
            "token_owners": [],
            "token_slots": [],
            "token_card_types": [],
            "token_energy_types": [],
            "token_weaknesses": [],
            "token_resistances": [],
        }

        def owner_id(record: Any) -> int:
            player_index = _value(record, "playerIndex")
            if player_index is None:
                return 0
            return 1 if int(player_index) == your_index else 2

        def add_card(record: Any, *, kind: int, zone: int, owner: int, slot: int = 0) -> None:
            if record is None or len(token_fields["tokens"]) >= MAX_STATE_TOKENS:
                return
            card_id = int(_value(record, "id", 0) or 0)
            card = self._cards.get(card_id)
            features = self._token_features(card)
            token_fields["tokens"].append(features)
            token_fields["token_card_ids"].append(card_id)
            token_fields["token_kinds"].append(kind)
            token_fields["token_zones"].append(zone)
            token_fields["token_owners"].append(owner)
            token_fields["token_slots"].append(slot)
            token_fields["token_card_types"].append(int(_value(card, "cardType", 0) or 0))
            token_fields["token_energy_types"].append(int(_value(card, "energyType", 0) or 0))
            token_fields["token_weaknesses"].append(int(_value(card, "weakness", 0) or 0))
            token_fields["token_resistances"].append(int(_value(card, "resistance", 0) or 0))

        def add_pokemon(
            pokemon: Any,
            *,
            zone: int,
            owner: int,
            slot: int,
            statuses: tuple[bool, bool, bool, bool, bool],
        ) -> None:
            if pokemon is None or len(token_fields["tokens"]) >= MAX_STATE_TOKENS:
                return
            card_id = int(_value(pokemon, "id", 0) or 0)
            card = self._cards.get(card_id)
            features = self._token_features(card, pokemon=pokemon, statuses=statuses)
            token_fields["tokens"].append(features)
            token_fields["token_card_ids"].append(card_id)
            token_fields["token_kinds"].append(2)
            token_fields["token_zones"].append(zone)
            token_fields["token_owners"].append(owner)
            token_fields["token_slots"].append(slot)
            token_fields["token_card_types"].append(int(_value(card, "cardType", 0) or 0))
            token_fields["token_energy_types"].append(int(_value(card, "energyType", 0) or 0))
            token_fields["token_weaknesses"].append(int(_value(card, "weakness", 0) or 0))
            token_fields["token_resistances"].append(int(_value(card, "resistance", 0) or 0))
            for attached in _value(pokemon, "energyCards", []) or []:
                add_card(attached, kind=3, zone=9, owner=owner, slot=slot)
            for attached in _value(pokemon, "tools", []) or []:
                add_card(attached, kind=4, zone=10, owner=owner, slot=slot)
            for attached in _value(pokemon, "preEvolution", []) or []:
                add_card(attached, kind=5, zone=11, owner=owner, slot=slot)

        # Decision-specific cards come first so they survive the token cap.
        context_card = selection.get("contextCard")
        add_card(context_card, kind=1, zone=12, owner=owner_id(context_card))
        add_card(selection.get("effect"), kind=1, zone=13, owner=owner_id(selection.get("effect")))
        for card in state.get("stadium") or []:
            add_card(card, kind=1, zone=1, owner=owner_id(card))

        for owner, player_index in enumerate(
            (your_index, 1 - your_index), start=1
        ):
            player = state["players"][player_index]
            statuses = tuple(
                bool(player.get(name))
                for name in ("poisoned", "burned", "asleep", "paralyzed", "confused")
            )
            for pokemon in player.get("active") or []:
                add_pokemon(pokemon, zone=4, owner=owner, slot=0, statuses=statuses)
            for bench_index, pokemon in enumerate(player.get("bench") or []):
                add_pokemon(
                    pokemon,
                    zone=5,
                    owner=owner,
                    slot=min(bench_index + 1, 15),
                    statuses=(False, False, False, False, False),
                )

        own_player = state["players"][your_index]
        for card in own_player.get("hand") or []:
            add_card(card, kind=1, zone=2, owner=1)
        for card in state.get("looking") or []:
            add_card(card, kind=1, zone=14, owner=owner_id(card))
        for card in selection.get("deck") or []:
            add_card(card, kind=1, zone=8, owner=owner_id(card))
        for owner, player_index in enumerate(
            (your_index, 1 - your_index), start=1
        ):
            player = state["players"][player_index]
            for card in player.get("prize") or []:
                add_card(card, kind=1, zone=6, owner=owner)
            # The newest discard cards are usually the most decision-relevant.
            for card in (player.get("discard") or [])[-48:]:
                add_card(card, kind=1, zone=3, owner=owner)

        options = selection["option"]
        decision.version = self.version
        for name, values in token_fields.items():
            setattr(decision, name, values)
        decision.option_card_ids = [
            self._option_card_id(observation, option) or 0 for option in options
        ]
        decision.option_attack_ids = [int(option.get("attackId") or 0) for option in options]
        decision.option_special_conditions = [
            int(option.get("specialConditionType") or 0) for option in options
        ]
        return decision

    @staticmethod
    def _token_features(
        card: Any,
        *,
        pokemon: Any | None = None,
        statuses: tuple[bool, bool, bool, bool, bool] = (False, False, False, False, False),
    ) -> list[float]:
        current_hp = float(_value(pokemon, "hp", 0) or 0)
        max_hp = float(_value(pokemon, "maxHp", 0) or 0)
        energies = _value(pokemon, "energies", []) or []
        energy_cards = _value(pokemon, "energyCards", []) or []
        tools = _value(pokemon, "tools", []) or []
        evolutions = _value(pokemon, "preEvolution", []) or []
        return [
            min(float(_value(card, "hp", 0) or 0) / 400.0, 1.0),
            min(float(_value(card, "retreatCost", 0) or 0) / 5.0, 1.0),
            float(bool(_value(card, "basic", False))),
            float(bool(_value(card, "stage1", False))),
            float(bool(_value(card, "stage2", False))),
            float(bool(_value(card, "ex", False))),
            float(bool(_value(card, "megaEx", False))),
            float(bool(_value(card, "tera", False))),
            float(bool(_value(card, "aceSpec", False))),
            min(len(_value(card, "skills", []) or []) / 4.0, 1.0),
            min(len(_value(card, "attacks", []) or []) / 4.0, 1.0),
            min(current_hp / 400.0, 1.0),
            min(max_hp / 400.0, 1.0),
            _safe_ratio(current_hp, max_hp),
            float(bool(_value(pokemon, "appearThisTurn", False))),
            min(len(energies) / 10.0, 1.0),
            min(len(energy_cards) / 10.0, 1.0),
            min(len(tools) / 4.0, 1.0),
            min(len(evolutions) / 3.0, 1.0),
            *(float(value) for value in statuses),
        ]


def _semantic_text_features(text: str) -> list[float]:
    normalized = text.lower().replace("’", "'").replace("é", "e")
    return [float(bool(re.search(pattern, normalized))) for pattern in _SEMANTIC_PATTERNS]


def _merge_semantics(first: list[float], second: list[float]) -> list[float]:
    return [max(left, right) for left, right in zip(first, second, strict=True)]


def pack_semantic_features(features: list[float]) -> list[float]:
    """Pack 32 binary tags into one bitmask while retaining eight numeric values."""
    if len(features) != SEMANTIC_FEATURE_SIZE:
        raise ValueError(f"Expected {SEMANTIC_FEATURE_SIZE} semantic features")
    mask = sum(
        (1 << index) for index, value in enumerate(features[: len(SEMANTIC_TAGS)]) if value
    )
    return [mask, *features[len(SEMANTIC_TAGS) :]]


def expand_semantic_features(features: list[float]) -> list[float]:
    """Expand either the compact JSON representation or an already-dense vector."""
    if len(features) == SEMANTIC_FEATURE_SIZE:
        return features
    if len(features) != PACKED_SEMANTIC_FEATURE_SIZE:
        raise ValueError("Semantic features must use the packed or dense V3 schema")
    mask = int(features[0])
    tags = [float(bool(mask & (1 << index))) for index in range(len(SEMANTIC_TAGS))]
    return [*tags, *(float(value) for value in features[1:])]


def structured_semantic_features(
    card: Any | None,
    attacks: list[Any] | tuple[Any, ...] = (),
) -> list[float]:
    """Convert official effect text and printed values into a stable semantic schema."""
    skills = list(_value(card, "skills", []) or [])
    attacks = list(attacks)
    texts = [str(_value(item, "text", "") or "") for item in [*skills, *attacks]]
    combined_text = " ".join(text for text in texts if text)
    tags = _semantic_text_features(combined_text)
    printed_damage = [float(_value(attack, "damage", 0) or 0) for attack in attacks]
    energy_costs = [len(_value(attack, "energies", []) or []) for attack in attacks]
    numeric_values = [float(value) for value in re.findall(r"\b\d+\b", combined_text)]
    numeric = [
        float(bool(combined_text)),
        min(max(printed_damage, default=0.0) / 400.0, 1.0),
        min(max(numeric_values, default=0.0) / 400.0, 1.0),
        min(min(energy_costs, default=0) / 5.0, 1.0),
        min(max(energy_costs, default=0) / 5.0, 1.0),
        min(len(attacks) / 4.0, 1.0),
        min(len(skills) / 4.0, 1.0),
        min(len(combined_text) / 500.0, 1.0),
    ]
    features = [*tags, *numeric]
    if len(features) != SEMANTIC_FEATURE_SIZE:
        raise AssertionError(f"Expected {SEMANTIC_FEATURE_SIZE} semantic features")
    return features


class FeatureEncoderV3(FeatureEncoderV2):
    """Add structured effect semantics and recent visible engine events to V2 tokens."""

    version = 3

    def __init__(
        self,
        card_catalog: Mapping[int, Any],
        attack_catalog: Mapping[int, Any],
    ) -> None:
        super().__init__(card_catalog, attack_catalog)
        self._card_semantics = {
            int(card_id): structured_semantic_features(
                card,
                [
                    attack_catalog[attack_id]
                    for attack_id in (_value(card, "attacks", []) or [])
                    if attack_id in attack_catalog
                ],
            )
            for card_id, card in card_catalog.items()
        }
        self._card_skill_semantics = {
            int(card_id): structured_semantic_features(card)
            for card_id, card in card_catalog.items()
        }
        self._attack_semantics = {
            int(attack_id): structured_semantic_features(None, [attack])
            for attack_id, attack in attack_catalog.items()
        }
        self._empty_semantics = [0.0] * SEMANTIC_FEATURE_SIZE

    def encode(self, observation: dict) -> EncodedDecision:
        decision = super().encode(observation)
        state = observation["current"]
        your_index = int(state["yourIndex"])
        decision.version = self.version
        decision.token_semantics = [
            pack_semantic_features(
                self._card_semantics.get(card_id, self._empty_semantics)
            )
            for card_id in decision.token_card_ids or []
        ]
        decision.option_semantics = []
        for card_id, attack_id in zip(
            decision.option_card_ids or [],
            decision.option_attack_ids or [],
            strict=True,
        ):
            attack_semantics = self._attack_semantics.get(attack_id, self._empty_semantics)
            card_semantics = (
                self._card_skill_semantics.get(card_id, self._empty_semantics)
                if attack_id
                else self._card_semantics.get(card_id, self._empty_semantics)
            )
            decision.option_semantics.append(
                pack_semantic_features(_merge_semantics(card_semantics, attack_semantics))
            )

        history = list(observation.get("logs") or [])[-MAX_HISTORY_TOKENS:]
        history_count = len(history)
        decision.history_features = []
        decision.history_types = []
        decision.history_owners = []
        decision.history_card_ids = []
        decision.history_target_card_ids = []
        decision.history_attack_ids = []
        decision.history_from_zones = []
        decision.history_to_zones = []
        for index, event in enumerate(history):
            player_index = _value(event, "playerIndex")
            owner = (
                0
                if player_index is None
                else (1 if int(player_index) == your_index else 2)
            )
            has_basic = _value(event, "hasBasicPokemon")
            value = float(_value(event, "value", 0) or 0)
            decision.history_features.append(
                [
                    max(-1.0, min(value / 400.0, 1.0)),
                    float(bool(_value(event, "putDamageCounter", False))),
                    float(has_basic is not None),
                    float(bool(has_basic)),
                    (index + 1) / max(history_count, 1),
                ]
            )
            decision.history_types.append(int(_value(event, "type", 0) or 0))
            decision.history_owners.append(owner)
            decision.history_card_ids.append(int(_value(event, "cardId", 0) or 0))
            decision.history_target_card_ids.append(
                int(_value(event, "cardIdTarget", 0) or 0)
            )
            decision.history_attack_ids.append(int(_value(event, "attackId", 0) or 0))
            decision.history_from_zones.append(int(_value(event, "fromArea", 0) or 0))
            decision.history_to_zones.append(int(_value(event, "toArea", 0) or 0))
        return decision


def build_feature_encoder(
    version: int,
    card_catalog: Mapping[int, Any],
    attack_catalog: Mapping[int, Any],
) -> FeatureEncoder:
    encoder_types = {
        1: FeatureEncoder,
        2: FeatureEncoderV2,
        3: FeatureEncoderV3,
    }
    try:
        encoder_type = encoder_types[version]
    except KeyError as error:
        raise ValueError(f"Unsupported encoder version: {version}") from error
    return encoder_type(card_catalog, attack_catalog)
