"""Index-independent semantic directives for replaying a searched turn plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from poketcg.agents.rule_agent import AreaType, OptionType


def _record(
    observation: dict,
    option: dict,
    *,
    target: bool,
) -> dict | None:
    state = observation["current"]
    your_index = int(state["yourIndex"])
    raw_player = option.get("playerIndex")
    player_index = your_index if raw_player is None else int(raw_player)
    if target:
        area = option.get("inPlayArea", option.get("area"))
        index = option.get("inPlayIndex", option.get("index"))
    else:
        area = option.get("area")
        index = option.get("index")
        if area is None and int(option["type"]) in {
            int(OptionType.PLAY),
            int(OptionType.ATTACH),
            int(OptionType.EVOLVE),
        }:
            area = int(AreaType.HAND)
    if index is None:
        return None
    if area == int(AreaType.DECK):
        records = observation["select"].get("deck")
    else:
        player = state["players"][player_index]
        records = {
            int(AreaType.HAND): player.get("hand"),
            int(AreaType.DISCARD): player.get("discard"),
            int(AreaType.ACTIVE): player.get("active"),
            int(AreaType.BENCH): player.get("bench"),
            int(AreaType.PRIZE): player.get("prize"),
        }.get(area)
    position = int(index)
    if records is None or not 0 <= position < len(records):
        return None
    return records[position]


def _integer(value) -> int | None:
    return int(value) if value is not None else None


def _record_value(record: dict | None, name: str) -> int | None:
    if not record:
        return None
    return _integer(record.get(name))


def _relative_player(observation: dict, option: dict) -> int | None:
    raw_player = option.get("playerIndex")
    if raw_player is None:
        return None
    your_index = int(observation["current"]["yourIndex"])
    return 0 if int(raw_player) == your_index else 1


@dataclass(frozen=True, slots=True)
class SemanticOption:
    """Stable meaning of one selected option, independent of its list index."""

    option_type: int
    card_id: int | None
    attack_id: int | None
    player_relation: int | None
    area: int | None
    in_play_area: int | None
    target_card_id: int | None
    target_serial_hint: int | None
    number: int | None
    count: int | None
    special_condition: int | None
    index_hint: int | None
    in_play_index_hint: int | None

    def semantic_key(self) -> tuple:
        """Exclude physical serial and slot hints from cross-world identity."""
        return (
            self.option_type,
            self.card_id,
            self.attack_id,
            self.player_relation,
            self.area,
            self.in_play_area,
            self.target_card_id,
            self.number,
            self.count,
            self.special_condition,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> SemanticOption:
        return cls(**value)


@dataclass(frozen=True, slots=True)
class SemanticAction:
    """One cardinality-valid action in a particular selection context."""

    context: int
    options: tuple[SemanticOption, ...]

    def semantic_key(self) -> tuple:
        keys = [option.semantic_key() for option in self.options]
        return self.context, tuple(sorted(keys, key=repr))

    def to_dict(self) -> dict:
        return {
            "context": self.context,
            "options": [option.to_dict() for option in self.options],
        }

    @classmethod
    def from_dict(cls, value: dict) -> SemanticAction:
        return cls(
            context=int(value["context"]),
            options=tuple(
                SemanticOption.from_dict(option) for option in value["options"]
            ),
        )


@dataclass(frozen=True, slots=True)
class SemanticTurnPlan:
    """A fixed semantic action sequence proposed without held-out worlds."""

    actions: tuple[SemanticAction, ...]

    def semantic_key(self) -> tuple:
        return tuple(action.semantic_key() for action in self.actions)

    def to_dict(self) -> dict:
        return {"actions": [action.to_dict() for action in self.actions]}

    @classmethod
    def from_dict(cls, value: dict) -> SemanticTurnPlan:
        return cls(
            actions=tuple(
                SemanticAction.from_dict(action) for action in value["actions"]
            )
        )


def semantic_option(observation: dict, option: dict) -> SemanticOption:
    """Resolve card and target identities while the source observation is available."""
    source = _record(observation, option, target=False)
    target = _record(observation, option, target=True)
    card_id = _integer(option.get("cardId"))
    if card_id is None:
        card_id = _record_value(source, "id")
    return SemanticOption(
        option_type=int(option["type"]),
        card_id=card_id,
        attack_id=_integer(option.get("attackId")),
        player_relation=_relative_player(observation, option),
        area=_integer(option.get("area")),
        in_play_area=_integer(option.get("inPlayArea")),
        target_card_id=_record_value(target, "id"),
        target_serial_hint=_record_value(target, "serial"),
        number=_integer(option.get("number")),
        count=_integer(option.get("count")),
        special_condition=_integer(option.get("specialCondition")),
        index_hint=_integer(option.get("index")),
        in_play_index_hint=_integer(option.get("inPlayIndex")),
    )


def semantic_action(
    observation: dict, action: list[int] | tuple[int, ...]
) -> SemanticAction:
    options = observation["select"]["option"]
    if any(index < 0 or index >= len(options) for index in action):
        raise IndexError("Action references an invalid option")
    return SemanticAction(
        context=int(observation["select"]["context"]),
        options=tuple(semantic_option(observation, options[index]) for index in action),
    )


def _match_score(planned: SemanticOption, actual: SemanticOption) -> int | None:
    if planned.option_type != actual.option_type:
        return None
    required = (
        "card_id",
        "attack_id",
        "player_relation",
        "area",
        "in_play_area",
        "target_card_id",
        "number",
        "count",
        "special_condition",
    )
    score = 1
    for name in required:
        expected = getattr(planned, name)
        observed = getattr(actual, name)
        if expected is not None:
            if observed != expected:
                return None
            score += 2
    if (
        planned.target_serial_hint is not None
        and planned.target_serial_hint == actual.target_serial_hint
    ):
        score += 4
    if planned.index_hint is not None and planned.index_hint == actual.index_hint:
        score += 1
    if (
        planned.in_play_index_hint is not None
        and planned.in_play_index_hint == actual.in_play_index_hint
    ):
        score += 1
    return score


def resolve_semantic_action(
    observation: dict, directive: SemanticAction
) -> list[int] | None:
    """Map a semantic directive back to legal indices, or fail closed."""
    selection = observation["select"]
    if int(selection["context"]) != directive.context:
        return None
    minimum = int(selection["minCount"])
    maximum = int(selection["maxCount"])
    if not minimum <= len(directive.options) <= maximum:
        return None
    if not directive.options:
        return []
    actual = [
        semantic_option(observation, option) for option in selection["option"]
    ]
    selected: list[int] = []
    for planned in directive.options:
        matches = [
            (score, -index, index)
            for index, candidate in enumerate(actual)
            if index not in selected
            and (score := _match_score(planned, candidate)) is not None
        ]
        if not matches:
            return None
        selected.append(max(matches)[2])
    return sorted(selected)
