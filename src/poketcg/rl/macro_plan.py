"""Library-Out turn plans and their public execution state.

The first macro-oracle prototype inferred generic intents from effect tags.
This module keeps that schema readable, but new plans are generated from the
actual Great Tusk / Crustle deck loop:

    enable Land Collapse -> mill four -> buy another turn -> repeat.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any

from poketcg.agents.rule_agent import OptionType, SelectContext

from .features import SEMANTIC_TAGS, structured_semantic_features
from .paired_rollout import RootCandidate
from .semantic_plan import SemanticAction, semantic_action


def _value(record: Any, name: str, default: Any = None) -> Any:
    if record is None:
        return default
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


class MacroPlanType(StrEnum):
    """Deck-level intent retained across all decisions in one turn."""

    BASELINE_V1 = "baseline_v1"

    # Library-Out v2 strategy ontology.
    MILL_FOUR_NOW = "mill_four_now"
    FIND_ANCIENT_SUPPORTER = "find_ancient_supporter"
    PREPARE_NEXT_GREAT_TUSK = "prepare_next_great_tusk"
    BUILD_CRUSTLE_WALL = "build_crustle_wall"
    ENABLE_NEUTRALIZATION_WALL = "enable_neutralization_wall"
    GUST_STALL_TARGET = "gust_stall_target"
    HAND_DISRUPTION_STALL = "hand_disruption_stall"
    HEAL_OR_ROTATE_WALL = "heal_or_rotate_wall"
    PRIZE_RACE_PIVOT = "prize_race_pivot"
    PRESERVE_DECK_AND_CHAIN = "preserve_deck_and_chain"

    # Read-only compatibility with macro-oracle v1 datasets.
    ADVANCE_WIN_CONDITION = "advance_win_condition"
    BUILD_DRAW_ENGINE = "build_draw_engine"
    SEARCH_AND_DEPLOY = "search_and_deploy"
    RECOVER_RESOURCES = "recover_resources"
    DISRUPT_OPPONENT = "disrupt_opponent"
    STABILIZE_BOARD = "stabilize_board"
    TAKE_SAFE_PRIZE = "take_safe_prize"
    PRESERVE_RESOURCES = "preserve_resources"


class MacroTermination(StrEnum):
    """Boundary at which the first macro-policy version releases ownership."""

    TURN_END = "turn_end"


@dataclass(frozen=True, slots=True)
class LibraryOutStrategyProfile:
    """Competition card identities used by the Library-Out macro policy."""

    strategy_version: str = "libraryout_v2"
    great_tusk: int = 58
    dwebble: int = 344
    crustle: int = 345
    terrakion: int = 607
    buddy_buddy_poffin: int = 1086
    ultra_ball: int = 1121
    pokegear: int = 1122
    switch: int = 1123
    fighting_gong: int = 1142
    jumbo_ice_cream: int = 1147
    poke_pad: int = 1152
    boss_orders: int = 1182
    explorers_guidance: int = 1185
    colress_tenacity: int = 1194
    xerosic_machinations: int = 1197
    lisia_appeal: int = 1204
    neutralization_zone: int = 1247
    mist_energy: int = 11
    rock_fighting_energy: int = 20
    land_collapse: int = 62
    giant_tusk: int = 63

    @property
    def ancient_supporters(self) -> frozenset[int]:
        # The official metadata omits regulation/Ancient labels. In this deck,
        # Explorer's Guidance is the Ancient Supporter that powers Land Collapse.
        return frozenset({self.explorers_guidance})

    @property
    def energy_cards(self) -> frozenset[int]:
        return frozenset({self.mist_energy, self.rock_fighting_energy})

    @property
    def gust_cards(self) -> frozenset[int]:
        return frozenset({self.boss_orders, self.lisia_appeal})

    @property
    def pokemon_search_cards(self) -> frozenset[int]:
        return frozenset(
            {
                self.buddy_buddy_poffin,
                self.ultra_ball,
                self.fighting_gong,
                self.poke_pad,
            }
        )


_PLAN_TAGS: dict[MacroPlanType, tuple[str, ...]] = {
    MacroPlanType.BASELINE_V1: (),
    MacroPlanType.MILL_FOUR_NOW: (
        "mill_deck",
        "discard",
        "deck_manipulation",
    ),
    MacroPlanType.FIND_ANCIENT_SUPPORTER: (
        "search_deck",
        "put_into_hand",
        "deck_manipulation",
    ),
    MacroPlanType.PREPARE_NEXT_GREAT_TUSK: (
        "search_deck",
        "put_into_hand",
        "attach_energy",
        "bench_target",
    ),
    MacroPlanType.BUILD_CRUSTLE_WALL: (
        "search_deck",
        "put_into_hand",
        "evolve",
        "bench_target",
        "damage_reduction_or_prevention",
    ),
    MacroPlanType.ENABLE_NEUTRALIZATION_WALL: (
        "search_deck",
        "put_into_hand",
        "damage_reduction_or_prevention",
    ),
    MacroPlanType.GUST_STALL_TARGET: (
        "switch",
        "opponent_target",
    ),
    MacroPlanType.HAND_DISRUPTION_STALL: ("hand_disruption",),
    MacroPlanType.HEAL_OR_ROTATE_WALL: (
        "switch",
        "retreat",
        "heal",
        "damage_reduction_or_prevention",
    ),
    MacroPlanType.PRIZE_RACE_PIVOT: (
        "knock_out",
        "prize",
        "damage_scaling",
        "opponent_target",
    ),
    MacroPlanType.PRESERVE_DECK_AND_CHAIN: (),
    # Legacy values are retained only for loading old records.
    MacroPlanType.ADVANCE_WIN_CONDITION: (
        "mill_deck",
        "discard",
        "deck_manipulation",
    ),
    MacroPlanType.BUILD_DRAW_ENGINE: (
        "draw",
        "search_deck",
        "put_into_hand",
        "deck_manipulation",
    ),
    MacroPlanType.SEARCH_AND_DEPLOY: (
        "search_deck",
        "put_into_hand",
        "evolve",
        "attach_energy",
        "bench_target",
    ),
    MacroPlanType.RECOVER_RESOURCES: (
        "recover_discard",
        "heal",
        "put_into_hand",
    ),
    MacroPlanType.DISRUPT_OPPONENT: (
        "hand_disruption",
        "switch",
        "opponent_target",
    ),
    MacroPlanType.STABILIZE_BOARD: (
        "switch",
        "retreat",
        "damage_reduction_or_prevention",
        "heal",
    ),
    MacroPlanType.TAKE_SAFE_PRIZE: (
        "knock_out",
        "prize",
        "damage_scaling",
        "opponent_target",
    ),
    MacroPlanType.PRESERVE_RESOURCES: (),
}


_PLAN_PRIORITY = {
    MacroPlanType.MILL_FOUR_NOW: 100,
    MacroPlanType.FIND_ANCIENT_SUPPORTER: 90,
    MacroPlanType.PREPARE_NEXT_GREAT_TUSK: 80,
    MacroPlanType.BUILD_CRUSTLE_WALL: 70,
    MacroPlanType.ENABLE_NEUTRALIZATION_WALL: 65,
    MacroPlanType.GUST_STALL_TARGET: 60,
    MacroPlanType.HAND_DISRUPTION_STALL: 55,
    MacroPlanType.HEAL_OR_ROTATE_WALL: 50,
    MacroPlanType.PRIZE_RACE_PIVOT: 35,
    MacroPlanType.PRESERVE_DECK_AND_CHAIN: 20,
}


@dataclass(frozen=True, slots=True)
class PlanOption:
    """One public-information macro candidate with an explicit root action."""

    plan_type: MacroPlanType
    root_action: SemanticAction
    primary_card_id: int | None
    target_card_id: int | None
    attack_id: int | None
    desired_tags: tuple[str, ...]
    preserve_card_ids: tuple[int, ...]
    require_attack: bool
    termination: MacroTermination
    maximum_steps: int
    source_action: tuple[int, ...]
    sources: tuple[str, ...]
    strategy_version: str = "generic_v1"
    preferred_card_ids: tuple[int, ...] = ()
    preferred_attack_ids: tuple[int, ...] = ()
    preconditions: tuple[str, ...] = ()
    success_conditions: tuple[str, ...] = ()
    public_signals: tuple[str, ...] = ()
    feasibility_score: float = 1.0

    def semantic_key(self) -> tuple:
        return (
            self.plan_type.value,
            self.root_action.semantic_key(),
            self.primary_card_id,
            self.target_card_id,
            self.attack_id,
            self.desired_tags,
            self.preserve_card_ids,
            self.require_attack,
            self.termination.value,
            self.maximum_steps,
            self.strategy_version,
            self.preferred_card_ids,
            self.preferred_attack_ids,
            self.preconditions,
            self.success_conditions,
            self.public_signals,
            round(self.feasibility_score, 6),
        )

    @property
    def plan_id(self) -> str:
        payload = json.dumps(
            self.semantic_key(), separators=(",", ":"), default=str
        )
        return hashlib.blake2b(payload.encode(), digest_size=12).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_type": self.plan_type.value,
            "root_action": self.root_action.to_dict(),
            "primary_card_id": self.primary_card_id,
            "target_card_id": self.target_card_id,
            "attack_id": self.attack_id,
            "desired_tags": list(self.desired_tags),
            "preserve_card_ids": list(self.preserve_card_ids),
            "require_attack": self.require_attack,
            "termination": self.termination.value,
            "maximum_steps": self.maximum_steps,
            "source_action": list(self.source_action),
            "sources": list(self.sources),
            "strategy_version": self.strategy_version,
            "preferred_card_ids": list(self.preferred_card_ids),
            "preferred_attack_ids": list(self.preferred_attack_ids),
            "preconditions": list(self.preconditions),
            "success_conditions": list(self.success_conditions),
            "public_signals": list(self.public_signals),
            "feasibility_score": self.feasibility_score,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PlanOption:
        return cls(
            plan_type=MacroPlanType(value["plan_type"]),
            root_action=SemanticAction.from_dict(value["root_action"]),
            primary_card_id=value.get("primary_card_id"),
            target_card_id=value.get("target_card_id"),
            attack_id=value.get("attack_id"),
            desired_tags=tuple(value.get("desired_tags") or ()),
            preserve_card_ids=tuple(
                int(item) for item in value.get("preserve_card_ids") or ()
            ),
            require_attack=bool(value.get("require_attack", False)),
            termination=MacroTermination(
                value.get("termination", MacroTermination.TURN_END.value)
            ),
            maximum_steps=int(value.get("maximum_steps", 32)),
            source_action=tuple(
                int(item) for item in value.get("source_action") or ()
            ),
            sources=tuple(str(item) for item in value.get("sources") or ()),
            strategy_version=str(value.get("strategy_version", "generic_v1")),
            preferred_card_ids=tuple(
                int(item) for item in value.get("preferred_card_ids") or ()
            ),
            preferred_attack_ids=tuple(
                int(item) for item in value.get("preferred_attack_ids") or ()
            ),
            preconditions=tuple(
                str(item) for item in value.get("preconditions") or ()
            ),
            success_conditions=tuple(
                str(item) for item in value.get("success_conditions") or ()
            ),
            public_signals=tuple(
                str(item) for item in value.get("public_signals") or ()
            ),
            feasibility_score=float(value.get("feasibility_score", 1.0)),
        )


@dataclass(frozen=True, slots=True)
class PlanProgress:
    """Public, serializable state carried by a plan-conditioned executor."""

    owner_player: int
    start_turn: int
    decisions: int = 0
    contexts: tuple[int, ...] = ()
    option_types: tuple[int, ...] = ()
    played_card_ids: tuple[int, ...] = ()
    attack_ids: tuple[int, ...] = ()
    plan_hits: int = 0

    @classmethod
    def start(cls, observation: dict) -> PlanProgress:
        current = observation["current"]
        return cls(
            owner_player=int(current["yourIndex"]),
            start_turn=int(current["turn"]),
        )

    @property
    def attacked(self) -> bool:
        return bool(self.attack_ids)

    def active(self, observation: dict, plan: PlanOption) -> bool:
        current = observation["current"]
        return (
            int(current["result"]) == -1
            and int(current["yourIndex"]) == self.owner_player
            and int(current["turn"]) == self.start_turn
            and self.decisions < plan.maximum_steps
        )

    def advance(
        self, plan: PlanOption, action: SemanticAction
    ) -> PlanProgress:
        option_types = tuple(option.option_type for option in action.options)
        card_ids = tuple(
            option.card_id
            for option in action.options
            if option.card_id is not None
            and option.option_type
            in {
                int(OptionType.PLAY),
                int(OptionType.ABILITY),
                int(OptionType.ATTACH),
                int(OptionType.EVOLVE),
            }
        )
        attack_ids = tuple(
            option.attack_id
            for option in action.options
            if option.attack_id is not None
            or option.option_type == int(OptionType.ATTACK)
        )
        normalized_attacks = tuple(
            int(attack_id or 0) for attack_id in attack_ids
        )
        return replace(
            self,
            decisions=self.decisions + 1,
            contexts=(*self.contexts, action.context),
            option_types=(*self.option_types, *option_types),
            played_card_ids=(*self.played_card_ids, *card_ids),
            attack_ids=(*self.attack_ids, *normalized_attacks),
            plan_hits=self.plan_hits + int(action_alignment(plan, action) > 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _option_card_ids(action: SemanticAction) -> set[int]:
    return {
        int(option.card_id)
        for option in action.options
        if option.card_id is not None
    }


def _option_attack_ids(action: SemanticAction) -> set[int]:
    return {
        int(option.attack_id)
        for option in action.options
        if option.attack_id is not None
    }


def action_alignment(plan: PlanOption, action: SemanticAction) -> float:
    """Score observable agreement between one action and a persistent plan."""
    score = 0.0
    option_types = {option.option_type for option in action.options}
    card_ids = _option_card_ids(action)
    attack_ids = _option_attack_ids(action)
    if plan.primary_card_id is not None and plan.primary_card_id in card_ids:
        score += 2.0
    if plan.target_card_id is not None and plan.target_card_id in card_ids:
        score += 1.0
    if plan.attack_id is not None and plan.attack_id in attack_ids:
        score += 3.0
    score += 1.5 * len(card_ids.intersection(plan.preferred_card_ids))
    score += 4.0 * len(attack_ids.intersection(plan.preferred_attack_ids))
    resource_loss_contexts = {
        int(SelectContext.DISCARD),
        int(SelectContext.DISCARD_ENERGY_CARD),
        int(SelectContext.DISCARD_TOOL_CARD),
        int(SelectContext.DISCARD_CARD_OR_ATTACHED_CARD),
        int(SelectContext.DISCARD_ENERGY),
    }
    if action.context in resource_loss_contexts:
        score -= 2.0 * len(card_ids.intersection(plan.preserve_card_ids))
    if plan.require_attack and int(OptionType.ATTACK) in option_types:
        score += 3.0
    return score


class MacroPlanGenerator:
    """Generate Library-Out plans from legal root actions and public state."""

    def __init__(
        self,
        card_catalog: Mapping[int, Any],
        attack_catalog: Mapping[int, Any],
        *,
        maximum_steps: int = 32,
        strategy_profile: LibraryOutStrategyProfile | None = None,
    ) -> None:
        if maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive")
        self._cards = card_catalog
        self._attacks = attack_catalog
        self._maximum_steps = maximum_steps
        self.profile = strategy_profile or LibraryOutStrategyProfile()

    def tags(self, directive: SemanticAction) -> tuple[str, ...]:
        """Return structured effect tags visible in one semantic action."""
        enabled: set[str] = set()
        for option in directive.options:
            card = self._cards.get(int(option.card_id or 0))
            attack = self._attacks.get(int(option.attack_id or 0))
            features = structured_semantic_features(
                card,
                [attack] if attack is not None else (),
            )
            enabled.update(
                SEMANTIC_TAGS[index]
                for index, value in enumerate(features[: len(SEMANTIC_TAGS)])
                if value
            )
        return tuple(sorted(enabled))

    @staticmethod
    def _field(player: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        return [
            pokemon
            for zone in ("active", "bench")
            for pokemon in (player.get(zone) or ())
            if pokemon
        ]

    @staticmethod
    def _record_id(record: Any) -> int | None:
        raw = _value(record, "id")
        return int(raw) if raw is not None else None

    def _public_signals(self, observation: dict) -> tuple[str, ...]:
        state = observation["current"]
        owner = int(state["yourIndex"])
        own = state["players"][owner]
        opponent = state["players"][1 - owner]
        active = next((item for item in own.get("active") or () if item), None)
        active_id = self._record_id(active)
        active_energy = len(_value(active, "energies", ()) or ())
        own_field = {
            self._record_id(item) for item in self._field(own)
        }
        opponent_has_ex = any(
            bool(_value(self._cards.get(self._record_id(item) or 0), "ex", False))
            for item in self._field(opponent)
        )
        signals = {
            f"own_deck={int(own.get('deckCount', 0))}",
            f"opponent_deck={int(opponent.get('deckCount', 0))}",
            f"own_prizes={len(own.get('prize') or ())}",
            f"opponent_prizes={len(opponent.get('prize') or ())}",
            f"opponent_hand={int(opponent.get('handCount', 0))}",
            f"active={active_id or 0}",
            f"active_energy={active_energy}",
        }
        if bool(state.get("supporterPlayed")):
            signals.add("supporter_already_played")
        if self.profile.great_tusk in own_field:
            signals.add("great_tusk_in_play")
        if self.profile.dwebble in own_field:
            signals.add("dwebble_in_play")
        if self.profile.crustle in own_field:
            signals.add("crustle_in_play")
        if opponent_has_ex:
            signals.add("visible_opponent_ex")
        return tuple(sorted(signals))

    def _plan_contract(
        self, plan_type: MacroPlanType
    ) -> tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[str, ...],
        tuple[str, ...],
        bool,
    ]:
        p = self.profile
        contracts = {
            MacroPlanType.BASELINE_V1: ((), (), (), (), (), False),
            MacroPlanType.MILL_FOUR_NOW: (
                (
                    p.explorers_guidance,
                    p.great_tusk,
                    p.fighting_gong,
                    p.switch,
                    *sorted(p.energy_cards),
                ),
                (p.land_collapse,),
                (
                    p.explorers_guidance,
                    p.great_tusk,
                    p.pokegear,
                    *sorted(p.energy_cards),
                ),
                (
                    "Great Tusk can become Active with two Energy",
                    "an Ancient Supporter can be or was played this turn",
                ),
                ("play Ancient Supporter", "use Land Collapse"),
                True,
            ),
            MacroPlanType.FIND_ANCIENT_SUPPORTER: (
                (p.pokegear, p.explorers_guidance),
                (),
                (p.great_tusk, *sorted(p.energy_cards)),
                ("supporter has not already been spent this turn",),
                ("put Explorer's Guidance into hand",),
                False,
            ),
            MacroPlanType.PREPARE_NEXT_GREAT_TUSK: (
                (
                    p.great_tusk,
                    p.fighting_gong,
                    p.ultra_ball,
                    p.poke_pad,
                    *sorted(p.energy_cards),
                ),
                (),
                (p.explorers_guidance,),
                ("Great Tusk or its Energy remains accessible",),
                ("bench Great Tusk", "attach toward Land Collapse"),
                False,
            ),
            MacroPlanType.BUILD_CRUSTLE_WALL: (
                (
                    p.dwebble,
                    p.crustle,
                    p.buddy_buddy_poffin,
                    p.ultra_ball,
                    p.poke_pad,
                ),
                (),
                (p.explorers_guidance, p.great_tusk),
                ("Dwebble or Crustle remains accessible",),
                ("bench Dwebble", "evolve Dwebble into Crustle"),
                False,
            ),
            MacroPlanType.ENABLE_NEUTRALIZATION_WALL: (
                (
                    p.colress_tenacity,
                    p.neutralization_zone,
                    *sorted(p.energy_cards),
                ),
                (),
                (),
                ("Neutralization Zone remains accessible",),
                ("put Neutralization Zone into play",),
                False,
            ),
            MacroPlanType.GUST_STALL_TARGET: (
                tuple(sorted(p.gust_cards)),
                (),
                (),
                ("opponent has a Bench target worth trapping",),
                ("force a low-pressure or high-retreat Active target",),
                False,
            ),
            MacroPlanType.HAND_DISRUPTION_STALL: (
                (p.xerosic_machinations,),
                (),
                (),
                ("opponent hand contains more than three cards",),
                ("reduce opponent hand to three cards",),
                False,
            ),
            MacroPlanType.HEAL_OR_ROTATE_WALL: (
                (
                    p.jumbo_ice_cream,
                    p.switch,
                    p.crustle,
                    p.great_tusk,
                ),
                (),
                (),
                ("a damaged wall or useful Bench pivot exists",),
                ("deny a prize and retain another mill turn",),
                False,
            ),
            MacroPlanType.PRIZE_RACE_PIVOT: (
                (p.great_tusk, p.terrakion, *sorted(p.energy_cards)),
                (p.giant_tusk,),
                (),
                ("a knockout changes the race more than one mill turn",),
                ("take a decisive prize or win the game"),
                True,
            ),
            MacroPlanType.PRESERVE_DECK_AND_CHAIN: (
                (),
                (),
                (p.explorers_guidance, p.ultra_ball),
                ("self-deck-out risk is material",),
                ("end safely with next-turn mill resources intact",),
                False,
            ),
        }
        return contracts[plan_type]

    def _feasibility(
        self,
        observation: dict,
        plan_type: MacroPlanType,
        directive: SemanticAction,
    ) -> float:
        if plan_type is MacroPlanType.BASELINE_V1:
            return 1.0
        state = observation["current"]
        owner = int(state["yourIndex"])
        own = state["players"][owner]
        opponent = state["players"][1 - owner]
        own_deck = int(own.get("deckCount", 0))
        opponent_deck = int(opponent.get("deckCount", 0))
        opponent_hand = int(opponent.get("handCount", 0))
        own_field = {
            self._record_id(item) for item in self._field(own)
        }
        active = next((item for item in own.get("active") or () if item), None)
        active_id = self._record_id(active)
        active_energy = len(_value(active, "energies", ()) or ())
        score = 1.0
        if plan_type is MacroPlanType.MILL_FOUR_NOW:
            if opponent_deck <= 0:
                return 0.0
            great_tusks = [
                item
                for item in self._field(own)
                if self._record_id(item) == self.profile.great_tusk
            ]
            can_attach = not bool(state.get("energyAttached"))
            can_fund_attack = any(
                len(_value(item, "energies", ()) or ())
                + int(can_attach)
                >= 2
                for item in great_tusks
            )
            if not can_fund_attack:
                return 0.0
            card_ids = _option_card_ids(directive)
            attack_ids = _option_attack_ids(directive)
            if self.profile.land_collapse in attack_ids:
                if not bool(state.get("supporterPlayed")):
                    return 0.0
                # The public state reports only "a Supporter was played", not
                # whether it carried the Ancient label.
                score = 0.8
            elif self.profile.explorers_guidance in card_ids:
                score = 0.75
            else:
                score = 0.65
            if active_id == self.profile.great_tusk:
                score += 0.1
            if active_energy >= 2:
                score += 0.1
        elif plan_type is MacroPlanType.FIND_ANCIENT_SUPPORTER:
            score = 0.2 if bool(state.get("supporterPlayed")) else 1.0
        elif plan_type is MacroPlanType.HAND_DISRUPTION_STALL:
            score = min(1.0, max(0.0, (opponent_hand - 3) / 3))
        elif plan_type is MacroPlanType.GUST_STALL_TARGET:
            score = 1.0 if any(opponent.get("bench") or ()) else 0.0
        elif plan_type is MacroPlanType.ENABLE_NEUTRALIZATION_WALL:
            visible_ex = any(
                bool(
                    _value(
                        self._cards.get(self._record_id(item) or 0),
                        "ex",
                        False,
                    )
                )
                for item in self._field(opponent)
            )
            score = 1.0 if visible_ex else 0.55
        elif plan_type is MacroPlanType.BUILD_CRUSTLE_WALL:
            score = 1.0 if self.profile.crustle not in own_field else 0.45
        elif plan_type is MacroPlanType.PRESERVE_DECK_AND_CHAIN:
            score = min(1.0, max(0.0, (10 - own_deck) / 7))
        elif plan_type is MacroPlanType.PRIZE_RACE_PIVOT:
            score = 0.45
            if _option_attack_ids(directive) - {self.profile.land_collapse}:
                score = 0.9
        return round(min(1.0, max(0.0, score)), 6)

    def _plan(
        self,
        observation: dict,
        candidate: RootCandidate,
        plan_type: MacroPlanType,
        *,
        extra_sources: tuple[str, ...] = (),
    ) -> PlanOption:
        directive = semantic_action(observation, candidate.action)
        root_tags = self.tags(directive)
        desired_tags = tuple(
            sorted({*root_tags, *_PLAN_TAGS[plan_type]})
        )
        first = directive.options[0] if directive.options else None
        (
            preferred_cards,
            preferred_attacks,
            preserve_cards,
            preconditions,
            success_conditions,
            require_attack,
        ) = self._plan_contract(plan_type)
        return PlanOption(
            plan_type=plan_type,
            root_action=directive,
            primary_card_id=first.card_id if first is not None else None,
            target_card_id=(
                first.target_card_id if first is not None else None
            ),
            attack_id=first.attack_id if first is not None else None,
            desired_tags=desired_tags,
            preserve_card_ids=preserve_cards,
            require_attack=require_attack,
            termination=MacroTermination.TURN_END,
            maximum_steps=self._maximum_steps,
            source_action=candidate.action,
            sources=tuple(sorted({*candidate.sources, *extra_sources})),
            strategy_version=self.profile.strategy_version,
            preferred_card_ids=preferred_cards,
            preferred_attack_ids=preferred_attacks,
            preconditions=preconditions,
            success_conditions=success_conditions,
            public_signals=self._public_signals(observation),
            feasibility_score=self._feasibility(
                observation, plan_type, directive
            ),
        )

    def _types(
        self,
        directive: SemanticAction,
    ) -> tuple[MacroPlanType, ...]:
        """Map a legal root action to one or more deck-specific intents."""
        p = self.profile
        option_types = {option.option_type for option in directive.options}
        card_ids = _option_card_ids(directive)
        attack_ids = _option_attack_ids(directive)
        target_ids = {
            int(option.target_card_id)
            for option in directive.options
            if option.target_card_id is not None
        }
        plans: list[MacroPlanType] = []

        if p.land_collapse in attack_ids:
            plans.append(MacroPlanType.MILL_FOUR_NOW)
        if attack_ids - {p.land_collapse}:
            plans.append(MacroPlanType.PRIZE_RACE_PIVOT)
        if p.explorers_guidance in card_ids:
            plans.extend(
                (
                    MacroPlanType.MILL_FOUR_NOW,
                    MacroPlanType.PREPARE_NEXT_GREAT_TUSK,
                )
            )
        if p.pokegear in card_ids:
            plans.append(MacroPlanType.FIND_ANCIENT_SUPPORTER)
        if card_ids & {
            p.great_tusk,
            p.fighting_gong,
            p.ultra_ball,
            p.poke_pad,
        }:
            plans.append(MacroPlanType.PREPARE_NEXT_GREAT_TUSK)
        if card_ids & {
            p.dwebble,
            p.crustle,
            p.buddy_buddy_poffin,
            p.ultra_ball,
            p.poke_pad,
        }:
            plans.append(MacroPlanType.BUILD_CRUSTLE_WALL)
        if card_ids & {p.colress_tenacity, p.neutralization_zone}:
            plans.append(MacroPlanType.ENABLE_NEUTRALIZATION_WALL)
        if card_ids & p.gust_cards:
            plans.append(MacroPlanType.GUST_STALL_TARGET)
        if p.xerosic_machinations in card_ids:
            plans.append(MacroPlanType.HAND_DISRUPTION_STALL)
        if card_ids & {p.jumbo_ice_cream, p.switch} or int(
            OptionType.RETREAT
        ) in option_types:
            plans.append(MacroPlanType.HEAL_OR_ROTATE_WALL)
        if card_ids & p.energy_cards:
            if p.great_tusk in target_ids:
                plans.extend(
                    (
                        MacroPlanType.MILL_FOUR_NOW,
                        MacroPlanType.PREPARE_NEXT_GREAT_TUSK,
                    )
                )
            elif p.terrakion in target_ids:
                plans.append(MacroPlanType.PRIZE_RACE_PIVOT)
        if int(OptionType.END) in option_types:
            plans.append(MacroPlanType.PRESERVE_DECK_AND_CHAIN)

        return tuple(dict.fromkeys(plans))

    def generate(
        self,
        observation: dict,
        candidates: Sequence[RootCandidate],
        *,
        baseline_action: Sequence[int],
        maximum: int,
    ) -> list[PlanOption]:
        if maximum <= 0:
            raise ValueError("maximum must be positive")
        baseline = tuple(sorted(int(index) for index in baseline_action))
        original = list(candidates)
        baseline_candidate = next(
            (
                candidate
                for candidate in original
                if candidate.action == baseline
            ),
            RootCandidate(baseline, ("v1_fallback",)),
        )
        baseline_plan = self._plan(
            observation,
            baseline_candidate,
            MacroPlanType.BASELINE_V1,
        )

        # The V1 root is intentionally considered again under explicit intent.
        # This distinguishes "same first action, different continuation" from
        # strict V1 and is essential for detecting joint-action improvements.
        inferred: list[PlanOption] = []
        ordered = [
            baseline_candidate,
            *[
                candidate
                for candidate in original
                if candidate.action != baseline
            ],
        ]
        for candidate in ordered:
            directive = semantic_action(observation, candidate.action)
            for plan_type in self._types(directive):
                plan = self._plan(
                    observation,
                    candidate,
                    plan_type,
                    extra_sources=("libraryout_strategy",),
                )
                if plan.feasibility_score > 0:
                    inferred.append(plan)

        inferred.sort(
            key=lambda plan: (
                -_PLAN_PRIORITY[plan.plan_type],
                -plan.feasibility_score,
                plan.source_action,
                plan.plan_type.value,
            )
        )
        unique: dict[tuple, PlanOption] = {
            baseline_plan.semantic_key(): baseline_plan
        }
        for plan in inferred:
            unique.setdefault(plan.semantic_key(), plan)
            if len(unique) == maximum:
                break
        return list(unique.values())


class HeuristicPlanExecutor:
    """Rank beam candidates using public plan state and Library-Out intent."""

    def __init__(self, plan_generator: MacroPlanGenerator) -> None:
        self._plan_generator = plan_generator

    def rank(
        self,
        observation: dict,
        plan: PlanOption,
        progress: PlanProgress,
        candidates: Sequence[RootCandidate],
    ) -> list[RootCandidate]:
        if plan.plan_type is MacroPlanType.BASELINE_V1:
            return list(candidates)

        p = self._plan_generator.profile

        def score(item: tuple[int, RootCandidate]) -> tuple[float, int]:
            index, candidate = item
            directive = semantic_action(observation, candidate.action)
            value = action_alignment(plan, directive)
            tags = set(self._plan_generator.tags(directive))
            value += 0.75 * len(tags.intersection(plan.desired_tags))
            option_types = {
                option.option_type for option in directive.options
            }
            card_ids = _option_card_ids(directive)
            attack_ids = _option_attack_ids(directive)
            if (
                plan.require_attack
                and not progress.attacked
                and int(OptionType.END) in option_types
            ):
                value -= 8.0
            if plan.plan_type is MacroPlanType.MILL_FOUR_NOW:
                if p.explorers_guidance in card_ids:
                    value += 5.0
                if p.land_collapse in attack_ids:
                    value += 10.0
                if (
                    int(OptionType.ATTACK) in option_types
                    and p.land_collapse not in attack_ids
                ):
                    value -= 7.0
            elif plan.plan_type is MacroPlanType.FIND_ANCIENT_SUPPORTER:
                if p.explorers_guidance in card_ids:
                    value += 8.0
            elif plan.plan_type is MacroPlanType.BUILD_CRUSTLE_WALL:
                if card_ids & {p.dwebble, p.crustle}:
                    value += 5.0
            elif plan.plan_type is MacroPlanType.ENABLE_NEUTRALIZATION_WALL:
                if p.neutralization_zone in card_ids:
                    value += 7.0
            elif plan.plan_type is MacroPlanType.GUST_STALL_TARGET:
                if card_ids & p.gust_cards:
                    value += 5.0
            elif plan.plan_type is MacroPlanType.HAND_DISRUPTION_STALL:
                if p.xerosic_machinations in card_ids:
                    value += 6.0
            elif plan.plan_type is MacroPlanType.PRESERVE_DECK_AND_CHAIN:
                if int(OptionType.END) in option_types:
                    value += 4.0
            return value, -index

        return [
            candidate
            for _, candidate in sorted(
                enumerate(candidates), key=score, reverse=True
            )
        ]
