"""Explicit deck-aware tactical planning and neural-policy integration."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from poketcg.rl.action_space import deterministic_subset, sample_subset

from .bc_agent import BCPolicyAgent, PolicyValueEvaluation
from .rule_agent import AreaType, OptionType, RuleAgent, SelectContext, _value


@dataclass(frozen=True, slots=True)
class DeckTacticalProfile:
    """Card identities needed to turn generic tactical features into a deck plan."""

    name: str
    energy_id: int
    primary_basic_id: int
    primary_attacker_id: int
    secondary_basic_id: int
    secondary_attacker_id: int
    draw_engine_id: int
    bridge_attacker_id: int
    switch_id: int
    damage_boost_id: int
    energy_search_id: int
    pokemon_search_ids: frozenset[int]
    hp_tool_id: int
    gust_id: int
    draw_supporter_ids: frozenset[int]
    stadium_id: int
    primary_attack_ids: frozenset[int]
    bridge_attack_ids: frozenset[int]
    energy_acceleration_attack_ids: frozenset[int]


MEGA_LUCARIO_PROFILE = DeckTacticalProfile(
    name="mega-lucario-ex",
    energy_id=6,
    primary_basic_id=677,
    primary_attacker_id=678,
    secondary_basic_id=673,
    secondary_attacker_id=674,
    draw_engine_id=675,
    bridge_attacker_id=676,
    switch_id=1123,
    damage_boost_id=1141,
    energy_search_id=1142,
    pokemon_search_ids=frozenset({1102, 1142, 1152}),
    hp_tool_id=1159,
    gust_id=1182,
    draw_supporter_ids=frozenset({1192, 1227}),
    stadium_id=1252,
    primary_attack_ids=frozenset({982, 983}),
    bridge_attack_ids=frozenset({980}),
    energy_acceleration_attack_ids=frozenset({982}),
)


@dataclass(frozen=True, slots=True)
class TacticalPlan:
    """One-turn attack plan shared by MAIN and its follow-up selections."""

    turn: int
    attacker_serial: int | None
    attacker_card_id: int | None
    attacker_area: int | None
    attacker_index: int | None
    evolve_card_id: int | None
    attack_id: int | None
    target_serial: int | None
    target_card_id: int | None
    target_area: int | None
    target_index: int | None
    expected_damage: int
    target_hp: int
    prizes: int
    energy_missing: int
    requires_switch: bool
    requires_gust: bool
    needs_damage_boost: bool
    score: float

    @property
    def knockout(self) -> bool:
        return self.target_hp > 0 and self.expected_damage >= self.target_hp


@dataclass(frozen=True, slots=True)
class PlannerEvaluation:
    scores: tuple[float, ...]
    action: tuple[int, ...]
    confidence: float
    handled: bool
    plan: TacticalPlan | None


@dataclass(frozen=True, slots=True)
class _Attacker:
    pokemon: dict
    area: int
    index: int
    card_id: int
    evolve_card_id: int | None = None


def _card_id(card: dict | None) -> int | None:
    if not card:
        return None
    value = card.get("id")
    return int(value) if value is not None else None


def _serial(card: dict | None) -> int | None:
    if not card:
        return None
    value = card.get("serial")
    return int(value) if value is not None else None


def _enum_context_name(context: int) -> str:
    try:
        return SelectContext(context).name
    except ValueError:
        return str(context)


def _field(player: dict) -> list[tuple[int, int, dict]]:
    result: list[tuple[int, int, dict]] = []
    for index, pokemon in enumerate(player.get("active") or []):
        if pokemon:
            result.append((int(AreaType.ACTIVE), index, pokemon))
    for index, pokemon in enumerate(player.get("bench") or []):
        if pokemon:
            result.append((int(AreaType.BENCH), index, pokemon))
    return result


class TacticalPlannerAgent:
    """Plan attacks explicitly, then score every legal option against that plan."""

    name = "tactical-planner"

    def __init__(
        self,
        *,
        card_catalog: Mapping[int, Any],
        attack_catalog: Mapping[int, Any],
        profile: DeckTacticalProfile = MEGA_LUCARIO_PROFILE,
        seed: int | None = None,
    ) -> None:
        self._cards = card_catalog
        self._attacks = attack_catalog
        self.profile = profile
        self._fallback = RuleAgent(
            card_catalog=card_catalog,
            attack_catalog=attack_catalog,
            seed=seed,
        )
        self._rng = random.Random(seed)
        self._plan: TacticalPlan | None = None
        self._turn = -1
        self._decisions = 0
        self._handled = 0
        self._high_confidence = 0

    def reset_episode(self) -> None:
        self._plan = None
        self._turn = -1

    def metrics(self) -> dict[str, float | int]:
        decisions = max(self._decisions, 1)
        return {
            "decisions": self._decisions,
            "handled": self._handled,
            "handled_rate": round(self._handled / decisions, 6),
            "high_confidence": self._high_confidence,
            "high_confidence_rate": round(self._high_confidence / decisions, 6),
        }

    def routing_reason(
        self,
        observation: dict,
        evaluation: PlannerEvaluation,
        *,
        threshold: float,
        allow_confidence: bool = False,
    ) -> str | None:
        """Return why this decision is safe enough to override the neural policy."""
        if not evaluation.handled:
            return None
        context = int(observation["select"]["context"])
        priority_contexts = {
            int(SelectContext.SETUP_ACTIVE_POKEMON),
            int(SelectContext.SETUP_BENCH_POKEMON),
            int(SelectContext.TO_HAND),
            int(SelectContext.ATTACH_FROM),
            int(SelectContext.ATTACH_TO),
        }
        if context in priority_contexts:
            return f"context:{_enum_context_name(context)}"
        if context == int(SelectContext.MAIN) and len(evaluation.action) == 1:
            option = observation["select"]["option"][evaluation.action[0]]
            if (
                int(option["type"]) == int(OptionType.ABILITY)
                and self._option_card_id(observation, option) == self.profile.draw_engine_id
            ):
                return "main:draw-engine-ability"
        if allow_confidence and evaluation.confidence >= threshold:
            return "confidence"
        return None

    def choose_action(self, observation: dict) -> list[int]:
        evaluation = self.evaluate(observation, persist=True)
        self._decisions += 1
        self._handled += int(evaluation.handled)
        self._high_confidence += int(evaluation.confidence >= 0.75)
        return list(evaluation.action)

    def score_options(self, observation: dict) -> list[float]:
        return list(self.evaluate(observation, persist=True).scores)

    def evaluate(self, observation: dict, *, persist: bool) -> PlannerEvaluation:
        selection = observation.get("select")
        if selection is None:
            raise ValueError("TacticalPlannerAgent received deck selection")
        state = observation["current"]
        turn = int(state["turn"])
        context = int(selection["context"])
        if persist and turn != self._turn:
            self._plan = None
            self._turn = turn

        local_plan = self._build_plan(observation)
        if context == int(SelectContext.MAIN):
            plan = local_plan
            if persist:
                self._plan = plan
        elif persist and self._plan is not None and self._plan.turn == turn:
            plan = self._plan
        else:
            plan = local_plan

        fallback = self._fallback.score_options(observation)
        if context == int(SelectContext.MAIN):
            scores, handled, confidence = self._score_main(observation, plan, fallback)
        else:
            scores, handled, confidence = self._score_follow_up(
                observation, plan, fallback
            )
        action = self._select_action(selection, scores)
        return PlannerEvaluation(
            scores=tuple(scores),
            action=tuple(action),
            confidence=confidence,
            handled=handled,
            plan=plan,
        )

    @staticmethod
    def _select_action(selection: dict, scores: list[float]) -> list[int]:
        if not scores:
            raise ValueError("Planner received an empty option list")
        logits = torch.tensor(scores, dtype=torch.float32)
        return deterministic_subset(
            logits,
            int(selection["minCount"]),
            int(selection["maxCount"]),
        )

    def _build_plan(self, observation: dict) -> TacticalPlan | None:
        state = observation["current"]
        selection = observation["select"]
        player_index = int(state["yourIndex"])
        own = state["players"][player_index]
        opponent = state["players"][1 - player_index]
        options = selection["option"]
        hand = own.get("hand") or []
        hand_counts = Counter(_card_id(card) for card in hand if card)
        can_attach = not bool(state.get("energyAttached")) and hand_counts[
            self.profile.energy_id
        ] > 0
        can_switch = any(
            int(option["type"]) == int(OptionType.RETREAT)
            or (
                int(option["type"]) == int(OptionType.PLAY)
                and self._option_card_id(observation, option) == self.profile.switch_id
            )
            for option in options
        )
        can_gust = any(
            (
                int(option["type"]) == int(OptionType.PLAY)
                and self._option_card_id(observation, option) == self.profile.gust_id
            )
            or (
                int(option["type"]) == int(OptionType.EVOLVE)
                and self._option_card_id(observation, option)
                == self.profile.secondary_attacker_id
            )
            for option in options
        )
        damage_boost_available = hand_counts[self.profile.damage_boost_id] > 0
        active_legal_attacks = {
            int(option["attackId"])
            for option in options
            if int(option["type"]) == int(OptionType.ATTACK)
            and option.get("attackId") is not None
        }

        attackers = [
            _Attacker(pokemon, area, index, int(pokemon["id"]))
            for area, index, pokemon in _field(own)
        ]
        for option in options:
            if int(option["type"]) != int(OptionType.EVOLVE):
                continue
            pokemon = self._target_pokemon(observation, option)
            evolved_id = self._option_card_id(observation, option)
            if pokemon is None or evolved_id is None:
                continue
            attackers.append(
                _Attacker(
                    pokemon=pokemon,
                    area=int(option.get("inPlayArea") or 0),
                    index=int(option.get("inPlayIndex") or 0),
                    card_id=evolved_id,
                    evolve_card_id=evolved_id,
                )
            )

        targets = _field(opponent)
        best: TacticalPlan | None = None
        for attacker in attackers:
            if attacker.area != int(AreaType.ACTIVE) and not can_switch:
                continue
            card = self._cards.get(attacker.card_id)
            attack_ids = list(_value(card, "attacks", []) or [])
            energy_count = len(attacker.pokemon.get("energies") or [])
            for raw_attack_id in attack_ids:
                attack_id = int(raw_attack_id)
                attack = self._attacks.get(attack_id)
                if attack is None:
                    continue
                if (
                    attack_id in self.profile.bridge_attack_ids
                    and not self._has_bench_card(own, self.profile.draw_engine_id)
                ):
                    continue
                energy_cost = len(_value(attack, "energies", []) or [])
                energy_missing = max(0, energy_cost - energy_count)
                # The engine only exposes attacks that are legal *right now*.
                # Keep a currently hidden attack when this turn's attachment is
                # exactly what will make it legal; otherwise a missing attack with
                # sufficient energy represents a real restriction (for example a
                # once-per-two-turn attack) and must not be planned.
                if (
                    attacker.area == int(AreaType.ACTIVE)
                    and active_legal_attacks
                    and attack_id not in active_legal_attacks
                    and energy_missing == 0
                ):
                    continue
                if energy_missing > int(can_attach):
                    continue
                base_damage = int(_value(attack, "damage", 0) or 0)
                if base_damage <= 0:
                    continue
                for target_area, target_index, target in targets:
                    requires_gust = target_area == int(AreaType.BENCH)
                    if requires_gust and not can_gust:
                        continue
                    damage = self._damage(attack_id, base_damage, target, boost=0)
                    boosted_damage = self._damage(
                        attack_id,
                        base_damage,
                        target,
                        boost=30 if damage_boost_available else 0,
                    )
                    target_hp = int(target.get("hp") or 0)
                    needs_boost = damage < target_hp <= boosted_damage
                    expected_damage = boosted_damage if needs_boost else damage
                    prizes = self._prize_count(target)
                    score = self._attack_plan_score(
                        state,
                        attacker,
                        target,
                        attack_id=attack_id,
                        expected_damage=expected_damage,
                        prizes=prizes,
                        energy_missing=energy_missing,
                        requires_gust=requires_gust,
                    )
                    candidate = TacticalPlan(
                        turn=int(state["turn"]),
                        attacker_serial=_serial(attacker.pokemon),
                        attacker_card_id=attacker.card_id,
                        attacker_area=attacker.area,
                        attacker_index=attacker.index,
                        evolve_card_id=attacker.evolve_card_id,
                        attack_id=attack_id,
                        target_serial=_serial(target),
                        target_card_id=_card_id(target),
                        target_area=target_area,
                        target_index=target_index,
                        expected_damage=expected_damage,
                        target_hp=target_hp,
                        prizes=prizes,
                        energy_missing=energy_missing,
                        requires_switch=attacker.area != int(AreaType.ACTIVE),
                        requires_gust=requires_gust,
                        needs_damage_boost=needs_boost,
                        score=score,
                    )
                    if best is None or candidate.score > best.score:
                        best = candidate
        return best

    def _attack_plan_score(
        self,
        state: dict,
        attacker: _Attacker,
        target: dict,
        *,
        attack_id: int,
        expected_damage: int,
        prizes: int,
        energy_missing: int,
        requires_gust: bool,
    ) -> float:
        hp = max(int(target.get("hp") or 0), 1)
        knockout = expected_damage >= hp
        opponent_prizes = len(state["players"][1 - int(state["yourIndex"])].get("prize") or [])
        terminal = knockout and prizes >= opponent_prizes
        score = min(expected_damage / hp, 1.0) * 500.0
        score += len(target.get("energies") or []) * 90.0
        score += prizes * (1_400.0 if knockout else 250.0)
        score += 700.0 if knockout else 0.0
        score += 100_000.0 if terminal else 0.0
        score -= energy_missing * 180.0
        score -= 90.0 if attacker.area != int(AreaType.ACTIVE) else 0.0
        score -= 60.0 if requires_gust else 0.0
        if attacker.card_id == self.profile.primary_attacker_id:
            score += 140.0
        elif attacker.card_id == self.profile.bridge_attacker_id:
            score += 60.0
        if attacker.evolve_card_id is not None:
            score -= 30.0
        if attack_id in self.profile.energy_acceleration_attack_ids:
            own = state["players"][int(state["yourIndex"])]
            discard_energy = sum(
                _card_id(card) == self.profile.energy_id
                for card in own.get("discard") or []
            )
            bench_deficit = sum(
                max(0, self._energy_goal(_card_id(pokemon)) - len(pokemon.get("energies") or []))
                for pokemon in own.get("bench") or []
                if pokemon
            )
            score += min(3, discard_energy, bench_deficit) * 220.0
        return score

    def _score_main(
        self,
        observation: dict,
        plan: TacticalPlan | None,
        fallback: list[float],
    ) -> tuple[list[float], bool, float]:
        state = observation["current"]
        own = state["players"][int(state["yourIndex"])]
        hand = own.get("hand") or []
        hand_count = len(hand)
        field_counts = Counter(_card_id(pokemon) for _, _, pokemon in _field(own))
        scores = [score * 0.01 for score in fallback]
        handled = False
        plan_hits = 0

        for index, option in enumerate(observation["select"]["option"]):
            option_type = int(option["type"])
            card_id = self._option_card_id(observation, option)
            score = scores[index]
            if option_type == int(OptionType.ABILITY):
                pokemon = self._option_card(observation, option)
                if _card_id(pokemon) == self.profile.draw_engine_id:
                    # If the engine exposes Lunar Cycle, its Solrock/energy and
                    # once-per-turn preconditions have already been satisfied.
                    score = 9_500.0
                    handled = True
            elif option_type == int(OptionType.EVOLVE):
                target = self._target_pokemon(observation, option)
                if plan and card_id == plan.evolve_card_id and self._matches_serial(
                    target, plan.attacker_serial
                ):
                    score = 9_000.0
                    plan_hits += 1
                elif card_id == self.profile.primary_attacker_id:
                    score = 7_600.0
                elif card_id == self.profile.secondary_attacker_id:
                    score = 7_300.0 + (900.0 if plan and plan.requires_gust else 0.0)
                handled = handled or card_id in {
                    self.profile.primary_attacker_id,
                    self.profile.secondary_attacker_id,
                }
            elif option_type == int(OptionType.ATTACH):
                target = self._target_pokemon(observation, option)
                if card_id == self.profile.hp_tool_id:
                    score = (
                        8_900.0
                        if _card_id(target) == self.profile.primary_attacker_id
                        else 4_000.0 + self._development_target_score(target)
                    )
                    handled = True
                elif plan and plan.energy_missing and self._matches_serial(
                    target, plan.attacker_serial
                ):
                    score = 8_700.0
                    plan_hits += 1
                else:
                    score = 5_500.0 + self._development_target_score(target)
                handled = True
            elif option_type == int(OptionType.RETREAT):
                if plan and plan.requires_switch:
                    score = 8_300.0
                    plan_hits += 1
                    handled = True
                else:
                    score = -500.0
            elif option_type == int(OptionType.PLAY):
                if card_id == self.profile.switch_id:
                    score = 8_400.0 if plan and plan.requires_switch else -500.0
                    plan_hits += int(score > 0)
                elif card_id == self.profile.gust_id:
                    score = 8_200.0 if plan and plan.requires_gust else -500.0
                    plan_hits += int(score > 0)
                elif card_id == self.profile.damage_boost_id:
                    score = 8_100.0 if plan and plan.needs_damage_boost else 2_500.0
                    plan_hits += int(plan is not None and plan.needs_damage_boost)
                elif card_id in self.profile.pokemon_search_ids:
                    score = 6_800.0 + self._setup_need_score(field_counts)
                elif card_id in self.profile.draw_supporter_ids:
                    score = 6_200.0 + max(0, 6 - hand_count) * 120.0
                    if plan and plan.requires_gust:
                        score = 1_000.0
                elif card_id == self.profile.stadium_id:
                    score = 3_200.0 if not state.get("stadium") else -200.0
                elif card_id in {
                    self.profile.primary_basic_id,
                    self.profile.secondary_basic_id,
                    self.profile.draw_engine_id,
                    self.profile.bridge_attacker_id,
                }:
                    score = 6_500.0 + self._bench_card_score(card_id, field_counts)
                handled = handled or card_id is not None
            elif option_type == int(OptionType.ATTACK):
                attack_id = int(option.get("attackId") or 0)
                if plan and attack_id == plan.attack_id and not plan.requires_switch:
                    score = 7_000.0 + (1_500.0 if plan.knockout else 0.0)
                    plan_hits += 1
                else:
                    score = 3_000.0 + self._attack_damage(attack_id)
                handled = True
            elif option_type == int(OptionType.END):
                score = -1_000.0
            scores[index] = score

        confidence = self._confidence(scores, handled, plan_hits > 0)
        return scores, handled, confidence

    def _score_follow_up(
        self,
        observation: dict,
        plan: TacticalPlan | None,
        fallback: list[float],
    ) -> tuple[list[float], bool, float]:
        selection = observation["select"]
        context = int(selection["context"])
        state = observation["current"]
        own = state["players"][int(state["yourIndex"])]
        field_counts = Counter(_card_id(pokemon) for _, _, pokemon in _field(own))
        hand_counts = Counter(_card_id(card) for card in own.get("hand") or [] if card)
        scores = [score * 0.01 for score in fallback]
        handled = False
        plan_match = False

        for index, option in enumerate(selection["option"]):
            option_type = int(option["type"])
            card = self._option_card(observation, option)
            card_id = _card_id(card)
            score = scores[index]
            if context in {
                int(SelectContext.SWITCH),
                int(SelectContext.TO_ACTIVE),
                int(SelectContext.ATTACH_FROM),
            }:
                target = self._target_pokemon(observation, option) or card
                if plan and self._matches_serial(target, plan.attacker_serial):
                    score = 10_000.0
                    plan_match = True
                else:
                    score = self._development_target_score(target)
                handled = True
            elif context == int(SelectContext.SETUP_ACTIVE_POKEMON):
                first = int(state.get("firstPlayer", -1)) == int(state["yourIndex"])
                priorities = {
                    self.profile.primary_basic_id: 900.0 if first else 750.0,
                    self.profile.bridge_attacker_id: 800.0 if first else 950.0,
                    self.profile.secondary_basic_id: 650.0,
                    self.profile.draw_engine_id: 500.0,
                }
                score = priorities.get(card_id, score)
                handled = card_id in priorities
            elif context == int(SelectContext.SETUP_BENCH_POKEMON):
                score = 400.0 + self._bench_card_score(card_id, field_counts)
                handled = card_id is not None
            elif context == int(SelectContext.ATTACH_TO):
                score = 1_000.0 if card_id == self.profile.energy_id else 100.0
                handled = True
            elif context == int(SelectContext.TO_HAND):
                score = self._search_target_score(
                    card_id,
                    field_counts=field_counts,
                    hand_counts=hand_counts,
                    plan=plan,
                )
                handled = card_id is not None
            elif context in {
                int(SelectContext.DISCARD),
                int(SelectContext.DISCARD_ENERGY),
                int(SelectContext.DISCARD_ENERGY_CARD),
                int(SelectContext.DISCARD_CARD_OR_ATTACHED_CARD),
                int(SelectContext.DISCARD_TOOL_CARD),
            }:
                score = -self._resource_value(card_id, field_counts, hand_counts)
                handled = card_id is not None or option_type == int(OptionType.ENERGY)
            elif context == int(SelectContext.ATTACK):
                attack_id = int(option.get("attackId") or 0)
                score = (
                    10_000.0
                    if plan and attack_id == plan.attack_id
                    else self._attack_damage(attack_id)
                )
                plan_match = plan_match or bool(plan and attack_id == plan.attack_id)
                handled = True
            elif context == int(SelectContext.ACTIVATE):
                score = 1_000.0 if option_type == int(OptionType.YES) else 0.0
                handled = option_type in {int(OptionType.YES), int(OptionType.NO)}
            elif option_type == int(OptionType.NUMBER):
                score = float(option.get("number") or 0)
                handled = True
            scores[index] = score

        confidence = self._confidence(scores, handled, plan_match)
        return scores, handled, confidence

    @staticmethod
    def _confidence(scores: list[float], handled: bool, plan_match: bool) -> float:
        if not handled or not scores:
            return 0.0
        if len(scores) == 1:
            return 1.0
        ordered = sorted(scores, reverse=True)
        gap = max(0.0, ordered[0] - ordered[1])
        scale = max(abs(ordered[0]), 1.0)
        margin = 1.0 - math.exp(-gap / scale * 4.0)
        base = 0.72 if plan_match else 0.52
        return min(0.99, base + 0.35 * margin)

    def _damage(self, attack_id: int, damage: int, target: dict, *, boost: int) -> int:
        raw = damage + boost
        if attack_id in self.profile.bridge_attack_ids:
            return raw
        card = self._cards.get(_card_id(target))
        weakness = _value(card, "weakness")
        resistance = _value(card, "resistance")
        if weakness is not None and int(weakness) == 6:
            raw *= 2
        elif resistance is not None and int(resistance) == 6:
            raw = max(0, raw - 30)
        return raw

    def _prize_count(self, pokemon: dict) -> int:
        card = self._cards.get(_card_id(pokemon))
        if bool(_value(card, "megaEx", False)):
            return 3
        if bool(_value(card, "ex", False)):
            return 2
        return 1

    def _option_card_id(self, observation: dict, option: dict) -> int | None:
        return _card_id(self._option_card(observation, option)) or (
            int(option["cardId"]) if option.get("cardId") is not None else None
        )

    def _option_card(self, observation: dict, option: dict) -> dict | None:
        index = option.get("index")
        if index is None:
            return None
        index = int(index)
        area = option.get("area")
        state = observation["current"]
        raw_player_index = option.get("playerIndex")
        player_index = int(
            state["yourIndex"] if raw_player_index is None else raw_player_index
        )
        if area is None and int(option["type"]) in {
            int(OptionType.PLAY),
            int(OptionType.ATTACH),
            int(OptionType.EVOLVE),
        }:
            records = state["players"][player_index].get("hand")
        elif area == int(AreaType.DECK):
            records = observation["select"].get("deck")
        elif area == 12:
            records = state.get("looking")
        elif area == 7:
            records = state.get("stadium")
        else:
            player = state["players"][player_index]
            records = {
                int(AreaType.HAND): player.get("hand"),
                int(AreaType.DISCARD): player.get("discard"),
                int(AreaType.ACTIVE): player.get("active"),
                int(AreaType.BENCH): player.get("bench"),
                int(AreaType.PRIZE): player.get("prize"),
            }.get(area)
        if records is None or not 0 <= index < len(records):
            return None
        return records[index]

    @staticmethod
    def _target_pokemon(observation: dict, option: dict) -> dict | None:
        area = option.get("inPlayArea", option.get("area"))
        index = option.get("inPlayIndex", option.get("index"))
        if area not in {int(AreaType.ACTIVE), int(AreaType.BENCH)} or index is None:
            return None
        state = observation["current"]
        raw_player_index = option.get("playerIndex")
        player_index = int(
            state["yourIndex"] if raw_player_index is None else raw_player_index
        )
        player = state["players"][player_index]
        records = (
            player.get("active")
            if area == int(AreaType.ACTIVE)
            else player.get("bench")
        )
        if records is None or not 0 <= int(index) < len(records):
            return None
        return records[int(index)]

    @staticmethod
    def _matches_serial(card: dict | None, serial: int | None) -> bool:
        return serial is not None and _serial(card) == serial

    @staticmethod
    def _has_bench_card(player: dict, card_id: int) -> bool:
        return any(_card_id(card) == card_id for card in player.get("bench") or [])

    @staticmethod
    def _has_field_card(player: dict, card_id: int) -> bool:
        return any(_card_id(card) == card_id for _, _, card in _field(player))

    @staticmethod
    def _hand_count(player: dict, card_id: int) -> int:
        return sum(_card_id(card) == card_id for card in player.get("hand") or [])

    def _attack_damage(self, attack_id: int) -> float:
        return float(_value(self._attacks.get(attack_id), "damage", 0) or 0)

    def _development_target_score(self, pokemon: dict | None) -> float:
        card_id = _card_id(pokemon)
        energies = len((pokemon or {}).get("energies") or [])
        goal = self._energy_goal(card_id)
        priorities = {
            self.profile.primary_attacker_id: 900.0,
            self.profile.primary_basic_id: 820.0,
            self.profile.secondary_attacker_id: 760.0,
            self.profile.secondary_basic_id: 700.0,
            self.profile.bridge_attacker_id: 450.0,
            self.profile.draw_engine_id: -400.0,
        }
        base = priorities.get(card_id, 200.0)
        if goal <= 0:
            return base
        deficit = goal - energies
        if deficit <= 0:
            return base - 1_200.0 - energies * 50.0
        return base + deficit * 180.0 - energies * 20.0

    def _energy_goal(self, card_id: int | None) -> int:
        if card_id in {self.profile.primary_basic_id, self.profile.primary_attacker_id}:
            return 2
        if card_id in {
            self.profile.secondary_basic_id,
            self.profile.secondary_attacker_id,
        }:
            return 3
        if card_id == self.profile.bridge_attacker_id:
            return 1
        return 0

    def _setup_need_score(self, field_counts: Counter[int | None]) -> float:
        score = 0.0
        if not field_counts[self.profile.primary_basic_id] and not field_counts[
            self.profile.primary_attacker_id
        ]:
            score += 500.0
        if not field_counts[self.profile.draw_engine_id]:
            score += 180.0
        if not field_counts[self.profile.bridge_attacker_id]:
            score += 160.0
        return score

    def _bench_card_score(
        self, card_id: int | None, field_counts: Counter[int | None]
    ) -> float:
        if card_id is None:
            return 0.0
        base = {
            self.profile.primary_basic_id: 600.0,
            self.profile.secondary_basic_id: 400.0,
            self.profile.draw_engine_id: 350.0,
            self.profile.bridge_attacker_id: 380.0,
        }.get(card_id, 100.0)
        copies = field_counts[card_id]
        if card_id in {self.profile.draw_engine_id, self.profile.bridge_attacker_id}:
            return base if copies == 0 else -8_000.0
        if card_id == self.profile.primary_basic_id and (
            copies + field_counts[self.profile.primary_attacker_id]
        ) >= 2:
            return -8_000.0
        if card_id == self.profile.secondary_basic_id and (
            copies + field_counts[self.profile.secondary_attacker_id]
        ) >= 1:
            return -2_000.0
        return base - copies * 450.0

    def _search_target_score(
        self,
        card_id: int | None,
        *,
        field_counts: Counter[int | None],
        hand_counts: Counter[int | None],
        plan: TacticalPlan | None,
    ) -> float:
        if card_id is None:
            return 0.0
        if card_id == self.profile.energy_id:
            return 900.0 if plan and plan.energy_missing else 350.0
        if card_id == self.profile.primary_attacker_id:
            return 900.0 if field_counts[self.profile.primary_basic_id] else 250.0
        if card_id == self.profile.secondary_attacker_id:
            return 700.0 if field_counts[self.profile.secondary_basic_id] else 180.0
        if card_id == self.profile.primary_basic_id:
            return 850.0 - 350.0 * (
                field_counts[card_id] + field_counts[self.profile.primary_attacker_id]
            )
        if card_id in {self.profile.draw_engine_id, self.profile.bridge_attacker_id}:
            other = (
                self.profile.bridge_attacker_id
                if card_id == self.profile.draw_engine_id
                else self.profile.draw_engine_id
            )
            return 700.0 + 150.0 * bool(field_counts[other]) - 500.0 * bool(
                field_counts[card_id]
            )
        return 300.0 - hand_counts[card_id] * 100.0

    def _resource_value(
        self,
        card_id: int | None,
        field_counts: Counter[int | None],
        hand_counts: Counter[int | None],
    ) -> float:
        if card_id is None:
            return 100.0
        if card_id == self.profile.energy_id:
            return 850.0 if hand_counts[card_id] <= 1 else 300.0
        if card_id == self.profile.primary_attacker_id and field_counts[
            self.profile.primary_basic_id
        ]:
            return 900.0
        if card_id == self.profile.secondary_attacker_id and field_counts[
            self.profile.secondary_basic_id
        ]:
            return 750.0
        if card_id in self.profile.draw_supporter_ids:
            return 250.0 + 100.0 * (hand_counts[card_id] <= 1)
        if card_id == self.profile.stadium_id and hand_counts[card_id] > 1:
            return 80.0
        return 400.0 - 100.0 * (hand_counts[card_id] > 1)


class PlannerPolicyAgent:
    """Route high-confidence decisions to a planner and expose blended MCTS priors."""

    name = "planner-policy"

    def __init__(
        self,
        policy: BCPolicyAgent,
        planner: TacticalPlannerAgent,
        *,
        planner_threshold: float = 0.7,
        planner_weight: float = 4.0,
        confidence_routing: bool = True,
        deterministic: bool = True,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= planner_threshold <= 1.0:
            raise ValueError("planner_threshold must be in [0, 1]")
        if planner_weight < 0:
            raise ValueError("planner_weight must be non-negative")
        self._policy = policy
        self._planner = planner
        self._threshold = planner_threshold
        self._weight = planner_weight
        self._confidence_routing = confidence_routing
        self._deterministic = deterministic
        self._rng = random.Random(seed)
        self._planner_routes = 0
        self._policy_routes = 0
        self._route_reasons: Counter[str] = Counter()

    @property
    def action_space_version(self) -> int:
        return self._policy.action_space_version

    def reset_episode(self) -> None:
        self._planner.reset_episode()

    def metrics(self) -> dict[str, Any]:
        total = max(self._planner_routes + self._policy_routes, 1)
        return {
            "planner_routes": self._planner_routes,
            "policy_routes": self._policy_routes,
            "planner_route_rate": round(self._planner_routes / total, 6),
            "planner_route_reasons": dict(sorted(self._route_reasons.items())),
            "planner": self._planner.metrics(),
        }

    def evaluate(self, observation: dict) -> PolicyValueEvaluation:
        neural = self._policy.evaluate(observation)
        planned = self._planner.evaluate(observation, persist=False)
        logits = neural.logits.clone()
        if planned.handled and len(planned.scores) > 1 and self._weight > 0:
            raw = torch.tensor(planned.scores, dtype=logits.dtype)
            normalized = (raw - raw.mean()) / raw.std(unbiased=False).clamp_min(1e-6)
            logits[: len(raw)] += self._weight * planned.confidence * normalized
        return PolicyValueEvaluation(
            logits=logits,
            value=neural.value,
            minimum=neural.minimum,
            maximum=neural.maximum,
        )

    def choose_action(self, observation: dict) -> list[int]:
        planned = self._planner.evaluate(observation, persist=True)
        route_reason = self._planner.routing_reason(
            observation,
            planned,
            threshold=self._threshold,
            allow_confidence=self._confidence_routing,
        )
        if route_reason is not None:
            self._planner_routes += 1
            self._route_reasons[route_reason] += 1
            return list(planned.action)
        self._policy_routes += 1
        if self._deterministic:
            return self._policy.choose_action(observation)
        evaluation = self.evaluate(observation)
        return sample_subset(
            evaluation.logits,
            evaluation.minimum,
            evaluation.maximum,
            rng=self._rng,
        )
