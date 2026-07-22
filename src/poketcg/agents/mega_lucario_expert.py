"""Native, resettable port of the public Mega Lucario ex rule expert."""

from __future__ import annotations

import importlib
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MegaLucarioAttackPlan:
    """Turn-local attack target selected by the public expert heuristic."""

    attacker: int = -1
    target: int = -1
    attack_index: int = -1
    remain_hp: float = -1
    energy: bool = False


class MegaLucarioExpertAgent:
    """Exact native policy port of the public Mega Lucario ex notebook.

    The notebook stores its attack plan in module globals. This implementation
    keeps the same state per instance and exposes ``reset_episode`` so local
    evaluation, search rollouts, and Kaggle runtime cannot leak state between
    games.
    """

    name = "mega-lucario-expert-native"

    MAKUHITA = 673
    HARIYAMA = 674
    LUNATONE = 675
    SOLROCK = 676
    RIOLU = 677
    MEGA_LUCARIO_EX = 678
    DUSK_BALL = 1102
    SWITCH = 1123
    PREMIUM_POWER_PRO = 1141
    FIGHTING_GONG = 1142
    POKE_PAD = 1152
    HERO_CAPE = 1159
    BOSS_ORDERS = 1182
    CARMINE = 1192
    LILLIE_DETERMINATION = 1227
    GRAVITY_MOUNTAIN = 1252
    BASIC_FIGHTING_ENERGY = 6
    MEGA_BRAVE_ATTACK = 983

    def __init__(
        self,
        *,
        card_catalog: Mapping[int, Any],
        deck: list[int] | None = None,
        api_module: Any | None = None,
    ) -> None:
        self._cards = dict(card_catalog)
        self._deck = list(deck) if deck is not None else None
        self._api = api_module or importlib.import_module("cg.api")
        self.reset_episode()

    def reset_episode(self) -> None:
        self._plan = MegaLucarioAttackPlan()
        self._pre_turn = 0
        self._ability_used = False

    def _get_card(
        self,
        observation: Any,
        area: Any,
        index: int,
        player_index: int,
    ) -> Any | None:
        areas = self._api.AreaType
        player = observation.current.players[player_index]
        if area == areas.DECK:
            return observation.select.deck[index]
        if area == areas.HAND:
            return player.hand[index]
        if area == areas.DISCARD:
            return player.discard[index]
        if area == areas.ACTIVE:
            return player.active[index]
        if area == areas.BENCH:
            return player.bench[index]
        if area == areas.PRIZE:
            return player.prize[index]
        if area == areas.STADIUM:
            return observation.current.stadium[index]
        if area == areas.LOOKING:
            return observation.current.looking[index]
        return None

    def _prize_count(self, pokemon: Any) -> int:
        data = self._cards[pokemon.id]
        count = 3 if data.megaEx else 2 if data.ex else 1
        for card in pokemon.energyCards:
            if card.id == 12:
                count -= 1
        for card in pokemon.tools:
            if card.id == 1172 and "Lillie" in data.name:
                count -= 1
        return max(0, count)

    def _pokemon_score(self, pokemon: Any) -> int:
        data = self._cards[pokemon.id]
        score = self._prize_count(pokemon) * 1000
        score += len(pokemon.energies) * 150
        score += len(pokemon.tools) * 100
        if data.stage2:
            score += 250
        elif data.stage1:
            score += 130
        if pokemon.id in {173, 174, 190, 1071}:
            score -= 200
        if pokemon.id == 112 and len(pokemon.energies) >= 1:
            score += 300
        score += pokemon.hp
        return score

    def _energy_score(
        self,
        pokemon: Any,
        *,
        active: bool,
        attacker1: bool,
        attacker2: bool,
    ) -> int:
        energy_count = len(pokemon.energies)
        score = 8000
        if active:
            score += 10
        if pokemon.id in {self.MAKUHITA, self.HARIYAMA}:
            if pokemon.id == self.HARIYAMA:
                score += 1
            if energy_count < 3:
                score += 100
            if attacker2:
                score -= 50
        elif pokemon.id == self.LUNATONE:
            score -= 100
        elif pokemon.id == self.SOLROCK:
            if energy_count < 1:
                score += 20
            else:
                score -= 100
        elif pokemon.id in {self.RIOLU, self.MEGA_LUCARIO_EX}:
            if pokemon.id == self.MEGA_LUCARIO_EX:
                score += 1
            if energy_count < 2:
                score += 100
            if attacker1:
                score -= 50
        return score

    def choose_action(self, observation_dict: dict) -> list[int]:
        observation = self._api.to_observation_class(observation_dict)
        if observation.select is None:
            if self._deck is None:
                raise ValueError("MegaLucarioExpertAgent has no configured deck")
            self.reset_episode()
            return list(self._deck)

        state = observation.current
        select = observation.select
        context = select.context
        my_index = state.yourIndex
        my_state = state.players[my_index]
        opponent_state = state.players[1 - my_index]
        my_prize = len(my_state.prize)

        if self._pre_turn != state.turn:
            self._pre_turn = state.turn
            self._plan = MegaLucarioAttackPlan()
            self._ability_used = False

        field_counts: defaultdict[int, int] = defaultdict(int)
        hand_counts: defaultdict[int, int] = defaultdict(int)
        discard_counts: defaultdict[int, int] = defaultdict(int)
        attacker1 = False
        attacker2 = False
        for card in my_state.active + my_state.bench:
            if card is None:
                continue
            field_counts[card.id] += 1
            if card.id in {self.MAKUHITA, self.HARIYAMA}:
                if len(card.energies) >= 3:
                    attacker2 = True
            elif (
                card.id in {self.RIOLU, self.MEGA_LUCARIO_EX}
                and len(card.energies) >= 2
            ):
                attacker1 = True
        for card in my_state.hand:
            hand_counts[card.id] += 1
        for card in my_state.discard:
            discard_counts[card.id] += 1

        stadium_id = 0
        for card in state.stadium:
            stadium_id = card.id

        areas = self._api.AreaType
        card_types = self._api.CardType
        energy_types = self._api.EnergyType
        contexts = self._api.SelectContext
        option_types = self._api.OptionType
        can_attack = False
        if context == contexts.MAIN:
            can_switch = False
            can_op_switch = False
            can_use_mega_brave = False
            for option in select.option:
                if option.type == option_types.PLAY:
                    card = self._get_card(
                        observation, areas.HAND, option.index, my_index
                    )
                    if card.id == self.SWITCH:
                        can_switch = True
                    elif card.id == self.BOSS_ORDERS:
                        can_op_switch = True
                elif option.type == option_types.EVOLVE:
                    card = self._get_card(
                        observation, areas.HAND, option.index, my_index
                    )
                    if card.id == self.HARIYAMA:
                        can_op_switch = True
                elif option.type == option_types.RETREAT:
                    can_switch = True
                elif option.type == option_types.ATTACK:
                    can_attack = True
                    if option.attackId == self.MEGA_BRAVE_ATTACK:
                        can_use_mega_brave = True

            my_cards = [my_state.active[0], *my_state.bench]
            opponent_cards = [opponent_state.active[0], *opponent_state.bench]
            if state.turn >= 2:
                best_score = -1.0
                for attacker_index, my_pokemon in enumerate(my_cards):
                    if attacker_index != 0 and not can_switch:
                        break
                    for attack_index in range(2):
                        energy_required = 0
                        base_damage = 0
                        base_score = 0
                        if my_pokemon.id == self.MEGA_LUCARIO_EX:
                            if attack_index == 0:
                                energy_required = 1
                                base_damage = 130
                                base_score += 60 * min(
                                    3,
                                    discard_counts[self.BASIC_FIGHTING_ENERGY],
                                )
                            else:
                                energy_required = 2
                                base_damage = 270
                            if my_prize in {2, 3}:
                                base_score -= 500
                        elif attack_index == 1:
                            break
                        elif my_pokemon.id == self.HARIYAMA:
                            energy_required = 3
                            base_damage = 210
                        elif my_pokemon.id == self.MAKUHITA:
                            for option in select.option:
                                if option.type == option_types.EVOLVE:
                                    index = option.inPlayIndex
                                    if option.inPlayArea == areas.BENCH:
                                        index += 1
                                    if index == attacker_index:
                                        break
                            else:
                                break
                            base_score -= 100
                            energy_required = 3
                            base_damage = 210
                        elif my_pokemon.id == self.SOLROCK:
                            if field_counts[self.LUNATONE] >= 1:
                                energy_required = 1
                                base_damage = 70
                        if base_damage <= 0:
                            continue

                        more_energy = False
                        energy_count = len(my_pokemon.energies)
                        if (
                            attack_index == 1
                            and attacker_index == 0
                            and energy_count >= 2
                            and not can_use_mega_brave
                        ):
                            break
                        if energy_count < energy_required:
                            if (
                                hand_counts[self.BASIC_FIGHTING_ENERGY] >= 1
                                and not state.energyAttached
                            ):
                                energy_count += 1
                                if energy_count < energy_required:
                                    continue
                                more_energy = True
                            else:
                                continue

                        for target_index, opponent_pokemon in enumerate(
                            opponent_cards
                        ):
                            if target_index != 0 and not can_op_switch:
                                break
                            damage = base_damage
                            data = self._cards[opponent_pokemon.id]
                            if data.weakness == energy_types.FIGHTING:
                                damage *= 2
                            elif data.resistance == energy_types.FIGHTING:
                                damage -= 30
                            prize = 0
                            score = float(self._pokemon_score(opponent_pokemon))
                            if opponent_pokemon.hp <= damage:
                                prize = self._prize_count(opponent_pokemon)
                            else:
                                score *= damage / opponent_pokemon.hp
                            score += base_score
                            if len(opponent_state.prize) <= prize:
                                score = 50000
                            if attacker_index == 0:
                                score += 220
                            if target_index == 0:
                                score += 300
                            score += energy_count
                            if best_score < score:
                                best_score = score
                                self._plan.attacker = attacker_index
                                self._plan.target = target_index
                                self._plan.attack_index = attack_index
                                self._plan.remain_hp = opponent_pokemon.hp - damage
                                self._plan.energy = more_energy

        scores: list[float] = []
        for option in select.option:
            score: float = 0
            if option.type == option_types.NUMBER:
                score = option.number
            elif option.type == option_types.YES:
                score = 1
            elif option.type == option_types.CARD:
                card = self._get_card(
                    observation,
                    option.area,
                    option.index,
                    option.playerIndex,
                )
                if card is not None:
                    energy_count = 0
                    if isinstance(card, self._api.Pokemon):
                        energy_count = len(card.energies)
                    if context in {contexts.SWITCH, contexts.TO_ACTIVE}:
                        if option.playerIndex == my_index:
                            score += energy_count * 2
                            if option.index == self._plan.attacker - 1:
                                score += 100
                            if card.id == self.MEGA_LUCARIO_EX:
                                score += 8 if my_prize in {2, 3} else 20
                            elif card.id == self.HARIYAMA and energy_count >= 2:
                                score += 15
                            elif card.id == self.MAKUHITA and energy_count >= 2:
                                score += 10
                            elif card.id == self.SOLROCK:
                                score += 5
                            elif card.id == self.RIOLU:
                                score += 4
                        elif option.index == self._plan.target - 1:
                            score += 100
                    elif context == contexts.SETUP_ACTIVE_POKEMON:
                        if card.id == self.SOLROCK:
                            score = 2 if state.firstPlayer == my_index else 4
                        elif card.id == self.RIOLU:
                            score = 3
                        elif card.id == self.MAKUHITA:
                            score = 1
                    elif context == contexts.TO_HAND:
                        score = 200 - hand_counts[card.id] * 100
                        if card.id == self.MAKUHITA:
                            score += -10 if field_counts[card.id] >= 1 else 10
                        elif card.id == self.HARIYAMA:
                            score += 20 if field_counts[self.MAKUHITA] >= 1 else -20
                        elif card.id == self.LUNATONE:
                            score += -250 if field_counts[card.id] >= 1 else 60
                        elif card.id == self.SOLROCK:
                            score += -250 if field_counts[card.id] >= 1 else 50
                        elif card.id == self.RIOLU:
                            count = (
                                field_counts[card.id]
                                + field_counts[self.MEGA_LUCARIO_EX]
                            )
                            if count >= 2:
                                score -= 150
                            elif count >= 1:
                                score -= 3
                            else:
                                score += 40
                        elif card.id == self.MEGA_LUCARIO_EX:
                            score += 40 if field_counts[self.RIOLU] >= 1 else -15
                        elif card.id == self.BASIC_FIGHTING_ENERGY:
                            score += (
                                30
                                if not self._ability_used
                                or not state.energyAttached
                                else -1
                            )
                    elif context == contexts.ATTACH_FROM:
                        score = self._energy_score(
                            card,
                            active=option.area == areas.ACTIVE,
                            attacker1=attacker1,
                            attacker2=attacker2,
                        )
            elif option.type == option_types.PLAY:
                card = self._get_card(
                    observation, areas.HAND, option.index, my_index
                )
                data = self._cards[card.id]
                if data.cardType == card_types.POKEMON:
                    score = 20000
                    if card.id in {self.LUNATONE, self.SOLROCK}:
                        if field_counts[card.id] >= 1:
                            score = -1
                    elif (
                        card.id == self.RIOLU
                        and field_counts[card.id]
                        + field_counts[self.MEGA_LUCARIO_EX]
                        >= 2
                    ):
                        score = -1
                else:
                    score = 10000
                    if card.id == self.SWITCH:
                        score = -1 if self._plan.attacker <= 0 else 6000
                    elif card.id == self.PREMIUM_POWER_PRO:
                        if state.supporterPlayed and self._plan.remain_hp <= 0:
                            score = -1
                        elif not can_attack:
                            if (
                                not state.supporterPlayed
                                and hand_counts[self.CARMINE] > 0
                                and hand_counts[self.LILLIE_DETERMINATION] == 0
                            ):
                                score = 3050
                            else:
                                score = -1
                        else:
                            score = 5000
                    elif card.id == self.BOSS_ORDERS:
                        score = 3200 if self._plan.target >= 1 else -1
                    elif card.id == self.CARMINE:
                        score = 3000
                    elif card.id == self.LILLIE_DETERMINATION:
                        score = 3100
                    elif card.id == self.GRAVITY_MOUNTAIN and stadium_id == 0:
                        score = -1
            elif option.type == option_types.ATTACH:
                card = self._get_card(
                    observation, areas.HAND, option.index, my_index
                )
                pokemon = self._get_card(
                    observation,
                    option.inPlayArea,
                    option.inPlayIndex,
                    my_index,
                )
                if card.id == self.HERO_CAPE:
                    score = 7000
                    if pokemon.id == self.RIOLU:
                        score += 100
                    elif pokemon.id == self.MEGA_LUCARIO_EX:
                        score += 200
                else:
                    score = self._energy_score(
                        pokemon,
                        active=option.inPlayArea == areas.ACTIVE,
                        attacker1=attacker1,
                        attacker2=attacker2,
                    )
                    if option.inPlayArea == areas.ACTIVE:
                        if self._plan.attacker == 0 and self._plan.energy:
                            score += 200
                    elif (
                        self._plan.attacker == 1 + option.inPlayIndex
                        and self._plan.energy
                    ):
                        score += 200
            elif option.type == option_types.EVOLVE:
                pokemon = self._get_card(
                    observation,
                    option.inPlayArea,
                    option.inPlayIndex,
                    my_index,
                )
                score = 9000 + len(pokemon.energies)
                if pokemon.id == self.MAKUHITA and self._plan.target == 0:
                    score = -1
            elif option.type == option_types.ABILITY:
                card = self._get_card(
                    observation, option.area, option.index, my_index
                )
                score = 1 if card.id == 1267 else 30000
            elif option.type == option_types.RETREAT:
                score = 2000 if self._plan.attacker >= 1 else -1
            elif option.type == option_types.ATTACK:
                score = 1000
                if self._plan.attack_index == 1:
                    if option.attackId == self.MEGA_BRAVE_ATTACK:
                        score += 100
                elif option.attackId != self.MEGA_BRAVE_ATTACK:
                    score += 100
            scores.append(score)

        descending = [
            index
            for index, _ in sorted(
                enumerate(scores),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        if context == contexts.MAIN:
            option = select.option[descending[0]]
            if option.type == option_types.ABILITY:
                card = self._get_card(
                    observation, option.area, option.index, my_index
                )
                if card.id == self.LUNATONE:
                    self._ability_used = True
        return descending[: select.maxCount]
