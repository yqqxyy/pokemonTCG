from poketcg.agents import RuleAgent
from poketcg.agents.rule_agent import AreaType, OptionType, SelectContext


def _observation(options: list[dict], *, context: int, minimum: int = 1, maximum: int = 1):
    return {
        "select": {
            "context": context,
            "minCount": minimum,
            "maxCount": maximum,
            "option": options,
            "deck": None,
        },
        "current": {
            "yourIndex": 0,
            "players": [
                {"hand": [{"id": 10}], "active": [], "bench": [], "discard": [], "prize": []},
                {"hand": None, "active": [], "bench": [], "discard": [], "prize": []},
            ],
        },
    }


def test_rule_prioritizes_development_before_attack_and_end() -> None:
    observation = _observation(
        [
            {"type": OptionType.END},
            {"type": OptionType.ATTACK, "attackId": 5},
            {"type": OptionType.ATTACH},
            {"type": OptionType.ABILITY},
        ],
        context=SelectContext.MAIN,
    )
    agent = RuleAgent(attack_catalog={5: {"damage": 200, "energies": []}}, seed=1)

    assert agent.choose_action(observation) == [3]


def test_rule_chooses_highest_base_damage_attack() -> None:
    observation = _observation(
        [
            {"type": OptionType.ATTACK, "attackId": 1},
            {"type": OptionType.ATTACK, "attackId": 2},
        ],
        context=SelectContext.ATTACK,
    )
    agent = RuleAgent(
        attack_catalog={
            1: {"damage": 30, "energies": [1]},
            2: {"damage": 90, "energies": [1, 2]},
        },
        seed=1,
    )

    assert agent.choose_action(observation) == [1]


def test_rule_prefers_yes() -> None:
    observation = _observation(
        [{"type": OptionType.NO}, {"type": OptionType.YES}],
        context=SelectContext.ACTIVATE,
    )

    assert RuleAgent(seed=1).choose_action(observation) == [1]


def test_rule_chooses_high_hp_setup_pokemon() -> None:
    observation = _observation(
        [
            {"type": OptionType.CARD, "area": AreaType.HAND, "index": 0, "playerIndex": 0},
            {"type": OptionType.CARD, "area": AreaType.HAND, "index": 1, "playerIndex": 0},
        ],
        context=SelectContext.SETUP_ACTIVE_POKEMON,
    )
    observation["current"]["players"][0]["hand"] = [{"id": 10}, {"id": 11}]
    agent = RuleAgent(card_catalog={10: {"hp": 60}, 11: {"hp": 120}}, seed=1)

    assert agent.choose_action(observation) == [1]


def test_rule_uses_minimum_count_when_discarding() -> None:
    observation = _observation(
        [{"type": OptionType.CARD} for _ in range(3)],
        context=SelectContext.DISCARD,
        minimum=1,
        maximum=3,
    )

    assert len(RuleAgent(seed=1).choose_action(observation)) == 1


def test_rule_exposes_equal_scores_before_random_tie_breaking() -> None:
    observation = _observation(
        [{"type": OptionType.CARD}, {"type": OptionType.CARD}],
        context=7,
    )

    assert RuleAgent(seed=1).score_options(observation) == [0.0, 0.0]
