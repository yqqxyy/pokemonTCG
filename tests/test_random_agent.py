from poketcg.agents import RandomAgent


def _observation(*, minimum: int = 1, maximum: int = 2, option_count: int = 4) -> dict:
    return {
        "select": {
            "minCount": minimum,
            "maxCount": maximum,
            "option": [{} for _ in range(option_count)],
        }
    }


def test_random_agent_is_seeded_and_legal() -> None:
    first = RandomAgent(seed=7).choose_action(_observation())
    second = RandomAgent(seed=7).choose_action(_observation())

    assert first == second
    assert len(first) == 2
    assert len(set(first)) == 2
    assert all(0 <= index < 4 for index in first)


def test_random_agent_rejects_deck_selection() -> None:
    observation = {"select": None}

    try:
        RandomAgent(seed=7).choose_action(observation)
    except ValueError as exc:
        assert "deck-selection" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

