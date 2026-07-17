from pathlib import Path

import pytest

from poketcg.agents import RandomAgent
from poketcg.engine import OfficialEngine
from poketcg.match import play_match


def test_load_deck_requires_60_cards(tmp_path: Path) -> None:
    deck_path = tmp_path / "deck.csv"
    deck_path.write_text("1\n2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 60"):
        OfficialEngine.load_deck(deck_path)


@pytest.mark.integration
def test_official_engine_completes_random_match() -> None:
    engine = OfficialEngine()
    deck = engine.load_deck(engine.sample_deck_path)

    result = play_match(
        engine,
        deck,
        deck,
        RandomAgent(seed=10),
        RandomAgent(seed=11),
        max_decisions=10_000,
    )

    assert result.winner in {0, 1, 2}
    assert result.decisions > 0
    assert result.turns >= 0

