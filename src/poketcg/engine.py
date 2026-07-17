"""Thin adapter around the competition's official native simulator."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

from .paths import resolve_official_dir


class OfficialEngine:
    """Load and expose the official simulator without copying restricted files."""

    def __init__(self, official_dir: str | Path | None = None) -> None:
        self.official_dir = resolve_official_dir(official_dir)
        self._game = self._load_game_module()
        self._api = importlib.import_module("cg.api")

    def _load_game_module(self) -> ModuleType:
        directory = str(self.official_dir)
        if directory not in sys.path:
            sys.path.insert(0, directory)
        return importlib.import_module("cg.game")

    @staticmethod
    def load_deck(path: str | Path) -> list[int]:
        """Read and validate a 60-card ID deck file."""
        deck_path = Path(path).expanduser().resolve()
        values = [line.strip() for line in deck_path.read_text().splitlines() if line.strip()]
        try:
            deck = [int(value) for value in values]
        except ValueError as exc:
            raise ValueError(f"Deck contains a non-integer card ID: {deck_path}") from exc
        if len(deck) != 60:
            raise ValueError(f"Deck must contain exactly 60 card IDs; found {len(deck)}.")
        return deck

    @property
    def sample_deck_path(self) -> Path:
        return self.official_dir / "deck.csv"

    def card_catalog(self) -> dict[int, object]:
        """Return official card metadata keyed by card ID."""
        return {card.cardId: card for card in self._api.all_card_data()}

    def attack_catalog(self) -> dict[int, object]:
        """Return official attack metadata keyed by attack ID."""
        return {attack.attackId: attack for attack in self._api.all_attack()}

    def start(self, deck0: list[int], deck1: list[int]):
        return self._game.battle_start(deck0, deck1)

    def select(self, action: list[int]) -> dict:
        return self._game.battle_select(action)

    def finish(self) -> None:
        self._game.battle_finish()
