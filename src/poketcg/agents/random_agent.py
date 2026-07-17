"""Random legal-action baseline."""

from __future__ import annotations

import random


class RandomAgent:
    """Choose a uniformly random legal set of option indices."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose_action(self, observation: dict) -> list[int]:
        selection = observation.get("select")
        if selection is None:
            raise ValueError("RandomAgent received the initial deck-selection observation.")

        option_count = len(selection["option"])
        count = int(selection["maxCount"])
        if count < int(selection["minCount"]) or count > option_count:
            raise ValueError("Simulator returned inconsistent selection bounds.")

        return self._rng.sample(range(option_count), count)

