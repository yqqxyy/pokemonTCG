"""Common agent interface."""

from __future__ import annotations

from typing import Protocol


class Agent(Protocol):
    """An agent that chooses indices from the simulator's current options."""

    name: str

    def choose_action(self, observation: dict) -> list[int]:
        """Return legal option indices for one simulator decision."""
        ...

