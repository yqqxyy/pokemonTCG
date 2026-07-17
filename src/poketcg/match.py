"""Local match execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

from .agents.base import Agent
from .engine import OfficialEngine


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Compact metrics from one completed match."""

    game: int
    winner: int
    turns: int
    decisions: int
    elapsed_ms: float
    player0: str
    player1: str
    agent_seed0: int | None
    agent_seed1: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_action(observation: dict, action: list[int]) -> None:
    selection = observation["select"]
    if selection is None:
        raise ValueError("A decision action cannot be used for deck selection.")
    if not isinstance(action, list) or not all(isinstance(index, int) for index in action):
        raise TypeError("Agent action must be list[int].")
    if len(action) != len(set(action)):
        raise ValueError("Agent action contains duplicate option indices.")

    minimum = int(selection["minCount"])
    maximum = int(selection["maxCount"])
    if not minimum <= len(action) <= maximum:
        raise ValueError(f"Action length {len(action)} is outside [{minimum}, {maximum}].")

    option_count = len(selection["option"])
    if any(index < 0 or index >= option_count for index in action):
        raise IndexError(f"Action contains an index outside [0, {option_count}).")


def play_match(
    engine: OfficialEngine,
    deck0: list[int],
    deck1: list[int],
    agent0: Agent,
    agent1: Agent,
    *,
    game: int = 0,
    agent_seed0: int | None = None,
    agent_seed1: int | None = None,
    max_decisions: int = 10_000,
) -> MatchResult:
    """Play one sequential match and always release native simulator memory."""
    started_at = perf_counter()
    observation, start_data = engine.start(deck0, deck1)
    if observation is None:
        raise RuntimeError(
            "Official simulator failed to start "
            f"(errorPlayer={start_data.errorPlayer}, errorType={start_data.errorType})."
        )

    decisions = 0
    try:
        while int(observation["current"]["result"]) == -1:
            if decisions >= max_decisions:
                raise RuntimeError(f"Match exceeded {max_decisions} decisions.")

            player_index = int(observation["current"]["yourIndex"])
            agent = agent0 if player_index == 0 else agent1
            action = agent.choose_action(observation)
            _validate_action(observation, action)
            observation = engine.select(action)
            decisions += 1

        state = observation["current"]
        return MatchResult(
            game=game,
            winner=int(state["result"]),
            turns=int(state["turn"]),
            decisions=decisions,
            elapsed_ms=round((perf_counter() - started_at) * 1_000, 3),
            player0=agent0.name,
            player1=agent1.name,
            agent_seed0=agent_seed0,
            agent_seed1=agent_seed1,
        )
    finally:
        engine.finish()

