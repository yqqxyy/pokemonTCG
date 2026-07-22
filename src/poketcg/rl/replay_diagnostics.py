"""Summarize Kaggle CABT replay JSON files for matchup failure analysis."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

PLAY = 7
ATTACH = 8
ATTACK = 13
END = 14
GREAT_TUSK = 58
CRUSTLE = 345
LAND_COLLAPSE = 62
EXPLORERS_GUIDANCE = 1185


def _normalized_name(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _player_index(replay: dict[str, Any], team: str) -> int:
    target = _normalized_name(team)
    names = replay.get("info", {}).get("TeamNames", [])
    matches = [index for index, name in enumerate(names) if _normalized_name(str(name)) == target]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one team named {team!r}, found {names!r}")
    return matches[0]


def _visual_frames(replay: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for step in replay.get("steps", []):
        for record in step:
            for frame in record.get("visualize") or []:
                if isinstance(frame, dict):
                    yield frame


def _initial_frame(replay: dict[str, Any]) -> dict[str, Any]:
    for frame in _visual_frames(replay):
        action = frame.get("action")
        current = frame.get("current")
        if (
            isinstance(action, list)
            and len(action) == 2
            and all(isinstance(deck, list) and len(deck) == 60 for deck in action)
            and isinstance(current, dict)
        ):
            return frame
    raise ValueError("Replay does not contain the two initial 60-card decks")


def _final_state(replay: dict[str, Any]) -> dict[str, Any]:
    final: dict[str, Any] | None = None
    for frame in _visual_frames(replay):
        current = frame.get("current")
        if isinstance(current, dict) and current.get("players"):
            final = current
    if final is None:
        raise ValueError("Replay does not contain a visualized game state")
    return final


def _card_names(initial: dict[str, Any]) -> dict[int, str]:
    names: dict[int, str] = {}
    current = initial.get("current") or {}
    for player in current.get("players") or []:
        for zone_name in ("deck", "hand", "active", "bench", "discard", "prize"):
            for card in player.get(zone_name) or []:
                if isinstance(card, dict) and card.get("id") is not None:
                    card_id = int(card["id"])
                    if card.get("name"):
                        names[card_id] = str(card["name"])
                    else:
                        names.setdefault(card_id, str(card_id))
    return names


def _archetype(card_names: Iterable[str]) -> str:
    lowered = {name.casefold() for name in card_names}

    def has(fragment: str) -> bool:
        return any(fragment in name for name in lowered)

    if has("great tusk") and has("crustle"):
        return "great_tusk_crustle"
    if has("mega kangaskhan ex") and has("crustle"):
        return "kangaskhan_crustle"
    if has("cornerstone mask ogerpon ex") and has("crustle"):
        return "ogerpon_crustle"
    if has("crustle"):
        return "crustle_wall"
    if has("mega starmie ex"):
        return "mega_starmie"
    if has("team rocket's spidops"):
        return "rocket_spidops"
    if has("mega lucario ex"):
        return "mega_lucario"
    if has("cynthia's garchomp ex"):
        return "cynthia_garchomp"
    if has("marnie's grimmsnarl ex"):
        return "marnie_grimmsnarl"
    if has("alakazam"):
        return "alakazam"
    if has("archaludon"):
        return "archaludon"
    if has("dragapult"):
        return "dragapult"
    if has("thwackey"):
        return "festival_thwackey"
    return "other"


def _selected_options(
    replay: dict[str, Any], player_index: int
) -> Iterable[tuple[int, dict[str, Any], dict[str, Any]]]:
    """Yield ``(step, option, observation)`` for the player's real decisions."""
    for step_index, step in enumerate(replay.get("steps", [])):
        if player_index >= len(step):
            continue
        record = step[player_index]
        action = record.get("action") or []
        observation = record.get("observation") or {}
        selection = observation.get("select")
        if not isinstance(selection, dict) or len(action) == 60:
            continue
        options = selection.get("option") or []
        for selected_index in action:
            if (
                isinstance(selected_index, int)
                and 0 <= selected_index < len(options)
                and isinstance(options[selected_index], dict)
            ):
                yield step_index, options[selected_index], observation


def _selected_play_card(
    option: dict[str, Any], observation: dict[str, Any], player_index: int
) -> int | None:
    if option.get("type") != PLAY:
        return None
    index = option.get("index")
    players = (observation.get("current") or {}).get("players") or []
    if not isinstance(index, int) or player_index >= len(players):
        return None
    hand = players[player_index].get("hand") or []
    if not 0 <= index < len(hand) or not isinstance(hand[index], dict):
        return None
    card_id = hand[index].get("id")
    return int(card_id) if card_id is not None else None


def _field_count(player: dict[str, Any]) -> int:
    return sum(card is not None for card in (player.get("active") or [])) + sum(
        card is not None for card in (player.get("bench") or [])
    )


def _failure_tags(
    *,
    archetype: str,
    terminal_signals: list[str],
    land_collapse_uses: int,
    first_land_collapse_turn: int | None,
    opponent_final_deck: int,
) -> list[str]:
    tags = list(terminal_signals)
    if "crustle" in archetype:
        tags.append("wall_or_mirror_matchup")
    if land_collapse_uses == 0:
        tags.append("mill_engine_never_online")
    elif first_land_collapse_turn is not None and first_land_collapse_turn >= 5:
        tags.append("slow_mill_setup")
    if 0 < opponent_final_deck <= 5:
        tags.append("near_mill_finish")
    elif opponent_final_deck >= 20:
        tags.append("low_library_pressure")
    return list(dict.fromkeys(tags))


def analyze_replay(path: str | Path, *, team: str = "yqqxyy") -> dict[str, Any]:
    replay_path = Path(path).expanduser().resolve()
    replay = json.loads(replay_path.read_text())
    player_index = _player_index(replay, team)
    opponent_index = 1 - player_index
    team_names = replay.get("info", {}).get("TeamNames", [])
    rewards = replay.get("rewards") or [0, 0]
    initial = _initial_frame(replay)
    names = _card_names(initial)
    decks = initial["action"]
    opponent_deck = Counter(int(card_id) for card_id in decks[opponent_index])
    opponent_card_names = [names.get(card_id, str(card_id)) for card_id in opponent_deck]
    archetype = _archetype(opponent_card_names)

    option_counts: Counter[str] = Counter()
    played_cards: Counter[str] = Counter()
    attack_ids: Counter[int] = Counter()
    first_land_collapse_turn: int | None = None
    decision_steps: set[int] = set()
    for step_index, option, observation in _selected_options(replay, player_index):
        decision_steps.add(step_index)
        option_type = int(option.get("type", -1))
        option_counts[str(option_type)] += 1
        if option_type == ATTACK and option.get("attackId") is not None:
            attack_id = int(option["attackId"])
            attack_ids[attack_id] += 1
            if attack_id == LAND_COLLAPSE and first_land_collapse_turn is None:
                turn = (observation.get("current") or {}).get("turn")
                first_land_collapse_turn = int(turn) if turn is not None else None
        card_id = _selected_play_card(option, observation, player_index)
        if card_id is not None:
            played_cards[names.get(card_id, str(card_id))] += 1

    final = _final_state(replay)
    players = final.get("players") or []
    if len(players) != 2:
        raise ValueError("Final replay state does not contain two players")
    ours = players[player_index]
    opponent = players[opponent_index]
    our_reward = float(rewards[player_index])
    opponent_reward = float(rewards[opponent_index])
    winner = player_index if our_reward > opponent_reward else opponent_index
    loser = opponent_index if winner == player_index else player_index
    terminal_signals: list[str] = []
    if len(players[winner].get("prize") or []) == 0:
        terminal_signals.append("prize_race")
    if _field_count(players[loser]) == 0:
        terminal_signals.append("board_out")
    if int(players[loser].get("deckCount") or 0) == 0:
        terminal_signals.append("deck_out")
    if not terminal_signals:
        terminal_signals.append("terminal_state_not_exposed")

    land_collapse_uses = attack_ids[LAND_COLLAPSE]
    failure_tags = (
        _failure_tags(
            archetype=archetype,
            terminal_signals=terminal_signals,
            land_collapse_uses=land_collapse_uses,
            first_land_collapse_turn=first_land_collapse_turn,
            opponent_final_deck=int(opponent.get("deckCount") or 0),
        )
        if our_reward < opponent_reward
        else []
    )
    deck_cards = [
        {"id": card_id, "name": names.get(card_id, str(card_id)), "count": count}
        for card_id, count in sorted(
            opponent_deck.items(), key=lambda item: (-item[1], names.get(item[0], ""))
        )
    ]
    return {
        "path": str(replay_path),
        "episode_id": replay.get("info", {}).get("EpisodeId", replay.get("id")),
        "seed": replay.get("configuration", {}).get("seed"),
        "team": str(team_names[player_index]).strip(),
        "opponent": str(team_names[opponent_index]).strip(),
        "player_index": player_index,
        "outcome": "win" if our_reward > opponent_reward else "loss",
        "reward": our_reward,
        "opponent_archetype": archetype,
        "opponent_deck": deck_cards,
        "steps": len(replay.get("steps", [])),
        "decisions": len(decision_steps),
        "final_turn": int(final.get("turn") or 0),
        "terminal_signals": terminal_signals,
        "failure_tags": failure_tags,
        "our_final": {
            "deck_count": int(ours.get("deckCount") or 0),
            "prize_count": len(ours.get("prize") or []),
            "field_count": _field_count(ours),
        },
        "opponent_final": {
            "deck_count": int(opponent.get("deckCount") or 0),
            "prize_count": len(opponent.get("prize") or []),
            "field_count": _field_count(opponent),
        },
        "strategy": {
            "land_collapse_uses": land_collapse_uses,
            "first_land_collapse_turn": first_land_collapse_turn,
            "explorers_guidance_plays": played_cards[
                names.get(EXPLORERS_GUIDANCE, "Explorer’s Guidance")
            ],
            "crustle_in_deck": sum(
                count
                for card_id, count in Counter(decks[player_index]).items()
                if card_id == CRUSTLE
            ),
            "selected_option_types": dict(sorted(option_counts.items())),
            "attack_ids": {str(key): value for key, value in sorted(attack_ids.items())},
            "played_cards": dict(played_cards.most_common()),
        },
    }


def _input_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    unique = {path: None for path in files}
    if not unique:
        raise ValueError("No replay JSON files found")
    return list(unique)


def build_report(paths: Iterable[str | Path], *, team: str = "yqqxyy") -> dict[str, Any]:
    episodes = [analyze_replay(path, team=team) for path in _input_files(paths)]
    by_archetype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tag_counts: Counter[str] = Counter()
    for episode in episodes:
        by_archetype[episode["opponent_archetype"]].append(episode)
        tag_counts.update(episode["failure_tags"])

    archetype_summary: dict[str, Any] = {}
    for name, group in sorted(by_archetype.items()):
        wins = sum(episode["outcome"] == "win" for episode in group)
        archetype_summary[name] = {
            "games": len(group),
            "wins": wins,
            "losses": len(group) - wins,
            "win_rate": round(wins / len(group), 6),
            "mean_land_collapse_uses": round(
                mean(episode["strategy"]["land_collapse_uses"] for episode in group),
                3,
            ),
            "mean_opponent_final_deck": round(
                mean(episode["opponent_final"]["deck_count"] for episode in group),
                3,
            ),
        }

    wins = sum(episode["outcome"] == "win" for episode in episodes)
    return {
        "format": "poketcg-kaggle-replay-diagnostics-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "team": team,
        "summary": {
            "games": len(episodes),
            "wins": wins,
            "losses": len(episodes) - wins,
            "win_rate": round(wins / len(episodes), 6),
            "by_archetype": archetype_summary,
            "failure_tags": dict(tag_counts.most_common()),
        },
        "episodes": episodes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--team", default="yqqxyy")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(args.input, team=args.team)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), **report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
