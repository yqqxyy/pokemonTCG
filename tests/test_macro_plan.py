from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from poketcg.mcts import HiddenStateGuess, SearchPosition
from poketcg.rl.collect_turn_synergy import (
    compact_macro_diagnostic,
    summarize_macro_oracle,
)
from poketcg.rl.macro_data import (
    prepare_macro_executor_data,
    prepare_macro_plan_value_data,
)
from poketcg.rl.macro_oracle import MacroPlanOracleEvaluator
from poketcg.rl.macro_plan import (
    HeuristicPlanExecutor,
    MacroPlanGenerator,
    MacroPlanType,
    PlanOption,
    PlanProgress,
)
from poketcg.rl.paired_rollout import RootCandidate
from poketcg.rl.plan_dagger import ClosedLoopPlanDAggerEvaluator
from poketcg.rl.plan_dagger_data import prepare_plan_dagger_data
from poketcg.rl.semantic_plan import semantic_action


def _observation(state: str, *, result: int = -1) -> dict:
    return {
        "state": state,
        "current": {
            "result": result,
            "yourIndex": 0,
            "turn": 3,
            "supporterPlayed": False,
            "players": [
                {
                    "active": [{"id": 58, "energies": [20, 20]}],
                    "bench": [{"id": 344}],
                    "deckCount": 20,
                    "discard": [],
                    "prize": [None] * 6,
                    "handCount": 5,
                    "hand": [],
                },
                {
                    "active": [{"id": 607, "energies": []}],
                    "bench": [{"id": 344}],
                    "deckCount": 18,
                    "discard": [],
                    "prize": [None] * 6,
                    "handCount": 6,
                    "hand": None,
                },
            ],
        },
        "select": {
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": 14},
                {"type": 7, "cardId": 1122},
            ],
        },
    }


class MacroBackend:
    def begin(
        self, observation: dict, hidden: HiddenStateGuess
    ) -> SearchPosition:
        return SearchPosition(1, _observation("root"))

    def step(self, search_id: int, action: list[int]) -> SearchPosition:
        transitions = {
            (1, (0,)): SearchPosition(
                3, _observation("baseline_loss", result=1)
            ),
            (1, (1,)): SearchPosition(2, _observation("setup")),
            (2, (0,)): SearchPosition(
                4, _observation("followup_loss", result=1)
            ),
            (2, (1,)): SearchPosition(
                5, _observation("joint_win", result=0)
            ),
        }
        return transitions[(search_id, tuple(action))]

    def end(self) -> None:
        return None


class MacroDeterminizer:
    def sample(self, observation: dict) -> HiddenStateGuess:
        return HiddenStateGuess([], [], [], [], [], [], "fake")


@dataclass
class BaselinePolicy:
    def choose_action(self, observation: dict) -> list[int]:
        return [0]


@dataclass
class DeviatingPlanStudent:
    def choose_action(
        self,
        observation: dict,
        plan: PlanOption,
        progress: PlanProgress,
    ) -> list[int]:
        return [0]


def _candidates(observation: dict) -> list[RootCandidate]:
    return [
        RootCandidate((0,), ("baseline",)),
        RootCandidate((1,), ("macro",)),
    ]


def test_plan_option_round_trip_keeps_public_execution_contract() -> None:
    generator = MacroPlanGenerator(
        {
            1122: {
                "skills": [
                    {
                        "text": (
                            "Look at the top 7 cards of your deck. "
                            "Put a Supporter into your hand."
                        )
                    }
                ]
            }
        },
        {},
        maximum_steps=7,
    )
    plan = generator.generate(
        _observation("root"),
        _candidates(_observation("root")),
        baseline_action=[0],
        maximum=2,
    )[1]

    restored = PlanOption.from_dict(plan.to_dict())

    assert restored == plan
    assert restored.plan_id == plan.plan_id
    assert restored.plan_type is MacroPlanType.FIND_ANCIENT_SUPPORTER
    assert restored.strategy_version == "libraryout_v2"
    assert 1185 in restored.preferred_card_ids
    assert "search_deck" in restored.desired_tags
    progress = PlanProgress.start(_observation("root"))
    assert progress.active(_observation("root"), restored)


def test_macro_oracle_records_joint_teacher_trajectory(
    tmp_path: Path,
) -> None:
    evaluator = MacroPlanOracleEvaluator(
        MacroDeterminizer(),
        BaselinePolicy,
        BaselinePolicy,
        _candidates,
        plan_generator=MacroPlanGenerator({}, {}, maximum_steps=4),
        decision_encoder=lambda observation: {
            "state": observation["state"]
        },
        determinizations=2,
        plan_pool_size=2,
        beam_width=4,
        branch_width=2,
        max_plan_steps=4,
        backend=MacroBackend(),
    )

    result = evaluator.evaluate(_observation("root"))

    assert result["diagnostic_kind"] == "macro_plan_oracle_v2_libraryout"
    assert result["candidate_plans"] == 2
    assert result["mean_one_step_gain"] == 0.0
    assert result["mean_full_turn_gain"] == 2.0
    assert result["mean_synergy_gain"] == 2.0
    assert result["joint_rescue_rate"] == 1.0
    sample = result["samples"][0]
    assert sample["best_macro"]["best_trajectory"]["paired_advantage"] == 2.0
    assert [
        step["action"]
        for step in sample["best_macro"]["best_trajectory"]["steps"]
    ] == [[1], [1]]
    assert [
        step["decision"]["state"]
        for step in sample["best_macro"]["best_trajectory"]["steps"]
    ] == ["root", "setup"]

    source = tmp_path / "oracle.jsonl"
    output = tmp_path / "executor.jsonl"
    plan_output = tmp_path / "plan_value.jsonl"
    source.write_text(
        json.dumps(
            {
                "state_id": "state-1",
                "collector_seed": 7,
                "game": 9,
                "decision_index": 3,
                "opponent": "fake",
                "player": 0,
                "turn": 3,
                "selection_reason": "test",
                "decision": {"state": "root"},
                "diagnostic": result,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = prepare_macro_executor_data(source, output)
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]

    assert summary["roots"] == 1
    assert summary["split_groups"] == 1
    assert summary["executor_rows"] == len(rows)
    expected_trajectories = sum(
        len(sample["plans"])
        for sample in result["samples"]
        if "error" not in sample
    )
    expected_rows = sum(
        len(plan["best_trajectory"]["steps"])
        for sample in result["samples"]
        if "error" not in sample
        for plan in sample["plans"]
    )
    all_beam_rows = sum(
        len(trajectory["steps"])
        for sample in result["samples"]
        if "error" not in sample
        for plan in sample["plans"]
        for trajectory in plan["trajectories"]
    )
    all_beam_trajectories = sum(
        len(plan["trajectories"])
        for sample in result["samples"]
        if "error" not in sample
        for plan in sample["plans"]
    )
    assert summary["trajectories"] == expected_trajectories
    assert len(rows) == expected_rows
    assert len(rows) < all_beam_rows
    assert {
        row["trajectory_selection"] for row in rows
    } == {"best_per_plan_per_hidden_world"}
    assert any(row["macro_synergy"] == 2.0 for row in rows)
    assert set(rows[0]["executor_input"]) == {
        "decision",
        "plan",
        "progress",
    }
    assert "determinization_id" not in rows[0]["executor_input"]

    plan_summary = prepare_macro_plan_value_data(source, plan_output)
    plan_rows = [
        json.loads(line)
        for line in plan_output.read_text(encoding="utf-8").splitlines()
    ]

    assert plan_summary["roots"] == 1
    assert plan_summary["plan_rows"] == result["candidate_plans"]
    assert len(plan_rows) == result["candidate_plans"]
    assert set(plan_rows[0]["selector_input"]) == {"decision", "plan"}
    assert "labels" not in plan_rows[0]["selector_input"]
    assert "determinization_id" not in plan_rows[0]["selector_input"]
    assert all(
        row["labels"]["effective_pairs"] == 2 for row in plan_rows
    )
    assert any(
        row["labels"]["oracle_paired_advantage"] == 2.0
        for row in plan_rows
    )

    compact_source = tmp_path / "oracle_compact.jsonl"
    compact_output = tmp_path / "executor_compact.jsonl"
    compact_record = json.loads(source.read_text(encoding="utf-8"))
    compact_diagnostic = compact_macro_diagnostic(
        copy.deepcopy(compact_record["diagnostic"])
    )
    compact_record["diagnostic"] = compact_diagnostic
    compact_source.write_text(
        json.dumps(compact_record) + "\n",
        encoding="utf-8",
    )

    compact_summary = prepare_macro_executor_data(
        compact_source, compact_output
    )
    compact_rows = [
        json.loads(line)
        for line in compact_output.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    oracle_summary = summarize_macro_oracle(compact_source)

    assert all(
        "trajectories" not in plan
        for sample in compact_diagnostic["samples"]
        for plan in sample["plans"]
    )
    assert compact_summary["trajectories"] == summary["trajectories"]
    assert compact_summary["executor_rows"] == summary["executor_rows"]
    assert [
        (row["plan_id"], row["determinization_id"], row["target_action"])
        for row in compact_rows
    ] == [
        (row["plan_id"], row["determinization_id"], row["target_action"])
        for row in rows
    ]
    assert (
        oracle_summary["beam_trajectories"]
        == all_beam_trajectories
    )
    assert oracle_summary["beam_executor_rows"] == all_beam_rows
    assert oracle_summary["teacher_trajectories"] == expected_trajectories


def test_libraryout_generator_keeps_same_root_with_explicit_intent() -> None:
    observation = _observation("strategy")
    observation["current"]["players"][0]["deckCount"] = 6
    observation["select"]["option"] = [
        {"type": 7, "cardId": 1185},
        {"type": 7, "cardId": 1122},
        {"type": 7, "cardId": 1197},
        {"type": 7, "cardId": 1121},
        {"type": 14},
    ]
    candidates = [
        RootCandidate((index,), ("test",))
        for index in range(len(observation["select"]["option"]))
    ]

    plans = MacroPlanGenerator({}, {}).generate(
        observation,
        candidates,
        baseline_action=[0],
        maximum=10,
    )

    assert plans[0].plan_type is MacroPlanType.BASELINE_V1
    same_root = [
        plan
        for plan in plans
        if plan.source_action == (0,)
        and plan.plan_type is MacroPlanType.MILL_FOUR_NOW
    ]
    assert len(same_root) == 1
    assert same_root[0].require_attack
    assert 62 in same_root[0].preferred_attack_ids
    emitted = {plan.plan_type for plan in plans}
    assert {
        MacroPlanType.FIND_ANCIENT_SUPPORTER,
        MacroPlanType.HAND_DISRUPTION_STALL,
        MacroPlanType.PREPARE_NEXT_GREAT_TUSK,
        MacroPlanType.BUILD_CRUSTLE_WALL,
        MacroPlanType.PRESERVE_DECK_AND_CHAIN,
    } <= emitted


def test_libraryout_generator_prunes_useless_hand_disruption() -> None:
    observation = _observation("small-hand")
    observation["current"]["players"][1]["handCount"] = 3
    observation["select"]["option"] = [
        {"type": 14},
        {"type": 7, "cardId": 1197},
    ]

    plans = MacroPlanGenerator({}, {}).generate(
        observation,
        [
            RootCandidate((0,), ("baseline",)),
            RootCandidate((1,), ("xerosic",)),
        ],
        baseline_action=[0],
        maximum=4,
    )

    assert MacroPlanType.HAND_DISRUPTION_STALL not in {
        plan.plan_type for plan in plans
    }


def test_libraryout_generator_does_not_call_setup_only_explorer_mill_four() -> None:
    observation = _observation("no-tusk")
    observation["current"]["players"][0]["active"] = [
        {"id": 607, "energies": [20]}
    ]
    observation["current"]["players"][0]["bench"] = [{"id": 344}]
    observation["select"]["option"] = [
        {"type": 14},
        {"type": 7, "cardId": 1185},
    ]

    plans = MacroPlanGenerator({}, {}).generate(
        observation,
        [
            RootCandidate((0,), ("baseline",)),
            RootCandidate((1,), ("explorer",)),
        ],
        baseline_action=[0],
        maximum=5,
    )

    explorer_plans = [
        plan for plan in plans if plan.source_action == (1,)
    ]
    assert {plan.plan_type for plan in explorer_plans} == {
        MacroPlanType.PREPARE_NEXT_GREAT_TUSK
    }


def test_crustle_plan_guides_ultra_ball_follow_up() -> None:
    root = _observation("ultra-ball")
    root["select"]["option"] = [
        {"type": 14},
        {"type": 7, "cardId": 1121},
    ]
    generator = MacroPlanGenerator({}, {})
    plans = generator.generate(
        root,
        [
            RootCandidate((0,), ("baseline",)),
            RootCandidate((1,), ("ultra-ball",)),
        ],
        baseline_action=[0],
        maximum=6,
    )
    crustle_plan = next(
        plan
        for plan in plans
        if plan.plan_type is MacroPlanType.BUILD_CRUSTLE_WALL
    )
    follow_up = _observation("search-target")
    follow_up["select"]["option"] = [
        {"type": 3, "cardId": 58},
        {"type": 3, "cardId": 345},
    ]
    candidates = [
        RootCandidate((0,), ("great-tusk",)),
        RootCandidate((1,), ("crustle",)),
    ]

    ranked = HeuristicPlanExecutor(generator).rank(
        follow_up,
        crustle_plan,
        PlanProgress.start(root),
        candidates,
    )

    assert ranked[0].action == (1,)


def test_plan_option_loads_legacy_v1_payload() -> None:
    observation = _observation("legacy")
    generator = MacroPlanGenerator({}, {})
    payload = generator.generate(
        observation,
        [RootCandidate((0,), ("baseline",))],
        baseline_action=[0],
        maximum=1,
    )[0].to_dict()
    for field in (
        "strategy_version",
        "preferred_card_ids",
        "preferred_attack_ids",
        "preconditions",
        "success_conditions",
        "public_signals",
        "feasibility_score",
    ):
        payload.pop(field)
    payload["plan_type"] = "search_and_deploy"

    restored = PlanOption.from_dict(payload)

    assert restored.plan_type is MacroPlanType.SEARCH_AND_DEPLOY
    assert restored.strategy_version == "generic_v1"


def test_macro_training_data_rejects_legacy_oracle(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.jsonl"
    output = tmp_path / "executor.jsonl"
    source.write_text(
        json.dumps(
            {
                "state_id": "legacy",
                "collector_seed": 1,
                "game": 1,
                "decision_index": 1,
                "opponent": "fake",
                "player": 0,
                "turn": 3,
                "selection_reason": "test",
                "diagnostic": {
                    "diagnostic_kind": "macro_plan_oracle_v1",
                    "plans": [],
                    "samples": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="accepts only"):
        prepare_macro_executor_data(source, output)


def test_closed_loop_plan_dagger_relabels_student_visited_state(
    tmp_path: Path,
) -> None:
    evaluator = ClosedLoopPlanDAggerEvaluator(
        MacroDeterminizer(),
        BaselinePolicy,
        BaselinePolicy,
        _candidates,
        plan_generator=MacroPlanGenerator({}, {}, maximum_steps=4),
        decision_encoder=lambda observation: {
            "state": observation["state"]
        },
        student_policy=DeviatingPlanStudent(),
        beta=0.0,
        dagger_plan_limit=1,
        determinizations=2,
        plan_pool_size=2,
        beam_width=4,
        branch_width=2,
        max_plan_steps=4,
        backend=MacroBackend(),
    )

    result = evaluator.evaluate(_observation("root"))

    assert result["diagnostic_kind"] == "closed_loop_plan_dagger_v1"
    assert result["visited_states"] == 2
    assert result["realized_beta"] == 0.0
    assert result["semantic_disagreement_rate"] == 1.0
    plan = result["samples"][0]["plans"][0]
    assert plan["labels"][0]["decision"]["state"] == "setup"
    assert plan["labels"][0]["student_action"] == [0]
    assert plan["labels"][0]["target_action"] == [1]
    assert plan["mixed_return"] == -1.0
    assert plan["oracle_return"] == 1.0
    assert plan["oracle_gap"] == 2.0

    raw = tmp_path / "dagger_raw.jsonl"
    output = tmp_path / "dagger_executor.jsonl"
    raw.write_text(
        json.dumps(
            {
                "state_id": "root-1",
                "collector_seed": 17,
                "game": 4,
                "decision_index": 2,
                "opponent": "fake",
                "player": 0,
                "turn": 3,
                "selection_reason": "test",
                "diagnostic": result,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = prepare_plan_dagger_data(raw, output)
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert summary["executor_rows"] == 2
    assert summary["semantic_disagreement_rate"] == 1.0
    assert rows[0]["trajectory_selection"] == (
        "closed_loop_dagger_visited_state"
    )
    assert rows[0]["executor_input"]["progress"]["decisions"] == 1


def test_plan_dagger_treats_root_completed_plan_as_skip() -> None:
    observation = _observation("root")
    evaluator = ClosedLoopPlanDAggerEvaluator(
        MacroDeterminizer(),
        BaselinePolicy,
        BaselinePolicy,
        _candidates,
        plan_generator=MacroPlanGenerator({}, {}, maximum_steps=4),
        decision_encoder=lambda value: {"state": value["state"]},
        student_policy=DeviatingPlanStudent(),
        beta=0.5,
        dagger_plan_limit=1,
        determinizations=1,
        plan_pool_size=2,
        beam_width=2,
        branch_width=2,
        max_plan_steps=4,
        backend=MacroBackend(),
    )
    generated = evaluator._plan_generator.generate(
        observation,
        _candidates(observation),
        baseline_action=[0],
        maximum=2,
    )
    root_completed_plan = replace(
        generated[1],
        root_action=semantic_action(observation, [0]),
        source_action=(0,),
    )
    root = evaluator._backend.begin(
        observation,
        MacroDeterminizer().sample(observation),
    )
    try:
        result = evaluator._roll_in_plan(
            root,
            root_completed_plan,
            root_player=0,
            baseline_return=-1.0,
        )
    finally:
        evaluator._backend.end()

    assert result["skipped_reason"] == "no_post_root_decision"
    assert result["visited_states"] == 0
    assert result["labels"] == []
    assert evaluator._branch_errors == {}
