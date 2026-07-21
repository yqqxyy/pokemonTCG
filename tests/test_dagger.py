from __future__ import annotations

import random

import pytest

from poketcg.rl.collect_dagger import (
    _finalize_statistics,
    _merge_statistics,
    _new_statistics,
    _update_context,
    _validate_configuration,
    choose_dagger_action,
)


def test_dagger_beta_extremes_choose_expected_actor() -> None:
    student = [0]
    expert = [1]

    assert choose_dagger_action(
        student, expert, beta=1.0, rng=random.Random(1)
    ) == ([1], "expert")
    assert choose_dagger_action(
        student, expert, beta=0.0, rng=random.Random(1)
    ) == ([0], "student")


def test_dagger_rejects_invalid_beta() -> None:
    with pytest.raises(ValueError, match="beta"):
        choose_dagger_action([0], [1], beta=1.1, rng=random.Random(1))
    with pytest.raises(ValueError, match="beta"):
        _validate_configuration(("rule",), beta=-0.1)


def test_dagger_statistics_report_disagreement_and_realized_beta() -> None:
    statistics = _new_statistics(("rule",))
    statistics["games"] = 1
    statistics["examples"] = 2
    statistics["student_decisions"] = 2
    statistics["expert_executions"] = 1
    statistics["student_executions"] = 1
    statistics["disagreements"] = 1
    statistics["student_outcomes"]["win"] = 1
    _update_context(statistics, 0, disagreed=True, execution_source="expert")
    _update_context(statistics, 0, disagreed=False, execution_source="student")

    summary = _finalize_statistics(statistics)

    assert summary["realized_beta"] == 0.5
    assert summary["disagreement_rate"] == 0.5
    assert summary["contexts"]["0"]["realized_beta"] == 0.5
    assert summary["contexts"]["0"]["disagreement_rate"] == 0.5


def test_dagger_worker_statistics_merge_contexts() -> None:
    first = _new_statistics(("rule",))
    second = _new_statistics(("rule",))
    for statistics, source in ((first, "expert"), (second, "student")):
        statistics["games"] = 1
        statistics["examples"] = 1
        statistics["student_decisions"] = 1
        statistics[f"{source}_executions"] = 1
        statistics["disagreements"] = int(source == "expert")
        _update_context(
            statistics,
            0,
            disagreed=source == "expert",
            execution_source=source,
        )

    merged = _merge_statistics((first, second), ("rule",))
    summary = _finalize_statistics(merged)

    assert summary["games"] == 2
    assert summary["examples"] == 2
    assert summary["realized_beta"] == 0.5
    assert summary["disagreement_rate"] == 0.5
