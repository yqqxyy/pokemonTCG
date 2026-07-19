import pytest

from poketcg.rl.value_diagnostics import (
    ValueRecord,
    high_confidence_errors,
    prize_phase,
    summarize_calibration,
    summarize_trajectories,
    trajectory_endpoints,
)


def _record(
    predicted_return: float,
    outcome: float,
    *,
    game: int,
    player: int = 0,
    decision: int = 0,
) -> ValueRecord:
    return ValueRecord(
        game=game,
        player=player,
        decision=decision,
        turn=decision + 1,
        turn_action_count=0,
        context=0,
        context_name="UNKNOWN",
        option_count=2,
        chosen_action=[0],
        action_probability=0.8,
        policy_entropy=0.5,
        predicted_return=predicted_return,
        outcome=outcome,
        own_prizes_remaining=6,
        opponent_prizes_remaining=6,
        phase="early",
    )


def test_prize_phase_uses_leading_players_progress() -> None:
    assert prize_phase(0, 0) == "setup"
    assert prize_phase(6, 6) == "early"
    assert prize_phase(5, 6) == "early"
    assert prize_phase(4, 6) == "mid"
    assert prize_phase(6, 3) == "mid"
    assert prize_phase(2, 6) == "late"


def test_perfect_value_predictions_are_perfectly_calibrated() -> None:
    records = [
        _record(-1.0, -1.0, game=0),
        _record(1.0, 1.0, game=1),
    ]

    summary = summarize_calibration(records, bins=4)

    assert summary["mae"] == 0.0
    assert summary["rmse"] == 0.0
    assert summary["brier_score"] == 0.0
    assert summary["pearson_correlation"] == 1.0
    assert summary["explained_variance"] == 1.0
    assert summary["calibration_intercept"] == 0.0
    assert summary["calibration_slope"] == 1.0
    assert summary["expected_calibration_error"] == 0.0


def test_constant_value_predictions_explain_no_outcome_variance() -> None:
    records = [
        _record(0.0, -1.0, game=0),
        _record(0.0, 1.0, game=1),
    ]

    summary = summarize_calibration(records, bins=4)

    assert summary["bias"] == 0.0
    assert summary["mae"] == 1.0
    assert summary["brier_score"] == 0.25
    assert summary["pearson_correlation"] == 0.0
    assert summary["explained_variance"] == 0.0
    assert summary["calibration_slope"] == 0.0


def test_trajectory_summary_measures_change_toward_outcome() -> None:
    records = [
        _record(-0.2, 1.0, game=0, decision=0),
        _record(0.1, 1.0, game=0, decision=1),
        _record(0.7, 1.0, game=0, decision=2),
        _record(0.4, -1.0, game=1, decision=0),
        _record(0.1, -1.0, game=1, decision=1),
        _record(-0.5, -1.0, game=1, decision=2),
    ]

    summary = summarize_trajectories(records)

    assert summary["games"] == 2
    assert summary["mean_net_change_toward_outcome"] == pytest.approx(0.9)
    assert summary["mean_absolute_step_change"] == pytest.approx(0.45)
    assert summary["mean_sign_flips"] == 1.0
    assert summary["by_outcome"]["win"]["mean_final_value"] == 0.7
    assert summary["by_outcome"]["loss"]["mean_final_value"] == -0.5

    initial, final = trajectory_endpoints(records)
    assert [record.predicted_return for record in initial] == [-0.2, 0.4]
    assert [record.predicted_return for record in final] == [0.7, -0.5]


def test_high_confidence_errors_only_include_wrong_sign_predictions() -> None:
    records = [
        _record(0.9, -1.0, game=0),
        _record(-0.8, 1.0, game=1),
        _record(0.7, -1.0, game=2),
        _record(0.95, 1.0, game=3),
    ]

    errors = high_confidence_errors(records, threshold=0.75, limit=10)

    assert [item["game"] for item in errors] == [0, 1]
