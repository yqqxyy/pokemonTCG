import pytest

from poketcg.rl.coverage_diagnostics import (
    CoverageRecord,
    classify_selection,
    summarize_coverage,
)


def _selection(option_count: int, minimum: int, maximum: int) -> dict:
    return {
        "option": list(range(option_count)),
        "minCount": minimum,
        "maxCount": maximum,
    }


@pytest.mark.parametrize(
    ("option_count", "minimum", "maximum", "classification", "valid_action_count"),
    [
        (1, 1, 1, "forced", 1),
        (3, 1, 1, "neural", 3),
        (3, 2, 2, "resolver", 3),
        (3, 0, 1, "resolver", 4),
        (3, 3, 3, "forced", 1),
        (0, 0, 0, "forced", 1),
    ],
)
def test_classify_selection(
    option_count: int,
    minimum: int,
    maximum: int,
    classification: str,
    valid_action_count: int,
) -> None:
    shape = classify_selection(_selection(option_count, minimum, maximum))

    assert shape.classification == classification
    assert shape.valid_action_count == valid_action_count


def test_classify_selection_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="bounds"):
        classify_selection(_selection(2, 1, 3))


def _record(
    *,
    game: int,
    classification: str,
    option_count: int,
    valid_action_count: int,
    log_types: list[int],
    outcome: float,
) -> CoverageRecord:
    return CoverageRecord(
        game=game,
        player=0,
        context=1,
        context_name="TEST",
        select_type=2,
        select_type_name="CARD",
        classification=classification,
        option_count=option_count,
        minimum=1,
        maximum=1,
        valid_action_count=valid_action_count,
        log_types=log_types,
        outcome=outcome,
    )


def test_summarize_coverage_excludes_forced_decisions_from_strategic_rate() -> None:
    records = [
        _record(
            game=0,
            classification="forced",
            option_count=1,
            valid_action_count=1,
            log_types=[],
            outcome=1.0,
        ),
        _record(
            game=0,
            classification="neural",
            option_count=3,
            valid_action_count=3,
            log_types=[1, 2],
            outcome=1.0,
        ),
        _record(
            game=1,
            classification="resolver",
            option_count=3,
            valid_action_count=4,
            log_types=[2],
            outcome=-1.0,
        ),
    ]

    summary = summarize_coverage(records)

    assert summary["games_observed"] == 2
    assert summary["forced_decisions"] == 1
    assert summary["neural_decisions"] == 1
    assert summary["resolver_decisions"] == 1
    assert summary["raw_neural_coverage"] == pytest.approx(1 / 3, abs=1e-6)
    assert summary["strategic_neural_coverage"] == 0.5
    assert summary["decisions_with_logs_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert summary["log_events"] == 3
    assert summary["observed_game_win_rate"] == 0.5
