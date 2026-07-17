from poketcg.rl.evaluate_panel import wilson_interval


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_interval(90, 100)

    assert low < 0.9 < high
    assert 0.8 < low < 0.9
    assert 0.9 < high < 1.0


def test_wilson_interval_handles_boundaries() -> None:
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0

