from poketcg.agents import HybridPolicyAgent
from poketcg.rl.evaluate_panel import wilson_interval


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_interval(90, 100)

    assert low < 0.9 < high
    assert 0.8 < low < 0.9
    assert 0.9 < high < 1.0


def test_wilson_interval_handles_boundaries() -> None:
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0


class _RecordingPolicy:
    def __init__(self, action: list[int]) -> None:
        self.action = action
        self.calls = 0

    def choose_action(self, observation: dict) -> list[int]:
        self.calls += 1
        return self.action


def test_hybrid_policy_routes_single_and_multiselect_decisions() -> None:
    single = _RecordingPolicy([1])
    multiselect = _RecordingPolicy([0, 2])
    agent = HybridPolicyAgent.__new__(HybridPolicyAgent)
    agent._single_policy = single
    agent._multiselect_policy = multiselect

    assert agent.choose_action({"select": {"minCount": 1, "maxCount": 1}}) == [1]
    assert agent.choose_action({"select": {"minCount": 0, "maxCount": 2}}) == [0, 2]
    assert single.calls == 1
    assert multiselect.calls == 1
