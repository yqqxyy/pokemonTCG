from __future__ import annotations

from collections import Counter

from poketcg.rl.expert_parity import summarize_parity


def test_summarize_parity_reports_exact_and_context_rates() -> None:
    report = summarize_parity(
        decisions=10,
        non_forced_decisions=6,
        exact_matches=9,
        set_matches=10,
        context_counts=Counter({"MAIN": 7, "TO_HAND": 3}),
        context_mismatches=Counter({"MAIN": 1}),
        mismatches=[{"context": "MAIN"}],
    )

    assert report["exact_match_rate"] == 0.9
    assert report["set_match_rate"] == 1.0
    assert report["context_counts"] == {"MAIN": 7, "TO_HAND": 3}
    assert report["context_mismatches"] == {"MAIN": 1}
