from poketcg.rl.planner_diagnostics import DecisionRecord, summarize_decisions


def _record(
    *,
    context: int = 0,
    context_name: str = "MAIN",
    route: bool,
    planner: bool,
    policy: bool,
) -> DecisionRecord:
    return DecisionRecord(
        player=0,
        context=context,
        context_name=context_name,
        planner_handled=True,
        planner_confidence=0.95,
        routed_to_planner=route,
        route_reason="confidence" if route else None,
        planner_agrees=planner,
        policy_agrees=policy,
        hybrid_agrees=planner if route else policy,
        planner_action="planner",
        policy_action="policy",
        expert_action="expert",
    )


def test_summarize_decisions_measures_helpful_and_harmful_replacements() -> None:
    records = [
        _record(route=True, planner=True, policy=True),
        _record(route=True, planner=True, policy=False),
        _record(route=True, planner=False, policy=True),
        _record(route=False, planner=False, policy=False),
    ]

    summary = summarize_decisions(records)["overall"]

    assert summary["decisions"] == 4
    assert summary["planner_route_rate"] == 0.75
    assert summary["planner_expert_agreement"] == 0.5
    assert summary["policy_expert_agreement"] == 0.5
    assert summary["hybrid_expert_agreement"] == 0.5
    assert summary["routed_planner_only_correct"] == 1
    assert summary["routed_policy_only_correct"] == 1
    assert summary["routed_net_replacements"] == 0


def test_summarize_decisions_splits_contexts_and_seats() -> None:
    records = [
        _record(route=False, planner=True, policy=False),
        DecisionRecord(
            player=1,
            context=35,
            context_name="ATTACK",
            planner_handled=True,
            planner_confidence=0.8,
            routed_to_planner=False,
            route_reason=None,
            planner_agrees=True,
            policy_agrees=False,
            hybrid_agrees=False,
            planner_action="ATTACK:983",
            policy_action="ATTACK:982",
            expert_action="ATTACK:983",
        ),
    ]

    summary = summarize_decisions(records)

    assert summary["contexts"]["MAIN"]["decisions"] == 1
    assert summary["contexts"]["ATTACK"]["decisions"] == 1
    assert summary["seats"]["player0"]["decisions"] == 1
    assert summary["seats"]["player1"]["decisions"] == 1
