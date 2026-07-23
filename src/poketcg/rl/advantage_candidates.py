"""Candidate-set construction shared by paired collection and online reranking."""

from __future__ import annotations

from .paired_rollout import RootCandidate


def root_candidates(
    baseline_action: list[int],
    rule_scores: list[float],
    model_logits: list[float],
    option_types: list[int],
    *,
    maximum: int = 3,
) -> list[RootCandidate]:
    """Combine rule, neural, and action-type-diverse exact-one candidates."""
    if len(baseline_action) != 1:
        raise ValueError("root candidate generation currently requires exact-one actions")
    option_count = len(rule_scores)
    if len(model_logits) != option_count or len(option_types) != option_count:
        raise ValueError("candidate feature vectors must match the option count")
    if maximum < 2:
        raise ValueError("maximum candidates must be at least two")
    baseline = baseline_action[0]
    sources: dict[int, set[str]] = {baseline: {"rule_choice"}}
    rule_order = sorted(
        range(option_count), key=lambda index: (rule_scores[index], -index), reverse=True
    )
    model_order = sorted(
        range(option_count), key=lambda index: (model_logits[index], -index), reverse=True
    )
    for index in rule_order[:3]:
        sources.setdefault(index, set()).add("rule_top")
    for index in model_order[:2]:
        sources.setdefault(index, set()).add("round0_top")

    ordered = [baseline]
    ordered.extend(index for index in model_order[:2] if index != baseline)
    ordered.extend(index for index in rule_order[:3] if index not in ordered)
    baseline_type = option_types[baseline]
    diverse = next(
        (
            index
            for index in rule_order
            if index != baseline and option_types[index] != baseline_type
        ),
        None,
    )
    if diverse is not None:
        sources.setdefault(diverse, set()).add("diverse_type")
        if diverse not in ordered[:maximum]:
            ordered.insert(min(2, len(ordered)), diverse)
    for index in rule_order:
        if index not in sources:
            sources[index] = {"rule_fallback"}
        if index not in ordered:
            ordered.append(index)

    unique = []
    for index in ordered:
        if index not in unique:
            unique.append(index)
        if len(unique) == maximum:
            break
    return [
        RootCandidate(action=(index,), sources=tuple(sorted(sources[index])))
        for index in unique
    ]
