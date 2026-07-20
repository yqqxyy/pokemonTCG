"""Cardinality-constrained set actions for single- and multi-select decisions."""

from __future__ import annotations

import math
import random

import torch
from torch import Tensor
from torch.distributions import Categorical


def legal_action_set_count(option_count: int, minimum: int, maximum: int) -> int:
    if not 0 <= minimum <= maximum <= option_count:
        raise ValueError("Selection bounds are inconsistent with the option count")
    return sum(math.comb(option_count, count) for count in range(minimum, maximum + 1))


def neural_selection(selection: dict, action_space_version: int) -> bool:
    """Return whether a checkpoint version learns this non-forced selection."""
    option_count = len(selection["option"])
    minimum = int(selection["minCount"])
    maximum = int(selection["maxCount"])
    if legal_action_set_count(option_count, minimum, maximum) <= 1:
        return False
    if action_space_version >= 2:
        return True
    return option_count > 1 and minimum == maximum == 1


def _log_prefix_partitions(logits: Tensor, maximum: int) -> list[list[Tensor]]:
    if logits.ndim != 1:
        raise ValueError("Subset logits must be one-dimensional")
    negative_infinity = logits.new_tensor(float("-inf"))
    zero = logits.new_zeros(())
    rows: list[list[Tensor]] = [[zero, *[negative_infinity] * maximum]]
    for item_index, logit in enumerate(logits, start=1):
        previous = rows[-1]
        current = [zero]
        for count in range(1, maximum + 1):
            excluded = previous[count]
            included = (
                previous[count - 1] + logit
                if count <= item_index
                else negative_infinity
            )
            current.append(torch.logaddexp(excluded, included))
        rows.append(current)
    return rows


def constrained_log_partition(logits: Tensor, minimum: int, maximum: int) -> Tensor:
    """Log sum of exp(set scores) over every cardinality-valid subset."""
    legal_action_set_count(logits.numel(), minimum, maximum)
    rows = _log_prefix_partitions(logits, maximum)
    return torch.logsumexp(torch.stack(rows[-1][minimum : maximum + 1]), dim=0)


def subset_log_probability(
    logits: Tensor,
    selected: Tensor,
    minimum: int,
    maximum: int,
) -> Tensor:
    if selected.shape != logits.shape or selected.dtype != torch.bool:
        raise ValueError("selected must be a boolean mask matching logits")
    selected_count = int(selected.sum())
    if not minimum <= selected_count <= maximum:
        raise ValueError("Selected subset violates cardinality bounds")
    return logits[selected].sum() - constrained_log_partition(logits, minimum, maximum)


def batch_subset_log_probabilities(
    logits: Tensor,
    option_mask: Tensor,
    selected_mask: Tensor,
    minimum: Tensor,
    maximum: Tensor,
) -> Tensor:
    values = []
    for row in range(logits.shape[0]):
        count = int(option_mask[row].sum())
        values.append(
            subset_log_probability(
                logits[row, :count],
                selected_mask[row, :count],
                int(minimum[row]),
                int(maximum[row]),
            )
        )
    return torch.stack(values)


def deterministic_subset(logits: Tensor, minimum: int, maximum: int) -> list[int]:
    """Return the highest-scoring legal subset under the additive set model."""
    legal_action_set_count(logits.numel(), minimum, maximum)
    detached = logits.detach()
    ranked = sorted(
        range(detached.numel()),
        key=lambda index: float(detached[index]),
        reverse=True,
    )
    cumulative = detached.new_zeros(())
    scores: dict[int, float] = {0: 0.0}
    for count, index in enumerate(ranked, start=1):
        cumulative = cumulative + detached[index]
        scores[count] = float(cumulative)
    selected_count = max(range(minimum, maximum + 1), key=scores.__getitem__)
    return sorted(ranked[:selected_count])


def sample_subset(
    logits: Tensor,
    minimum: int,
    maximum: int,
    *,
    rng: random.Random | None = None,
) -> list[int]:
    """Sample exactly from P(S) proportional to exp(sum(logit_i)) over legal sets."""
    legal_action_set_count(logits.numel(), minimum, maximum)
    detached = logits.detach().cpu()
    rows = _log_prefix_partitions(detached, maximum)
    count_logits = torch.stack(rows[-1][minimum : maximum + 1])
    probabilities = count_logits.softmax(dim=0).tolist()
    if rng is None:
        offset = int(torch.multinomial(torch.tensor(probabilities), 1).item())
    else:
        offset = rng.choices(range(len(probabilities)), weights=probabilities, k=1)[0]
    remaining = minimum + offset
    selected: list[int] = []
    for item_count in range(logits.numel(), 0, -1):
        if remaining == 0:
            break
        denominator = rows[item_count][remaining]
        included = detached[item_count - 1] + rows[item_count - 1][remaining - 1]
        probability = float(torch.exp(included - denominator).clamp(0.0, 1.0))
        draw = float(torch.rand(())) if rng is None else rng.random()
        if draw < probability:
            selected.append(item_count - 1)
            remaining -= 1
    if remaining:
        raise RuntimeError("Failed to sample a cardinality-constrained subset")
    return sorted(selected)


def constrained_entropy(logits: Tensor, minimum: int, maximum: int) -> Tensor:
    """Compute the exact entropy of the cardinality-constrained set distribution."""
    legal_action_set_count(logits.numel(), minimum, maximum)
    negative_infinity = logits.new_tensor(float("-inf"))
    zero = logits.new_zeros(())
    log_partitions = [zero, *[negative_infinity] * maximum]
    expected_scores = [zero for _ in range(maximum + 1)]
    for item_count, logit in enumerate(logits, start=1):
        previous_logs = log_partitions
        previous_scores = expected_scores
        log_partitions = [zero, *[negative_infinity] * maximum]
        expected_scores = [zero for _ in range(maximum + 1)]
        for count in range(1, min(item_count, maximum) + 1):
            included_log = previous_logs[count - 1] + logit
            included_score = previous_scores[count - 1] + logit
            if count == item_count:
                log_partitions[count] = included_log
                expected_scores[count] = included_score
                continue
            excluded_log = previous_logs[count]
            weights = torch.stack((excluded_log, included_log)).softmax(dim=0)
            log_partitions[count] = torch.logaddexp(excluded_log, included_log)
            expected_scores[count] = (
                weights[0] * previous_scores[count] + weights[1] * included_score
            )
    valid_logs = torch.stack(log_partitions[minimum : maximum + 1])
    valid_scores = torch.stack(expected_scores[minimum : maximum + 1])
    count_weights = valid_logs.softmax(dim=0)
    log_partition = torch.logsumexp(valid_logs, dim=0)
    expected_score = (count_weights * valid_scores).sum()
    return log_partition - expected_score


def batch_subset_entropies(
    logits: Tensor,
    option_mask: Tensor,
    minimum: Tensor,
    maximum: Tensor,
) -> Tensor:
    """Exact entropy for each cardinality-constrained set distribution in a batch."""
    values = []
    for row in range(logits.shape[0]):
        count = int(option_mask[row].sum())
        valid_logits = logits[row, :count]
        if int(minimum[row]) == int(maximum[row]) == 1:
            values.append(Categorical(logits=valid_logits).entropy())
        else:
            values.append(
                constrained_entropy(
                    valid_logits,
                    int(minimum[row]),
                    int(maximum[row]),
                )
            )
    return torch.stack(values)
