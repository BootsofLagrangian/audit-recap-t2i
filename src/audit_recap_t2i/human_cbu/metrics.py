"""Pure-Python agreement and uncertainty metrics for the human CBU audit.

The helpers in this module deliberately operate on plain sequences and mappings.
They do not depend on the storage layer, pandas, SciPy, or NumPy, and every
returned structure can be serialized with ``json.dumps(..., allow_nan=False)``.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Hashable, Mapping, Sequence
from itertools import combinations
from typing import Any, TypeAlias

Label: TypeAlias = str | int | float | bool
Record: TypeAlias = Mapping[str, Any]
KeySpec: TypeAlias = str | Sequence[str]
Statistic: TypeAlias = Callable[[Sequence[Record]], float | int | None]

DEFAULT_ABSTENTION_LABELS = frozenset({"uncertain", "unjudgeable", "image_unavailable"})


def _stable_sort_key(value: Any) -> tuple[str, str]:
    try:
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError):
        encoded = repr(value)
    return type(value).__name__, encoded


def _validate_json_scalar(value: Any, *, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError(f"{name} must be a finite JSON scalar, got {value!r}")


def _validate_hashable(value: Any, *, name: str) -> None:
    if not isinstance(value, Hashable):
        raise ValueError(f"{name} must be hashable, got {value!r}")


def _validate_weight(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}") from exc
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return weight


def _prepare_pairs(
    labels_a: Sequence[Label],
    labels_b: Sequence[Label],
    weights: Sequence[float] | None,
) -> tuple[list[Label], list[Label], list[float], bool, float]:
    first = list(labels_a)
    second = list(labels_b)
    if len(first) != len(second):
        raise ValueError(f"Label sequences must have the same length, got {len(first)} and {len(second)}")
    supplied = weights is not None
    raw_weights = [1.0] * len(first) if weights is None else list(weights)
    if len(raw_weights) != len(first):
        raise ValueError(f"weights must have length {len(first)}, got {len(raw_weights)}")
    validated = [_validate_weight(weight, name=f"weights[{index}]") for index, weight in enumerate(raw_weights)]
    normalized, _, weight_scale = _stabilize_weights(validated)
    return first, second, normalized, supplied, weight_scale


def _stabilize_weights(weights: Sequence[float]) -> tuple[list[float], float, float]:
    """Keep non-negative relative weights finite without changing their ratios."""

    values = list(weights)
    if not values:
        return [], 0.0, 1.0
    try:
        total = math.fsum(values)
    except OverflowError:
        total = math.inf
    if math.isfinite(total):
        return values, total, 1.0
    scale = max(values)
    if scale == 0:
        return values, 0.0, 1.0
    scaled = [value / scale for value in values]
    return scaled, math.fsum(scaled), scale


def _resolve_categories(
    labels_a: Sequence[Label],
    labels_b: Sequence[Label],
    categories: Sequence[Label] | None,
) -> list[Label]:
    observed = [*labels_a, *labels_b]
    for index, label in enumerate(observed):
        _validate_hashable(label, name=f"observed label {index}")
        _validate_json_scalar(label, name=f"observed label {index}")
    if categories is None:
        resolved = sorted(set(observed), key=_stable_sort_key)
    else:
        resolved = list(categories)
    for index, category in enumerate(resolved):
        _validate_hashable(category, name=f"categories[{index}]")
        _validate_json_scalar(category, name=f"categories[{index}]")
    if len(set(resolved)) != len(resolved):
        raise ValueError("categories must not contain duplicates")
    known = set(resolved)
    unknown = set(observed) - known
    if unknown:
        raise ValueError(f"Observed labels are missing from categories: {sorted(unknown, key=_stable_sort_key)!r}")
    return resolved


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _scaled_matrix(matrix: Sequence[Sequence[float | int]]) -> tuple[list[list[float]], float]:
    maximum = max((float(value) for row in matrix for value in row), default=0.0)
    if maximum == 0:
        return [[0.0 for _ in row] for row in matrix], 0.0
    scaled = [[float(value) / maximum for value in row] for row in matrix]
    return scaled, math.fsum(value for row in scaled for value in row)


def _exact_from_matrix(matrix: Sequence[Sequence[float | int]]) -> float | None:
    normalized, total = _scaled_matrix(matrix)
    if total == 0:
        return None
    return math.fsum(normalized[index][index] for index in range(len(normalized))) / total


def _kappa_from_matrix(matrix: Sequence[Sequence[float | int]]) -> float | None:
    normalized, total = _scaled_matrix(matrix)
    if total == 0:
        return None
    observed = math.fsum(normalized[index][index] for index in range(len(normalized))) / total
    row_proportions = [math.fsum(row) / total for row in normalized]
    column_totals = [
        math.fsum(normalized[row_index][column_index] for row_index in range(len(normalized)))
        for column_index in range(len(normalized))
    ]
    column_proportions = [column / total for column in column_totals]
    expected = math.fsum(row * column for row, column in zip(row_proportions, column_proportions, strict=True))
    denominator = 1.0 - expected
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return None
    return (observed - expected) / denominator


def _gwet_ac1_from_matrix(matrix: Sequence[Sequence[float | int]]) -> float | None:
    category_count = len(matrix)
    normalized, total = _scaled_matrix(matrix)
    if total == 0 or category_count <= 1:
        return None
    observed = math.fsum(normalized[index][index] for index in range(category_count)) / total
    row_totals = [math.fsum(row) for row in normalized]
    column_totals = [
        math.fsum(normalized[row_index][column_index] for row_index in range(category_count))
        for column_index in range(category_count)
    ]
    pooled = [(row + column) / (2.0 * total) for row, column in zip(row_totals, column_totals, strict=True)]
    expected = math.fsum(proportion * (1.0 - proportion) for proportion in pooled) / (category_count - 1)
    denominator = 1.0 - expected
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return None
    return (observed - expected) / denominator


def confusion_matrix(
    labels_a: Sequence[Label],
    labels_b: Sequence[Label],
    *,
    categories: Sequence[Label] | None = None,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Return a nominal confusion matrix with rater A on rows and rater B on columns."""

    first, second, normalized_weights, supplied, weight_scale = _prepare_pairs(labels_a, labels_b, weights)
    resolved = _resolve_categories(first, second, categories)
    positions = {category: index for index, category in enumerate(resolved)}
    if supplied:
        matrix: list[list[float | int]] = [[0.0 for _ in resolved] for _ in resolved]
    else:
        matrix = [[0 for _ in resolved] for _ in resolved]
    for label_a, label_b, weight in zip(first, second, normalized_weights, strict=True):
        row = positions[label_a]
        column = positions[label_b]
        matrix[row][column] += weight if supplied else 1
    return {
        "labels": resolved,
        "matrix": matrix,
        "n_pairs": len(first),
        "weight_total": math.fsum(normalized_weights),
        "weight_scale": weight_scale,
        "weighted": supplied,
    }


def exact_agreement(
    labels_a: Sequence[Label],
    labels_b: Sequence[Label],
    *,
    weights: Sequence[float] | None = None,
) -> float | None:
    """Return exact agreement, or ``None`` when the effective denominator is zero."""

    matrix = confusion_matrix(labels_a, labels_b, weights=weights)["matrix"]
    return _exact_from_matrix(matrix)


def cohen_kappa(
    labels_a: Sequence[Label],
    labels_b: Sequence[Label],
    *,
    categories: Sequence[Label] | None = None,
    weights: Sequence[float] | None = None,
) -> float | None:
    """Return unweighted nominal Cohen's kappa."""

    matrix = confusion_matrix(labels_a, labels_b, categories=categories, weights=weights)["matrix"]
    return _kappa_from_matrix(matrix)


def gwet_ac1(
    labels_a: Sequence[Label],
    labels_b: Sequence[Label],
    *,
    categories: Sequence[Label] | None = None,
    weights: Sequence[float] | None = None,
) -> float | None:
    """Return Gwet's nominal AC1 coefficient."""

    matrix = confusion_matrix(labels_a, labels_b, categories=categories, weights=weights)["matrix"]
    return _gwet_ac1_from_matrix(matrix)


def nominal_agreement(
    labels_a: Sequence[Label],
    labels_b: Sequence[Label],
    *,
    categories: Sequence[Label] | None = None,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Return the confusion matrix and three complementary nominal agreement statistics."""

    matrix_summary = confusion_matrix(labels_a, labels_b, categories=categories, weights=weights)
    matrix = matrix_summary["matrix"]
    return {
        "n_pairs": matrix_summary["n_pairs"],
        "weight_total": matrix_summary["weight_total"],
        "confusion_matrix": matrix_summary,
        "exact_agreement": _exact_from_matrix(matrix),
        "cohen_kappa": _kappa_from_matrix(matrix),
        "gwet_ac1": _gwet_ac1_from_matrix(matrix),
    }


def _raw_rate(count: int, total: int) -> float | None:
    return None if total == 0 else count / total


def _weighted_flag_rate(flags: Sequence[bool], weights: Sequence[float]) -> float | None:
    normalized, denominator, _ = _stabilize_weights(weights)
    if denominator == 0:
        return None
    return math.fsum(weight for flag, weight in zip(flags, normalized, strict=True) if flag) / denominator


def _abstention_counts(labels: Sequence[Label], abstention_labels: Collection[Label]) -> list[dict[str, Any]]:
    counts = Counter(label for label in labels if label in abstention_labels)
    return [{"label": label, "count": counts[label]} for label in sorted(counts, key=_stable_sort_key)]


def binary_agreement(
    labels_a: Sequence[Label],
    labels_b: Sequence[Label],
    *,
    positive_label: Label = "yes",
    negative_label: Label = "no",
    abstention_labels: Collection[Label] = DEFAULT_ABSTENTION_LABELS,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Report binary agreement while retaining explicit abstention rates.

    A pair enters the binary denominator only when both values are either
    ``positive_label`` or ``negative_label``. Values listed in
    ``abstention_labels`` are retained in the returned missingness accounting.
    Any other label raises an error instead of being silently discarded.
    """

    if positive_label == negative_label:
        raise ValueError("positive_label and negative_label must differ")
    abstentions = set(abstention_labels)
    if positive_label in abstentions or negative_label in abstentions:
        raise ValueError("positive_label and negative_label cannot be abstention labels")
    first, second, normalized_weights, supplied, weight_scale = _prepare_pairs(labels_a, labels_b, weights)
    allowed = {positive_label, negative_label} | abstentions
    for index, (label_a, label_b) in enumerate(zip(first, second, strict=True)):
        _validate_json_scalar(label_a, name=f"labels_a[{index}]")
        _validate_json_scalar(label_b, name=f"labels_b[{index}]")
        unknown = [label for label in (label_a, label_b) if label not in allowed]
        if unknown:
            raise ValueError(f"Unexpected non-binary label at pair {index}: {unknown!r}")

    first_abstained = [label in abstentions for label in first]
    second_abstained = [label in abstentions for label in second]
    either_abstained = [label_a or label_b for label_a, label_b in zip(first_abstained, second_abstained, strict=True)]
    both_abstained = [label_a and label_b for label_a, label_b in zip(first_abstained, second_abstained, strict=True)]
    binary_indices = [index for index, excluded in enumerate(either_abstained) if not excluded]
    binary_first = [first[index] for index in binary_indices]
    binary_second = [second[index] for index in binary_indices]
    binary_weights = [normalized_weights[index] for index in binary_indices]
    agreement = nominal_agreement(
        binary_first,
        binary_second,
        categories=[positive_label, negative_label],
        weights=binary_weights if supplied else None,
    )
    matrix = agreement["confusion_matrix"]["matrix"]
    both_positive = float(matrix[0][0])
    positive_negative = float(matrix[0][1])
    negative_positive = float(matrix[1][0])
    both_negative = float(matrix[1][1])
    disagreements = positive_negative + negative_positive
    positive_specific = _safe_ratio(2.0 * both_positive, 2.0 * both_positive + disagreements)
    negative_specific = _safe_ratio(2.0 * both_negative, 2.0 * both_negative + disagreements)

    n_total = len(first)
    n_binary = len(binary_indices)
    total_weight = math.fsum(normalized_weights)
    binary_weight = math.fsum(binary_weights)
    return {
        "positive_label": positive_label,
        "negative_label": negative_label,
        "n_total": n_total,
        "n_binary": n_binary,
        "n_excluded": n_total - n_binary,
        "weight_total": total_weight,
        "weight_binary": binary_weight,
        "weight_excluded": max(0.0, total_weight - binary_weight),
        "weight_scale": weight_scale,
        "binary_coverage": _raw_rate(n_binary, n_total),
        "weighted_binary_coverage": _safe_ratio(binary_weight, total_weight),
        "confusion_matrix": agreement["confusion_matrix"],
        "exact_agreement": agreement["exact_agreement"],
        "cohen_kappa": agreement["cohen_kappa"],
        "gwet_ac1": agreement["gwet_ac1"],
        "positive_agreement": positive_specific,
        "negative_agreement": negative_specific,
        "abstention": {
            "rater_a_count": sum(first_abstained),
            "rater_b_count": sum(second_abstained),
            "either_count": sum(either_abstained),
            "both_count": sum(both_abstained),
            "rater_a_rate": _raw_rate(sum(first_abstained), n_total),
            "rater_b_rate": _raw_rate(sum(second_abstained), n_total),
            "either_rate": _raw_rate(sum(either_abstained), n_total),
            "both_rate": _raw_rate(sum(both_abstained), n_total),
            "weighted_rater_a_rate": _weighted_flag_rate(first_abstained, normalized_weights),
            "weighted_rater_b_rate": _weighted_flag_rate(second_abstained, normalized_weights),
            "weighted_either_rate": _weighted_flag_rate(either_abstained, normalized_weights),
            "weighted_both_rate": _weighted_flag_rate(both_abstained, normalized_weights),
            "rater_a_labels": _abstention_counts(first, abstentions),
            "rater_b_labels": _abstention_counts(second, abstentions),
        },
    }


def consensus_label(
    labels: Sequence[Label | None],
    *,
    abstention_labels: Collection[Label] = DEFAULT_ABSTENTION_LABELS,
    minimum_raters: int = 2,
) -> Label | None:
    """Return a strict-majority consensus, with two-rater disagreements returning ``None``."""

    if minimum_raters < 1:
        raise ValueError("minimum_raters must be at least 1")
    observed = [label for label in labels if label is not None]
    if len(observed) < minimum_raters:
        return None
    for index, label in enumerate(observed):
        _validate_hashable(label, name=f"labels[{index}]")
        _validate_json_scalar(label, name=f"labels[{index}]")
    counts = Counter(observed)
    winner, winner_count = max(counts.items(), key=lambda pair: (pair[1], _stable_sort_key(pair[0])))
    tied = sum(count == winner_count for count in counts.values()) > 1
    if tied or winner_count <= len(observed) / 2 or winner in set(abstention_labels):
        return None
    return winner


def weighted_proportion(
    records: Sequence[Record],
    *,
    outcome_key: str | None = None,
    predicate: Callable[[Record], bool | None] | None = None,
    positive_value: Any = True,
    weight_key: str = "sampling_weight",
) -> float | None:
    """Return a design-weighted proportion using each row's sampling weight.

    Exactly one of ``outcome_key`` and ``predicate`` must be supplied. A
    ``None`` outcome is excluded from the denominator. Zero total eligible
    weight returns ``None``.
    """

    if (outcome_key is None) == (predicate is None):
        raise ValueError("Supply exactly one of outcome_key or predicate")
    eligible_outcomes: list[bool] = []
    eligible_weights: list[float] = []
    for index, record in enumerate(records):
        if weight_key not in record:
            raise KeyError(f"Record {index} is missing weight key {weight_key!r}")
        weight = _validate_weight(record[weight_key], name=f"records[{index}][{weight_key!r}]")
        if predicate is not None:
            outcome = predicate(record)
            if outcome is not None and not isinstance(outcome, bool):
                raise ValueError(f"predicate must return bool or None, got {outcome!r} for record {index}")
        else:
            value = record[outcome_key]  # type: ignore[index]
            outcome = None if value is None else value == positive_value
        if outcome is None:
            continue
        eligible_outcomes.append(outcome)
        eligible_weights.append(weight)
    normalized, denominator, _ = _stabilize_weights(eligible_weights)
    numerator = math.fsum(weight for outcome, weight in zip(eligible_outcomes, normalized, strict=True) if outcome)
    return _safe_ratio(numerator, denominator)


def _record_key(record: Record, key: KeySpec, *, name: str) -> Hashable:
    if isinstance(key, str):
        value: Any = record[key]
    else:
        value = tuple(record[field] for field in key)
    _validate_hashable(value, name=name)
    return value


def _agreement_bundle(
    first: Sequence[Label],
    second: Sequence[Label],
    *,
    categories: Sequence[Label],
    positive_label: Label | None,
    negative_label: Label | None,
    abstention_labels: Collection[Label],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "nominal": nominal_agreement(first, second, categories=categories),
    }
    if (positive_label is None) != (negative_label is None):
        raise ValueError("positive_label and negative_label must either both be set or both be None")
    if positive_label is not None and negative_label is not None:
        result["binary"] = binary_agreement(
            first,
            second,
            positive_label=positive_label,
            negative_label=negative_label,
            abstention_labels=abstention_labels,
        )
    else:
        result["binary"] = None
    return result


def _symmetrized_nominal_agreement(
    first: Sequence[Label],
    second: Sequence[Label],
    *,
    categories: Sequence[Label],
) -> dict[str, Any]:
    """Summarize unordered rater pairs without assigning persistent A/B roles."""

    resolved = _resolve_categories(first, second, categories)
    positions = {category: index for index, category in enumerate(resolved)}
    matrix = [[0.0 for _ in resolved] for _ in resolved]
    for label_a, label_b in zip(first, second, strict=True):
        row = positions[label_a]
        column = positions[label_b]
        if row == column:
            matrix[row][column] += 1.0
        else:
            matrix[row][column] += 0.5
            matrix[column][row] += 0.5
    matrix_summary = {
        "labels": resolved,
        "matrix": matrix,
        "n_pairs": len(first),
        "weight_total": float(len(first)),
        "weight_scale": 1.0,
        "weighted": True,
        "symmetrized_unordered_pairs": True,
    }
    return {
        "n_pairs": len(first),
        "weight_total": float(len(first)),
        "confusion_matrix": matrix_summary,
        "exact_agreement": _exact_from_matrix(matrix),
        "cohen_kappa": None,
        "gwet_ac1": _gwet_ac1_from_matrix(matrix),
        "method": "symmetrized_unordered_rater_pairs",
    }


def _pooled_agreement_bundle(
    first: Sequence[Label],
    second: Sequence[Label],
    *,
    categories: Sequence[Label],
    positive_label: Label | None,
    negative_label: Label | None,
    abstention_labels: Collection[Label],
) -> dict[str, Any]:
    nominal = _symmetrized_nominal_agreement(first, second, categories=categories)
    result: dict[str, Any] = {"nominal": nominal}
    if (positive_label is None) != (negative_label is None):
        raise ValueError("positive_label and negative_label must either both be set or both be None")
    if positive_label is None or negative_label is None:
        result["binary"] = None
        return result

    binary = binary_agreement(
        first,
        second,
        positive_label=positive_label,
        negative_label=negative_label,
        abstention_labels=abstention_labels,
    )
    abstentions = set(abstention_labels)
    eligible = [
        index
        for index, (label_a, label_b) in enumerate(zip(first, second, strict=True))
        if label_a not in abstentions and label_b not in abstentions
    ]
    binary_nominal = _symmetrized_nominal_agreement(
        [first[index] for index in eligible],
        [second[index] for index in eligible],
        categories=[positive_label, negative_label],
    )
    matrix = binary_nominal["confusion_matrix"]["matrix"]
    disagreements = float(matrix[0][1]) + float(matrix[1][0])
    combined_abstention_counts = Counter(label for label in [*first, *second] if label in abstentions)
    either_count = binary["abstention"]["either_count"]
    both_count = binary["abstention"]["both_count"]
    rating_count = 2 * len(first)
    abstention_rating_count = sum(combined_abstention_counts.values())
    binary.update(
        {
            "confusion_matrix": binary_nominal["confusion_matrix"],
            "exact_agreement": binary_nominal["exact_agreement"],
            "cohen_kappa": None,
            "gwet_ac1": binary_nominal["gwet_ac1"],
            "positive_agreement": _safe_ratio(
                2.0 * float(matrix[0][0]),
                2.0 * float(matrix[0][0]) + disagreements,
            ),
            "negative_agreement": _safe_ratio(
                2.0 * float(matrix[1][1]),
                2.0 * float(matrix[1][1]) + disagreements,
            ),
            "method": "symmetrized_unordered_rater_pairs",
            "abstention": {
                "rating_count": rating_count,
                "rating_abstention_count": abstention_rating_count,
                "rating_abstention_rate": _raw_rate(abstention_rating_count, rating_count),
                "pair_either_count": either_count,
                "pair_both_count": both_count,
                "pair_either_rate": _raw_rate(either_count, len(first)),
                "pair_both_rate": _raw_rate(both_count, len(first)),
                "labels": [
                    {"label": label, "count": combined_abstention_counts[label]}
                    for label in sorted(combined_abstention_counts, key=_stable_sort_key)
                ],
            },
        }
    )
    result["binary"] = binary
    return result


def _pairwise_kappa_mean(
    pairwise: Sequence[Mapping[str, Any]],
    *,
    section: str,
    weight_key: str,
) -> float | None:
    coefficients: list[float] = []
    weights: list[float] = []
    for entry in pairwise:
        summary = entry[section]
        coefficient = summary["cohen_kappa"]
        if coefficient is None:
            continue
        weight = float(summary[weight_key])
        if weight <= 0:
            continue
        coefficients.append(float(coefficient))
        weights.append(weight)
    denominator = math.fsum(weights)
    if denominator == 0:
        return None
    return (
        math.fsum(coefficient * weight for coefficient, weight in zip(coefficients, weights, strict=True)) / denominator
    )


def pre_adjudication_agreement(
    annotations: Sequence[Record],
    *,
    item_key: KeySpec = "item_id",
    rater_key: str = "participant_pseudonym",
    label_key: str = "label",
    categories: Sequence[Label] | None = None,
    positive_label: Label | None = "yes",
    negative_label: Label | None = "no",
    abstention_labels: Collection[Label] = DEFAULT_ABSTENTION_LABELS,
) -> dict[str, Any]:
    """Compute human-human agreement from independent ratings before adjudication.

    Every available rater pair on an item contributes once. Per-rater-pair
    summaries report Cohen's kappa. The pooled table is symmetrized because
    unordered human pairs do not have stable A/B roles; it therefore reports
    AC1 and the pair-count-weighted mean of per-rater-pair kappas, not a pooled
    Cohen kappa. Missing ``None`` labels are retained as missingness counts but
    do not form pairs. Sampling weights are intentionally not applied: this
    describes reliability in the rated validation sample.
    """

    ratings_by_item: dict[Hashable, dict[Label, Label]] = defaultdict(dict)
    seen: set[tuple[Hashable, Label]] = set()
    missing_labels = 0
    all_labels: list[Label] = []
    for index, annotation in enumerate(annotations):
        item = _record_key(annotation, item_key, name=f"annotations[{index}] item key")
        rater = annotation[rater_key]
        _validate_hashable(rater, name=f"annotations[{index}][{rater_key!r}]")
        _validate_json_scalar(rater, name=f"annotations[{index}][{rater_key!r}]")
        pair = (item, rater)
        if pair in seen:
            raise ValueError(f"Duplicate rating for item {item!r} and rater {rater!r}")
        seen.add(pair)
        label = annotation[label_key]
        if label is None:
            missing_labels += 1
            continue
        _validate_hashable(label, name=f"annotations[{index}][{label_key!r}]")
        _validate_json_scalar(label, name=f"annotations[{index}][{label_key!r}]")
        ratings_by_item[item][rater] = label
        all_labels.append(label)

    resolved_categories = _resolve_categories(all_labels, [], categories)
    pooled_first: list[Label] = []
    pooled_second: list[Label] = []
    by_pair: dict[tuple[Label, Label], tuple[list[Label], list[Label]]] = {}
    items_with_pairs = 0
    items_with_consensus = 0
    for item in sorted(ratings_by_item, key=_stable_sort_key):
        item_ratings = ratings_by_item[item]
        raters = sorted(item_ratings, key=_stable_sort_key)
        if len(raters) >= 2:
            items_with_pairs += 1
        if (
            consensus_label(
                list(item_ratings.values()),
                abstention_labels=abstention_labels,
                minimum_raters=2,
            )
            is not None
        ):
            items_with_consensus += 1
        for rater_a, rater_b in combinations(raters, 2):
            label_a = item_ratings[rater_a]
            label_b = item_ratings[rater_b]
            pooled_first.append(label_a)
            pooled_second.append(label_b)
            pair_first, pair_second = by_pair.setdefault((rater_a, rater_b), ([], []))
            pair_first.append(label_a)
            pair_second.append(label_b)

    pairwise = []
    for (rater_a, rater_b), (first, second) in sorted(by_pair.items(), key=lambda pair: _stable_sort_key(pair[0])):
        pairwise.append(
            {
                "rater_a": rater_a,
                "rater_b": rater_b,
                **_agreement_bundle(
                    first,
                    second,
                    categories=resolved_categories,
                    positive_label=positive_label,
                    negative_label=negative_label,
                    abstention_labels=abstention_labels,
                ),
            }
        )
    pooled = _pooled_agreement_bundle(
        pooled_first,
        pooled_second,
        categories=resolved_categories,
        positive_label=positive_label,
        negative_label=negative_label,
        abstention_labels=abstention_labels,
    )
    pooled["nominal"]["mean_pairwise_cohen_kappa"] = _pairwise_kappa_mean(
        pairwise,
        section="nominal",
        weight_key="n_pairs",
    )
    if pooled["binary"] is not None:
        pooled["binary"]["mean_pairwise_cohen_kappa"] = _pairwise_kappa_mean(
            pairwise,
            section="binary",
            weight_key="n_binary",
        )
    return {
        "n_annotations": len(annotations),
        "n_missing_labels": missing_labels,
        "n_items": len({pair[0] for pair in seen}),
        "n_rated_items": len(ratings_by_item),
        "n_items_with_pairs": items_with_pairs,
        "n_items_with_consensus": items_with_consensus,
        "consensus_rate": _raw_rate(items_with_consensus, len(ratings_by_item)),
        "n_pairwise_comparisons": len(pooled_first),
        "categories": resolved_categories,
        "pooled": pooled,
        "pairwise": pairwise,
    }


def _classification_rates(matrix: Sequence[Sequence[float | int]]) -> dict[str, float | None]:
    true_positive = float(matrix[0][0])
    false_negative = float(matrix[0][1])
    false_positive = float(matrix[1][0])
    true_negative = float(matrix[1][1])
    return {
        "positive_precision": _safe_ratio(true_positive, true_positive + false_positive),
        "positive_recall": _safe_ratio(true_positive, true_positive + false_negative),
        "negative_specificity": _safe_ratio(true_negative, true_negative + false_positive),
        "negative_predictive_value": _safe_ratio(true_negative, true_negative + false_negative),
    }


def _judge_group_summary(
    records: Sequence[Record],
    *,
    human_label_key: str,
    judge_label_key: str,
    weight_key: str,
    label_categories: Sequence[Label],
    positive_label: Label | None,
    negative_label: Label | None,
    abstention_labels: Collection[Label],
) -> dict[str, Any]:
    raw_weights = [_validate_weight(record[weight_key], name=f"record[{weight_key!r}]") for record in records]
    weights, weight_total, weight_scale = _stabilize_weights(raw_weights)
    human_missing = [record[human_label_key] is None for record in records]
    judge_missing = [record[judge_label_key] is None for record in records]
    compared_indices = [
        index
        for index, (missing_human, missing_judge) in enumerate(zip(human_missing, judge_missing, strict=True))
        if not missing_human and not missing_judge
    ]
    human = [records[index][human_label_key] for index in compared_indices]
    judge = [records[index][judge_label_key] for index in compared_indices]
    compared_weights = [weights[index] for index in compared_indices]
    unweighted_nominal = nominal_agreement(human, judge, categories=label_categories)
    weighted_nominal = nominal_agreement(human, judge, categories=label_categories, weights=compared_weights)
    unweighted_binary = None
    weighted_binary = None
    if (positive_label is None) != (negative_label is None):
        raise ValueError("positive_label and negative_label must either both be set or both be None")
    if positive_label is not None and negative_label is not None:
        unweighted_binary = binary_agreement(
            human,
            judge,
            positive_label=positive_label,
            negative_label=negative_label,
            abstention_labels=abstention_labels,
        )
        weighted_binary = binary_agreement(
            human,
            judge,
            positive_label=positive_label,
            negative_label=negative_label,
            abstention_labels=abstention_labels,
            weights=compared_weights,
        )
        unweighted_binary["classification_rates"] = _classification_rates(
            unweighted_binary["confusion_matrix"]["matrix"]
        )
        weighted_binary["classification_rates"] = _classification_rates(weighted_binary["confusion_matrix"]["matrix"])
    return {
        "n_records": len(records),
        "n_compared": len(compared_indices),
        "n_missing_human": sum(human_missing),
        "n_missing_judge": sum(judge_missing),
        "weight_total": weight_total,
        "weight_compared": math.fsum(compared_weights),
        "weight_scale": weight_scale,
        "missing_human_rate": _raw_rate(sum(human_missing), len(records)),
        "missing_judge_rate": _raw_rate(sum(judge_missing), len(records)),
        "weighted_missing_human_rate": _weighted_flag_rate(human_missing, weights),
        "weighted_missing_judge_rate": _weighted_flag_rate(judge_missing, weights),
        "unweighted": {
            "nominal": unweighted_nominal,
            "binary": unweighted_binary,
        },
        "weighted": {
            "nominal": weighted_nominal,
            "binary": weighted_binary,
        },
    }


def _group_summaries(
    records: Sequence[Record],
    *,
    group_keys: Sequence[str],
    summary: Callable[[Sequence[Record]], dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Record]] = defaultdict(list)
    for record in records:
        group = tuple(record[key] for key in group_keys)
        for key, value in zip(group_keys, group, strict=True):
            _validate_json_scalar(value, name=f"group field {key!r}")
        grouped[group].append(record)
    output = []
    for group, group_records in sorted(grouped.items(), key=lambda pair: _stable_sort_key(pair[0])):
        identifiers = dict(zip(group_keys, group, strict=True))
        output.append({**identifiers, **summary(group_records)})
    return output


def judge_human_summary(
    records: Sequence[Record],
    *,
    human_label_key: str = "human_label",
    judge_label_key: str = "judge_label",
    category_key: str = "category",
    surface_key: str = "surface",
    judge_key: str | None = None,
    weight_key: str = "sampling_weight",
    label_categories: Sequence[Label] | None = None,
    positive_label: Label | None = "yes",
    negative_label: Label | None = "no",
    abstention_labels: Collection[Label] = DEFAULT_ABSTENTION_LABELS,
) -> dict[str, Any]:
    """Summarize judge-versus-human agreement overall and by audit groups.

    Rows with a missing human consensus or judge answer are excluded from
    agreement denominators but retained in missingness rates. Set both binary
    labels to ``None`` for genuinely nominal tasks such as yes/partial/no.
    """

    rows = list(records)
    all_compared_labels: list[Label] = []
    for index, record in enumerate(rows):
        if weight_key not in record:
            raise KeyError(f"Record {index} is missing weight key {weight_key!r}")
        _validate_weight(record[weight_key], name=f"records[{index}][{weight_key!r}]")
        for label_key in (human_label_key, judge_label_key):
            label = record[label_key]
            if label is not None:
                _validate_hashable(label, name=f"records[{index}][{label_key!r}]")
                _validate_json_scalar(label, name=f"records[{index}][{label_key!r}]")
                all_compared_labels.append(label)
    resolved_categories = _resolve_categories(all_compared_labels, [], label_categories)

    def summarize(group: Sequence[Record]) -> dict[str, Any]:
        return _judge_group_summary(
            group,
            human_label_key=human_label_key,
            judge_label_key=judge_label_key,
            weight_key=weight_key,
            label_categories=resolved_categories,
            positive_label=positive_label,
            negative_label=negative_label,
            abstention_labels=abstention_labels,
        )

    result: dict[str, Any] = {
        "label_categories": resolved_categories,
        "overall": summarize(rows),
        "by_category": _group_summaries(rows, group_keys=[category_key], summary=summarize),
        "by_surface": _group_summaries(rows, group_keys=[surface_key], summary=summarize),
        "by_category_surface": _group_summaries(
            rows,
            group_keys=[category_key, surface_key],
            summary=summarize,
        ),
    }
    if judge_key is not None:
        result["by_judge"] = _group_summaries(rows, group_keys=[judge_key], summary=summarize)
        result["by_judge_category_surface"] = _group_summaries(
            rows,
            group_keys=[judge_key, category_key, surface_key],
            summary=summarize,
        )
    return result


def _finite_statistic(value: float | int | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"statistic must return a number or None, got {value!r}") from exc
    return result if math.isfinite(result) else None


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def _json_group_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_group_value(part) for part in value]
    _validate_json_scalar(value, name="stratum")
    return value


def cluster_bootstrap_percentile(
    records: Sequence[Record],
    statistic: Statistic,
    *,
    cluster_key: KeySpec = "item_group",
    stratum_key: KeySpec | None = None,
    bootstrap_instance_key: str | None = "__bootstrap_instance__",
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
    return_distribution: bool = False,
) -> dict[str, Any]:
    """Compute a deterministic percentile CI by resampling whole image clusters.

    All rows in a selected image cluster travel together, so paired caption
    surfaces, claims, and raters remain paired. Each selected cluster occurrence
    receives a unique ``bootstrap_instance_key`` value. A statistic that groups
    by items should therefore use, for example,
    ``item_key=["__bootstrap_instance__", "item_id"]``; this preserves repeated
    cluster draws rather than collapsing or rejecting them.

    When ``stratum_key`` is given, the original number of image clusters is
    resampled independently within each stratum. The stratum must be constant
    for an entire image cluster. In particular, do not use category or surface
    as the stratum when one image cluster crosses those values.
    """

    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if isinstance(confidence_level, bool):
        raise ValueError("confidence_level must be a finite number strictly between 0 and 1")
    try:
        normalized_confidence = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_level must be a finite number strictly between 0 and 1") from exc
    if not math.isfinite(normalized_confidence) or not 0.0 < normalized_confidence < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    rows = list(records)
    if bootstrap_instance_key is not None:
        for index, record in enumerate(rows):
            if bootstrap_instance_key in record:
                raise ValueError(f"Record {index} already contains bootstrap instance key {bootstrap_instance_key!r}")
    clusters: dict[Hashable, list[Record]] = defaultdict(list)
    cluster_strata: dict[Hashable, Hashable | None] = {}
    row_clusters: list[Hashable] = []
    for index, record in enumerate(rows):
        cluster = _record_key(record, cluster_key, name=f"records[{index}] cluster key")
        row_clusters.append(cluster)
        stratum = (
            None if stratum_key is None else _record_key(record, stratum_key, name=f"records[{index}] stratum key")
        )
        if cluster in cluster_strata and cluster_strata[cluster] != stratum:
            raise ValueError(f"Cluster {cluster!r} appears in multiple strata")
        cluster_strata[cluster] = stratum
        clusters[cluster].append(record)

    by_stratum: dict[Hashable | None, list[Hashable]] = defaultdict(list)
    for cluster, stratum in cluster_strata.items():
        by_stratum[stratum].append(cluster)
    ordered_strata = sorted(by_stratum, key=_stable_sort_key)
    for stratum in ordered_strata:
        by_stratum[stratum].sort(key=_stable_sort_key)

    ordered_clusters = [cluster for stratum in ordered_strata for cluster in by_stratum[stratum]]
    original_instances = {cluster: index for index, cluster in enumerate(ordered_clusters)}

    def attach_instance(record: Record, instance: int) -> Record:
        if bootstrap_instance_key is None:
            return record
        return {**record, bootstrap_instance_key: instance}

    estimate_rows = [
        attach_instance(record, original_instances[cluster]) for record, cluster in zip(rows, row_clusters, strict=True)
    ]
    estimate = None if not rows else _finite_statistic(statistic(estimate_rows))
    generator = random.Random(seed)
    distribution: list[float] = []
    if clusters:
        for _ in range(n_resamples):
            sample: list[Record] = []
            bootstrap_instance = 0
            for stratum in ordered_strata:
                stratum_clusters = by_stratum[stratum]
                for _ in range(len(stratum_clusters)):
                    selected = stratum_clusters[generator.randrange(len(stratum_clusters))]
                    sample.extend(attach_instance(record, bootstrap_instance) for record in clusters[selected])
                    bootstrap_instance += 1
            replicate = _finite_statistic(statistic(sample))
            if replicate is not None:
                distribution.append(replicate)

    alpha = 1.0 - normalized_confidence
    output: dict[str, Any] = {
        "method": "image_cluster_percentile",
        "estimate": estimate,
        "confidence_level": normalized_confidence,
        "lower": _percentile(distribution, alpha / 2.0),
        "upper": _percentile(distribution, 1.0 - alpha / 2.0),
        "n_records": len(rows),
        "n_clusters": len(clusters),
        "n_strata": len(by_stratum),
        "clusters_per_stratum": [
            {
                "stratum": _json_group_value(stratum),
                "n_clusters": len(by_stratum[stratum]),
            }
            for stratum in ordered_strata
        ],
        "n_resamples": n_resamples,
        "n_valid_resamples": len(distribution),
        "seed": seed,
        "bootstrap_instance_key": bootstrap_instance_key,
    }
    if return_distribution:
        output["distribution"] = distribution
    return output
