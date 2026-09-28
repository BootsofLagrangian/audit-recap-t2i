"""Deterministic, privacy-aware exports for the human CBU audit.

The module consumes plain mappings, such as the output of
``HumanCBUStore.export_rows``.  It deliberately does not open the study
database: callers must make the private/public boundary explicit by passing
already-authorized rows.

Partial studies still produce machine-readable audit artifacts, but rebuttal
tables and prose are withheld until the supplied operational status confirms
that collection is closed, assignments are complete, disagreements are
resolved, and substantive image-support consensus exists.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from audit_recap_t2i.human_cbu.metrics import (
    cluster_bootstrap_percentile,
    consensus_label,
    judge_human_summary,
    nominal_agreement,
    pre_adjudication_agreement,
    weighted_proportion,
)

EXPORT_SCHEMA_VERSION = "human-cbu-export-v1"

DEFAULT_READINESS_THRESHOLDS: dict[str, Any] = {
    "minimum_substantive_overall": 120,
    "minimum_substantive_per_type": 5,
    "minimum_substantive_per_stratum": 2,
    "expected_proposed_types": [],
    "expected_strata": [],
    "require_complete_judge_coverage": True,
    "required_bootstrap_resamples": None,
    "required_bootstrap_seed": None,
}
MINIMUM_READY_BOOTSTRAP_RESAMPLES = 2
INTERVAL_SCOPE_NOTE = (
    "Image-cluster bootstrap intervals are model/superpopulation sensitivity "
    "intervals conditional on the fixed realized annotators; they are not "
    "design-consistent finite-population confidence intervals."
)

PHASE_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    "caption": {
        "caption_licensed": ("yes", "no", "uncertain"),
        "atomic_visual_claim": ("yes", "not_visual", "not_atomic", "uncertain"),
        "category_check": ("correct", "incorrect", "uncertain", "not_applicable"),
    },
    "image": {
        "image_support": (
            "yes",
            "no",
            "uncertain",
            "not_visual",
            "image_unavailable",
            "prefer_not_to_answer",
        ),
    },
}

FIELD_BINARY_LABELS: dict[str, tuple[str | None, str | None]] = {
    "caption_licensed": ("yes", "no"),
    "atomic_visual_claim": (None, None),
    "category_check": ("correct", "incorrect"),
    "image_support": ("yes", "no"),
}

ABSTENTION_LABELS = frozenset(
    {
        "uncertain",
        "unjudgeable",
        "not_visual",
        "not_applicable",
        "image_unavailable",
        "prefer_not_to_answer",
        "cannot_judge",
    }
)

# Public rows intentionally omit source surface, judge answers, timestamps,
# free text, internal assignment/annotation IDs, and image-group identity.
PUBLIC_COLUMNS = (
    "public_annotation_id",
    "public_item_id",
    "public_repeat_of",
    "public_participant_id",
    "phase",
    "category",
    "surface_group",
    "caption_licensed",
    "atomic_visual_claim",
    "category_check",
    "corrected_category",
    "image_support",
    "salient_coverage",
    "control_usefulness",
    "confidence",
    "sampling_weight",
    "population_size",
    "sample_probability",
    "stratum",
    "revision",
)

_PRIVATE_SENSITIVE_KEYS = frozenset(
    {
        "invite",
        "invite_code",
        "invite_digest",
        "session",
        "session_token",
        "token",
        "token_digest",
        "authorization",
        "cookie",
        "email",
        "name",
        "ip",
        "ip_address",
        "user_agent",
        "participant_profile",
    }
)

_SAFE_PROFILE_MARGINAL_FIELDS = frozenset(
    {
        "age_band",
        "author_status",
        "recruitment_source",
        "primary_language_group",
        "english_proficiency",
        "vision_status",
        "color_vision",
        "t2i_experience",
        "image_annotation_experience",
        "domain_experience",
        "device_class",
        "screen_size_band",
    }
)
PROFILE_MINIMUM_CELL_SIZE = 3


def _stable_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def _stable_value(value: Any) -> str:
    try:
        return _stable_json(value)
    except (TypeError, ValueError):
        return repr(value)


def _sort_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            str(row.get("item_id", "")),
            str(row.get("phase", "")),
            str(row.get("participant_pseudonym", "")),
            str(row.get("annotation_id", "")),
        ),
    )


def _public_id(namespace: str, kind: str, value: Any) -> str | None:
    if value is None:
        return None
    material = f"{namespace}\0{kind}\0{value}".encode()
    return hashlib.blake2b(material, digest_size=16).hexdigest()


def _surface_group(row: Mapping[str, Any]) -> str:
    stratum = row.get("stratum")
    if isinstance(stratum, str) and "/" in stratum:
        return stratum.split("/", 1)[0]
    surface_group = row.get("surface_group")
    if surface_group is not None:
        return str(surface_group)
    return str(row.get("surface") or "")


def _strip_sensitive(value: Any) -> Any:
    """Recursively remove credential and direct-identity fields.

    Store exports do not contain these fields.  The recursive guard prevents a
    future caller from accidentally copying such fields into the private audit
    bundle when using the neutral mapping API directly.
    """

    if isinstance(value, Mapping):
        return {
            str(key): _strip_sensitive(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).lower() not in _PRIVATE_SENSITIVE_KEYS
        }
    if isinstance(value, list):
        return [_strip_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_sensitive(item) for item in value]
    return value


def public_annotation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    public_id_namespace: str,
) -> list[dict[str, Any]]:
    """Project annotations to a publication-safe, pseudonymous allow-list."""

    if not isinstance(public_id_namespace, str) or not public_id_namespace.strip():
        raise ValueError("public_id_namespace must be a non-empty string")
    output = []
    for row in _sort_rows(rows):
        projected = {
            "public_annotation_id": _public_id(
                public_id_namespace,
                "annotation",
                row.get("annotation_id")
                or (
                    row.get("item_id"),
                    row.get("phase"),
                    row.get("participant_pseudonym"),
                ),
            ),
            "public_item_id": _public_id(public_id_namespace, "item", row.get("item_id")),
            "public_repeat_of": _public_id(public_id_namespace, "item", row.get("repeat_of")),
            "public_participant_id": _public_id(
                public_id_namespace,
                "participant",
                row.get("participant_pseudonym"),
            ),
            "phase": row.get("phase"),
            "category": row.get("category"),
            "surface_group": _surface_group(row),
            "caption_licensed": row.get("caption_licensed"),
            "atomic_visual_claim": row.get("atomic_visual_claim"),
            "category_check": row.get("category_check"),
            "corrected_category": row.get("corrected_category"),
            "image_support": row.get("image_support"),
            "salient_coverage": row.get("salient_coverage"),
            "control_usefulness": row.get("control_usefulness"),
            "confidence": row.get("confidence"),
            "sampling_weight": row.get("sampling_weight", row.get("weight")),
            "population_size": row.get("population_size"),
            "sample_probability": row.get("sample_probability"),
            "stratum": row.get("stratum"),
            "revision": row.get("revision"),
        }
        output.append(projected)
    return output


def private_audit_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return stable private audit rows with credential/identity fields removed."""

    return [_strip_sensitive(row) for row in _sort_rows(rows)]


def _strict_consensus(labels: Sequence[Any]) -> Any:
    # Empty abstention_labels retains agreed initial outcomes such as unanimous
    # ``uncertain``. Such outcomes remain visible in nominal metrics and are
    # excluded only from binary agreement denominators.
    return consensus_label(labels, abstention_labels=(), minimum_raters=2)


def _one_consensus(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> tuple[Any, str]:
    adjudicated = {
        row["adjudication"][field]
        for row in rows
        if isinstance(row.get("adjudication"), Mapping) and field in row["adjudication"]
    }
    if len(adjudicated) > 1:
        raise ValueError(f"conflicting adjudications for {rows[0].get('item_id')!r}/{field}")
    if adjudicated:
        return next(iter(adjudicated)), "adjudication"
    labels = [row.get(field) for row in rows if row.get(field) is not None]
    consensus = _strict_consensus(labels)
    return consensus, "initial_agreement" if consensus is not None else "none"


def _consistent(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    values = {_stable_value(row.get(key)): row.get(key) for row in rows}
    if len(values) > 1:
        raise ValueError(f"inconsistent {key!r} for item {rows[0].get('item_id')!r}")
    return next(iter(values.values()), None)


def consensus_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse independent base-item annotations to adjudicated/agreed labels."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("repeat_of") is not None:
            continue
        item_id = str(row.get("item_id") or "")
        phase = str(row.get("phase") or "")
        if phase not in PHASE_FIELDS or not item_id:
            continue
        grouped[(item_id, phase)].append(row)

    output: list[dict[str, Any]] = []
    for (item_id, phase), item_rows in sorted(grouped.items()):
        first = item_rows[0]
        collapsed: dict[str, Any] = {
            "item_id": item_id,
            "phase": phase,
            "item_group": _consistent(item_rows, "item_group") or item_id,
            "category": _consistent(item_rows, "category"),
            "proposed_category": _consistent(item_rows, "category"),
            "surface": _consistent(item_rows, "surface"),
            "surface_group": _surface_group(first),
            "stratum": _consistent(item_rows, "stratum"),
            "sampling_weight": _consistent(item_rows, "sampling_weight"),
            "population_size": _consistent(item_rows, "population_size"),
            "sample_probability": _consistent(item_rows, "sample_probability"),
            "qwen_answer": _consistent(item_rows, "qwen_answer"),
            "gemma_answer": _consistent(item_rows, "gemma_answer"),
            "n_annotations": len(item_rows),
        }
        if collapsed["sampling_weight"] is None:
            collapsed["sampling_weight"] = _consistent(item_rows, "weight")
        sources = []
        for field in PHASE_FIELDS[phase]:
            collapsed[field], source = _one_consensus(item_rows, field)
            sources.append(source)
        if phase == "caption":
            collapsed["corrected_category"], corrected_source = _one_consensus(
                item_rows,
                "corrected_category",
            )
            if collapsed["category_check"] == "correct":
                collapsed["resolved_category"] = collapsed["proposed_category"]
            elif collapsed["category_check"] == "incorrect":
                collapsed["resolved_category"] = collapsed["corrected_category"]
                sources.append(corrected_source)
            else:
                collapsed["resolved_category"] = None
        collapsed["consensus_source"] = (
            "adjudication"
            if "adjudication" in sources
            else "initial_agreement"
            if "initial_agreement" in sources
            else "none"
        )
        output.append(collapsed)
    return output


def _field_agreement(
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    field: str,
    categories: Sequence[str],
    bootstrap_resamples: int | None = None,
    bootstrap_seed: int = 1477,
) -> dict[str, Any]:
    positive, negative = FIELD_BINARY_LABELS[field]
    annotations = [
        row for row in rows if row.get("repeat_of") is None and row.get("phase") == phase and row.get(field) is not None
    ]
    summary = pre_adjudication_agreement(
        annotations,
        item_key="item_id",
        rater_key="participant_pseudonym",
        label_key=field,
        categories=categories,
        positive_label=positive,
        negative_label=negative,
        abstention_labels=ABSTENTION_LABELS,
    )
    if bootstrap_resamples is not None:

        def exact_agreement(sample: Sequence[Mapping[str, Any]]) -> float | None:
            replicate = pre_adjudication_agreement(
                sample,
                item_key=("__bootstrap_instance__", "item_id"),
                rater_key="participant_pseudonym",
                label_key=field,
                categories=categories,
                positive_label=positive,
                negative_label=negative,
                abstention_labels=ABSTENTION_LABELS,
            )
            return replicate["pooled"]["nominal"]["exact_agreement"]

        summary["cluster_bootstrap_95"] = cluster_bootstrap_percentile(
            annotations,
            exact_agreement,
            cluster_key="item_group",
            n_resamples=bootstrap_resamples,
            confidence_level=0.95,
            seed=bootstrap_seed,
        )
    return summary


def _judge_label(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("answer", "support", "image_support"):
            if value.get(key) is not None:
                return value[key]
        return None
    return value


def _support_predicate(row: Mapping[str, Any]) -> bool | None:
    label = row.get("human_label")
    if label == "yes":
        return True
    if label == "no":
        return False
    return None


def _valid_sampling_weight(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _unit_weighted_proportion(rows: Sequence[Mapping[str, Any]]) -> float | None:
    unit_rows = [{**row, "_unit_weight": 1.0} for row in rows]
    return weighted_proportion(unit_rows, predicate=_support_predicate, weight_key="_unit_weight")


def _weighted_support_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rows = list(records)
    valid_weights = all(_valid_sampling_weight(row.get("sampling_weight")) for row in rows)
    weighted = weighted_proportion(rows, predicate=_support_predicate) if rows and valid_weights else None
    bootstrap = None
    if rows and valid_weights:
        bootstrap = cluster_bootstrap_percentile(
            rows,
            lambda sample: weighted_proportion(sample, predicate=_support_predicate),
            cluster_key="item_group",
            n_resamples=n_resamples,
            confidence_level=0.95,
            seed=bootstrap_seed,
        )
    return {
        "n_items": len(rows),
        "n_substantive": sum(_support_predicate(row) is not None for row in rows),
        "unweighted": _unit_weighted_proportion(rows),
        "design_weighted": weighted,
        "cluster_bootstrap_95": bootstrap,
    }


def _group_support(
    records: Sequence[Mapping[str, Any]],
    key: str,
    *,
    n_resamples: int,
    bootstrap_seed: int,
    expected_groups: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    groups: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for group in expected_groups:
        groups[group]
    for row in records:
        groups[row.get(key)].append(row)
    return [
        {
            key: group,
            **_weighted_support_summary(
                group_rows,
                n_resamples=n_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
        }
        for group, group_rows in sorted(groups.items(), key=lambda pair: _stable_value(pair[0]))
    ]


def _equal_cell_support_macro(
    records: Sequence[Mapping[str, Any]],
    *,
    declared_strata: Sequence[str],
    n_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Average all declared cell estimates and bootstrap their shared cluster draw."""

    strata = list(declared_strata)
    allowed = set(strata)
    rows = [row for row in records if row.get("stratum") in allowed]

    def cell_estimates(sample: Sequence[Mapping[str, Any]]) -> list[float] | None:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in sample:
            grouped[str(row["stratum"])].append(row)
        estimates: list[float] = []
        for stratum in strata:
            estimate = _unit_weighted_proportion(grouped[stratum])
            if estimate is None:
                return None
            estimates.append(float(estimate))
        return estimates

    def statistic(sample: Sequence[Mapping[str, Any]]) -> float | None:
        estimates = cell_estimates(sample)
        return None if not estimates else math.fsum(estimates) / len(estimates)

    observed_cells = {str(row["stratum"]) for row in rows}
    substantive_by_cell = Counter(str(row["stratum"]) for row in rows if _support_predicate(row) is not None)
    populated = [stratum for stratum in strata if substantive_by_cell[stratum] > 0]
    estimate = statistic(rows)
    bootstrap = cluster_bootstrap_percentile(
        rows,
        statistic,
        cluster_key="item_group",
        n_resamples=n_resamples,
        confidence_level=0.95,
        seed=bootstrap_seed,
    )
    return {
        "estimate": estimate,
        "n_cells_declared": len(strata),
        "n_cells_observed": len(observed_cells),
        "n_cells_reported": len(strata),
        "n_empty_cells": len(strata) - len(populated),
        "n_cells_in_macro": len(populated),
        "n_substantive": sum(substantive_by_cell.values()),
        "weighting": (
            "equal weight over all declared strata; the estimate requires at "
            "least one substantive yes/no item in every declared cell"
        ),
        "cluster_bootstrap_95": bootstrap,
    }


def _weighted_ordinal_median(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    weight_key: str = "sampling_weight",
) -> int | None:
    weighted_values = sorted(
        (
            int(row[value_key]),
            float(row[weight_key]),
        )
        for row in rows
        if isinstance(row.get(value_key), int)
        and not isinstance(row.get(value_key), bool)
        and 1 <= int(row[value_key]) <= 5
        and _valid_sampling_weight(row.get(weight_key))
    )
    total = math.fsum(weight for _, weight in weighted_values)
    if total == 0:
        return None
    cumulative = 0.0
    for value, weight in weighted_values:
        cumulative += weight
        if cumulative >= total / 2:
            return value
    return weighted_values[-1][0]


def _raw_usefulness_records(
    rows: Sequence[Mapping[str, Any]],
    support_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project eligible raw ratings while preserving one design weight per item."""

    supported = {
        str(row["item_id"]): row
        for row in support_records
        if row.get("extraction_valid") is True and row.get("human_label") == "yes"
    }
    eligible_by_item: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        item_id = str(row.get("item_id") or "")
        if (
            row.get("repeat_of") is None
            and row.get("phase") == "image"
            and row.get("image_support") == "yes"
            and item_id in supported
        ):
            eligible_by_item[item_id].append(row)

    output: list[dict[str, Any]] = []
    for item_id, item_rows in sorted(eligible_by_item.items()):
        resolved = supported[item_id]
        item_weight = float(resolved["sampling_weight"])
        rating_weight = item_weight / len(item_rows)
        for row in sorted(
            item_rows,
            key=lambda value: (
                str(value.get("participant_pseudonym") or ""),
                str(value.get("annotation_id") or ""),
            ),
        ):
            output.append(
                {
                    "item_id": item_id,
                    "item_group": resolved.get("item_group"),
                    "participant_pseudonym": row.get("participant_pseudonym"),
                    "category": resolved.get("category"),
                    "surface_group": resolved.get("surface_group"),
                    "control_usefulness": row.get("control_usefulness"),
                    "sampling_weight": rating_weight,
                    "extraction_valid": True,
                    "human_label": "yes",
                }
            )
    return output


def _usefulness_ordinal_agreement(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe pre-adjudication ordinal proximity without averaging scores."""

    ratings_by_item: dict[str, dict[str, int]] = defaultdict(dict)
    for row in records:
        value = row.get("control_usefulness")
        participant = str(row.get("participant_pseudonym") or "")
        if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 5 and participant:
            item_id = str(row.get("item_id") or "")
            if participant in ratings_by_item[item_id]:
                raise ValueError(f"duplicate usefulness rating for item {item_id!r} and participant {participant!r}")
            ratings_by_item[item_id][participant] = value

    distances = [
        abs(first - second)
        for item_id in sorted(ratings_by_item)
        for first, second in combinations(
            [
                value
                for _, value in sorted(
                    ratings_by_item[item_id].items(),
                    key=lambda pair: pair[0],
                )
            ],
            2,
        )
    ]
    ordered = sorted(distances)
    if not ordered:
        median_distance: float | None = None
    elif len(ordered) % 2:
        median_distance = float(ordered[len(ordered) // 2])
    else:
        middle = len(ordered) // 2
        median_distance = (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "n_numeric_pairs": len(distances),
        "exact_rate": None if not distances else sum(distance == 0 for distance in distances) / len(distances),
        "within_one_rate": None if not distances else sum(distance <= 1 for distance in distances) / len(distances),
        "median_absolute_difference": median_distance,
        "weighted": False,
    }


def _usefulness_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize optional control usefulness without creating a composite score.

    The input contains raw ratings from annotators who individually selected
    image support for a claim whose resolved extraction and support labels are
    both valid.  Each item's design weight is split equally across its eligible
    raw ratings, preventing items with extra ratings from receiving extra mass.
    ``cannot_judge`` and missing optional scores remain visible.
    """

    rows = [row for row in records if row.get("extraction_valid") is True and row.get("human_label") == "yes"]
    total_weight = math.fsum(
        float(row["sampling_weight"]) for row in rows if _valid_sampling_weight(row.get("sampling_weight"))
    )

    def bucket(row: Mapping[str, Any]) -> str:
        value = row.get("control_usefulness")
        if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 5:
            return str(value)
        if value == "cannot_judge":
            return "cannot_judge"
        return "missing"

    counts = Counter(bucket(row) for row in rows)
    weighted_counts = {
        label: math.fsum(
            float(row["sampling_weight"])
            for row in rows
            if bucket(row) == label and _valid_sampling_weight(row.get("sampling_weight"))
        )
        for label in ("1", "2", "3", "4", "5", "cannot_judge", "missing")
    }
    distribution = {
        label: None if total_weight == 0 else weighted_counts[label] / total_weight
        for label in ("1", "2", "3", "4", "5", "cannot_judge", "missing")
    }
    return {
        "conditioning": (
            "raw ratings with individual image_support=yes on resolved extraction-valid and image-supported claims"
        ),
        "weighting": "each supported item's design weight is divided equally over eligible raw ratings",
        "n_supported_items": len({str(row.get("item_id")) for row in rows}),
        "n_eligible_annotations": len(rows),
        "n_rated": sum(counts[str(score)] for score in range(1, 6)),
        "n_cannot_judge": counts["cannot_judge"],
        "n_missing": counts["missing"],
        "design_weighted_distribution": distribution,
        "design_weighted_ordinal_median": _weighted_ordinal_median(
            rows,
            value_key="control_usefulness",
        ),
    }


def _group_usefulness(
    records: Sequence[Mapping[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    groups: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[row.get(key)].append(row)
    return [
        {key: group, **_usefulness_summary(group_rows)}
        for group, group_rows in sorted(groups.items(), key=lambda pair: _stable_value(pair[0]))
    ]


def _salient_coverage_records(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Collapse claim-sampled coverage ratings to one participant/pair record.

    The base sample is stratified over claims, not image-caption pairs.  A
    caption can therefore appear under more than one sampled claim.  Keeping
    every such row would pseudo-replicate the same whole-caption judgment.
    We deterministically retain the lexicographically first base item within
    each participant × image-group × surface cell and report the collapse
    counts.  No claim-level design weights are applied to this pair-level,
    sample-conditional analysis.
    """

    eligible = [
        row
        for row in rows
        if row.get("phase") == "image"
        and row.get("repeat_of") is None
        and row.get("image_support") not in {"image_unavailable", "prefer_not_to_answer"}
    ]
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        participant = str(row.get("participant_pseudonym") or "")
        item_group = str(row.get("item_group") or row.get("item_id") or "")
        surface = str(row.get("surface") or row.get("surface_group") or "")
        if participant and item_group and surface:
            grouped[(participant, item_group, surface)].append(row)

    output: list[dict[str, Any]] = []
    collapsed_rows = 0
    for (participant, item_group, surface), pair_rows in sorted(grouped.items()):
        ordered = sorted(
            pair_rows,
            key=lambda row: (
                str(row.get("item_id") or ""),
                str(row.get("annotation_id") or ""),
            ),
        )
        collapsed_rows += len(ordered) - 1
        chosen = ordered[0]
        output.append(
            {
                "pair_key": (item_group, surface),
                "participant_pseudonym": participant,
                "surface_group": _surface_group(chosen),
                "salient_coverage": chosen.get("salient_coverage"),
                "n_claim_rows_for_pair": len(ordered),
            }
        )
    return output, {
        "n_input_claim_annotations": len(eligible),
        "n_participant_pair_records": len(output),
        "n_duplicate_claim_rows_collapsed": collapsed_rows,
    }


def _salient_coverage_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [
        int(row["salient_coverage"])
        for row in records
        if isinstance(row.get("salient_coverage"), int)
        and not isinstance(row.get("salient_coverage"), bool)
        and 1 <= int(row["salient_coverage"]) <= 5
    ]
    counts = Counter(values)
    ordered = sorted(values)
    median: float | None
    if not ordered:
        median = None
    elif len(ordered) % 2:
        median = float(ordered[len(ordered) // 2])
    else:
        middle = len(ordered) // 2
        median = (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "n_participant_pair_records": len(records),
        "n_rated": len(values),
        "n_missing": len(records) - len(values),
        "unweighted_distribution": {
            str(score): None if not values else counts[score] / len(values)
            for score in range(1, 6)
        },
        "unweighted_ordinal_median": median,
    }


def _salient_coverage_agreement(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ratings_by_pair: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for row in records:
        value = row.get("salient_coverage")
        participant = str(row.get("participant_pseudonym") or "")
        pair_key = row.get("pair_key")
        if (
            isinstance(pair_key, tuple)
            and len(pair_key) == 2
            and isinstance(value, int)
            and not isinstance(value, bool)
            and 1 <= value <= 5
            and participant
        ):
            ratings_by_pair[pair_key][participant] = value
    distances = [
        abs(first - second)
        for pair_key in sorted(ratings_by_pair)
        for first, second in combinations(
            [value for _, value in sorted(ratings_by_pair[pair_key].items())],
            2,
        )
    ]
    ordered = sorted(distances)
    median = (
        None
        if not ordered
        else float(ordered[len(ordered) // 2])
        if len(ordered) % 2
        else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
    )
    return {
        "n_numeric_pairs": len(distances),
        "exact_rate": None if not distances else sum(value == 0 for value in distances) / len(distances),
        "within_one_rate": None if not distances else sum(value <= 1 for value in distances) / len(distances),
        "median_absolute_difference": median,
        "weighted": False,
    }


def _judge_nominal_exact(records: Sequence[Mapping[str, Any]]) -> float | None:
    eligible = [
        row
        for row in records
        if row.get("human_label") is not None
        and row.get("judge_label") is not None
        and _valid_sampling_weight(row.get("sampling_weight"))
    ]
    summary = nominal_agreement(
        [row["human_label"] for row in eligible],
        [row["judge_label"] for row in eligible],
        categories=PHASE_FIELDS["image"]["image_support"],
        weights=[float(row["sampling_weight"]) for row in eligible],
    )
    return summary["exact_agreement"]


def _normalize_readiness_thresholds(
    thresholds: Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized = {
        key: list(value) if isinstance(value, list) else value for key, value in DEFAULT_READINESS_THRESHOLDS.items()
    }
    supplied = {} if thresholds is None else dict(thresholds)
    unknown = set(supplied) - set(normalized)
    if unknown:
        raise ValueError(f"unknown readiness threshold(s): {', '.join(sorted(unknown))}")
    normalized.update(supplied)
    for key in (
        "minimum_substantive_overall",
        "minimum_substantive_per_type",
        "minimum_substantive_per_stratum",
    ):
        value = normalized[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    if type(normalized["require_complete_judge_coverage"]) is not bool:
        raise ValueError("require_complete_judge_coverage must be boolean")
    required_resamples = normalized["required_bootstrap_resamples"]
    if required_resamples is not None and (
        isinstance(required_resamples, bool)
        or not isinstance(required_resamples, int)
        or required_resamples < MINIMUM_READY_BOOTSTRAP_RESAMPLES
    ):
        raise ValueError(
            f"required_bootstrap_resamples must be null or an integer >= {MINIMUM_READY_BOOTSTRAP_RESAMPLES}"
        )
    required_seed = normalized["required_bootstrap_seed"]
    if required_seed is not None and (isinstance(required_seed, bool) or not isinstance(required_seed, int)):
        raise ValueError("required_bootstrap_seed must be null or an integer")
    for key in ("expected_proposed_types", "expected_strata"):
        values = normalized[key]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"{key} must be a sequence of unique non-empty strings")
        if any(not isinstance(value, str) or not value for value in values) or len(set(values)) != len(values):
            raise ValueError(f"{key} must be a sequence of unique non-empty strings")
        normalized[key] = sorted(values)
    return normalized


def _operational_readiness(
    rows: Sequence[Mapping[str, Any]],
    status: Mapping[str, Any] | None,
    collapsed: Sequence[Mapping[str, Any]],
    support_records: Sequence[Mapping[str, Any]],
    *,
    readiness_thresholds: Mapping[str, Any] | None,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    thresholds = _normalize_readiness_thresholds(readiness_thresholds)
    reasons: list[str] = []
    if bootstrap_resamples < MINIMUM_READY_BOOTSTRAP_RESAMPLES:
        reasons.append(
            f"bootstrap_resamples_below_readiness_minimum:{bootstrap_resamples}/{MINIMUM_READY_BOOTSTRAP_RESAMPLES}"
        )
    required_resamples = thresholds["required_bootstrap_resamples"]
    if required_resamples is not None and bootstrap_resamples != required_resamples:
        reasons.append(f"bootstrap_resamples_mismatch:{bootstrap_resamples}/{required_resamples}")
    required_seed = thresholds["required_bootstrap_seed"]
    if required_seed is not None and bootstrap_seed != required_seed:
        reasons.append(f"bootstrap_seed_mismatch:{bootstrap_seed}/{required_seed}")
    if not rows:
        reasons.append("no_annotations")
    if status is None:
        reasons.append("operational_status_missing")
    else:
        study = status.get("study")
        study_state = study.get("status") if isinstance(study, Mapping) else None
        if study_state not in {"closed", "archived"}:
            reasons.append(f"study_not_closed:{study_state or 'unknown'}")
        assignments = status.get("assignments")
        for phase in PHASE_FIELDS:
            phase_assignments = assignments.get(phase, {}) if isinstance(assignments, Mapping) else {}
            pending = phase_assignments.get("pending")
            total = phase_assignments.get("total")
            completed = phase_assignments.get("completed")
            if not isinstance(total, int) or not isinstance(completed, int):
                reasons.append(f"assignment_status_missing:{phase}")
            elif completed != total or (isinstance(pending, int) and pending != 0):
                reasons.append(f"assignments_incomplete:{phase}")
        agreement = status.get("agreement")
        for phase in PHASE_FIELDS:
            phase_agreement = agreement.get(phase, {}) if isinstance(agreement, Mapping) else {}
            unresolved = phase_agreement.get("unresolved_disagreements")
            if not isinstance(unresolved, int):
                reasons.append(f"disagreement_status_missing:{phase}")
            elif unresolved:
                reasons.append(f"unresolved_disagreements:{phase}:{unresolved}")
            panel_shortfalls = phase_agreement.get("eligible_panel_shortfalls")
            if not isinstance(panel_shortfalls, int):
                reasons.append(f"eligible_panel_status_missing:{phase}")
            elif panel_shortfalls:
                reasons.append(f"eligible_panel_shortfalls:{phase}:{panel_shortfalls}")

    base_rows = [row for row in rows if row.get("repeat_of") is None]
    missing_weights = {
        row.get("item_id")
        for row in base_rows
        if not _valid_sampling_weight(
            row.get("sampling_weight") if row.get("sampling_weight") is not None else row.get("weight")
        )
    }
    if missing_weights:
        reasons.append(f"missing_sampling_weights:{len(missing_weights)}")

    observed_items = {row.get("item_id") for row in rows if row.get("item_id") is not None}
    status_items = status.get("items") if isinstance(status, Mapping) else None
    if isinstance(status_items, int) and len(observed_items) != status_items:
        reasons.append(f"observed_items_mismatch:{len(observed_items)}/{status_items}")

    for phase, fields in PHASE_FIELDS.items():
        phase_rows = [row for row in collapsed if row.get("phase") == phase]
        for field in fields:
            unresolved = sum(row.get(field) is None for row in phase_rows)
            if unresolved:
                reasons.append(f"unresolved_consensus:{phase}:{field}:{unresolved}")
    unresolved_corrections = sum(
        row.get("phase") == "caption"
        and row.get("category_check") == "incorrect"
        and row.get("resolved_category") is None
        for row in collapsed
    )
    if unresolved_corrections:
        reasons.append(f"unresolved_consensus:caption:corrected_category:{unresolved_corrections}")

    substantive = [row for row in support_records if row.get("image_support") in {"yes", "no"}]
    if not substantive:
        reasons.append("no_substantive_extraction_valid_image_consensus")
    minimum_overall = int(thresholds["minimum_substantive_overall"])
    if len(substantive) < minimum_overall:
        reasons.append(f"substantive_below_minimum:overall:{len(substantive)}/{minimum_overall}")

    image_rows = [row for row in collapsed if row.get("phase") == "image"]
    observed_types = {str(row.get("category")) for row in image_rows if row.get("category") is not None}
    declared_types = set(thresholds["expected_proposed_types"])
    expected_types = declared_types or observed_types
    for category in sorted(declared_types - observed_types):
        reasons.append(f"missing_design_type:{category}")
    for category in sorted(observed_types - declared_types) if declared_types else ():
        reasons.append(f"unexpected_design_type:{category}")
    substantive_by_type = Counter(row.get("proposed_category") for row in substantive)
    minimum_per_type = int(thresholds["minimum_substantive_per_type"])
    if minimum_per_type:
        for category in sorted(expected_types, key=_stable_value):
            observed = substantive_by_type[category]
            if observed < minimum_per_type:
                reasons.append(f"substantive_below_minimum:type:{category}:{observed}/{minimum_per_type}")

    observed_strata = {str(row.get("stratum")) for row in image_rows if row.get("stratum") is not None}
    declared_strata = set(thresholds["expected_strata"])
    expected_strata = declared_strata or observed_strata
    for stratum in sorted(declared_strata - observed_strata):
        reasons.append(f"missing_design_stratum:{stratum}")
    for stratum in sorted(observed_strata - declared_strata) if declared_strata else ():
        reasons.append(f"unexpected_design_stratum:{stratum}")
    substantive_by_stratum = Counter(row.get("stratum") for row in substantive)
    minimum_per_stratum = int(thresholds["minimum_substantive_per_stratum"])
    if minimum_per_stratum:
        for stratum in sorted(expected_strata, key=_stable_value):
            observed = substantive_by_stratum[stratum]
            if observed < minimum_per_stratum:
                reasons.append(f"substantive_below_minimum:stratum:{stratum}:{observed}/{minimum_per_stratum}")

    for judge in ("qwen", "gemma"):
        missing = sum(_judge_label(row.get(f"{judge}_answer")) is None for row in support_records)
        if thresholds["require_complete_judge_coverage"]:
            if missing:
                reasons.append(
                    f"incomplete_judge_coverage:{judge}:{len(support_records) - missing}/{len(support_records)}"
                )
        elif support_records and missing == len(support_records):
            reasons.append(f"no_judge_answers:{judge}")

    return {
        "ready": not reasons,
        "state": "ready" if not reasons else "not_ready",
        "provisional": bool(reasons),
        "reasons": reasons,
        "thresholds": thresholds,
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "minimum_ready_resamples": MINIMUM_READY_BOOTSTRAP_RESAMPLES,
            "required_resamples": required_resamples,
            "required_seed": required_seed,
        },
    }


def _sample_summary(
    rows: Sequence[Mapping[str, Any]],
    consensus: Sequence[Mapping[str, Any]],
    status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    items = {str(row.get("item_id")): row for row in rows if row.get("item_id") is not None}
    repeats = {item_id for item_id, row in items.items() if row.get("repeat_of") is not None}
    base = set(items) - repeats
    participants = {row.get("participant_pseudonym") for row in rows if row.get("participant_pseudonym")}
    phase_annotations = Counter(str(row.get("phase")) for row in rows)
    phase_consensus = Counter(
        str(row.get("phase"))
        for row in consensus
        if any(row.get(field) is not None for field in PHASE_FIELDS.get(str(row.get("phase")), {}))
    )
    return {
        "status_item_count": status.get("items") if isinstance(status, Mapping) else None,
        "observed_items": len(items),
        "observed_base_items": len(base),
        "observed_repeat_items": len(repeats),
        "participants": len(participants),
        "annotations_by_phase": dict(sorted(phase_annotations.items())),
        "consensus_items_by_phase": dict(sorted(phase_consensus.items())),
        "image_clusters": len(
            {
                row.get("item_group")
                for row in consensus
                if row.get("phase") == "image" and row.get("item_group") is not None
            }
        ),
    }


def _nonnegative_count(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _privacy_safe_participant_summary(
    participant_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the aggregate-only participant-flow contract.

    Participant composition must come from the store's aggregate query rather
    than annotation rows.  That prevents joint profiles or stable study
    pseudonyms from crossing into supported exports.
    """

    if participant_summary is None:
        return {
            "identity": "study-local pseudonyms",
            "direct_identifiers_collected": False,
            "flow": {
                "invited_total": None,
                "current_status": None,
                "consented_current": None,
                "completed": None,
                "procedurally_completed_total": None,
                "analyzed": None,
            },
            "profile_marginals": {
                "n_participants": None,
                "joint_profiles_released": False,
                "fields": {},
            },
            "aggregate_summary_supplied": False,
        }

    summary = dict(participant_summary)
    allowed_top_level = {
        "identity",
        "direct_identifiers_collected",
        "flow",
        "profile_marginals",
    }
    unknown = set(summary) - allowed_top_level
    if unknown:
        raise ValueError("participant_summary contains non-aggregate field(s): " + ", ".join(sorted(unknown)))
    if summary.get("direct_identifiers_collected") is not False:
        raise ValueError("participant_summary must declare direct_identifiers_collected=false")
    if summary.get("identity", "study-local pseudonyms") != "study-local pseudonyms":
        raise ValueError("participant_summary.identity must be 'study-local pseudonyms'")

    raw_flow = summary.get("flow")
    if not isinstance(raw_flow, Mapping):
        raise ValueError("participant_summary.flow must be a mapping")
    allowed_flow = {
        "invited_total",
        "current_status",
        "consented_current",
        "completed",
        "procedurally_completed_total",
        "analyzed",
    }
    unknown_flow = set(raw_flow) - allowed_flow
    if unknown_flow:
        raise ValueError("participant_summary.flow contains unsupported field(s): " + ", ".join(sorted(unknown_flow)))
    missing_flow = allowed_flow - set(raw_flow)
    if missing_flow:
        raise ValueError("participant_summary.flow is missing field(s): " + ", ".join(sorted(missing_flow)))
    flow: dict[str, Any] = {}
    for key in (
        "invited_total",
        "consented_current",
        "completed",
        "procedurally_completed_total",
        "analyzed",
    ):
        if key in raw_flow:
            flow[key] = _nonnegative_count(raw_flow[key], name=f"participant_summary.flow.{key}")
    statuses = raw_flow.get("current_status", {})
    if not isinstance(statuses, Mapping):
        raise ValueError("participant_summary.flow.current_status must be a mapping")
    allowed_statuses = {"invited", "active", "declined", "withdrawn"}
    unknown_statuses = set(statuses) - allowed_statuses
    if unknown_statuses:
        raise ValueError(
            "participant_summary.flow.current_status contains unsupported status(es): "
            + ", ".join(sorted(unknown_statuses))
        )
    missing_statuses = allowed_statuses - set(statuses)
    if missing_statuses:
        raise ValueError(
            "participant_summary.flow.current_status is missing status(es): " + ", ".join(sorted(missing_statuses))
        )
    flow["current_status"] = {
        str(status): _nonnegative_count(
            count,
            name=f"participant_summary.flow.current_status.{status}",
        )
        for status, count in sorted(statuses.items())
    }

    raw_marginals = summary.get("profile_marginals")
    if not isinstance(raw_marginals, Mapping):
        raise ValueError("participant_summary.profile_marginals must be a mapping")
    if raw_marginals.get("joint_profiles_released") is not False:
        raise ValueError("participant_summary must not release joint profiles")
    raw_fields = raw_marginals.get("fields", {})
    if not isinstance(raw_fields, Mapping):
        raise ValueError("participant_summary.profile_marginals.fields must be a mapping")
    fields: dict[str, Any] = {}
    for field, raw_field in sorted(raw_fields.items(), key=lambda pair: str(pair[0])):
        if field not in _SAFE_PROFILE_MARGINAL_FIELDS:
            raise ValueError(f"participant profile marginal {field!r} is not an approved coarse field")
        if not isinstance(raw_field, Mapping):
            raise ValueError(f"participant profile marginal {field!r} must be a mapping")
        if set(raw_field) - {"counts", "missing", "multi_select"}:
            raise ValueError(f"participant profile marginal {field!r} has unsupported fields")
        raw_counts = raw_field.get("counts", [])
        if not isinstance(raw_counts, Sequence) or isinstance(raw_counts, (str, bytes)):
            raise ValueError(f"participant profile marginal {field!r}.counts must be a sequence")
        counts = []
        for index, entry in enumerate(raw_counts):
            if not isinstance(entry, Mapping) or set(entry) != {"value", "count"}:
                raise ValueError(f"participant profile marginal {field!r}.counts[{index}] is malformed")
            value = entry["value"]
            if not isinstance(value, str):
                raise ValueError(f"participant profile marginal {field!r}.counts[{index}].value must be a string")
            counts.append(
                {
                    "value": value,
                    "count": _nonnegative_count(
                        entry["count"],
                        name=(f"participant_summary.profile_marginals.fields.{field}.counts[{index}].count"),
                    ),
                }
            )
        multi_select = raw_field.get("multi_select", False)
        if type(multi_select) is not bool:
            raise ValueError(f"participant profile marginal {field!r}.multi_select must be boolean")
        fields[field] = {
            "counts": sorted(counts, key=lambda entry: entry["value"]),
            "missing": _nonnegative_count(
                raw_field.get("missing", 0),
                name=f"participant_summary.profile_marginals.fields.{field}.missing",
            ),
            "multi_select": multi_select,
        }
    n_participants = _nonnegative_count(
        raw_marginals.get("n_participants"),
        name="participant_summary.profile_marginals.n_participants",
    )
    invited_total = flow["invited_total"]
    consented_current = flow["consented_current"]
    completed = flow["completed"]
    procedurally_completed_total = flow["procedurally_completed_total"]
    analyzed = flow["analyzed"]
    status_total = sum(flow["current_status"].values())
    if status_total != invited_total:
        raise ValueError("participant current-status total must equal invited_total")
    if flow["current_status"]["active"] != consented_current:
        raise ValueError("participant active status count must equal consented_current")
    if not completed <= analyzed <= consented_current <= invited_total:
        raise ValueError("participant flow must satisfy completed <= analyzed <= consented_current <= invited_total")
    if not completed <= procedurally_completed_total <= invited_total:
        raise ValueError("participant flow must satisfy completed <= procedurally_completed_total <= invited_total")
    if n_participants != analyzed:
        raise ValueError("profile marginal participant count must equal analyzed")
    for field, marginal in fields.items():
        counts = [entry["count"] for entry in marginal["counts"]]
        if any(count > n_participants for count in counts):
            raise ValueError(f"participant profile marginal {field!r} count exceeds analyzed")
        if marginal["missing"] > n_participants:
            raise ValueError(f"participant profile marginal {field!r} missing exceeds analyzed")
        if not marginal["multi_select"] and sum(counts) + marginal["missing"] != n_participants:
            raise ValueError(f"participant profile marginal {field!r} does not cover analyzed participants")
    release_fields: dict[str, Any] = {}
    for field, marginal in fields.items():
        positive_cells = [entry["count"] for entry in marginal["counts"] if entry["count"] > 0]
        if marginal["missing"] > 0:
            positive_cells.append(marginal["missing"])
        suppress_field = any(count < PROFILE_MINIMUM_CELL_SIZE for count in positive_cells)
        if suppress_field:
            release_fields[field] = {
                "counts": [],
                "missing": None,
                "multi_select": marginal["multi_select"],
                "suppressed": True,
                "minimum_cell_size": PROFILE_MINIMUM_CELL_SIZE,
            }
        else:
            release_fields[field] = {
                **marginal,
                "suppressed": False,
                "minimum_cell_size": PROFILE_MINIMUM_CELL_SIZE,
            }
    return {
        "identity": "study-local pseudonyms",
        "direct_identifiers_collected": False,
        "flow": flow,
        "profile_marginals": {
            "n_participants": n_participants,
            "joint_profiles_released": False,
            "fields": release_fields,
        },
        "aggregate_summary_supplied": True,
    }


def compute_export_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    status: Mapping[str, Any] | None,
    participant_summary: Mapping[str, Any] | None = None,
    study_provenance: Mapping[str, Any] | None = None,
    readiness_thresholds: Mapping[str, Any] | None = None,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 1477,
) -> dict[str, Any]:
    """Compute a strict-JSON metrics bundle from neutral annotation mappings."""

    ordered = _sort_rows(rows)
    collapsed = consensus_rows(ordered)
    normalized_readiness = _normalize_readiness_thresholds(readiness_thresholds)
    caption_consensus = {str(row["item_id"]): row for row in collapsed if row["phase"] == "caption"}
    image_consensus = [row for row in collapsed if row["phase"] == "image"]

    human_human = {
        phase: {
            field: _field_agreement(
                ordered,
                phase=phase,
                field=field,
                categories=categories,
            )
            for field, categories in fields.items()
        }
        for phase, fields in PHASE_FIELDS.items()
    }

    all_support_records = []
    for row in image_consensus:
        caption = caption_consensus.get(str(row["item_id"]), {})
        all_support_records.append(
            {
                **row,
                "proposed_category": row.get("category"),
                "category": caption.get("resolved_category"),
                "human_label": row.get("image_support"),
                "extraction_valid": (
                    caption.get("caption_licensed") == "yes" and caption.get("atomic_visual_claim") == "yes"
                ),
            }
        )
    support_records = [row for row in all_support_records if row["extraction_valid"]]
    typed_support_records = [row for row in support_records if row.get("category") is not None]
    extraction_valid_ids = {str(row["item_id"]) for row in support_records}
    human_human["image"]["image_support_extraction_valid"] = _field_agreement(
        [row for row in ordered if row.get("phase") == "image" and str(row.get("item_id")) in extraction_valid_ids],
        phase="image",
        field="image_support",
        categories=PHASE_FIELDS["image"]["image_support"],
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    readiness = _operational_readiness(
        ordered,
        status,
        collapsed,
        support_records,
        readiness_thresholds=normalized_readiness,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    missing_weight_count = 0
    for row in support_records:
        if not _valid_sampling_weight(row.get("sampling_weight")):
            missing_weight_count += 1
            row["sampling_weight"] = 1.0
    usefulness_records = _raw_usefulness_records(ordered, support_records)
    coverage_records, coverage_collapse = _salient_coverage_records(ordered)

    judges: dict[str, Any] = {}
    for judge in ("qwen", "gemma"):
        judge_rows = [
            {
                **row,
                "judge_label": _judge_label(row.get(f"{judge}_answer")),
            }
            for row in support_records
        ]
        summary = judge_human_summary(
            judge_rows,
            human_label_key="human_label",
            judge_label_key="judge_label",
            category_key="category",
            surface_key="surface_group",
            weight_key="sampling_weight",
            label_categories=PHASE_FIELDS["image"]["image_support"],
            positive_label="yes",
            negative_label="no",
            abstention_labels=ABSTENTION_LABELS,
        )
        summary["overall_cluster_bootstrap_95"] = cluster_bootstrap_percentile(
            judge_rows,
            _judge_nominal_exact,
            cluster_key="item_group",
            n_resamples=bootstrap_resamples,
            confidence_level=0.95,
            seed=bootstrap_seed,
        )
        summary["overall"]["cluster_bootstrap_95"] = summary["overall_cluster_bootstrap_95"]
        summary["by_category"] = [group for group in summary["by_category"] if group.get("category") is not None]
        summary["by_category_surface"] = [
            group for group in summary["by_category_surface"] if group.get("category") is not None
        ]
        for group_key, output_key in (
            ("category", "by_category"),
            ("surface_group", "by_surface"),
        ):
            for group in summary[output_key]:
                group_rows = [row for row in judge_rows if row.get(group_key) == group.get(group_key)]
                group["cluster_bootstrap_95"] = cluster_bootstrap_percentile(
                    group_rows,
                    _judge_nominal_exact,
                    cluster_key="item_group",
                    n_resamples=bootstrap_resamples,
                    confidence_level=0.95,
                    seed=bootstrap_seed,
                )
        judges[judge] = summary

    support_by_stratum = _group_support(
        support_records,
        "stratum",
        n_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        expected_groups=sorted(
            {
                *normalized_readiness["expected_strata"],
                *(row.get("stratum") for row in all_support_records if row.get("stratum") is not None),
            },
            key=_stable_value,
        ),
    )
    macro_strata = normalized_readiness["expected_strata"] or [
        str(entry["stratum"]) for entry in support_by_stratum if entry.get("stratum") is not None
    ]
    equal_stratum_macro = _equal_cell_support_macro(
        support_records,
        declared_strata=macro_strata,
        n_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    sample_summary = _sample_summary(ordered, collapsed, status)
    annotator_summary = _privacy_safe_participant_summary(participant_summary)
    if (
        annotator_summary["aggregate_summary_supplied"]
        and annotator_summary["flow"]["analyzed"] != sample_summary["participants"]
    ):
        raise ValueError("participant flow analyzed count must equal unique annotators in exported rows")

    normalized_provenance = None
    if study_provenance is not None:
        if not isinstance(study_provenance, Mapping) or set(study_provenance) != {
            "ethics_review_basis",
            "participant_notice_version",
        }:
            raise ValueError("study_provenance must contain exactly the declared public provenance fields")
        normalized_provenance = {}
        for field in ("ethics_review_basis", "participant_notice_version"):
            value = study_provenance[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"study_provenance.{field} must be a non-empty string")
            normalized_provenance[field] = value.strip()

    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "study_provenance": normalized_provenance,
        "analysis_status": readiness,
        "sample": sample_summary,
        "annotators": annotator_summary,
        "weighting": {
            "primary": "inverse-probability design weight",
            "repeat_items_in_primary_metrics": False,
            "missing_weights_unit_fallback_count": missing_weight_count,
        },
        "human_human_pre_adjudication": human_human,
        "human_consensus": {
            "caption_items": sum(row["phase"] == "caption" for row in collapsed),
            "image_items": len(image_consensus),
            "extraction_valid_items": len(support_records),
            "image_items_excluded_invalid_extraction": len(all_support_records) - len(support_records),
            "image_items_excluded_unresolved_category_from_per_type": (
                len(support_records) - len(typed_support_records)
            ),
            "image_substantive_items": sum(row.get("image_support") in {"yes", "no"} for row in support_records),
            "image_abstention_items": sum(row.get("image_support") in ABSTENTION_LABELS for row in support_records),
            "image_unresolved_items": sum(row.get("image_support") is None for row in support_records),
        },
        "human_supported_rate": {
            "overall": _weighted_support_summary(
                support_records,
                n_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "by_category": _group_support(
                typed_support_records,
                "category",
                n_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "by_surface_group": _group_support(
                support_records,
                "surface_group",
                n_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
            ),
            "by_stratum": support_by_stratum,
            "equal_stratum_macro": equal_stratum_macro,
        },
        "exploratory_control_usefulness": {
            "overall": _usefulness_summary(usefulness_records),
            "by_category": _group_usefulness(
                [row for row in usefulness_records if row.get("category") is not None],
                "category",
            ),
            "by_surface_group": _group_usefulness(usefulness_records, "surface_group"),
            "pre_adjudication_ordinal_agreement": _usefulness_ordinal_agreement(usefulness_records),
            "reporting_rule": (
                "exploratory ordinal result; report the full distribution and weighted median only; "
                "never combine with extraction validity or image support"
            ),
        },
        "exploratory_salient_content_coverage": {
            "overall": _salient_coverage_summary(coverage_records),
            "by_surface_group": [
                {
                    "surface_group": group,
                    **_salient_coverage_summary(group_rows),
                }
                for group, group_rows in sorted(
                    (
                        (group, [row for row in coverage_records if row.get("surface_group") == group])
                        for group in {row.get("surface_group") for row in coverage_records}
                    ),
                    key=lambda pair: _stable_value(pair[0]),
                )
            ],
            "pre_adjudication_ordinal_agreement": _salient_coverage_agreement(
                coverage_records
            ),
            "deduplication": coverage_collapse,
            "reporting_rule": (
                "exploratory sample-conditional ordinal result; one deterministic "
                "record per participant and image-caption pair; unweighted because "
                "the source sample was stratified over claims rather than pairs; "
                "do not call this exhaustive image recall"
            ),
        },
        "judge_human": judges,
        "method": {
            "consensus": (
                "adjudication when present, otherwise agreed initial labels "
                "(strict >50% consensus with at least two ratings)"
            ),
            "agreement": "pre-adjudication; Gwet AC1 primary sensitivity to prevalence, with exact and Cohen kappa",
            "binary_abstentions": sorted(ABSTENTION_LABELS),
            "uncertainty": "95% percentile bootstrap over image clusters",
            "uncertainty_interpretation": INTERVAL_SCOPE_NOTE,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": bootstrap_seed,
            "per_type_grouping": (
                "human-resolved category; original proposed category and sampling stratum retained separately"
            ),
        },
    }


def _csv_text(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _metric_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for phase, fields in metrics["human_human_pre_adjudication"].items():
        for field, summary in fields.items():
            nominal = summary["pooled"]["nominal"]
            binary = summary["pooled"]["binary"]
            values = {
                "exact_agreement": nominal["exact_agreement"],
                "mean_pairwise_cohen_kappa": nominal["mean_pairwise_cohen_kappa"],
                "gwet_ac1": nominal["gwet_ac1"],
                "positive_agreement": None if binary is None else binary["positive_agreement"],
                "negative_agreement": None if binary is None else binary["negative_agreement"],
            }
            for metric, value in values.items():
                output.append(
                    {
                        "family": "human_human_pre_adjudication",
                        "phase": phase,
                        "field": field,
                        "judge": "",
                        "category": "",
                        "surface_group": "",
                        "metric": metric,
                        "value": value,
                        "n": nominal["n_pairs"],
                        "weighted": False,
                        "provisional": metrics["analysis_status"]["provisional"],
                    }
                )
            interval = summary.get("cluster_bootstrap_95")
            if interval is not None:
                for bound in ("lower", "upper"):
                    output.append(
                        {
                            "family": "human_human_pre_adjudication",
                            "phase": phase,
                            "field": field,
                            "judge": "",
                            "category": "",
                            "surface_group": "",
                            "metric": f"exact_agreement_cluster_bootstrap_95_{bound}",
                            "value": interval[bound],
                            "n": nominal["n_pairs"],
                            "weighted": False,
                            "provisional": metrics["analysis_status"]["provisional"],
                        }
                    )
    for judge, summary in metrics["judge_human"].items():
        scopes = [("overall", "", "", summary["overall"])]
        scopes.extend(("category", entry["category"], "", entry) for entry in summary["by_category"])
        scopes.extend(("surface_group", "", entry["surface_group"], entry) for entry in summary["by_surface"])
        for _scope, category, surface_group, group in scopes:
            nominal = group["weighted"]["nominal"]
            binary = group["weighted"]["binary"]
            values = {
                "exact_agreement": nominal["exact_agreement"],
                "cohen_kappa": nominal["cohen_kappa"],
                "gwet_ac1": nominal["gwet_ac1"],
                "positive_agreement": None if binary is None else binary["positive_agreement"],
                "negative_agreement": None if binary is None else binary["negative_agreement"],
            }
            for metric, value in values.items():
                output.append(
                    {
                        "family": "judge_human",
                        "phase": "image",
                        "field": "image_support",
                        "judge": judge,
                        "category": category,
                        "surface_group": surface_group,
                        "metric": metric,
                        "value": value,
                        "n": group["n_compared"],
                        "weighted": True,
                        "provisional": metrics["analysis_status"]["provisional"],
                    }
                )
            interval = group.get("cluster_bootstrap_95")
            if interval is not None:
                for metric in ("lower", "upper"):
                    output.append(
                        {
                            "family": "judge_human",
                            "phase": "image",
                            "field": "image_support",
                            "judge": judge,
                            "category": category,
                            "surface_group": surface_group,
                            "metric": f"exact_agreement_cluster_bootstrap_95_{metric}",
                            "value": interval[metric],
                            "n": group["n_compared"],
                            "weighted": True,
                            "provisional": metrics["analysis_status"]["provisional"],
                        }
                    )
    support_scopes = [("", "", metrics["human_supported_rate"]["overall"])]
    support_scopes.extend((entry["category"], "", entry) for entry in metrics["human_supported_rate"]["by_category"])
    support_scopes.extend(
        ("", entry["surface_group"], entry) for entry in metrics["human_supported_rate"]["by_surface_group"]
    )
    for entry in metrics["human_supported_rate"]["by_stratum"]:
        stratum = str(entry["stratum"])
        surface_group, separator, category = stratum.partition("/")
        support_scopes.append(
            (
                category if separator else "",
                surface_group if separator else stratum,
                entry,
            )
        )
    for category, surface_group, summary in support_scopes:
        output.append(
            {
                "family": "human_supported_rate",
                "phase": "image",
                "field": "image_support",
                "judge": "",
                "category": category,
                "surface_group": surface_group,
                "metric": "design_weighted_proportion",
                "value": summary["design_weighted"],
                "n": summary["n_substantive"],
                "weighted": True,
                "provisional": metrics["analysis_status"]["provisional"],
            }
        )
        interval = summary.get("cluster_bootstrap_95")
        if interval is not None:
            for bound in ("lower", "upper"):
                output.append(
                    {
                        "family": "human_supported_rate",
                        "phase": "image",
                        "field": "image_support",
                        "judge": "",
                        "category": category,
                        "surface_group": surface_group,
                        "metric": f"design_weighted_cluster_bootstrap_95_{bound}",
                        "value": interval[bound],
                        "n": summary["n_substantive"],
                        "weighted": True,
                        "provisional": metrics["analysis_status"]["provisional"],
                    }
                )
    macro = metrics["human_supported_rate"]["equal_stratum_macro"]
    output.append(
        {
            "family": "human_supported_rate",
            "phase": "image",
            "field": "image_support",
            "judge": "",
            "category": "",
            "surface_group": "",
            "metric": "equal_stratum_macro_unweighted_proportion",
            "value": macro["estimate"],
            "n": macro["n_cells_in_macro"],
            "weighted": False,
            "provisional": metrics["analysis_status"]["provisional"],
        }
    )
    macro_interval = macro.get("cluster_bootstrap_95")
    if macro_interval is not None:
        for bound in ("lower", "upper"):
            output.append(
                {
                    "family": "human_supported_rate",
                    "phase": "image",
                    "field": "image_support",
                    "judge": "",
                    "category": "",
                    "surface_group": "",
                    "metric": f"equal_stratum_macro_cluster_bootstrap_95_{bound}",
                    "value": macro_interval[bound],
                    "n": macro["n_cells_in_macro"],
                    "weighted": False,
                    "provisional": metrics["analysis_status"]["provisional"],
                }
            )
    usefulness = metrics["exploratory_control_usefulness"]
    usefulness_scopes = [("", "", usefulness["overall"])]
    usefulness_scopes.extend((entry["category"], "", entry) for entry in usefulness["by_category"])
    usefulness_scopes.extend(("", entry["surface_group"], entry) for entry in usefulness["by_surface_group"])
    for category, surface_group, summary in usefulness_scopes:
        for label, value in summary["design_weighted_distribution"].items():
            output.append(
                {
                    "family": "exploratory_control_usefulness",
                    "phase": "image",
                    "field": "control_usefulness",
                    "judge": "",
                    "category": category,
                    "surface_group": surface_group,
                    "metric": f"design_weighted_share:{label}",
                    "value": value,
                    "n": summary["n_eligible_annotations"],
                    "weighted": True,
                    "provisional": metrics["analysis_status"]["provisional"],
                }
            )
        output.append(
            {
                "family": "exploratory_control_usefulness",
                "phase": "image",
                "field": "control_usefulness",
                "judge": "",
                "category": category,
                "surface_group": surface_group,
                "metric": "design_weighted_ordinal_median",
                "value": summary["design_weighted_ordinal_median"],
                "n": summary["n_rated"],
                "weighted": True,
                "provisional": metrics["analysis_status"]["provisional"],
            }
        )
    ordinal = usefulness["pre_adjudication_ordinal_agreement"]
    for metric in ("exact_rate", "within_one_rate", "median_absolute_difference"):
        output.append(
            {
                "family": "exploratory_control_usefulness",
                "phase": "image",
                "field": "control_usefulness",
                "judge": "",
                "category": "",
                "surface_group": "",
                "metric": f"pre_adjudication_{metric}",
                "value": ordinal[metric],
                "n": ordinal["n_numeric_pairs"],
                "weighted": False,
                "provisional": metrics["analysis_status"]["provisional"],
            }
        )
    coverage = metrics["exploratory_salient_content_coverage"]
    coverage_scopes = [("", coverage["overall"])]
    coverage_scopes.extend(
        (entry["surface_group"], entry)
        for entry in coverage["by_surface_group"]
    )
    for surface_group, summary in coverage_scopes:
        for label, value in summary["unweighted_distribution"].items():
            output.append(
                {
                    "family": "exploratory_salient_content_coverage",
                    "phase": "image",
                    "field": "salient_coverage",
                    "judge": "",
                    "category": "",
                    "surface_group": surface_group,
                    "metric": f"unweighted_share:{label}",
                    "value": value,
                    "n": summary["n_rated"],
                    "weighted": False,
                    "provisional": metrics["analysis_status"]["provisional"],
                }
            )
        output.append(
            {
                "family": "exploratory_salient_content_coverage",
                "phase": "image",
                "field": "salient_coverage",
                "judge": "",
                "category": "",
                "surface_group": surface_group,
                "metric": "unweighted_ordinal_median",
                "value": summary["unweighted_ordinal_median"],
                "n": summary["n_rated"],
                "weighted": False,
                "provisional": metrics["analysis_status"]["provisional"],
            }
        )
    coverage_ordinal = coverage["pre_adjudication_ordinal_agreement"]
    for metric in ("exact_rate", "within_one_rate", "median_absolute_difference"):
        output.append(
            {
                "family": "exploratory_salient_content_coverage",
                "phase": "image",
                "field": "salient_coverage",
                "judge": "",
                "category": "",
                "surface_group": "",
                "metric": f"pre_adjudication_{metric}",
                "value": coverage_ordinal[metric],
                "n": coverage_ordinal["n_numeric_pairs"],
                "weighted": False,
                "provisional": metrics["analysis_status"]["provisional"],
            }
        )
    for metric, value in coverage["deduplication"].items():
        output.append(
            {
                "family": "exploratory_salient_content_coverage",
                "phase": "image",
                "field": "salient_coverage",
                "judge": "",
                "category": "",
                "surface_group": "",
                "metric": metric,
                "value": value,
                "n": coverage["deduplication"]["n_input_claim_annotations"],
                "weighted": False,
                "provisional": metrics["analysis_status"]["provisional"],
            }
        )
    return output


def _matrix_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    def append_matrix(
        matrix: Mapping[str, Any],
        *,
        family: str,
        phase: str,
        field: str,
        judge: str = "",
        category: str = "",
        surface_group: str = "",
    ) -> None:
        labels = matrix["labels"]
        for row_index, row_label in enumerate(labels):
            for column_index, column_label in enumerate(labels):
                output.append(
                    {
                        "family": family,
                        "phase": phase,
                        "field": field,
                        "judge": judge,
                        "category": category,
                        "surface_group": surface_group,
                        "rater_a_label": row_label,
                        "rater_b_label": column_label,
                        "count": matrix["matrix"][row_index][column_index],
                        "weighted": matrix["weighted"],
                    }
                )

    for phase, fields in metrics["human_human_pre_adjudication"].items():
        for field, summary in fields.items():
            append_matrix(
                summary["pooled"]["nominal"]["confusion_matrix"],
                family="human_human_pre_adjudication",
                phase=phase,
                field=field,
            )
    for judge, summary in metrics["judge_human"].items():
        append_matrix(
            summary["overall"]["weighted"]["nominal"]["confusion_matrix"],
            family="judge_human",
            phase="image",
            field="image_support",
            judge=judge,
        )
        for entry in summary["by_category"]:
            append_matrix(
                entry["weighted"]["nominal"]["confusion_matrix"],
                family="judge_human",
                phase="image",
                field="image_support",
                judge=judge,
                category=str(entry["category"]),
            )
    return output


def _sample_flow_rows(
    rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    status: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    sample = metrics["sample"]
    flow = [
        {"phase": "sample", "stage": "observed_base_items", "count": sample["observed_base_items"], "note": ""},
        {
            "phase": "sample",
            "stage": "observed_hidden_repeat_items",
            "count": sample["observed_repeat_items"],
            "note": "excluded from primary estimates",
        },
        {"phase": "sample", "stage": "image_clusters", "count": sample["image_clusters"], "note": ""},
        {"phase": "sample", "stage": "participants", "count": sample["participants"], "note": "pseudonymous"},
    ]
    participant_flow = metrics["annotators"].get("flow", {})
    for stage in (
        "invited_total",
        "consented_current",
        "completed",
        "procedurally_completed_total",
        "analyzed",
    ):
        if participant_flow.get(stage) is not None:
            flow.append(
                {
                    "phase": "participants",
                    "stage": stage,
                    "count": participant_flow[stage],
                    "note": "aggregate participant flow",
                }
            )
    current_status = participant_flow.get("current_status", {})
    if isinstance(current_status, Mapping):
        for status_name, count in sorted(current_status.items()):
            flow.append(
                {
                    "phase": "participants",
                    "stage": f"current_status:{status_name}",
                    "count": count,
                    "note": "aggregate current status",
                }
            )
    for phase in PHASE_FIELDS:
        assignments = status.get("assignments", {}).get(phase, {}) if isinstance(status, Mapping) else {}
        flow.extend(
            [
                {
                    "phase": phase,
                    "stage": "planned_assignments",
                    "count": assignments.get("total"),
                    "note": "unknown without operational status" if status is None else "",
                },
                {
                    "phase": phase,
                    "stage": "completed_annotations",
                    "count": sample["annotations_by_phase"].get(phase, 0),
                    "note": "",
                },
                {
                    "phase": phase,
                    "stage": "consensus_items",
                    "count": sample["consensus_items_by_phase"].get(phase, 0),
                    "note": "adjudication or agreed initial labels",
                },
            ]
        )
    image = metrics["human_consensus"]
    flow.extend(
        [
            {
                "phase": "image",
                "stage": "substantive_consensus_items",
                "count": image["image_substantive_items"],
                "note": "yes/no denominator for binary judge agreement",
            },
            {
                "phase": "image",
                "stage": "abstention_consensus_items",
                "count": image["image_abstention_items"],
                "note": "excluded from binary denominator",
            },
            {
                "phase": "image",
                "stage": "unresolved_consensus_items",
                "count": image["image_unresolved_items"],
                "note": "",
            },
        ]
    )
    return flow


def _format_estimate(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _rebuttal_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    human = metrics["human_human_pre_adjudication"]["image"]["image_support_extraction_valid"]["pooled"]
    rows = [
        {
            "comparison": "Human–human (pre-adjudication)",
            "scope": "Extraction-valid",
            "n": human["nominal"]["n_pairs"],
            "exact": human["nominal"]["exact_agreement"],
            "exact_ci": metrics["human_human_pre_adjudication"]["image"]["image_support_extraction_valid"].get(
                "cluster_bootstrap_95"
            ),
            "ac1": human["nominal"]["gwet_ac1"],
            "positive": human["binary"]["positive_agreement"],
            "negative": human["binary"]["negative_agreement"],
        }
    ]
    for judge, summary in metrics["judge_human"].items():
        groups = [("All", summary["overall"])]
        groups.extend((str(entry["category"]), entry) for entry in summary["by_category"])
        for scope, group in groups:
            nominal = group["weighted"]["nominal"]
            binary = group["weighted"]["binary"]
            exact_ci = group.get("cluster_bootstrap_95")
            rows.append(
                {
                    "comparison": f"{judge.title()}–human",
                    "scope": scope,
                    "n": group["n_compared"],
                    "exact": nominal["exact_agreement"],
                    "exact_ci": exact_ci,
                    "ac1": nominal["gwet_ac1"],
                    "positive": binary["positive_agreement"],
                    "negative": binary["negative_agreement"],
                }
            )
    return rows


def _rebuttal_markdown(metrics: Mapping[str, Any]) -> str:
    status = metrics["analysis_status"]
    if not status["ready"]:
        reasons = "\n".join(f"- `{reason}`" for reason in status["reasons"])
        return (
            "# Human-CBU rebuttal table\n\n"
            "> **NOT READY — do not quote partial values in the rebuttal.**\n\n"
            "The deterministic export gate is closed for the following reasons:\n\n"
            f"{reasons}\n"
        )
    lines = [
        "# Human-CBU rebuttal table",
        "",
        "| Comparison | Scope | n | Exact agreement (95% cluster CI) "
        "| Gwet AC1 | Positive agreement | Negative agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _rebuttal_rows(metrics):
        interval = row["exact_ci"]
        exact = _format_estimate(row["exact"])
        if interval is not None:
            exact = f"{exact} [{_format_estimate(interval['lower'])}, {_format_estimate(interval['upper'])}]"
        lines.append(
            "| {comparison} | {scope} | {n} | {exact} | {ac1} | {positive} | {negative} |".format(
                comparison=row["comparison"],
                scope=row["scope"],
                n=row["n"],
                exact=exact,
                ac1=_format_estimate(row["ac1"]),
                positive=_format_estimate(row["positive"]),
                negative=_format_estimate(row["negative"]),
            )
        )
    lines.extend(
        [
            "",
            "Values are generated from blinded annotations. Judge–human rows use inverse-probability "
            "sampling weights; binary agreement excludes declared abstention labels. Gwet AC1 is "
            "reported alongside exact, positive, and negative agreement because support prevalence "
            "can make Cohen's kappa unstable.",
            "",
            INTERVAL_SCOPE_NOTE,
            "",
        ]
    )
    return "\n".join(lines)


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _rebuttal_latex(metrics: Mapping[str, Any]) -> str:
    status = metrics["analysis_status"]
    if not status["ready"]:
        reasons = "; ".join(status["reasons"])
        return (
            "% NOT READY: do not quote partial values in the rebuttal.\n"
            "\\begin{tabular}{ll}\n"
            "\\toprule\n"
            f"Status & NOT READY ({_latex_escape(reasons)}) \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
        )
    lines = [
        "% Generated by audit_recap_t2i.human_cbu.export; do not hand-edit.",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Comparison & Scope & $n$ & Exact (95\\% CI) & Gwet AC1 & Pos. agr. & Neg. agr. \\\\",
        "\\midrule",
    ]
    for row in _rebuttal_rows(metrics):
        interval = row["exact_ci"]
        exact = _format_estimate(row["exact"])
        if interval is not None:
            exact = f"{exact} [{_format_estimate(interval['lower'])}, {_format_estimate(interval['upper'])}]"
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                _latex_escape(row["comparison"]),
                _latex_escape(row["scope"]),
                row["n"],
                exact,
                _format_estimate(row["ac1"]),
                _format_estimate(row["positive"]),
                _format_estimate(row["negative"]),
            )
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            f"\\par\\footnotesize {_latex_escape(INTERVAL_SCOPE_NOTE)}",
            "",
        ]
    )
    return "\n".join(lines)


def _rebuttal_summary(metrics: Mapping[str, Any]) -> str:
    status = metrics["analysis_status"]
    if not status["ready"]:
        reasons = "\n".join(f"- `{reason}`" for reason in status["reasons"])
        return (
            "# Human-CBU rebuttal summary\n\n"
            "> **NOT READY. Do not use this file as empirical evidence.**\n\n"
            "Collection or adjudication is incomplete, so no result prose was generated:\n\n"
            f"{reasons}\n"
        )

    human = metrics["human_human_pre_adjudication"]["image"]["image_support_extraction_valid"]["pooled"]
    qwen = metrics["judge_human"]["qwen"]["overall"]["weighted"]
    gemma = metrics["judge_human"]["gemma"]["overall"]["weighted"]
    sample = metrics["sample"]
    support = metrics["human_consensus"]
    return (
        "# Human-CBU rebuttal summary\n\n"
        "<!-- Generated from metrics.json; do not replace values by hand. -->\n\n"
        f"We added a blinded human-anchored validation over {sample['observed_base_items']} sampled "
        f"claims ({sample['participants']} pseudonymous annotators; hidden repeats are excluded from "
        f"primary estimates). For image support, {support['image_substantive_items']} items received "
        "a substantive yes/no consensus. Pre-adjudication human–human exact agreement was "
        f"{_format_estimate(human['nominal']['exact_agreement'])} (Gwet AC1 "
        f"{_format_estimate(human['nominal']['gwet_ac1'])}). Design-weighted exact agreement with "
        f"human consensus was {_format_estimate(qwen['nominal']['exact_agreement'])} for Qwen and "
        f"{_format_estimate(gemma['nominal']['exact_agreement'])} for Gemma. Per-type results, "
        "including count, relation, and text-rendering claims, are reported in the generated table. "
        "These values provide a human anchor for judge-conditional audit measurements; they do not "
        "certify the full corpus or establish downstream T2I utility. "
        f"{INTERVAL_SCOPE_NOTE}\n"
    )


def build_export_bundle(
    rows: Sequence[Mapping[str, Any]],
    *,
    status: Mapping[str, Any] | None,
    participant_summary: Mapping[str, Any] | None = None,
    study_provenance: Mapping[str, Any] | None = None,
    readiness_thresholds: Mapping[str, Any] | None = None,
    include_public_rows: bool = False,
    public_id_namespace: str | None = None,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 1477,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Build all requested export files in memory.

    Returning text separately from I/O makes deterministic behavior easy to
    test and allows callers to atomically publish the complete bundle.
    """

    if type(include_public_rows) is not bool:
        raise ValueError("include_public_rows must be boolean")
    if include_public_rows and (not isinstance(public_id_namespace, str) or not public_id_namespace.strip()):
        raise ValueError("public_id_namespace must be a non-empty string when public rows are included")
    ordered = _sort_rows(rows)
    metrics = compute_export_metrics(
        ordered,
        status=status,
        participant_summary=participant_summary,
        study_provenance=study_provenance,
        readiness_thresholds=readiness_thresholds,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    private_rows = private_audit_rows(ordered)
    metric_rows = _metric_rows(metrics)
    matrix_rows = _matrix_rows(metrics)
    flow_rows = _sample_flow_rows(ordered, metrics, status)

    files = {
        "annotations.private.jsonl": "".join(f"{_stable_json(row)}\n" for row in private_rows),
        "sample_flow.csv": _csv_text(("phase", "stage", "count", "note"), flow_rows),
        "metrics.json": f"{_stable_json(metrics, indent=2)}\n",
        "metrics.csv": _csv_text(
            (
                "family",
                "phase",
                "field",
                "judge",
                "category",
                "surface_group",
                "metric",
                "value",
                "n",
                "weighted",
                "provisional",
            ),
            metric_rows,
        ),
        "confusion_matrices.csv": _csv_text(
            (
                "family",
                "phase",
                "field",
                "judge",
                "category",
                "surface_group",
                "rater_a_label",
                "rater_b_label",
                "count",
                "weighted",
            ),
            matrix_rows,
        ),
        "rebuttal_table.md": _rebuttal_markdown(metrics),
        "rebuttal_table.tex": _rebuttal_latex(metrics),
        "rebuttal_summary.md": _rebuttal_summary(metrics),
    }
    if include_public_rows:
        public_rows = public_annotation_rows(
            ordered,
            public_id_namespace=str(public_id_namespace),
        )
        files["annotations.public.csv"] = _csv_text(PUBLIC_COLUMNS, public_rows)
    return files, metrics


def write_export_bundle(
    rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    status: Mapping[str, Any] | None,
    participant_summary: Mapping[str, Any] | None = None,
    study_provenance: Mapping[str, Any] | None = None,
    readiness_thresholds: Mapping[str, Any] | None = None,
    include_public_rows: bool = False,
    public_id_namespace: str | None = None,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 1477,
) -> dict[str, Any]:
    """Write a complete export bundle using per-file atomic replacement."""

    files, metrics = build_export_bundle(
        rows,
        status=status,
        participant_summary=participant_summary,
        study_provenance=study_provenance,
        readiness_thresholds=readiness_thresholds,
        include_public_rows=include_public_rows,
        public_id_namespace=public_id_namespace,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if not include_public_rows:
        # Reusing an export directory after a withdrawal or a privacy-policy
        # change must not leave a stale row-level release beside the new
        # aggregate-only bundle.
        (destination / "annotations.public.csv").unlink(missing_ok=True)
    for name, content in sorted(files.items()):
        target = destination / name
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)
        if name == "annotations.private.jsonl":
            target.chmod(0o600)
    return {
        "output_dir": str(destination),
        "files": sorted(files),
        "analysis_status": metrics["analysis_status"],
    }
