from __future__ import annotations

import json
from collections import Counter

import pytest

from audit_recap_t2i.human_cbu.metrics import (
    binary_agreement,
    cluster_bootstrap_percentile,
    cohen_kappa,
    confusion_matrix,
    consensus_label,
    exact_agreement,
    gwet_ac1,
    judge_human_summary,
    nominal_agreement,
    pre_adjudication_agreement,
    weighted_proportion,
)


def _assert_strict_json(value) -> None:
    json.dumps(value, allow_nan=False)


def test_nominal_confusion_and_agreement_metrics_match_hand_calculation() -> None:
    human = ["yes", "yes", "no", "no"]
    judge = ["yes", "no", "no", "no"]

    confusion = confusion_matrix(human, judge, categories=["yes", "no"])
    assert confusion == {
        "labels": ["yes", "no"],
        "matrix": [[1, 1], [0, 2]],
        "n_pairs": 4,
        "weight_total": 4.0,
        "weight_scale": 1.0,
        "weighted": False,
    }
    assert exact_agreement(human, judge) == pytest.approx(0.75)
    assert cohen_kappa(human, judge, categories=["yes", "no"]) == pytest.approx(0.5)
    assert gwet_ac1(human, judge, categories=["yes", "no"]) == pytest.approx(9 / 17)

    summary = nominal_agreement(human, judge, categories=["yes", "no"])
    assert summary["exact_agreement"] == pytest.approx(0.75)
    assert summary["cohen_kappa"] == pytest.approx(0.5)
    assert summary["gwet_ac1"] == pytest.approx(9 / 17)
    _assert_strict_json(summary)


def test_nominal_metrics_support_design_weights() -> None:
    first = ["yes", "yes", "no"]
    second = ["yes", "no", "no"]
    weights = [1.0, 3.0, 2.0]

    summary = nominal_agreement(first, second, categories=["yes", "no"], weights=weights)
    assert summary["confusion_matrix"]["matrix"] == [[1.0, 3.0], [0.0, 2.0]]
    assert summary["confusion_matrix"]["weighted"] is True
    assert summary["exact_agreement"] == pytest.approx(0.5)


def test_nominal_metrics_are_scale_invariant_for_extreme_finite_weights() -> None:
    first = ["yes", "yes", "no", "no"]
    second = ["yes", "no", "yes", "no"]
    base = nominal_agreement(first, second, categories=["yes", "no"], weights=[1.0, 2.0, 3.0, 4.0])
    tiny = nominal_agreement(
        first,
        second,
        categories=["yes", "no"],
        weights=[1e-300, 2e-300, 3e-300, 4e-300],
    )
    assert tiny["exact_agreement"] == pytest.approx(base["exact_agreement"])
    assert tiny["cohen_kappa"] == pytest.approx(base["cohen_kappa"])
    assert tiny["gwet_ac1"] == pytest.approx(base["gwet_ac1"])

    overflow_sum = nominal_agreement(
        ["yes", "no"],
        ["yes", "no"],
        categories=["yes", "no"],
        weights=[1e308, 1e308],
    )
    assert overflow_sum["exact_agreement"] == 1.0
    assert overflow_sum["confusion_matrix"]["weight_scale"] == 1e308
    assert overflow_sum["confusion_matrix"]["matrix"] == [[1.0, 0.0], [0.0, 1.0]]
    _assert_strict_json([tiny, overflow_sum])


def test_nominal_degenerate_denominators_return_none() -> None:
    empty = nominal_agreement([], [], categories=["yes", "no"])
    assert empty["exact_agreement"] is None
    assert empty["cohen_kappa"] is None
    assert empty["gwet_ac1"] is None

    unanimous = nominal_agreement(["yes", "yes"], ["yes", "yes"], categories=["yes", "no"])
    assert unanimous["exact_agreement"] == 1.0
    assert unanimous["cohen_kappa"] is None
    assert unanimous["gwet_ac1"] == 1.0

    one_observed_category = nominal_agreement(["yes"], ["yes"])
    assert one_observed_category["gwet_ac1"] is None

    zero_weight = nominal_agreement(["yes"], ["yes"], categories=["yes", "no"], weights=[0.0])
    assert zero_weight["exact_agreement"] is None
    assert zero_weight["cohen_kappa"] is None
    assert zero_weight["gwet_ac1"] is None
    _assert_strict_json([empty, unanimous, one_observed_category, zero_weight])


def test_nominal_validation_rejects_mismatched_or_unknown_data() -> None:
    with pytest.raises(ValueError, match="same length"):
        confusion_matrix(["yes"], [])
    with pytest.raises(ValueError, match="missing from categories"):
        confusion_matrix(["partial"], ["yes"], categories=["yes", "no"])
    with pytest.raises(ValueError, match="duplicates"):
        confusion_matrix(["yes"], ["yes"], categories=["yes", "yes"])
    with pytest.raises(ValueError, match="non-negative"):
        confusion_matrix(["yes"], ["yes"], weights=[-1.0])
    with pytest.raises(ValueError, match="length"):
        confusion_matrix(["yes"], ["yes"], weights=[])


def test_binary_agreement_reports_specific_agreement_and_abstentions() -> None:
    first = ["yes", "yes", "no", "no", "uncertain", "unjudgeable", "yes"]
    second = ["yes", "no", "no", "no", "yes", "unjudgeable", "image_unavailable"]

    result = binary_agreement(first, second)

    assert result["n_total"] == 7
    assert result["n_binary"] == 4
    assert result["n_excluded"] == 3
    assert result["binary_coverage"] == pytest.approx(4 / 7)
    assert result["confusion_matrix"]["matrix"] == [[1, 1], [0, 2]]
    assert result["exact_agreement"] == pytest.approx(0.75)
    assert result["positive_agreement"] == pytest.approx(2 / 3)
    assert result["negative_agreement"] == pytest.approx(4 / 5)
    assert result["abstention"]["rater_a_count"] == 2
    assert result["abstention"]["rater_b_count"] == 2
    assert result["abstention"]["either_count"] == 3
    assert result["abstention"]["both_count"] == 1
    assert result["abstention"]["rater_a_labels"] == [
        {"label": "uncertain", "count": 1},
        {"label": "unjudgeable", "count": 1},
    ]
    assert result["abstention"]["rater_b_labels"] == [
        {"label": "image_unavailable", "count": 1},
        {"label": "unjudgeable", "count": 1},
    ]
    _assert_strict_json(result)


def test_binary_agreement_retains_weighted_abstention_rates() -> None:
    result = binary_agreement(
        ["yes", "uncertain", "no"],
        ["yes", "no", "no"],
        weights=[1.0, 7.0, 2.0],
    )

    assert result["binary_coverage"] == pytest.approx(2 / 3)
    assert result["weighted_binary_coverage"] == pytest.approx(0.3)
    assert result["abstention"]["either_rate"] == pytest.approx(1 / 3)
    assert result["abstention"]["weighted_either_rate"] == pytest.approx(0.7)
    assert result["confusion_matrix"]["matrix"] == [[1.0, 0.0], [0.0, 2.0]]
    assert result["exact_agreement"] == 1.0


def test_binary_agreement_rejects_unexpected_substantive_labels() -> None:
    with pytest.raises(ValueError, match="Unexpected non-binary label"):
        binary_agreement(["partial"], ["yes"])
    with pytest.raises(ValueError, match="must differ"):
        binary_agreement(["yes"], ["yes"], positive_label="yes", negative_label="yes")
    with pytest.raises(ValueError, match="cannot be abstention"):
        binary_agreement(["yes"], ["yes"], abstention_labels={"yes"})


def test_consensus_requires_a_strict_non_abstention_majority() -> None:
    assert consensus_label(["yes", "yes"]) == "yes"
    assert consensus_label(["yes", "no"]) is None
    assert consensus_label(["yes", "yes", "no"]) == "yes"
    assert consensus_label(["yes", "no", "uncertain"]) is None
    assert consensus_label(["uncertain", "uncertain"]) is None
    assert consensus_label(["yes"]) is None
    assert consensus_label(["yes"], minimum_raters=1) == "yes"
    assert consensus_label([None, "yes", "yes"]) == "yes"
    with pytest.raises(ValueError, match="at least 1"):
        consensus_label(["yes"], minimum_raters=0)


def test_pre_adjudication_agreement_reports_pooled_and_rater_pair_results() -> None:
    annotations = [
        {"item_id": "i1", "participant_pseudonym": "r1", "label": "yes"},
        {"item_id": "i1", "participant_pseudonym": "r2", "label": "yes"},
        {"item_id": "i1", "participant_pseudonym": "r3", "label": "no"},
        {"item_id": "i2", "participant_pseudonym": "r1", "label": "no"},
        {"item_id": "i2", "participant_pseudonym": "r2", "label": "no"},
        {"item_id": "i2", "participant_pseudonym": "r3", "label": None},
        {"item_id": "i3", "participant_pseudonym": "r2", "label": "uncertain"},
        {"item_id": "i3", "participant_pseudonym": "r3", "label": "uncertain"},
    ]

    result = pre_adjudication_agreement(
        annotations,
        categories=["yes", "no", "uncertain"],
    )

    assert result["n_annotations"] == 8
    assert result["n_missing_labels"] == 1
    assert result["n_items"] == 3
    assert result["n_rated_items"] == 3
    assert result["n_items_with_pairs"] == 3
    assert result["n_items_with_consensus"] == 2
    assert result["consensus_rate"] == pytest.approx(2 / 3)
    assert result["n_pairwise_comparisons"] == 5
    assert [(entry["rater_a"], entry["rater_b"]) for entry in result["pairwise"]] == [
        ("r1", "r2"),
        ("r1", "r3"),
        ("r2", "r3"),
    ]
    r1_r2 = result["pairwise"][0]
    assert r1_r2["nominal"]["n_pairs"] == 2
    assert r1_r2["nominal"]["exact_agreement"] == 1.0
    assert result["pooled"]["binary"]["n_binary"] == 4
    assert result["pooled"]["binary"]["n_excluded"] == 1
    _assert_strict_json(result)


def test_pre_adjudication_supports_composite_items_and_nominal_only_tasks() -> None:
    annotations = [
        {"image": "a", "claim": "1", "participant": "r1", "meaningful": "limited"},
        {"image": "a", "claim": "1", "participant": "r2", "meaningful": "yes"},
    ]
    result = pre_adjudication_agreement(
        annotations,
        item_key=["image", "claim"],
        rater_key="participant",
        label_key="meaningful",
        positive_label=None,
        negative_label=None,
    )
    assert result["pooled"]["binary"] is None
    assert result["pooled"]["nominal"]["confusion_matrix"]["labels"] == ["limited", "yes"]


def test_pre_adjudication_pooled_summary_is_invariant_to_rater_pseudonym_order() -> None:
    first_names = [
        {"item_id": "i1", "participant_pseudonym": "a", "label": "yes"},
        {"item_id": "i1", "participant_pseudonym": "b", "label": "no"},
        {"item_id": "i1", "participant_pseudonym": "c", "label": "uncertain"},
    ]
    renamed = [
        {"item_id": "i1", "participant_pseudonym": "a", "label": "no"},
        {"item_id": "i1", "participant_pseudonym": "b", "label": "uncertain"},
        {"item_id": "i1", "participant_pseudonym": "c", "label": "yes"},
    ]
    first = pre_adjudication_agreement(first_names, categories=["yes", "no", "uncertain"])
    second = pre_adjudication_agreement(renamed, categories=["yes", "no", "uncertain"])

    assert first["pooled"] == second["pooled"]
    assert first["pooled"]["nominal"]["cohen_kappa"] is None
    assert first["pooled"]["binary"]["abstention"]["rating_abstention_rate"] == pytest.approx(1 / 3)
    assert first["pooled"]["binary"]["abstention"]["pair_either_rate"] == pytest.approx(2 / 3)
    assert first["pooled"]["binary"]["abstention"]["labels"] == [{"label": "uncertain", "count": 2}]


def test_pre_adjudication_rejects_duplicate_item_rater_rows() -> None:
    duplicate = [
        {"item_id": "i1", "participant_pseudonym": "r1", "label": "yes"},
        {"item_id": "i1", "participant_pseudonym": "r1", "label": "no"},
    ]
    with pytest.raises(ValueError, match="Duplicate rating"):
        pre_adjudication_agreement(duplicate)


def test_weighted_proportion_uses_sampling_weight_and_excludes_missing() -> None:
    records = [
        {"supported": True, "sampling_weight": 1.0},
        {"supported": False, "sampling_weight": 3.0},
        {"supported": None, "sampling_weight": 100.0},
    ]
    assert weighted_proportion(records, outcome_key="supported") == pytest.approx(0.25)

    labels = [
        {"answer": "yes", "sampling_weight": 2.0},
        {"answer": "no", "sampling_weight": 6.0},
        {"answer": "uncertain", "sampling_weight": 99.0},
    ]
    estimate = weighted_proportion(
        labels,
        predicate=lambda row: None if row["answer"] == "uncertain" else row["answer"] == "yes",
    )
    assert estimate == pytest.approx(0.25)


def test_weighted_proportion_handles_zero_denominator_and_bad_inputs() -> None:
    assert weighted_proportion([{"value": True, "sampling_weight": 0.0}], outcome_key="value") is None
    assert weighted_proportion([{"value": None, "sampling_weight": 10.0}], outcome_key="value") is None
    with pytest.raises(ValueError, match="exactly one"):
        weighted_proportion([], outcome_key="value", predicate=lambda row: True)
    with pytest.raises(ValueError, match="exactly one"):
        weighted_proportion([])
    with pytest.raises(KeyError, match="sampling_weight"):
        weighted_proportion([{"value": True}], outcome_key="value")
    with pytest.raises(ValueError, match="bool or None"):
        weighted_proportion(
            [{"sampling_weight": 1.0}],
            predicate=lambda row: "yes",  # type: ignore[return-value]
        )


def test_judge_human_summary_groups_and_weights_binary_results() -> None:
    records = [
        {
            "category": "object",
            "surface": "ours",
            "human_label": "yes",
            "judge_label": "yes",
            "sampling_weight": 1.0,
        },
        {
            "category": "object",
            "surface": "ours",
            "human_label": "no",
            "judge_label": "yes",
            "sampling_weight": 3.0,
        },
        {
            "category": "count",
            "surface": "reference",
            "human_label": "no",
            "judge_label": "no",
            "sampling_weight": 2.0,
        },
        {
            "category": "count",
            "surface": "reference",
            "human_label": "uncertain",
            "judge_label": "no",
            "sampling_weight": 4.0,
        },
        {
            "category": "count",
            "surface": "reference",
            "human_label": None,
            "judge_label": "yes",
            "sampling_weight": 5.0,
        },
    ]

    result = judge_human_summary(records)
    overall = result["overall"]

    assert result["label_categories"] == ["no", "uncertain", "yes"]
    assert overall["n_records"] == 5
    assert overall["n_compared"] == 4
    assert overall["n_missing_human"] == 1
    assert overall["weighted_missing_human_rate"] == pytest.approx(1 / 3)
    assert overall["unweighted"]["nominal"]["exact_agreement"] == pytest.approx(0.5)
    assert overall["weighted"]["nominal"]["exact_agreement"] == pytest.approx(0.3)
    assert overall["unweighted"]["binary"]["n_binary"] == 3
    assert overall["unweighted"]["binary"]["exact_agreement"] == pytest.approx(2 / 3)
    assert overall["weighted"]["binary"]["exact_agreement"] == pytest.approx(0.5)
    assert overall["weighted"]["binary"]["classification_rates"]["positive_precision"] == pytest.approx(0.25)
    assert overall["weighted"]["binary"]["classification_rates"]["positive_recall"] == 1.0
    assert len(result["by_category"]) == 2
    assert len(result["by_surface"]) == 2
    assert len(result["by_category_surface"]) == 2
    object_ours = next(
        group for group in result["by_category_surface"] if group["category"] == "object" and group["surface"] == "ours"
    )
    assert object_ours["n_records"] == 2
    assert object_ours["weighted"]["binary"]["exact_agreement"] == pytest.approx(0.25)
    _assert_strict_json(result)


def test_judge_human_summary_can_separate_judges_and_disable_binary_metrics() -> None:
    rows = [
        {
            "judge": "qwen",
            "category": "relation",
            "surface": "ours",
            "human": "partial",
            "answer": "yes",
            "sampling_weight": 1.0,
        },
        {
            "judge": "gemma",
            "category": "relation",
            "surface": "ours",
            "human": "partial",
            "answer": "partial",
            "sampling_weight": 1.0,
        },
    ]
    result = judge_human_summary(
        rows,
        human_label_key="human",
        judge_label_key="answer",
        judge_key="judge",
        positive_label=None,
        negative_label=None,
    )
    assert result["overall"]["unweighted"]["binary"] is None
    assert [group["judge"] for group in result["by_judge"]] == ["gemma", "qwen"]
    assert len(result["by_judge_category_surface"]) == 2
    _assert_strict_json(result)


def test_judge_human_summary_rejects_partial_as_silent_binary_exclusion() -> None:
    rows = [
        {
            "category": "relation",
            "surface": "ours",
            "human_label": "partial",
            "judge_label": "yes",
            "sampling_weight": 1.0,
        }
    ]
    with pytest.raises(ValueError, match="Unexpected non-binary label"):
        judge_human_summary(rows)


def test_cluster_bootstrap_is_seeded_and_resamples_whole_image_clusters() -> None:
    records = [
        {"item_group": "a", "surface": "ours", "score": 1.0},
        {"item_group": "a", "surface": "reference", "score": 0.0},
        {"item_group": "b", "surface": "ours", "score": 3.0},
        {"item_group": "b", "surface": "reference", "score": 1.0},
        {"item_group": "c", "surface": "ours", "score": 5.0},
        {"item_group": "c", "surface": "reference", "score": 1.0},
    ]

    def paired_mean(sample):
        counts = Counter((row["item_group"], row["surface"]) for row in sample)
        for image in {row["item_group"] for row in sample}:
            assert counts[(image, "ours")] == counts[(image, "reference")]
        return sum(row["score"] for row in sample) / len(sample)

    first = cluster_bootstrap_percentile(
        records,
        paired_mean,
        n_resamples=200,
        confidence_level=0.9,
        seed=17,
        return_distribution=True,
    )
    second = cluster_bootstrap_percentile(
        records,
        paired_mean,
        n_resamples=200,
        confidence_level=0.9,
        seed=17,
        return_distribution=True,
    )
    different_seed = cluster_bootstrap_percentile(
        records,
        paired_mean,
        n_resamples=200,
        confidence_level=0.9,
        seed=18,
        return_distribution=True,
    )

    assert first == second
    assert first["distribution"] != different_seed["distribution"]
    assert first["estimate"] == pytest.approx(11 / 6)
    assert first["n_records"] == 6
    assert first["n_clusters"] == 3
    assert first["n_strata"] == 1
    assert first["n_valid_resamples"] == 200
    assert first["lower"] <= first["estimate"] <= first["upper"]
    _assert_strict_json(first)


def test_cluster_bootstrap_preserves_stratum_cluster_counts() -> None:
    records = [
        {"item_group": "a", "stratum": "object", "value": 1.0},
        {"item_group": "b", "stratum": "object", "value": 2.0},
        {"item_group": "c", "stratum": "count", "value": 10.0},
        {"item_group": "d", "stratum": "count", "value": 20.0},
        {"item_group": "e", "stratum": "count", "value": 30.0},
    ]

    def stratified_mean(sample):
        strata = Counter(row["stratum"] for row in sample)
        assert strata == {"count": 3, "object": 2}
        return sum(row["value"] for row in sample) / len(sample)

    result = cluster_bootstrap_percentile(
        records,
        stratified_mean,
        stratum_key="stratum",
        n_resamples=50,
        seed=9,
    )
    assert result["n_strata"] == 2
    assert result["clusters_per_stratum"] == [
        {"stratum": "count", "n_clusters": 3},
        {"stratum": "object", "n_clusters": 2},
    ]


def test_cluster_bootstrap_namespaces_duplicate_draws_for_human_agreement() -> None:
    records = [
        {
            "item_group": "image-a",
            "item_id": "claim-a",
            "participant_pseudonym": "r1",
            "label": "yes",
        },
        {
            "item_group": "image-a",
            "item_id": "claim-a",
            "participant_pseudonym": "r2",
            "label": "yes",
        },
        {
            "item_group": "image-b",
            "item_id": "claim-b",
            "participant_pseudonym": "r1",
            "label": "yes",
        },
        {
            "item_group": "image-b",
            "item_id": "claim-b",
            "participant_pseudonym": "r2",
            "label": "no",
        },
    ]

    def human_exact_agreement(sample):
        agreement = pre_adjudication_agreement(
            sample,
            item_key=["__bootstrap_instance__", "item_id"],
            categories=["yes", "no"],
        )
        return agreement["pooled"]["nominal"]["exact_agreement"]

    result = cluster_bootstrap_percentile(
        records,
        human_exact_agreement,
        n_resamples=100,
        seed=4,
    )
    assert result["estimate"] == pytest.approx(0.5)
    assert result["n_valid_resamples"] == 100
    assert result["bootstrap_instance_key"] == "__bootstrap_instance__"
    _assert_strict_json(result)


def test_cluster_bootstrap_handles_empty_and_invalid_replicates_without_nan() -> None:
    empty = cluster_bootstrap_percentile([], lambda sample: None, n_resamples=5)
    assert empty["estimate"] is None
    assert empty["lower"] is None
    assert empty["upper"] is None
    assert empty["n_valid_resamples"] == 0
    _assert_strict_json(empty)

    records = [{"item_group": "a", "value": 1.0}]
    invalid = cluster_bootstrap_percentile(records, lambda sample: float("nan"), n_resamples=5)
    assert invalid["estimate"] is None
    assert invalid["n_valid_resamples"] == 0
    assert invalid["lower"] is None
    assert invalid["upper"] is None
    _assert_strict_json(invalid)


def test_cluster_bootstrap_validates_parameters_and_stratum_membership() -> None:
    records = [
        {"item_group": "same", "stratum": "object", "value": 1.0},
        {"item_group": "same", "stratum": "count", "value": 2.0},
    ]
    with pytest.raises(ValueError, match="multiple strata"):
        cluster_bootstrap_percentile(records, lambda sample: 1.0, stratum_key="stratum", n_resamples=2)
    with pytest.raises(ValueError, match="positive integer"):
        cluster_bootstrap_percentile([], lambda sample: None, n_resamples=0)
    with pytest.raises(ValueError, match="positive integer"):
        cluster_bootstrap_percentile([], lambda sample: None, n_resamples=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="strictly between"):
        cluster_bootstrap_percentile([], lambda sample: None, confidence_level=1.0)
    with pytest.raises(ValueError, match="finite number"):
        cluster_bootstrap_percentile([], lambda sample: None, confidence_level="95%")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed must"):
        cluster_bootstrap_percentile([], lambda sample: None, seed=True)
    with pytest.raises(ValueError, match="already contains bootstrap instance key"):
        cluster_bootstrap_percentile(
            [{"item_group": "a", "__bootstrap_instance__": 0}],
            lambda sample: 1.0,
            n_resamples=2,
        )
