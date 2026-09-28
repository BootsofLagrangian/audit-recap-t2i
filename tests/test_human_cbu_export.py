from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import pytest

from audit_recap_t2i.human_cbu.export import (
    build_export_bundle,
    compute_export_metrics,
    consensus_rows,
    private_audit_rows,
    public_annotation_rows,
    write_export_bundle,
)

LENIENT_READINESS = {
    "minimum_substantive_overall": 1,
    "minimum_substantive_per_type": 0,
    "minimum_substantive_per_stratum": 0,
    "expected_proposed_types": ["attribute", "count"],
    "expected_strata": ["ours/attribute", "reference/count"],
    "require_complete_judge_coverage": True,
}


def _annotation(
    item: int,
    phase: str,
    rater: str,
    *,
    support: str,
    weight: float,
    category: str,
    qwen: str,
    gemma: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "study_id": "private-study",
        "annotation_id": f"annotation-{item}-{phase}-{rater}",
        "assignment_id": f"assignment-{item}-{phase}-{rater}",
        "item_id": f"private-item-{item}",
        "item_group": f"private-image-{item}",
        "participant_pseudonym": rater,
        "phase": phase,
        "category": category,
        "surface": "ours_cc12m" if item == 1 else "ref_cc12m_llavanext",
        "qwen_answer": {"answer": qwen, "model": "private-qwen-checkpoint"},
        "gemma_answer": {"answer": gemma, "model": "private-gemma-checkpoint"},
        "stratum": f"{'ours' if item == 1 else 'reference'}/{category}",
        "population": {"source": "private-cc12m"},
        "sample": {"version": "v1"},
        "weight": weight,
        "sampling_weight": weight,
        "population_size": 100,
        "sample_probability": 0.02,
        "repeat_of": None,
        "adjudication": None,
        "revision": 1,
        "created_at": "2026-07-27T12:00:00+00:00",
        "updated_at": "2026-07-27T12:01:00+00:00",
        "note": "free text stays private",
        "reason_tags": ["occlusion"],
    }
    if phase == "caption":
        row.update(
            {
                "caption_licensed": "yes",
                "atomic_visual_claim": "yes",
                "category_check": "correct",
                "corrected_category": None,
            }
        )
    else:
        row.update(
            {
                "image_support": support,
                "salient_coverage": 4 if item == 1 else 2,
                "control_usefulness": 4 if support == "yes" else None,
                "confidence": "high",
            }
        )
    return row


def _complete_rows() -> list[dict[str, Any]]:
    rows = []
    for item, support, weight, category, qwen, gemma in (
        (1, "yes", 2.0, "attribute", "yes", "no"),
        (2, "no", 3.0, "count", "yes", "no"),
    ):
        for phase in ("caption", "image"):
            for rater in ("P001", "P002"):
                rows.append(
                    _annotation(
                        item,
                        phase,
                        rater,
                        support=support,
                        weight=weight,
                        category=category,
                        qwen=qwen,
                        gemma=gemma,
                    )
                )
    return rows


def _status(
    *,
    closed: bool = True,
    pending: int = 0,
    unresolved: int = 0,
    panel_shortfalls: int = 0,
) -> dict[str, Any]:
    completed = 4 - pending
    return {
        "study": {"status": "closed" if closed else "open"},
        "items": 2,
        "assignments": {
            "caption": {"pending": pending, "completed": completed, "total": 4},
            "image": {"pending": pending, "completed": completed, "total": 4},
        },
        "agreement": {
            "caption": {
                "unresolved_disagreements": unresolved,
                "eligible_panel_shortfalls": panel_shortfalls,
            },
            "image": {
                "unresolved_disagreements": unresolved,
                "eligible_panel_shortfalls": panel_shortfalls,
            },
        },
    }


def _sixteen_cell_clustered_rows() -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    claim_types = (
        "object",
        "attribute",
        "relation",
        "count",
        "style",
        "camera",
        "lighting",
        "text_rendering",
    )
    rows: list[dict[str, Any]] = []
    item = 0
    for surface_group in ("ours", "reference"):
        for category in claim_types:
            for cluster, support in (("cluster-a", "no"), ("cluster-b", "yes")):
                item += 1
                for phase in ("caption", "image"):
                    for rater in ("P001", "P002"):
                        row = _annotation(
                            item,
                            phase,
                            rater,
                            support=support,
                            weight=1.0,
                            category=category,
                            qwen=support,
                            gemma=support,
                        )
                        row["item_group"] = cluster
                        row["surface"] = f"{surface_group}_cc12m"
                        row["stratum"] = f"{surface_group}/{category}"
                        rows.append(row)
    assignments_per_phase = item * 2
    status = {
        "study": {"status": "closed"},
        "items": item,
        "assignments": {
            phase: {
                "pending": 0,
                "completed": assignments_per_phase,
                "total": assignments_per_phase,
            }
            for phase in ("caption", "image")
        },
        "agreement": {
            phase: {
                "unresolved_disagreements": 0,
                "eligible_panel_shortfalls": 0,
            }
            for phase in ("caption", "image")
        },
    }
    readiness = {
        "minimum_substantive_overall": 1,
        "minimum_substantive_per_type": 0,
        "minimum_substantive_per_stratum": 0,
        "expected_proposed_types": list(claim_types),
        "expected_strata": [
            f"{surface_group}/{category}" for surface_group in ("ours", "reference") for category in claim_types
        ],
        "require_complete_judge_coverage": True,
    }
    return rows, status, readiness


def test_ready_metrics_are_design_weighted_and_machine_generated() -> None:
    metrics = compute_export_metrics(
        _complete_rows(),
        status=_status(),
        readiness_thresholds=LENIENT_READINESS,
        bootstrap_resamples=50,
        bootstrap_seed=7,
    )

    assert metrics["analysis_status"] == {
        "ready": True,
        "state": "ready",
        "provisional": False,
        "reasons": [],
        "thresholds": {
            "minimum_substantive_overall": 1,
            "minimum_substantive_per_type": 0,
            "minimum_substantive_per_stratum": 0,
            "expected_proposed_types": ["attribute", "count"],
            "expected_strata": ["ours/attribute", "reference/count"],
            "require_complete_judge_coverage": True,
            "required_bootstrap_resamples": None,
            "required_bootstrap_seed": None,
        },
        "bootstrap": {
            "resamples": 50,
            "seed": 7,
            "minimum_ready_resamples": 2,
            "required_resamples": None,
            "required_seed": None,
        },
    }
    human = metrics["human_human_pre_adjudication"]["image"]["image_support"]["pooled"]
    assert human["nominal"]["exact_agreement"] == 1.0
    assert human["nominal"]["gwet_ac1"] == 1.0
    assert metrics["human_supported_rate"]["overall"]["design_weighted"] == pytest.approx(0.4)
    assert metrics["judge_human"]["qwen"]["overall"]["weighted"]["nominal"]["exact_agreement"] == pytest.approx(0.4)
    assert metrics["judge_human"]["gemma"]["overall"]["weighted"]["nominal"]["exact_agreement"] == pytest.approx(0.6)
    assert metrics["human_supported_rate"]["overall"]["cluster_bootstrap_95"]["n_clusters"] == 2
    macro = metrics["human_supported_rate"]["equal_stratum_macro"]
    assert macro["estimate"] == 0.5
    assert macro["n_cells_declared"] == 2
    assert macro["n_cells_observed"] == 2
    assert macro["n_cells_reported"] == 2
    assert macro["n_empty_cells"] == 0
    assert macro["n_cells_in_macro"] == 2
    assert macro["n_substantive"] == 2
    assert macro["cluster_bootstrap_95"]["n_clusters"] == 2
    assert macro["cluster_bootstrap_95"]["n_resamples"] == 50
    assert {entry["stratum"] for entry in metrics["human_supported_rate"]["by_stratum"]} == {
        "ours/attribute",
        "reference/count",
    }
    usefulness = metrics["exploratory_control_usefulness"]["overall"]
    assert usefulness["n_supported_items"] == 1
    assert usefulness["n_eligible_annotations"] == 2
    assert usefulness["n_rated"] == 2
    assert usefulness["design_weighted_distribution"]["4"] == 1.0
    assert usefulness["design_weighted_ordinal_median"] == 4
    assert metrics["exploratory_control_usefulness"]["pre_adjudication_ordinal_agreement"] == {
        "n_numeric_pairs": 1,
        "exact_rate": 1.0,
        "within_one_rate": 1.0,
        "median_absolute_difference": 0.0,
        "weighted": False,
    }
    coverage = metrics["exploratory_salient_content_coverage"]
    assert coverage["overall"] == {
        "n_participant_pair_records": 4,
        "n_rated": 4,
        "n_missing": 0,
        "unweighted_distribution": {
            "1": 0.0,
            "2": 0.5,
            "3": 0.0,
            "4": 0.5,
            "5": 0.0,
        },
        "unweighted_ordinal_median": 3.0,
    }
    assert coverage["deduplication"] == {
        "n_input_claim_annotations": 4,
        "n_participant_pair_records": 4,
        "n_duplicate_claim_rows_collapsed": 0,
    }
    assert coverage["pre_adjudication_ordinal_agreement"]["exact_rate"] == 1.0
    assert "control_usefulness" not in metrics["human_human_pre_adjudication"]["image"]
    json.dumps(metrics, allow_nan=False)


def test_equal_macro_bootstraps_all_sixteen_cells_in_each_shared_cluster_draw() -> None:
    rows, status, readiness = _sixteen_cell_clustered_rows()
    metrics = compute_export_metrics(
        rows,
        status=status,
        readiness_thresholds=readiness,
        bootstrap_resamples=50,
        bootstrap_seed=1477,
    )

    macro = metrics["human_supported_rate"]["equal_stratum_macro"]
    interval = macro["cluster_bootstrap_95"]
    assert metrics["analysis_status"]["ready"] is True
    assert macro["estimate"] == 0.5
    assert macro["n_cells_declared"] == 16
    assert macro["n_cells_in_macro"] == 16
    assert interval["estimate"] == 0.5
    assert interval["n_clusters"] == 2
    assert interval["n_valid_resamples"] == 50
    assert interval["lower"] == 0.0
    assert interval["upper"] == 1.0


def test_headline_support_and_judge_agreement_condition_on_valid_extraction() -> None:
    rows = _complete_rows()
    for row in rows:
        if row["item_id"] == "private-item-2" and row["phase"] == "caption":
            row["atomic_visual_claim"] = "not_visual"
            row["category_check"] = "not_applicable"

    metrics = compute_export_metrics(
        rows,
        status=_status(),
        readiness_thresholds=LENIENT_READINESS,
        bootstrap_resamples=20,
        bootstrap_seed=7,
    )

    assert metrics["human_consensus"]["extraction_valid_items"] == 1
    assert metrics["human_consensus"]["image_items_excluded_invalid_extraction"] == 1
    all_human = metrics["human_human_pre_adjudication"]["image"]["image_support"]
    valid_human = metrics["human_human_pre_adjudication"]["image"]["image_support_extraction_valid"]
    assert all_human["n_items_with_pairs"] == 2
    assert valid_human["n_items_with_pairs"] == 1
    assert metrics["human_supported_rate"]["overall"]["design_weighted"] == 1.0
    assert metrics["judge_human"]["qwen"]["overall"]["weighted"]["nominal"]["exact_agreement"] == 1.0
    assert metrics["judge_human"]["gemma"]["overall"]["weighted"]["nominal"]["exact_agreement"] == 0.0
    assert metrics["exploratory_control_usefulness"]["overall"]["n_supported_items"] == 1


def test_usefulness_is_exploratory_ordinal_and_support_conditioned() -> None:
    rows = _complete_rows()
    for row in rows:
        if row["phase"] != "image":
            continue
        if row["item_id"] == "private-item-1":
            row["control_usefulness"] = 5
        else:
            # Even malformed legacy input cannot leak an unsupported score into
            # the exploratory summary.  The store rejects this combination.
            row["control_usefulness"] = 1

    metrics = compute_export_metrics(
        rows,
        status=_status(),
        readiness_thresholds=LENIENT_READINESS,
        bootstrap_resamples=20,
        bootstrap_seed=7,
    )

    usefulness = metrics["exploratory_control_usefulness"]["overall"]
    assert usefulness == {
        "conditioning": (
            "raw ratings with individual image_support=yes on resolved extraction-valid and image-supported claims"
        ),
        "weighting": ("each supported item's design weight is divided equally over eligible raw ratings"),
        "n_supported_items": 1,
        "n_eligible_annotations": 2,
        "n_rated": 2,
        "n_cannot_judge": 0,
        "n_missing": 0,
        "design_weighted_distribution": {
            "1": 0.0,
            "2": 0.0,
            "3": 0.0,
            "4": 0.0,
            "5": 1.0,
            "cannot_judge": 0.0,
            "missing": 0.0,
        },
        "design_weighted_ordinal_median": 5,
    }
    assert metrics["exploratory_control_usefulness"]["pre_adjudication_ordinal_agreement"] == {
        "n_numeric_pairs": 1,
        "exact_rate": 1.0,
        "within_one_rate": 1.0,
        "median_absolute_difference": 0.0,
        "weighted": False,
    }
    assert "mean" not in json.dumps(usefulness)

    files, _ = build_export_bundle(
        rows,
        status=_status(),
        readiness_thresholds=LENIENT_READINESS,
        public_id_namespace="release-v1",
        bootstrap_resamples=20,
        bootstrap_seed=7,
    )
    metric_rows = list(csv.DictReader(io.StringIO(files["metrics.csv"])))
    usefulness_rows = [row for row in metric_rows if row["family"] == "exploratory_control_usefulness"]
    assert usefulness_rows
    assert {row["metric"] for row in usefulness_rows} >= {
        "design_weighted_share:1",
        "design_weighted_share:5",
        "design_weighted_share:cannot_judge",
        "design_weighted_share:missing",
        "design_weighted_ordinal_median",
        "pre_adjudication_exact_rate",
        "pre_adjudication_within_one_rate",
        "pre_adjudication_median_absolute_difference",
    }


def test_optional_usefulness_missingness_remains_in_the_distribution() -> None:
    rows = _complete_rows()
    for row in rows:
        if row["phase"] == "image" and row["item_id"] == "private-item-1" and row["participant_pseudonym"] == "P002":
            row["control_usefulness"] = None

    metrics = compute_export_metrics(
        rows,
        status=_status(),
        readiness_thresholds=LENIENT_READINESS,
        bootstrap_resamples=10,
        bootstrap_seed=7,
    )

    usefulness = metrics["exploratory_control_usefulness"]["overall"]
    assert usefulness["n_supported_items"] == 1
    assert usefulness["n_eligible_annotations"] == 2
    assert usefulness["n_rated"] == 1
    assert usefulness["n_missing"] == 1
    assert usefulness["design_weighted_distribution"]["4"] == pytest.approx(0.5)
    assert usefulness["design_weighted_distribution"]["missing"] == pytest.approx(0.5)
    assert usefulness["design_weighted_ordinal_median"] == 4
    assert metrics["exploratory_control_usefulness"]["pre_adjudication_ordinal_agreement"]["n_numeric_pairs"] == 0


def test_public_export_is_allow_listed_and_private_export_strips_secrets() -> None:
    rows = _complete_rows()
    rows[0].update(
        {
            "email": "rater@example.test",
            "session_token": "secret",
            "participant_profile": {
                "author_status": "non_author",
                "english_proficiency": "fluent",
                "t2i_experience": "some",
            },
            "nested": {"user_agent": "browser", "kept": "audit"},
        }
    )

    public = public_annotation_rows(rows, public_id_namespace="release-v1")
    assert len(public) == len(rows)
    assert set(public[0]) == {
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
    }
    serialized_public = json.dumps(public)
    for forbidden in (
        "private-item",
        "private-image",
        "assignment-",
        "private-qwen",
        "free text",
        "rater@example",
        "secret",
        "user_agent",
        "created_at",
        "english_proficiency",
    ):
        assert forbidden not in serialized_public
    assert {row["surface_group"] for row in public} == {"ours", "reference"}
    assert {row["public_participant_id"] for row in public}.isdisjoint({"P001", "P002"})
    other_namespace = public_annotation_rows(rows, public_id_namespace="release-v2")
    assert {row["public_participant_id"] for row in public}.isdisjoint(
        {row["public_participant_id"] for row in other_namespace}
    )

    private = private_audit_rows(rows)
    serialized_private = json.dumps(private)
    assert "rater@example" not in serialized_private
    assert "session_token" not in serialized_private
    assert "user_agent" not in serialized_private
    assert '"kept": "audit"' in serialized_private
    assert "participant_profile" not in serialized_private
    assert "english_proficiency" not in serialized_private
    assert "private-qwen-checkpoint" in serialized_private


def test_bundle_has_exact_artifacts_and_is_deterministic() -> None:
    rows = _complete_rows()
    files_a, metrics_a = build_export_bundle(
        rows,
        status=_status(),
        readiness_thresholds=LENIENT_READINESS,
        public_id_namespace="release-v1",
        bootstrap_resamples=25,
        bootstrap_seed=3,
    )
    files_b, metrics_b = build_export_bundle(
        list(reversed(rows)),
        status=_status(),
        readiness_thresholds=LENIENT_READINESS,
        public_id_namespace="release-v1",
        bootstrap_resamples=25,
        bootstrap_seed=3,
    )

    assert files_a == files_b
    assert metrics_a == metrics_b
    assert set(files_a) == {
        "annotations.private.jsonl",
        "sample_flow.csv",
        "metrics.json",
        "metrics.csv",
        "confusion_matrices.csv",
        "rebuttal_table.md",
        "rebuttal_table.tex",
        "rebuttal_summary.md",
    }
    assert "NOT READY" not in files_a["rebuttal_table.md"]
    assert "Qwen–human" in files_a["rebuttal_table.md"]
    assert "0.400" in files_a["rebuttal_table.md"]
    assert "0.600" in files_a["rebuttal_table.md"]
    assert "do not certify the full corpus" in files_a["rebuttal_summary.md"]
    for name in ("rebuttal_table.md", "rebuttal_table.tex", "rebuttal_summary.md"):
        assert "model/superpopulation sensitivity" in files_a[name]
        assert "fixed realized annotators" in files_a[name]
        assert "not design-consistent finite-population" in files_a[name]
    assert (
        metrics_a["method"]["uncertainty_interpretation"]
        == "Image-cluster bootstrap intervals are model/superpopulation sensitivity "
        "intervals conditional on the fixed realized annotators; they are not "
        "design-consistent finite-population confidence intervals."
    )

    assert "annotations.public.csv" not in files_a
    private = [json.loads(line) for line in files_a["annotations.private.jsonl"].splitlines()]
    assert private[0]["surface"] in {"ours_cc12m", "ref_cc12m_llavanext"}


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (None, "operational_status_missing"),
        (_status(closed=False), "study_not_closed:open"),
        (_status(pending=1), "assignments_incomplete:caption"),
        (_status(unresolved=1), "unresolved_disagreements:caption:1"),
        (_status(panel_shortfalls=1), "eligible_panel_shortfalls:caption:1"),
    ],
)
def test_incomplete_study_withholds_rebuttal_numbers(
    status: dict[str, Any] | None,
    reason: str,
) -> None:
    files, metrics = build_export_bundle(
        _complete_rows(),
        status=status,
        readiness_thresholds=LENIENT_READINESS,
        public_id_namespace="release-v1",
        bootstrap_resamples=10,
    )

    assert metrics["analysis_status"]["ready"] is False
    assert reason in metrics["analysis_status"]["reasons"]
    assert "NOT READY" in files["rebuttal_table.md"]
    assert "do not use this file as empirical evidence" in files["rebuttal_summary.md"].lower()
    assert "Qwen–human" not in files["rebuttal_table.md"]


def test_no_annotations_is_explicitly_not_ready() -> None:
    files, metrics = build_export_bundle(
        [],
        status=_status(),
        readiness_thresholds=LENIENT_READINESS,
        public_id_namespace="release-v1",
        bootstrap_resamples=5,
    )

    assert metrics["analysis_status"]["ready"] is False
    assert "no_annotations" in metrics["analysis_status"]["reasons"]
    assert "no_substantive_extraction_valid_image_consensus" in metrics["analysis_status"]["reasons"]
    assert "annotations.public.csv" not in files
    json.loads(files["metrics.json"])


def test_write_bundle_creates_only_declared_files(tmp_path: Path) -> None:
    result = write_export_bundle(
        _complete_rows(),
        tmp_path / "export",
        status=_status(),
        readiness_thresholds=LENIENT_READINESS,
        public_id_namespace="release-v1",
        bootstrap_resamples=10,
    )

    destination = tmp_path / "export"
    assert result["analysis_status"]["ready"] is True
    assert sorted(path.name for path in destination.iterdir()) == result["files"]
    for path in destination.iterdir():
        assert path.stat().st_size > 0
    assert (destination / "annotations.private.jsonl").stat().st_mode & 0o777 == 0o600


def test_aggregate_rerun_removes_stale_public_rows(tmp_path: Path) -> None:
    destination = tmp_path / "reused-export"
    write_export_bundle(
        _complete_rows(),
        destination,
        status=_status(),
        readiness_thresholds=LENIENT_READINESS,
        include_public_rows=True,
        public_id_namespace="approved-release-v1",
        bootstrap_resamples=5,
    )
    assert (destination / "annotations.public.csv").is_file()

    result = write_export_bundle(
        _complete_rows(),
        destination,
        status=_status(),
        readiness_thresholds=LENIENT_READINESS,
        include_public_rows=False,
        bootstrap_resamples=5,
    )
    assert "annotations.public.csv" not in result["files"]
    assert not (destination / "annotations.public.csv").exists()


def test_public_namespace_is_required() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_export_bundle(
            _complete_rows(),
            status=_status(),
            readiness_thresholds=LENIENT_READINESS,
            include_public_rows=True,
            public_id_namespace="",
            bootstrap_resamples=5,
        )


def test_public_rows_are_explicit_opt_in_and_use_hashed_participants() -> None:
    rows = _complete_rows()
    files, _ = build_export_bundle(
        rows,
        status=_status(),
        readiness_thresholds=LENIENT_READINESS,
        include_public_rows=True,
        public_id_namespace="release-v1",
        bootstrap_resamples=5,
    )

    public = list(csv.DictReader(io.StringIO(files["annotations.public.csv"])))
    assert len(public) == len(rows)
    assert "participant_pseudonym" not in public[0]
    assert "public_participant_id" in public[0]
    assert {row["public_participant_id"] for row in public}.isdisjoint({"P001", "P002"})


def test_human_corrected_category_drives_per_type_results_but_not_design_stratum() -> None:
    rows = _complete_rows()
    for row in rows:
        if row["item_id"] == "private-item-1" and row["phase"] == "caption":
            row["category_check"] = "incorrect"
            row["corrected_category"] = "relation"

    collapsed = consensus_rows(rows)
    caption = next(row for row in collapsed if row["item_id"] == "private-item-1" and row["phase"] == "caption")
    assert caption["proposed_category"] == "attribute"
    assert caption["corrected_category"] == "relation"
    assert caption["resolved_category"] == "relation"
    assert caption["consensus_source"] == "initial_agreement"

    metrics = compute_export_metrics(
        rows,
        status=_status(),
        readiness_thresholds=LENIENT_READINESS,
        bootstrap_resamples=10,
        bootstrap_seed=7,
    )
    assert {entry["category"] for entry in metrics["human_supported_rate"]["by_category"]} == {"count", "relation"}
    assert {entry["category"] for entry in metrics["judge_human"]["qwen"]["by_category"]} == {"count", "relation"}
    assert {entry["stratum"] for entry in metrics["human_supported_rate"]["by_stratum"]} == {
        "ours/attribute",
        "reference/count",
    }
    assert "human-resolved category" in metrics["method"]["per_type_grouping"]


def test_uncertain_type_is_excluded_from_human_resolved_per_type_results() -> None:
    rows = _complete_rows()
    for row in rows:
        if row["item_id"] == "private-item-1" and row["phase"] == "caption":
            row["category_check"] = "uncertain"

    caption = next(
        row for row in consensus_rows(rows) if row["item_id"] == "private-item-1" and row["phase"] == "caption"
    )
    assert caption["resolved_category"] is None
    metrics = compute_export_metrics(
        rows,
        status=_status(),
        readiness_thresholds=LENIENT_READINESS,
        bootstrap_resamples=5,
    )
    assert metrics["human_consensus"]["image_items_excluded_unresolved_category_from_per_type"] == 1
    assert [entry["category"] for entry in metrics["human_supported_rate"]["by_category"]] == ["count"]
    assert [entry["category"] for entry in metrics["judge_human"]["qwen"]["by_category"]] == ["count"]


def test_declared_missing_design_cell_fails_closed_and_remains_in_macro() -> None:
    rows = [row for row in _complete_rows() if row["item_id"] != "private-item-2"]
    status = _status()
    status["items"] = 1
    for phase in ("caption", "image"):
        status["assignments"][phase] = {"pending": 0, "completed": 2, "total": 2}
    metrics = compute_export_metrics(
        rows,
        status=status,
        readiness_thresholds=LENIENT_READINESS,
        bootstrap_resamples=5,
    )
    assert "missing_design_type:count" in metrics["analysis_status"]["reasons"]
    assert "missing_design_stratum:reference/count" in metrics["analysis_status"]["reasons"]
    reference = next(
        entry for entry in metrics["human_supported_rate"]["by_stratum"] if entry["stratum"] == "reference/count"
    )
    assert reference["n_items"] == 0
    assert reference["design_weighted"] is None
    macro = metrics["human_supported_rate"]["equal_stratum_macro"]
    assert macro["n_empty_cells"] == 1
    assert macro["estimate"] is None
    assert macro["cluster_bootstrap_95"]["n_valid_resamples"] == 0


def test_agreement_csv_uses_pairwise_cohen_and_group_cis_are_exported() -> None:
    files, metrics = build_export_bundle(
        _complete_rows(),
        status=_status(),
        readiness_thresholds=LENIENT_READINESS,
        bootstrap_resamples=10,
        bootstrap_seed=7,
    )

    human_nominal = metrics["human_human_pre_adjudication"]["image"]["image_support_extraction_valid"]["pooled"][
        "nominal"
    ]
    assert human_nominal["cohen_kappa"] is None
    assert human_nominal["mean_pairwise_cohen_kappa"] == 1.0
    human_interval = metrics["human_human_pre_adjudication"]["image"]["image_support_extraction_valid"][
        "cluster_bootstrap_95"
    ]
    assert human_interval["estimate"] == 1.0
    assert human_interval["n_clusters"] == 2
    assert human_interval["n_resamples"] == 10
    assert human_interval["lower"] == 1.0
    assert human_interval["upper"] == 1.0
    for judge in ("qwen", "gemma"):
        for group in (
            *metrics["judge_human"][judge]["by_category"],
            *metrics["judge_human"][judge]["by_surface"],
        ):
            assert group["cluster_bootstrap_95"]["n_clusters"] == 1

    exported = list(csv.DictReader(io.StringIO(files["metrics.csv"])))
    human_rows = [
        row
        for row in exported
        if row["family"] == "human_human_pre_adjudication" and row["field"] == "image_support_extraction_valid"
    ]
    assert any(row["metric"] == "mean_pairwise_cohen_kappa" and float(row["value"]) == 1.0 for row in human_rows)
    assert not any(row["metric"] == "cohen_kappa" for row in human_rows)
    human_ci = [row for row in human_rows if row["metric"].startswith("exact_agreement_cluster_bootstrap_95_")]
    assert {row["metric"] for row in human_ci} == {
        "exact_agreement_cluster_bootstrap_95_lower",
        "exact_agreement_cluster_bootstrap_95_upper",
    }
    assert "Human–human (pre-adjudication) | Extraction-valid | 2 | 1.000 [1.000, 1.000]" in files["rebuttal_table.md"]
    category_ci = [
        row
        for row in exported
        if row["family"] == "judge_human"
        and row["category"]
        and row["metric"] == "exact_agreement_cluster_bootstrap_95_lower"
    ]
    surface_ci = [
        row
        for row in exported
        if row["family"] == "judge_human"
        and row["surface_group"]
        and not row["category"]
        and row["metric"] == "exact_agreement_cluster_bootstrap_95_lower"
    ]
    assert len(category_ci) == 4
    assert len(surface_ci) == 4


def test_readiness_thresholds_and_complete_judge_coverage_are_enforced() -> None:
    metrics = compute_export_metrics(
        _complete_rows(),
        status=_status(),
        readiness_thresholds={
            "minimum_substantive_overall": 3,
            "minimum_substantive_per_type": 2,
            "minimum_substantive_per_stratum": 2,
            "require_complete_judge_coverage": True,
        },
        bootstrap_resamples=5,
    )
    reasons = metrics["analysis_status"]["reasons"]
    assert "substantive_below_minimum:overall:2/3" in reasons
    assert "substantive_below_minimum:type:attribute:1/2" in reasons
    assert "substantive_below_minimum:type:count:1/2" in reasons
    assert "substantive_below_minimum:stratum:ours/attribute:1/2" in reasons
    assert "substantive_below_minimum:stratum:reference/count:1/2" in reasons

    incomplete = _complete_rows()
    for row in incomplete:
        if row["item_id"] == "private-item-2" and row["phase"] == "image":
            row["gemma_answer"] = None
    metrics = compute_export_metrics(
        incomplete,
        status=_status(),
        readiness_thresholds=LENIENT_READINESS,
        bootstrap_resamples=5,
    )
    assert "incomplete_judge_coverage:gemma:1/2" in metrics["analysis_status"]["reasons"]


def test_bootstrap_plan_is_machine_checked_and_one_resample_is_never_ready() -> None:
    one_resample = compute_export_metrics(
        _complete_rows(),
        status=_status(),
        readiness_thresholds=LENIENT_READINESS,
        bootstrap_resamples=1,
        bootstrap_seed=1477,
    )
    assert one_resample["analysis_status"]["ready"] is False
    assert "bootstrap_resamples_below_readiness_minimum:1/2" in one_resample["analysis_status"]["reasons"]

    frozen_plan = {
        **LENIENT_READINESS,
        "required_bootstrap_resamples": 10_000,
        "required_bootstrap_seed": 1477,
    }
    mismatch = compute_export_metrics(
        _complete_rows(),
        status=_status(),
        readiness_thresholds=frozen_plan,
        bootstrap_resamples=2,
        bootstrap_seed=7,
    )
    assert mismatch["analysis_status"]["ready"] is False
    assert "bootstrap_resamples_mismatch:2/10000" in mismatch["analysis_status"]["reasons"]
    assert "bootstrap_seed_mismatch:7/1477" in mismatch["analysis_status"]["reasons"]
    assert mismatch["analysis_status"]["bootstrap"] == {
        "resamples": 2,
        "seed": 7,
        "minimum_ready_resamples": 2,
        "required_resamples": 10_000,
        "required_seed": 1477,
    }

    accepted = compute_export_metrics(
        _complete_rows(),
        status=_status(),
        readiness_thresholds={
            **LENIENT_READINESS,
            "required_bootstrap_resamples": 2,
            "required_bootstrap_seed": 7,
        },
        bootstrap_resamples=2,
        bootstrap_seed=7,
    )
    assert accepted["analysis_status"]["ready"] is True


def test_participant_summary_is_explicit_aggregate_only_input() -> None:
    participant_summary = {
        "identity": "study-local pseudonyms",
        "direct_identifiers_collected": False,
        "flow": {
            "invited_total": 4,
            "current_status": {
                "invited": 1,
                "active": 2,
                "declined": 1,
                "withdrawn": 0,
            },
            "consented_current": 2,
            "completed": 2,
            "procedurally_completed_total": 3,
            "analyzed": 2,
        },
        "profile_marginals": {
            "n_participants": 2,
            "joint_profiles_released": False,
            "fields": {
                "author_status": {
                    "counts": [
                        {"value": "author", "count": 1},
                        {"value": "non_author", "count": 1},
                    ],
                    "missing": 0,
                    "multi_select": False,
                }
            },
        },
    }
    metrics = compute_export_metrics(
        _complete_rows(),
        status=_status(),
        participant_summary=participant_summary,
        readiness_thresholds=LENIENT_READINESS,
        bootstrap_resamples=5,
    )
    assert metrics["annotators"]["aggregate_summary_supplied"] is True
    assert metrics["annotators"]["flow"]["analyzed"] == 2
    author_marginal = metrics["annotators"]["profile_marginals"]["fields"]["author_status"]
    assert author_marginal["suppressed"] is True
    assert author_marginal["minimum_cell_size"] == 3
    assert author_marginal["counts"] == []
    assert author_marginal["missing"] is None
    assert "P001" not in json.dumps(metrics["annotators"])
    files, _ = build_export_bundle(
        _complete_rows(),
        status=_status(),
        participant_summary=participant_summary,
        readiness_thresholds=LENIENT_READINESS,
        bootstrap_resamples=5,
    )
    flow_rows = list(csv.DictReader(io.StringIO(files["sample_flow.csv"])))
    participant_stages = {row["stage"]: int(row["count"]) for row in flow_rows if row["phase"] == "participants"}
    assert participant_stages["invited_total"] == 4
    assert participant_stages["consented_current"] == 2
    assert participant_stages["completed"] == 2
    assert participant_stages["procedurally_completed_total"] == 3
    assert participant_stages["analyzed"] == 2

    unsafe = {**participant_summary, "participant_pseudonyms": ["P001"]}
    with pytest.raises(ValueError, match="non-aggregate"):
        compute_export_metrics(
            _complete_rows(),
            status=_status(),
            participant_summary=unsafe,
            readiness_thresholds=LENIENT_READINESS,
            bootstrap_resamples=5,
        )
    contradictory = {
        **participant_summary,
        "flow": {**participant_summary["flow"], "analyzed": 5},
    }
    with pytest.raises(ValueError, match="completed <= analyzed"):
        compute_export_metrics(
            _complete_rows(),
            status=_status(),
            participant_summary=contradictory,
            readiness_thresholds=LENIENT_READINESS,
            bootstrap_resamples=5,
        )
    analyzed_mismatch = {
        **participant_summary,
        "flow": {
            **participant_summary["flow"],
            "completed": 1,
            "analyzed": 1,
        },
        "profile_marginals": {
            **participant_summary["profile_marginals"],
            "n_participants": 1,
            "fields": {
                "author_status": {
                    "counts": [{"value": "non_author", "count": 1}],
                    "missing": 0,
                    "multi_select": False,
                }
            },
        },
    }
    with pytest.raises(ValueError, match="unique annotators"):
        compute_export_metrics(
            _complete_rows(),
            status=_status(),
            participant_summary=analyzed_mismatch,
            readiness_thresholds=LENIENT_READINESS,
            bootstrap_resamples=5,
        )
