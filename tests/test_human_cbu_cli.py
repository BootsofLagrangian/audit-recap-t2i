from __future__ import annotations

import copy
import hashlib
import json
import re
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from audit_recap_t2i.human_cbu.store import (
    ADJUDICATION_PACKET_VERSION,
    HumanCBUStore,
)
from scripts.human_cbu_eval import (
    build_from_config,
    command_adjudicate,
    command_adjudication_packets,
    command_assign,
    command_export,
    command_init_db,
    command_invites,
    command_participant_test,
    command_preview,
    command_provision_adjudicator,
    command_replace_participant,
    command_sample,
    command_seal,
    command_serve,
    is_loopback_host,
    normalize_store_items,
    parse_args,
    resolve_exact_span,
    secure_write_once_json,
    validate_bind_security,
    validate_frozen_sample_design,
)


def _builder_row(index: int, *, repeat_of: str | None = None, span: str = "red cat") -> dict[str, Any]:
    item_id = f"item-{index}"
    return {
        "item_id": item_id,
        "question_id": f"surface:{index}:u0000",
        "caption_id": f"surface:{index}",
        "extractor_request_id": f"extractor-{index}",
        "source_row": index,
        "caption": "A red cat sits on a mat.",
        "unit": "a red cat",
        "span": span,
        "target": "cat",
        "category": "attribute",
        "surface": "ours_cc12m",
        "surface_group": "ours",
        "family": "cc12m",
        "stratum": "ours/attribute",
        "image_key": "url_sha256:image",
        "judge_answers": {
            "qwen": {"answer": "yes", "confidence": 1.0},
            "gemma": {"answer": "no", "confidence": 1.0},
        },
        "repeat_of": repeat_of,
        "is_repeat": repeat_of is not None,
        "stratum_population": 10,
        "stratum_sample_size": 2,
        "sampling_weight": 0.0 if repeat_of is not None else 5.0,
    }


def _reports(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    asset = asset_root / "opaque.jpg"
    asset.write_bytes(b"synthetic-image-bytes")
    asset_sha256 = hashlib.sha256(asset.read_bytes()).hexdigest()
    canonical = {
        "requested_unique_samples": 1,
        "verified_unique_samples": 1,
        "failed_unique_samples": 0,
        "verified": [
            {
                "image_key": "url_sha256:image",
                "sample_id": "cc12m_202603:00001:000000001",
                "stable_image_id": "stable-image-group",
                "source_url_sha1": "a" * 40,
                "storage_uri": "wds://cc12m/00001.tar#000000001.jpg",
            }
        ],
        "failures": [],
    }
    materialized = {
        "requested_unique_images": 1,
        "materialized_unique_images": 1,
        "failed_unique_images": 0,
        "asset_root": str(asset_root.resolve()),
        "assets": [
            {
                "image_key": "url_sha256:image",
                "asset_relpath": "assets/opaque.jpg",
                "bytes": asset.stat().st_size,
                "sha256": asset_sha256,
            }
        ],
        "failures": [],
    }
    return canonical, materialized


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _hashed_adjudication_packet(
    marker: str,
    phase: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "packet_version": ADJUDICATION_PACKET_VERSION,
        "packet_id": "ap_" + marker * 64,
        "phase": phase,
        "evidence": evidence,
    }
    packet["packet_sha256"] = hashlib.sha256(
        json.dumps(
            packet,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return packet


def test_exact_span_resolution_never_guesses() -> None:
    assert resolve_exact_span("A red cat.", "red cat") == (
        {"start": 2, "end": 9},
        "exact_unique_text",
    )
    assert resolve_exact_span("cat and cat", "cat") == (None, "ambiguous")
    assert resolve_exact_span("A cat.", "dog") == (None, "not_found")
    assert resolve_exact_span("A cat.", "") == (None, "missing")
    assert resolve_exact_span("A cat.", {"start": 2, "end": 5}) == (
        {"start": 2, "end": 5},
        "provided_offsets",
    )


def test_normalization_uses_canonical_group_and_zero_weight_hidden_repeat(
    tmp_path: Path,
) -> None:
    canonical, materialized = _reports(tmp_path)
    base = _builder_row(0)
    repeat = _builder_row(1, repeat_of=base["item_id"], span="cat")
    normalized, report = normalize_store_items(
        [base, repeat],
        canonical_report=canonical,
        materialization_report=materialized,
    )

    assert report["base_items"] == 1
    assert report["repeat_items"] == 1
    assert report["unique_image_groups"] == 1
    assert normalized[0]["span"] == {"start": 2, "end": 9}
    assert normalized[1]["span"] == {"start": 6, "end": 9}
    assert {row["item_group"] for row in normalized} == {"stable-image-group"}
    assert normalized[0]["sample_probability"] == pytest.approx(0.2)
    assert normalized[0]["weight"] == 5.0
    assert normalized[1]["sample_probability"] is None
    assert normalized[1]["weight"] == 0.0
    assert normalized[1]["sample"]["base_sample_probability"] == pytest.approx(0.2)
    assert normalized[1]["repeat_of"] == base["item_id"]
    assert normalized[0]["image_asset"] == {
        "asset_relpath": "assets/opaque.jpg",
        "sha256": hashlib.sha256(b"synthetic-image-bytes").hexdigest(),
    }
    serialized_asset = json.dumps(normalized[0]["image_asset"])
    assert "surface" not in serialized_asset
    assert "caption" not in serialized_asset
    assert "qwen" not in serialized_asset
    assert "gemma" not in serialized_asset


def test_normalization_preserves_unresolved_span_without_false_offset(
    tmp_path: Path,
) -> None:
    canonical, materialized = _reports(tmp_path)
    row = _builder_row(0, span="cat")
    row["caption"] = "cat and cat"
    normalized, report = normalize_store_items(
        [row],
        canonical_report=canonical,
        materialization_report=materialized,
    )
    assert normalized[0]["span"] is None
    assert normalized[0]["sample"]["span_text"] == "cat"
    assert normalized[0]["sample"]["span_resolution"] == "ambiguous"
    assert report["span_resolution"] == {"ambiguous": 1}


def test_normalization_rejects_changed_materialized_image_bytes(
    tmp_path: Path,
) -> None:
    canonical, materialized = _reports(tmp_path)
    (tmp_path / "assets" / "opaque.jpg").write_bytes(b"tampered-image-bytes!")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        normalize_store_items(
            [_builder_row(0)],
            canonical_report=canonical,
            materialization_report=materialized,
        )


def test_secure_write_once_is_mode_0600_and_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "invites.private.json"
    secure_write_once_json(target, {"secret": "only-once"})
    assert target.stat().st_mode & 0o777 == 0o600
    assert json.loads(target.read_text()) == {"secret": "only-once"}
    with pytest.raises(FileExistsError):
        secure_write_once_json(target, {"secret": "replacement"})
    assert json.loads(target.read_text()) == {"secret": "only-once"}


@pytest.mark.parametrize(
    "host",
    ["localhost", "LOCALHOST", "127.0.0.1", "127.42.1.9", "::1"],
)
def test_loopback_host_allowlist(host: str) -> None:
    assert is_loopback_host(host)
    validate_bind_security(host, preview=True, tls_enabled=False)
    with pytest.raises(ValueError, match="production serving requires both"):
        validate_bind_security(host, preview=False, tls_enabled=False)
    validate_bind_security(host, preview=False, tls_enabled=True)


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "192.0.2.10", "example.test", "localhost.example.test"],
)
def test_nonloopback_bind_requires_production_tls(host: str) -> None:
    assert not is_loopback_host(host)
    with pytest.raises(ValueError, match="preview may bind only"):
        validate_bind_security(host, preview=True, tls_enabled=True)
    with pytest.raises(ValueError, match="production serving requires both"):
        validate_bind_security(host, preview=False, tls_enabled=False)
    validate_bind_security(host, preview=False, tls_enabled=True)


def test_preview_and_serve_reject_unsafe_bind_before_file_or_database_access() -> None:
    with pytest.raises(ValueError, match="preview may bind only"):
        command_preview(
            Namespace(
                item_json="/does/not/exist.json",
                image="/does/not/exist.jpg",
                phase="caption",
                host="0.0.0.0",
                port=8765,
            )
        )
    with pytest.raises(ValueError, match="production serving requires both"):
        command_serve(
            Namespace(
                db="/does/not/exist.sqlite3",
                study_id="audit",
                asset_root=None,
                host="192.0.2.10",
                port=8765,
                tls_cert=None,
                tls_key=None,
            )
        )


@pytest.mark.parametrize(
    "existing_name",
    ["items.private.jsonl", "study_manifest.json"],
)
def test_sample_refuses_any_existing_frozen_output(
    tmp_path: Path,
    existing_name: str,
) -> None:
    output = tmp_path / "frozen"
    output.mkdir()
    existing = output / existing_name
    existing.write_text("do-not-replace", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new version/destination"):
        command_sample(
            Namespace(
                config="/does/not/exist.yaml",
                output_dir=str(output),
                claims_per_stratum=None,
                no_fingerprint=False,
            )
        )
    assert existing.read_text(encoding="utf-8") == "do-not-replace"


def test_sample_publishes_frozen_pair_once_with_private_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("scripts.human_cbu_eval.load_config", lambda _: {})
    monkeypatch.setattr(
        "scripts.human_cbu_eval.build_from_config",
        lambda *_args, **_kwargs: ([{"item_id": "item-1"}], {"protocol": "v1"}),
    )
    output = tmp_path / "frozen-v1"
    arguments = Namespace(
        config="synthetic.yaml",
        output_dir=str(output),
        claims_per_stratum=None,
        no_fingerprint=False,
    )
    assert command_sample(arguments) == 0
    items = output / "items.private.jsonl"
    manifest = output / "study_manifest.json"
    assert output.stat().st_mode & 0o777 == 0o700
    assert items.stat().st_mode & 0o777 == 0o600
    assert manifest.stat().st_mode & 0o777 == 0o600
    assert json.loads(items.read_text(encoding="utf-8")) == {"item_id": "item-1"}
    items_sha256 = hashlib.sha256(items.read_bytes()).hexdigest()
    assert json.loads(manifest.read_text(encoding="utf-8")) == {
        "protocol": "v1",
        "items_sha256": items_sha256,
    }

    regenerated = tmp_path / "frozen-v1-regenerated"
    assert (
        command_sample(
            Namespace(
                config="synthetic.yaml",
                output_dir=str(regenerated),
                claims_per_stratum=None,
                no_fingerprint=False,
            )
        )
        == 0
    )
    assert (regenerated / "items.private.jsonl").read_bytes() == items.read_bytes()
    assert (regenerated / "study_manifest.json").read_bytes() == manifest.read_bytes()
    with pytest.raises(FileExistsError, match="new version/destination"):
        command_sample(arguments)


def test_build_rejects_claims_per_stratum_override_mismatch() -> None:
    config = {
        "study": {
            "surface_groups": {"ours": ["ours"]},
            "semantic_visual_claim_types": ["object"],
            "claims_per_stratum": 2,
        },
        "inputs": {"judges": {}},
        "images": {},
    }
    with pytest.raises(ValueError, match="does not match the frozen"):
        build_from_config(config, claims_per_stratum=1)


def test_frozen_design_rejects_inconsistent_repeat_surface() -> None:
    base = {
        "item_id": "base",
        "surface": "ours_a",
        "category": "attribute",
        "stratum": "ours/attribute",
        "item_group": "image-a",
        "repeat_of": None,
    }
    repeat = {
        **base,
        "item_id": "repeat",
        "surface": "ours_b",
        "repeat_of": "base",
    }
    with pytest.raises(ValueError, match="does not match.*on surface"):
        validate_frozen_sample_design(
            [base, repeat],
            surface_groups={"ours": ["ours_a", "ours_b"]},
            semantic_visual_claim_types=["attribute"],
            expected_strata=["ours/attribute"],
            claims_per_stratum=1,
        )


def _semantic_repeat_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    base = {
        "item_id": "base",
        "caption": "A red cat sits on a mat.",
        "unit": "a red cat",
        "target": "cat",
        "category": "attribute",
        "span": {"start": 2, "end": 9},
        "surface": "ours",
        "stratum": "ours/attribute",
        "item_group": "image-a",
        "image_locator": {
            "sample_id": "cc12m:1",
            "stable_image_id": "image-a",
        },
        "image_asset": {
            "asset_relpath": "assets/opaque.jpg",
            "sha256": "a" * 64,
        },
        "asset_status": "available",
        "qwen_answer": {
            "answer": "yes",
            "confidence": 0.9,
            "model": "qwen",
        },
        "gemma_answer": {
            "answer": "no",
            "confidence": 0.8,
            "model": "gemma",
        },
        "population": {
            "family": "cc12m",
            "surface_group": "ours",
            "stratum": "ours/attribute",
            "population_claims": 10,
        },
        "sample": {
            "question_id": "surface:1:u0000",
            "caption_id": "surface:1",
            "extractor_request_id": "extractor-1",
            "source_row": 1,
            "span_text": "red cat",
            "span_resolution": "exact_unique_text",
            "stratum_sample_size": 1,
            "base_sample_probability": 0.1,
            "is_repeat": False,
        },
        "weight": 10.0,
        "population_size": 10,
        "sample_probability": 0.1,
        "repeat_of": None,
    }
    repeat = copy.deepcopy(base)
    repeat.update(
        {
            "item_id": "repeat",
            "repeat_of": "base",
            "weight": 0.0,
            "sample_probability": None,
        }
    )
    repeat["sample"]["is_repeat"] = True
    return base, repeat


def test_frozen_design_allows_only_repeat_specific_id_order_and_weight_changes() -> None:
    base, repeat = _semantic_repeat_pair()
    base["position"] = 3
    repeat["position"] = 29
    base["order_key"] = "base-order"
    repeat["order_key"] = "repeat-order"
    base["sample"]["order"] = 3
    repeat["sample"]["order"] = 29

    report = validate_frozen_sample_design(
        [base, repeat],
        surface_groups={"ours": ["ours"]},
        semantic_visual_claim_types=["attribute"],
        expected_strata=["ours/attribute"],
        claims_per_stratum=1,
        expected_repeat_items=1,
    )
    assert report["repeat_items"] == 1


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("caption",), "A blue dog sits on a mat."),
        (("unit",), "a blue dog"),
        (("target",), "dog"),
        (("span", "start"), 3),
        (("sample", "span_text"), "blue dog"),
        (("sample", "question_id"), "surface:other:u0000"),
        (("sample", "caption_id"), "surface:other"),
        (("sample", "extractor_request_id"), "extractor-other"),
        (("qwen_answer", "answer"), "no"),
        (("qwen_answer", "confidence"), 0.1),
        (("gemma_answer", "model"), "different-gemma"),
        (("image_asset", "sha256"), "b" * 64),
    ],
)
def test_frozen_design_rejects_repeat_semantic_or_machine_payload_mutation(
    path: tuple[str, ...],
    replacement: Any,
) -> None:
    base, repeat = _semantic_repeat_pair()
    target: dict[str, Any] = repeat
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = replacement

    with pytest.raises(
        ValueError,
        match=rf"semantic payload.*{re.escape('.'.join(path))}",
    ):
        validate_frozen_sample_design(
            [base, repeat],
            surface_groups={"ours": ["ours"]},
            semantic_visual_claim_types=["attribute"],
            expected_strata=["ours/attribute"],
            claims_per_stratum=1,
            expected_repeat_items=1,
        )


def test_frozen_design_rejects_repeat_reassigned_to_another_declared_category() -> None:
    attribute, repeat = _semantic_repeat_pair()
    count = copy.deepcopy(attribute)
    count.update(
        {
            "item_id": "count-base",
            "category": "count",
            "stratum": "ours/count",
            "item_group": "image-b",
            "repeat_of": None,
        }
    )
    count["population"]["stratum"] = "ours/count"
    repeat["category"] = "count"
    repeat["stratum"] = "ours/count"
    repeat["population"]["stratum"] = "ours/count"

    with pytest.raises(ValueError, match="semantic payload.*category"):
        validate_frozen_sample_design(
            [attribute, count, repeat],
            surface_groups={"ours": ["ours"]},
            semantic_visual_claim_types=["attribute", "count"],
            expected_strata=["ours/attribute", "ours/count"],
            claims_per_stratum=1,
            expected_repeat_items=1,
        )


def test_frozen_design_rejects_missing_configured_hidden_repeats() -> None:
    base = {
        "item_id": "base",
        "surface": "ours",
        "category": "attribute",
        "stratum": "ours/attribute",
        "item_group": "image-a",
        "repeat_of": None,
    }
    with pytest.raises(ValueError, match="hidden-repeat count.*0/1"):
        validate_frozen_sample_design(
            [base],
            surface_groups={"ours": ["ours"]},
            semantic_visual_claim_types=["attribute"],
            expected_strata=["ours/attribute"],
            claims_per_stratum=1,
            expected_repeat_items=1,
        )


def test_frozen_design_rejects_inconsistent_inverse_probability_weight() -> None:
    base = {
        "item_id": "base",
        "surface": "ours",
        "category": "attribute",
        "stratum": "ours/attribute",
        "item_group": "image-a",
        "repeat_of": None,
        "population_size": 10,
        "sample_probability": 0.1,
        "weight": 9.0,
        "sample": {"stratum_sample_size": 1},
    }
    with pytest.raises(ValueError, match="inconsistent inverse-probability weight"):
        validate_frozen_sample_design(
            [base],
            surface_groups={"ours": ["ours"]},
            semantic_visual_claim_types=["attribute"],
            expected_strata=["ours/attribute"],
            claims_per_stratum=1,
            validate_sampling_metadata=True,
        )


@pytest.mark.parametrize("required_labels_per_item", [None, True, 1])
def test_init_rejects_unfrozen_or_unsafe_required_label_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    required_labels_per_item: Any,
) -> None:
    monkeypatch.setattr(
        "scripts.human_cbu_eval.load_config",
        lambda _: {
            "study": {
                "claims_per_stratum": 1,
                "required_labels_per_item": required_labels_per_item,
            },
            "inputs": {},
            "images": {},
        },
    )
    with pytest.raises(ValueError, match="required_labels_per_item.*integer >= 2"):
        command_init_db(
            Namespace(
                config="synthetic.yaml",
                sample_dir=str(tmp_path),
                db=str(tmp_path / "study.sqlite3"),
                items=None,
                sample_manifest=None,
                canonical_report=None,
                materialization_report=None,
                init_report=None,
            )
        )


def test_human_cbu_v1_requires_exactly_two_initial_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.human_cbu_eval.load_config",
        lambda _: {
            "study": {
                "protocol_version": "human_cbu_v1",
                "claims_per_stratum": 1,
                "required_labels_per_item": 3,
            },
            "inputs": {},
            "images": {},
        },
    )
    with pytest.raises(ValueError, match="requires exactly 2 initial labels"):
        command_init_db(
            Namespace(
                config="synthetic.yaml",
                sample_dir=str(tmp_path),
                db=str(tmp_path / "study.sqlite3"),
                items=None,
                sample_manifest=None,
                canonical_report=None,
                materialization_report=None,
                init_report=None,
            )
        )


@pytest.mark.parametrize(
    "adjudicator_pseudonym",
    [None, "", "contains whitespace", "a" * 65],
)
def test_human_cbu_v1_requires_predeclared_adjudicator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adjudicator_pseudonym: str | None,
) -> None:
    monkeypatch.setattr(
        "scripts.human_cbu_eval.load_config",
        lambda _: {
            "study": {
                "protocol_version": "human_cbu_v1",
                "claims_per_stratum": 1,
                "required_labels_per_item": 2,
                "adjudicator_pseudonym": adjudicator_pseudonym,
            },
            "inputs": {},
            "images": {},
        },
    )
    with pytest.raises(ValueError, match="requires study.adjudicator_pseudonym"):
        command_init_db(
            Namespace(
                config="synthetic.yaml",
                sample_dir=str(tmp_path),
                db=str(tmp_path / "study.sqlite3"),
                items=None,
                sample_manifest=None,
                canonical_report=None,
                materialization_report=None,
                init_report=None,
            )
        )


def test_human_cbu_v1_init_rejects_missing_hidden_repeat(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    canonical, materialized = _reports(sample_dir)
    config = {
        "study": {
            "id": "audit",
            "title": "Visual-Claim Annotation Study",
            "protocol_version": "human_cbu_v1",
            "consent_version": "consent-v1",
            "token_budget": 64,
            "seed": 1477,
            "claims_per_stratum": 1,
            "repeat_fraction": 1.0,
            "required_labels_per_item": 2,
            "adjudicator_pseudonym": "human-adjudicator-01",
            "surface_groups": {"ours": ["ours_cc12m"]},
            "semantic_visual_claim_types": ["attribute"],
        },
        "inputs": {},
        "images": {},
    }
    config_path = sample_dir / "config.yaml"
    _write_json(config_path, config)
    row = _builder_row(0)
    row["stratum_sample_size"] = 1
    row["sampling_weight"] = 10.0
    _write_jsonl(sample_dir / "items.private.jsonl", [row])
    items_path = sample_dir / "items.private.jsonl"
    manifest_path = sample_dir / "study_manifest.json"
    manifest = {
        "study_namespace": "audit",
        "surface_groups": {"ours": ["ours_cc12m"]},
        "semantic_visual_claim_types": ["attribute"],
        "expected_strata": ["ours/attribute"],
        "claims_per_stratum": 1,
        "repeat_fraction": 1.0,
        "sample": {"repeat_items": 1},
    }
    _write_json(manifest_path, manifest)
    _write_json(sample_dir / "canonical_manifest_report.private.json", canonical)
    _write_json(sample_dir / "materialization_report.private.json", materialized)

    arguments = Namespace(
        config=str(config_path),
        sample_dir=str(sample_dir),
        db=str(tmp_path / "study.sqlite3"),
        items=None,
        sample_manifest=None,
        canonical_report=None,
        materialization_report=None,
        init_report=None,
    )
    with pytest.raises(ValueError, match="must bind items.private.jsonl"):
        command_init_db(arguments)

    frozen_items_bytes = items_path.read_bytes()
    manifest["items_sha256"] = hashlib.sha256(frozen_items_bytes).hexdigest()
    _write_json(manifest_path, manifest)
    items_path.write_bytes(frozen_items_bytes + b"\n")
    with pytest.raises(ValueError, match="does not match the exact"):
        command_init_db(arguments)

    items_path.write_bytes(frozen_items_bytes)
    with pytest.raises(ValueError, match="hidden-repeat count.*0/1"):
        command_init_db(arguments)


def test_adjudication_packets_cli_writes_private_self_contained_packet_directories_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root = tmp_path / "source"
    source_asset = source_root / "assets" / "private-name.jpg"
    source_asset.parent.mkdir(parents=True)
    source_bytes = b"exact-frozen-image-bytes"
    source_asset.write_bytes(source_bytes)
    image_sha256 = hashlib.sha256(source_bytes).hexdigest()
    image_packet = _hashed_adjudication_packet(
        "c",
        "image",
        {
            "unit": "a red cat",
            "target": "cat",
            "category": "attribute",
            "image_file": "image.jpg",
            "image_sha256": image_sha256,
        },
    )
    semantic_categories = ["object", "attribute", "relation"]

    class FakeStore:
        def adjudication_packet_bundle_inputs(
            self,
            study_id: str,
            phase: str,
        ) -> dict[str, Any]:
            assert study_id == "audit"
            assert phase == "image"
            return {
                "packets": [image_packet],
                "image_sources": {
                    image_packet["packet_id"]: "assets/private-name.jpg",
                },
            }

        def get_study(self, study_id: str) -> dict[str, Any]:
            assert study_id == "audit"
            return {
                "metadata": {
                    "asset_serve_root": str(source_root),
                    "semantic_visual_claim_types": semantic_categories,
                }
            }

    monkeypatch.setattr("scripts.human_cbu_eval._open_store", lambda _: FakeStore())
    output = tmp_path / "adjudication-packets"
    arguments = Namespace(
        db="unused.sqlite3",
        study_id="audit",
        phase="image",
        output_dir=str(output),
        asset_root=None,
    )
    assert command_adjudication_packets(arguments) == 0
    assert output.stat().st_mode & 0o777 == 0o700
    packets_dir = output / "packets"
    packet_dir = packets_dir / image_packet["packet_id"]
    assert packets_dir.stat().st_mode & 0o777 == 0o700
    assert packet_dir.stat().st_mode & 0o777 == 0o700
    assert {path.name for path in output.iterdir()} == {"packets"}
    assert {path.name for path in packets_dir.iterdir()} == {image_packet["packet_id"]}
    assert {path.name for path in packet_dir.iterdir()} == {
        "packet.json",
        "review.html",
        "image.jpg",
    }

    packet_file = packet_dir / "packet.json"
    review_file = packet_dir / "review.html"
    copied_image = packet_dir / "image.jpg"
    assert packet_file.stat().st_mode & 0o777 == 0o600
    assert review_file.stat().st_mode & 0o777 == 0o600
    assert copied_image.stat().st_mode & 0o777 == 0o600
    assert copied_image.read_bytes() == source_bytes
    assert copied_image.stat().st_ino != source_asset.stat().st_ino
    assert json.loads(packet_file.read_text(encoding="utf-8")) == image_packet
    review_html = review_file.read_text(encoding="utf-8")
    assert "a red cat" in review_html
    assert "image.jpg" in review_html
    assert "private-name" not in review_html
    assert str(source_root) not in review_html
    assert not (packet_dir / "response.json").exists()

    report = json.loads(capsys.readouterr().out)
    assert report["packet_count"] == 1
    assert report["image_count"] == 1
    assert report["packet_directories"] == [str(packet_dir.resolve())]
    assert report["review_files"] == [str(review_file.resolve())]
    assert "exactly one packet directory" in report["response_workflow"]
    serialized = json.dumps(image_packet, sort_keys=True) + review_html + json.dumps(report)
    assert "private-name" not in serialized
    assert str(source_root) not in serialized
    assert not (output / "packets.json").exists()
    assert not (output / "images").exists()
    assert not (output / "response-templates").exists()
    assert not (output / "responses").exists()
    with pytest.raises(FileExistsError):
        command_adjudication_packets(arguments)


def test_caption_only_adjudication_packet_has_no_image_or_caption_leakage_outside_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _hashed_adjudication_packet(
        "d",
        "caption",
        {
            "unit": "a red cat",
            "target": "cat",
            "category": "attribute",
            "caption": "A red cat.",
            "span": "red cat",
        },
    )
    semantic_categories = ["object", "attribute", "relation"]

    class FakeStore:
        def adjudication_packet_bundle_inputs(
            self,
            study_id: str,
            phase: str,
        ) -> dict[str, Any]:
            assert study_id == "audit"
            assert phase == "caption"
            return {"packets": [packet], "image_sources": {}}

        def get_study(self, study_id: str) -> dict[str, Any]:
            assert study_id == "audit"
            return {
                "metadata": {
                    "semantic_visual_claim_types": semantic_categories,
                }
            }

    monkeypatch.setattr("scripts.human_cbu_eval._open_store", lambda _: FakeStore())
    output = tmp_path / "caption-only"
    assert (
        command_adjudication_packets(
            Namespace(
                db="unused.sqlite3",
                study_id="audit",
                phase="caption",
                output_dir=str(output),
                asset_root=None,
            )
        )
        == 0
    )
    packet_dir = output / "packets" / packet["packet_id"]
    assert {path.name for path in output.iterdir()} == {"packets"}
    assert {path.name for path in packet_dir.iterdir()} == {
        "packet.json",
        "review.html",
    }
    assert json.loads((packet_dir / "packet.json").read_text()) == packet
    assert (packet_dir / "review.html").stat().st_mode & 0o777 == 0o600
    assert "A <mark>red cat</mark>." in (packet_dir / "review.html").read_text(encoding="utf-8")
    assert not (packet_dir / "response.json").exists()
    assert not any(path.name.startswith("image.") for path in packet_dir.iterdir())
    assert not (output / "packets.json").exists()


def test_adjudication_packets_reject_mixed_phase_store_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caption_packet = _hashed_adjudication_packet(
        "a",
        "caption",
        {
            "unit": "a red cat",
            "target": "cat",
            "category": "attribute",
            "caption": "A red cat.",
            "span": "red cat",
        },
    )
    image_sha256 = hashlib.sha256(b"image").hexdigest()
    image_packet = _hashed_adjudication_packet(
        "b",
        "image",
        {
            "unit": "a red cat",
            "target": "cat",
            "category": "attribute",
            "image_file": "image.jpg",
            "image_sha256": image_sha256,
        },
    )

    class FakeStore:
        def adjudication_packet_bundle_inputs(
            self,
            study_id: str,
            phase: str,
        ) -> dict[str, Any]:
            assert phase == "caption"
            return {
                "packets": [caption_packet, image_packet],
                "image_sources": {image_packet["packet_id"]: "assets/image.jpg"},
            }

    monkeypatch.setattr("scripts.human_cbu_eval._open_store", lambda _: FakeStore())
    output = tmp_path / "mixed"
    with pytest.raises(ValueError, match="wrong phase"):
        command_adjudication_packets(
            Namespace(
                db="unused.sqlite3",
                study_id="audit",
                phase="caption",
                output_dir=str(output),
                asset_root=None,
            )
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("source_ref", "source_bytes", "frozen_sha256", "expected_error"),
    [
        (
            "../outside.jpg",
            b"outside",
            hashlib.sha256(b"outside").hexdigest(),
            "safe relative path",
        ),
        (
            "assets/source.jpg",
            b"changed bytes",
            hashlib.sha256(b"original bytes").hexdigest(),
            "frozen materialization SHA-256",
        ),
    ],
)
def test_adjudication_packets_reject_traversal_and_changed_image_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_ref: str,
    source_bytes: bytes,
    frozen_sha256: str,
    expected_error: str,
) -> None:
    source_root = tmp_path / "source"
    (source_root / "assets").mkdir(parents=True)
    if source_ref.startswith("../"):
        (tmp_path / "outside.jpg").write_bytes(source_bytes)
    else:
        (source_root / source_ref).write_bytes(source_bytes)
    packet = _hashed_adjudication_packet(
        "e",
        "image",
        {
            "unit": "a red cat",
            "target": "cat",
            "category": "attribute",
            "image_file": "image.jpg",
            "image_sha256": frozen_sha256,
        },
    )

    class FakeStore:
        def adjudication_packet_bundle_inputs(
            self,
            study_id: str,
            phase: str,
        ) -> dict[str, Any]:
            assert phase == "image"
            return {
                "packets": [packet],
                "image_sources": {packet["packet_id"]: source_ref},
            }

        def get_study(self, study_id: str) -> dict[str, Any]:
            return {
                "metadata": {
                    "asset_serve_root": str(source_root),
                    "semantic_visual_claim_types": ["object", "attribute", "relation"],
                }
            }

    monkeypatch.setattr("scripts.human_cbu_eval._open_store", lambda _: FakeStore())
    output = tmp_path / f"packets-{expected_error.split()[0]}"
    with pytest.raises(ValueError, match=expected_error):
        command_adjudication_packets(
            Namespace(
                db="unused.sqlite3",
                study_id="audit",
                phase="image",
                output_dir=str(output),
                asset_root=None,
            )
        )
    assert not list(output.rglob("packet.json"))
    assert not list(output.rglob("review.html"))
    assert not list(output.rglob("image.*"))


def test_provision_adjudicator_cli_writes_raw_credential_once_and_privately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStore:
        def provision_adjudicator(self, study_id: str) -> dict[str, str]:
            assert study_id == "audit"
            return {
                "study_id": study_id,
                "adjudicator_pseudonym": "human-adjudicator-01",
                "credential": "raw-secret-credential",
            }

    monkeypatch.setattr("scripts.human_cbu_eval._open_store", lambda _: FakeStore())
    output = tmp_path / "adjudicator-credential.private.json"
    arguments = Namespace(
        db="unused.sqlite3",
        study_id="audit",
        output=str(output),
    )
    assert command_provision_adjudicator(arguments) == 0
    assert output.stat().st_mode & 0o777 == 0o600
    assert json.loads(output.read_text(encoding="utf-8"))["credential"] == ("raw-secret-credential")
    with pytest.raises(FileExistsError):
        command_provision_adjudicator(arguments)


def test_adjudication_cli_parsers_scope_review_to_one_packet_directory() -> None:
    review = parse_args(
        [
            "adjudication-review",
            "--db",
            "study.sqlite3",
            "--study-id",
            "audit",
            "--packet-dir",
            "private/packets/ap_" + "a" * 64,
            "--credential-file",
            "adjudicator.private.json",
        ]
    )
    assert review.command == "adjudication-review"
    assert review.packet_dir.endswith("ap_" + "a" * 64)
    assert review.credential_file == "adjudicator.private.json"
    assert review.host == "127.0.0.1"
    assert review.port == 8766
    assert review.no_browser is False
    assert not hasattr(review, "phase")
    assert not hasattr(review, "bundle_dir")
    assert not hasattr(review, "response_json")

    adjudicate = parse_args(
        [
            "adjudicate",
            "--db",
            "study.sqlite3",
            "--study-id",
            "audit",
            "--phase",
            "image",
            "--packet-dir",
            "private/packets/ap_" + "a" * 64,
            "--credential-file",
            "adjudicator.private.json",
        ]
    )
    assert adjudicate.phase == "image"
    assert adjudicate.packet_dir.endswith("ap_" + "a" * 64)
    assert not hasattr(adjudicate, "bundle_dir")
    assert not hasattr(adjudicate, "response_json")


def test_participant_test_parser_pairs_additional_items_with_images() -> None:
    arguments = parse_args(
        [
            "participant-test",
            "--item-json",
            "first.json",
            "--image",
            "first.jpg",
            "--additional-item-json",
            "second.json",
            "--additional-image",
            "second.jpg",
            "--additional-item-json",
            "third.json",
            "--additional-image",
            "third.jpg",
            "--pairs",
            "3",
        ]
    )
    assert arguments.item_json == "first.json"
    assert arguments.image == "first.jpg"
    assert arguments.additional_item_json == ["second.json", "third.json"]
    assert arguments.additional_image == ["second.jpg", "third.jpg"]
    assert arguments.pairs == 3


def test_participant_test_command_rejects_unpaired_additional_fixture(
    tmp_path: Path,
) -> None:
    item_path = tmp_path / "item.json"
    item_path.write_text(
        json.dumps(
            {
                "category": "object",
                "unit": "sphere",
                "target": "sphere",
                "caption": "A sphere.",
                "span": "sphere",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="--additional-item-json and --additional-image",
    ):
        command_participant_test(
            Namespace(
                host="127.0.0.1",
                port=8768,
                item_json=str(item_path),
                image=str(tmp_path / "image.jpg"),
                additional_item_json=["second.json"],
                additional_image=[],
                pairs=2,
            )
        )


def test_adjudicate_cli_passes_only_packet_directory_and_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential_file = tmp_path / "adjudicator.private.json"
    _write_json(
        credential_file,
        {
            "study_id": "audit",
            "adjudicator_pseudonym": "human-adjudicator-01",
            "credential": "raw-secret-credential",
        },
    )
    credential_file.chmod(0o600)
    packet_dir = tmp_path / "private" / "packets" / ("ap_" + "a" * 64)
    packet_dir.mkdir(parents=True)
    observed: dict[str, Any] = {}

    class FakeStore:
        def get_study(self, study_id: str) -> dict[str, Any]:
            assert study_id == "audit"
            return {
                "protocol_version": "human_cbu_v1",
                "metadata": {
                    "adjudicator_pseudonym": "human-adjudicator-01",
                },
            }

        def adjudicate_packet_response(
            self,
            study_id: str,
            phase: str,
            *,
            packet_dir: str,
            adjudicator_credential: str,
        ) -> dict[str, Any]:
            observed.update(
                {
                    "study_id": study_id,
                    "phase": phase,
                    "packet_dir": packet_dir,
                    "adjudicator_credential": adjudicator_credential,
                }
            )
            return {"packet_id": "ap_" + "a" * 64, "phase": phase}

    monkeypatch.setattr("scripts.human_cbu_eval._open_store", lambda _: FakeStore())
    assert (
        command_adjudicate(
            Namespace(
                db="unused.sqlite3",
                study_id="audit",
                item_id=None,
                phase="image",
                packet_dir=str(packet_dir),
                payload_json=None,
                adjudicator_pseudonym=None,
                credential_file=str(credential_file),
            )
        )
        == 0
    )
    assert observed == {
        "study_id": "audit",
        "phase": "image",
        "packet_dir": str(packet_dir),
        "adjudicator_credential": "raw-secret-credential",
    }


def test_replace_participant_cli_requires_exact_source_confirmation() -> None:
    with pytest.raises(ValueError, match="must exactly match"):
        command_replace_participant(
            Namespace(
                db="unused.sqlite3",
                study_id="audit",
                source_participant_id="p_source",
                replacement_participant_id="p_reserve",
                confirm_source_participant_id="p_other",
            )
        )


def test_init_invite_assign_and_ethics_gated_seal(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    canonical, materialized = _reports(sample_dir)
    rows = [_builder_row(0)]
    config = {
        "study": {
            "id": "audit",
            "title": "Visual-Claim Annotation Study",
            "protocol_version": "human-cbu-v1",
            "consent_version": "consent-v1",
            "token_budget": 64,
            "seed": 1477,
            "claims_per_stratum": 1,
            "required_labels_per_item": 2,
            "surface_groups": {"ours": ["ours_cc12m"]},
            "semantic_visual_claim_types": ["attribute"],
            "collect": {"image_support": True},
        },
        "inputs": {},
        "images": {},
    }
    config_path = sample_dir / "config.yaml"
    config_path.write_text(
        """
study:
  id: audit
  title: Visual-Claim Annotation Study
  protocol_version: human-cbu-v1
  consent_version: consent-v1
  token_budget: 64
  seed: 1477
  claims_per_stratum: 1
  required_labels_per_item: 2
  semantic_visual_claim_types: [attribute]
  surface_groups:
    ours: [ours_cc12m]
  collect:
    image_support: true
inputs: {}
images: {}
""".lstrip(),
        encoding="utf-8",
    )
    _write_jsonl(sample_dir / "items.private.jsonl", rows)
    _write_json(
        sample_dir / "study_manifest.json",
        {
            "study_namespace": "audit",
            "surface_groups": {"ours": ["ours_cc12m"]},
            "semantic_visual_claim_types": ["attribute"],
            "expected_strata": ["ours/attribute"],
            "claims_per_stratum": 1,
        },
    )
    _write_json(
        sample_dir / "canonical_manifest_report.private.json",
        canonical,
    )
    _write_json(
        sample_dir / "materialization_report.private.json",
        materialized,
    )
    mismatched_row = _builder_row(0)
    mismatched_row["category"] = "count"
    mismatched_row["stratum"] = "ours/count"
    mismatched_items = sample_dir / "items-mismatched.private.jsonl"
    _write_jsonl(mismatched_items, [mismatched_row])
    with pytest.raises(ValueError, match="semantic_visual_claim_types"):
        command_init_db(
            Namespace(
                config=str(config_path),
                sample_dir=str(sample_dir),
                db=str(tmp_path / "mismatched.sqlite3"),
                items=str(mismatched_items),
                sample_manifest=None,
                canonical_report=None,
                materialization_report=None,
                init_report=None,
            )
        )
    assert not (tmp_path / "mismatched.sqlite3").exists()

    inconsistent_stratum = _builder_row(0)
    inconsistent_stratum["stratum"] = "reference/attribute"
    inconsistent_items = sample_dir / "items-inconsistent.private.jsonl"
    _write_jsonl(inconsistent_items, [inconsistent_stratum])
    with pytest.raises(ValueError, match="expected 'ours/attribute'"):
        command_init_db(
            Namespace(
                config=str(config_path),
                sample_dir=str(sample_dir),
                db=str(tmp_path / "inconsistent.sqlite3"),
                items=str(inconsistent_items),
                sample_manifest=None,
                canonical_report=None,
                materialization_report=None,
                init_report=None,
            )
        )

    two_per_cell_config = sample_dir / "config-two-per-cell.yaml"
    two_per_cell_config.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "claims_per_stratum: 1",
            "claims_per_stratum: 2",
        ),
        encoding="utf-8",
    )
    two_per_cell_manifest = sample_dir / "manifest-two-per-cell.json"
    _write_json(
        two_per_cell_manifest,
        {
            "study_namespace": "audit",
            "surface_groups": {"ours": ["ours_cc12m"]},
            "semantic_visual_claim_types": ["attribute"],
            "expected_strata": ["ours/attribute"],
            "claims_per_stratum": 2,
        },
    )
    with pytest.raises(ValueError, match="exactly study.claims_per_stratum"):
        command_init_db(
            Namespace(
                config=str(two_per_cell_config),
                sample_dir=str(sample_dir),
                db=str(tmp_path / "short-cell.sqlite3"),
                items=None,
                sample_manifest=str(two_per_cell_manifest),
                canonical_report=None,
                materialization_report=None,
                init_report=None,
            )
        )

    mismatched_manifest = sample_dir / "manifest-mismatched-budget.json"
    _write_json(
        mismatched_manifest,
        {
            "study_namespace": "audit",
            "surface_groups": {"ours": ["ours_cc12m"]},
            "semantic_visual_claim_types": ["attribute"],
            "expected_strata": ["ours/attribute"],
            "claims_per_stratum": 2,
        },
    )
    with pytest.raises(ValueError, match="manifest claims_per_stratum"):
        command_init_db(
            Namespace(
                config=str(config_path),
                sample_dir=str(sample_dir),
                db=str(tmp_path / "manifest-mismatch.sqlite3"),
                items=None,
                sample_manifest=str(mismatched_manifest),
                canonical_report=None,
                materialization_report=None,
                init_report=None,
            )
        )

    frozen_manifest_path = sample_dir / "study_manifest.json"
    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    frozen_manifest["items_sha256"] = hashlib.sha256((sample_dir / "items.private.jsonl").read_bytes()).hexdigest()
    _write_json(frozen_manifest_path, frozen_manifest)

    database = tmp_path / "study.sqlite3"
    participant_information_path = tmp_path / "participant-information.json"
    _write_json(
        participant_information_path,
        {
            "approved_version": "consent-v1",
            "duration": "Approximately 45 minutes.",
            "compensation": "As described in the approved recruitment notice.",
            "data_retention": "Pseudonymous responses are retained under the approved protocol.",
            "research_contact": "Use the contact in the institutionally approved information sheet.",
            "withdrawal_policy": "Participants may stop at any time; withdrawn responses are excluded.",
            "information_sheet_url": "https://example.edu/approved-human-cbu-information",
        },
    )
    assert (
        command_init_db(
            Namespace(
                config=str(config_path),
                sample_dir=str(sample_dir),
                db=str(database),
                items=None,
                sample_manifest=None,
                canonical_report=None,
                materialization_report=None,
                init_report=None,
            )
        )
        == 0
    )
    assert database.stat().st_mode & 0o777 == 0o600
    store = HumanCBUStore.open(database)
    draft = store.get_study("audit")
    assert draft["status"] == "draft"
    assert draft["metadata"]["readiness"]["expected_proposed_types"] == ["attribute"]
    assert draft["metadata"]["readiness"]["expected_strata"] == ["ours/attribute"]
    assert draft["metadata"]["required_labels_per_item"] == 2
    assert draft["metadata"]["claims_per_stratum"] == 1
    assert draft["metadata"]["semantic_visual_claim_types"] == ["attribute"]
    assert (
        draft["metadata"]["items_sha256"]
        == hashlib.sha256((sample_dir / "items.private.jsonl").read_bytes()).hexdigest()
    )
    assert draft["metadata"]["items_manifest_binding"]["verified"] is True
    private = store.get_private_item("audit", "item-0")
    assert private["span"] == "red cat"
    assert private["span_offsets"] == {"start": 2, "end": 9}
    assert private["item_group"] == "stable-image-group"
    export_dir = tmp_path / "draft-export"
    with pytest.raises(ValueError, match="closed or archived"):
        command_export(
            Namespace(
                db=str(database),
                study_id="audit",
                output_dir=str(export_dir),
                public_id_namespace=None,
                bootstrap_resamples=10,
                bootstrap_seed=1477,
                include_notes=False,
                include_exact_timestamps=False,
                include_public_rows=False,
            )
        )
    assert not export_dir.exists()

    invite_path = tmp_path / "invites.private.json"
    assert (
        command_invites(
            Namespace(
                db=str(database),
                study_id="audit",
                count=2,
                output=str(invite_path),
                expires_in_seconds=None,
            )
        )
        == 0
    )
    assert invite_path.stat().st_mode & 0o777 == 0o600
    assert len(json.loads(invite_path.read_text())["invites"]) == 2
    with pytest.raises(ValueError, match="frozen required_labels_per_item"):
        command_assign(
            Namespace(
                db=str(database),
                study_id="audit",
                config=str(config_path),
                labels_per_item=1,
                seed=None,
            )
        )
    assignments = HumanCBUStore.open(database).status_summary("audit")["assignments"]
    assert all(phase["total"] == 0 for phase in assignments.values())
    with pytest.raises(ValueError, match="--seed does not match the frozen assignment_seed"):
        command_assign(
            Namespace(
                db=str(database),
                study_id="audit",
                config=str(config_path),
                labels_per_item=None,
                seed="different-seed",
            )
        )
    assignments = HumanCBUStore.open(database).status_summary("audit")["assignments"]
    assert all(phase["total"] == 0 for phase in assignments.values())
    assert (
        command_assign(
            Namespace(
                db=str(database),
                study_id="audit",
                config=str(config_path),
                labels_per_item=None,
                seed=None,
            )
        )
        == 0
    )

    with pytest.raises(ValueError, match="nonempty"):
        command_seal(
            Namespace(
                db=str(database),
                study_id="audit",
                ethics_determination_id="   ",
                participant_information_json=str(participant_information_path),
            )
    )
    assert HumanCBUStore.open(database).get_study("audit")["status"] == "draft"
    for invalid_basis in (
        "review-not-required",
        "Review-Not-Required",
        "review_not_required",
        "review-not-required :   ",
    ):
        with pytest.raises(ValueError, match="authority or basis"):
            command_seal(
                Namespace(
                    db=str(database),
                    study_id="audit",
                    ethics_determination_id=invalid_basis,
                    participant_information_json=str(participant_information_path),
                )
            )
    assert HumanCBUStore.open(database).get_study("audit")["status"] == "draft"
    assert (
        command_seal(
            Namespace(
                db=str(database),
                study_id="audit",
                ethics_determination_id="IRB-EQUIVALENT-2026-1477",
                participant_information_json=str(participant_information_path),
            )
        )
        == 0
    )
    sealed = HumanCBUStore.open(database).get_study("audit")
    assert sealed["status"] == "ready"
    assert sealed["metadata"]["ethics_or_irb_equivalent_determination_id"] == "IRB-EQUIVALENT-2026-1477"
    assert sealed["metadata"]["participant_information"]["approved_version"] == "consent-v1"

    store = HumanCBUStore.open(database)
    store.set_study_status("audit", "open")
    invite_rows = json.loads(invite_path.read_text())["invites"]
    for invite in invite_rows:
        token = store.create_session(invite["invite_code"])["session_token"]
        store.record_consent_profile(
            token,
            consented=True,
            consent_version="consent-v1",
            profile={
                "author_status": "non_author",
                "recruitment_source": "academic_pool",
            },
        )
        caption_task = store.fetch_next_task(token)["task"]
        store.save_annotation(
            token,
            caption_task["assignment_id"],
            {
                "caption_licensed": "yes",
                "atomic_visual_claim": "yes",
                "category_check": "correct",
            },
        )
        image_task = store.fetch_next_task(token)["task"]
        store.save_annotation(
            token,
            image_task["assignment_id"],
            {"image_support": "yes"},
        )
    store.set_study_status("audit", "closed")
    completed_export = tmp_path / "completed-export"
    assert (
        command_export(
            Namespace(
                db=str(database),
                study_id="audit",
                output_dir=str(completed_export),
                public_id_namespace=None,
                bootstrap_resamples=10,
                bootstrap_seed=1477,
                include_notes=False,
                include_exact_timestamps=False,
                include_public_rows=False,
            )
        )
        == 0
    )
    assert not (completed_export / "annotations.public.csv").exists()
    metrics = json.loads((completed_export / "metrics.json").read_text())
    assert metrics["study_provenance"] == {
        "ethics_review_basis": "IRB-EQUIVALENT-2026-1477",
        "participant_notice_version": "consent-v1",
    }
    assert metrics["annotators"]["flow"]["completed"] == 2
    assert metrics["annotators"]["profile_marginals"]["n_participants"] == 2
    assert metrics["analysis_status"]["ready"] is False
    assert "substantive_below_minimum:overall:1/120" in metrics["analysis_status"]["reasons"]
    assert config["study"]["id"] == "audit"
