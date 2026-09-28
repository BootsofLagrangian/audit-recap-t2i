from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from audit_recap_t2i.human_cbu.assets import (
    materialize_selected_images,
    verify_canonical_manifest,
)
from audit_recap_t2i.human_cbu.builder import (
    build_sample,
    load_claim_records,
    parse_historical_wds_path,
    stable_image_key,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def claim_response(surface: str, caption_index: int, category: str) -> dict:
    caption_id = f"{surface}:{caption_index}"
    return {
        "ok": True,
        "parsed": {
            "caption_id": caption_id,
            "claimed_units": [
                {
                    "category": category,
                    "unit": f"{category} unit",
                    "span": f"{category} span",
                    "target": "scene",
                }
            ],
        },
        "request": {
            "request_id": f"extract-{caption_id}",
            "caption_id": caption_id,
            "surface": surface,
            "source_row": caption_index,
            "token_budget": 64,
            "caption": f"budgeted {category} span",
            "source_caption": f"full budgeted {category} span caption",
        },
    }


def judge_response(surface: str, caption_index: int, category: str, answer: str, model: str) -> dict:
    caption_id = f"{surface}:{caption_index}"
    question_id = f"{caption_id}:u0000"
    return {
        "ok": True,
        "model": model,
        "request_id": f"judge-{model}-{caption_id}",
        "parsed": {
            "caption_id": caption_id,
            "question_results": [{"question_id": question_id, "answer": answer, "confidence": 0.9}],
        },
        "request": {
            "caption_id": caption_id,
            "surface": surface,
            "source_row": caption_index,
            "token_budget": 64,
            "questions": [
                {
                    "question_id": question_id,
                    "category": category,
                    "question": f"Is {category} visible?",
                }
            ],
            "image_url": f"https://example.test/image-{caption_index}.jpg",
            "image_path": f"/missing/00001__{caption_index:09d}.jpg",
            "image_sha256": None,
            "pair_key": str(caption_index),
            "public_lookup_key": f"https://example.test/image-{caption_index}.jpg",
            "family": "cc12m",
        },
    }


def test_build_sample_balances_surface_group_and_category(tmp_path: Path) -> None:
    categories = ("object", "count")
    claims: list[dict] = []
    qwen: list[dict] = []
    gemma: list[dict] = []
    row_index = 0
    for surface in ("ours", "ref_a", "ref_b"):
        for category in (*categories, "relation"):
            for _ in range(4):
                claims.append(claim_response(surface, row_index, category))
                qwen.append(judge_response(surface, row_index, category, "yes", "qwen"))
                gemma.append(judge_response(surface, row_index, category, "no", "gemma"))
                row_index += 1
    claim_path = tmp_path / "claims.jsonl"
    qwen_path = tmp_path / "qwen.jsonl"
    gemma_path = tmp_path / "gemma.jsonl"
    write_jsonl(claim_path, claims)
    write_jsonl(qwen_path, qwen)
    write_jsonl(gemma_path, gemma)

    result = build_sample(
        claim_response_paths=[claim_path],
        judge_response_paths={"qwen": [qwen_path], "gemma": [gemma_path]},
        surface_groups={"ours": ["ours"], "reference": ["ref_a", "ref_b"]},
        semantic_visual_claim_types=["object", "count"],
        token_budget=64,
        claims_per_stratum=3,
        seed=1477,
        study_namespace="test-study",
        repeat_fraction=0.25,
        required_judges=("qwen", "gemma"),
        fingerprint_inputs=False,
    )

    base_items = [item for item in result.items if not item["is_repeat"]]
    repeats = [item for item in result.items if item["is_repeat"]]
    assert len(base_items) == 12
    assert len(repeats) == 3
    assert {item["stratum"] for item in base_items} == {
        "ours/object",
        "ours/count",
        "reference/object",
        "reference/count",
    }
    assert all(item["judge_answers"]["qwen"]["answer"] == "yes" for item in base_items)
    assert all(item["judge_answers"]["gemma"]["answer"] == "no" for item in base_items)
    assert all(item["sampling_weight"] > 0 for item in base_items)
    assert all(item["sampling_weight"] == 0 for item in repeats)
    assert all(item["repeat_of"] for item in repeats)
    assert result.report["semantic_visual_claim_types"] == ["object", "count"]
    assert result.report["expected_strata"] == [
        "ours/count",
        "ours/object",
        "reference/count",
        "reference/object",
    ]

    assert result.report["claim_load"]["claims"] == 24
    assert set(result.report["claim_load"]["categories"]) == {"object", "count"}

    with pytest.raises(ValueError, match="missing=.*lighting"):
        build_sample(
            claim_response_paths=[claim_path],
            judge_response_paths={"qwen": [qwen_path], "gemma": [gemma_path]},
            surface_groups={"ours": ["ours"], "reference": ["ref_a", "ref_b"]},
            semantic_visual_claim_types=["object", "count", "lighting"],
            token_budget=64,
            claims_per_stratum=3,
            seed=1477,
            study_namespace="test-study-missing-cell",
            required_judges=("qwen", "gemma"),
            fingerprint_inputs=False,
        )


def test_claim_retry_replaces_invalid_or_earlier_response(tmp_path: Path) -> None:
    first = claim_response("ours", 0, "object")
    replacement = claim_response("ours", 0, "count")
    base_path = tmp_path / "base.jsonl"
    retry_path = tmp_path / "retry.jsonl"
    write_jsonl(base_path, [first, {"ok": False, "parsed": None, "request": {}}])
    write_jsonl(retry_path, [replacement])

    claims, report = load_claim_records([base_path, retry_path], token_budget=64)

    assert claims["ours:0:u0000"]["category"] == "count"
    assert report["duplicate_captions_replaced"] == 1
    assert report["invalid_rows"] == 1


def test_claim_load_matches_paper_normalized_deduplication(tmp_path: Path) -> None:
    row = claim_response("ours", 0, "object")
    row["parsed"]["claimed_units"].append(
        {
            "category": "object",
            "unit": "  OBJECT   UNIT ",
            "span": "a second surface span",
            "target": "SCENE",
        }
    )
    path = tmp_path / "claims.jsonl"
    write_jsonl(path, [row])

    claims, report = load_claim_records([path], token_budget=64)

    assert list(claims) == ["ours:0:u0000"]
    assert report["normalized_duplicate_claims_ignored"] == 1


def test_stable_image_key_prefers_content_hash_and_never_numeric_pair_key() -> None:
    sha = "a" * 64
    key, basis = stable_image_key(
        {
            "image_sha256": sha,
            "public_lookup_key": "https://EXAMPLE.test/a.jpg#fragment",
            "pair_key": "12345",
        }
    )
    assert key == f"sha256:{sha}"
    assert basis == "content_sha256"

    url_key, url_basis = stable_image_key(
        {
            "image_sha256": None,
            "public_lookup_key": "https://EXAMPLE.test/a.jpg#fragment",
            "pair_key": "12345",
        }
    )
    assert url_key.startswith("url_sha256:")
    assert url_basis == "public_lookup_key"


def test_parse_historical_wds_path() -> None:
    assert parse_historical_wds_path("/old/cache/00153__000356174.jpg") == {
        "shard": "00153",
        "key": "000356174",
        "suffix": ".jpg",
    }
    assert parse_historical_wds_path("/old/cache/no-separator.jpg") is None


def test_materialize_selected_image_from_canonical_tar(tmp_path: Path) -> None:
    source_image = tmp_path / "source.jpg"
    Image.new("RGB", (32, 24), color=(25, 50, 75)).save(source_image)
    wds_root = tmp_path / "wds"
    wds_root.mkdir()
    tar_path = wds_root / "00001.tar"
    with tarfile.open(tar_path, "w") as archive:
        archive.add(source_image, arcname="000000123.jpg")
    asset_root = tmp_path / "study" / "assets"
    item = {
        "image_key": "url_sha256:" + "b" * 64,
        "image_sha256_expected": None,
        "wds_locator": {"shard": "00001", "key": "000000123", "suffix": ".jpg"},
    }

    report = materialize_selected_images([item], wds_root=wds_root, asset_root=asset_root, workers=1)

    assert report["materialized_unique_images"] == 1
    assert report["failed_unique_images"] == 0
    asset = report["assets"][0]
    assert Path(asset["path"]).exists()
    assert asset["width"] == 32
    assert asset["height"] == 24
    assert asset["bytes"] > 0


def test_canonical_manifest_verification_uses_exact_sample_and_url(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sample_id": "cc12m_202603:00001:000000123",
                    "canonical_url": "https://example.test/exact.jpg",
                    "source_url_sha1": "a" * 40,
                    "stable_image_id": f"source_url_sha1:{'a' * 40}",
                    "storage_uri": "/canonical/00001.tar#000000123.jpg",
                    "media_ext": ".jpg",
                    "width": 32,
                    "height": 24,
                    "bytes": 100,
                    "sha256_raw": None,
                },
                {
                    "sample_id": "cc12m_202603:00002:000000999",
                    "canonical_url": "https://example.test/exact.jpg",
                    "source_url_sha1": "a" * 40,
                    "stable_image_id": f"source_url_sha1:{'a' * 40}",
                    "storage_uri": "/canonical/00002.tar#000000999.jpg",
                    "media_ext": ".jpg",
                    "width": 32,
                    "height": 24,
                    "bytes": 100,
                    "sha256_raw": None,
                },
            ]
        ),
        manifest,
    )
    item = {
        "image_key": "url_sha256:" + "b" * 64,
        "image_url": "https://example.test/exact.jpg",
        "wds_locator": {"shard": "00001", "key": "000000123", "suffix": ".jpg"},
    }

    report = verify_canonical_manifest([item], manifest_path=manifest)

    assert report["verified_unique_samples"] == 1
    assert report["failed_unique_samples"] == 0
    assert report["verified"][0]["sample_id"] == "cc12m_202603:00001:000000123"


def test_canonical_manifest_verification_fails_closed_on_url_mismatch(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sample_id": "cc12m_202603:00001:000000123",
                    "canonical_url": "https://example.test/canonical.jpg",
                    "source_url_sha1": "a" * 40,
                    "stable_image_id": f"source_url_sha1:{'a' * 40}",
                    "storage_uri": "/canonical/00001.tar#000000123.jpg",
                    "media_ext": ".jpg",
                    "width": 32,
                    "height": 24,
                    "bytes": 100,
                    "sha256_raw": None,
                }
            ]
        ),
        manifest,
    )
    item = {
        "image_key": "url_sha256:" + "b" * 64,
        "image_url": "https://example.test/request.jpg",
        "wds_locator": {"shard": "00001", "key": "000000123", "suffix": ".jpg"},
    }

    report = verify_canonical_manifest([item], manifest_path=manifest)

    assert report["verified_unique_samples"] == 0
    assert report["failed_unique_samples"] == 1
    assert report["failures"][0]["reason"] == "canonical_url_mismatch"
