"""Build a blinded, stratified human-audit sample from released CBU artifacts."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

SEMANTIC_VISUAL_CLAIM_TYPES = (
    "object",
    "attribute",
    "relation",
    "count",
    "style",
    "camera",
    "lighting",
    "text_rendering",
)
JUDGE_ANSWERS = {"yes", "no", "uncertain"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_WDS_MEMBER_RE = re.compile(r"^(?P<shard>[^_][^_]*)__+(?P<key>[^/]+?)(?P<suffix>\.[A-Za-z0-9]+)$")


@dataclass(frozen=True)
class BuildResult:
    """In-memory study items and a reproducibility report."""

    items: list[dict[str, Any]]
    report: dict[str, Any]


def iter_jsonl(paths: Sequence[str | Path]) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from one or more JSONL files."""

    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"expected a JSON object at {path}:{line_number}")
                yield row


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash an artifact without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_fingerprints(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Return stable fingerprints for every input artifact."""

    result: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        stat = path.stat()
        result.append(
            {
                "path": str(path.resolve()),
                "bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    return result


def _request_from_response(row: Mapping[str, Any]) -> Mapping[str, Any]:
    request = row.get("request")
    return request if isinstance(request, Mapping) else {}


def _parsed_from_response(row: Mapping[str, Any]) -> Mapping[str, Any]:
    parsed = row.get("parsed")
    return parsed if isinstance(parsed, Mapping) else {}


def _is_valid_response(row: Mapping[str, Any]) -> bool:
    return row.get("ok") is True and isinstance(row.get("parsed"), Mapping)


def load_claim_records(
    paths: Sequence[str | Path],
    *,
    token_budget: int,
    allowed_categories: Sequence[str] = SEMANTIC_VISUAL_CLAIM_TYPES,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load the latest valid extractor record for every CBU question ID.

    Paths are applied in order. A later valid response replaces an earlier
    response for the same caption, which makes retry JSONL files composable.
    """

    allowed = set(allowed_categories)
    by_caption: dict[str, dict[str, Any]] = {}
    rows_seen = 0
    invalid_rows = 0
    mismatched_caption_ids = 0
    duplicate_captions = 0
    normalized_duplicate_claims = 0
    for row in iter_jsonl(paths):
        rows_seen += 1
        if not _is_valid_response(row):
            invalid_rows += 1
            continue
        request = _request_from_response(row)
        parsed = _parsed_from_response(row)
        if request.get("token_budget") != token_budget:
            continue
        caption_id = str(request.get("caption_id") or parsed.get("caption_id") or "")
        if not caption_id:
            invalid_rows += 1
            continue
        parsed_caption_id = str(parsed.get("caption_id") or caption_id)
        if parsed_caption_id != caption_id:
            mismatched_caption_ids += 1
            continue
        units = parsed.get("claimed_units")
        if not isinstance(units, list):
            invalid_rows += 1
            continue
        normalized_units: list[dict[str, str]] = []
        seen_claims: set[tuple[str, str, str]] = set()
        for index, raw_unit in enumerate(units):
            if not isinstance(raw_unit, Mapping):
                continue
            category = str(raw_unit.get("category") or "")
            if category not in allowed:
                continue
            unit = str(raw_unit.get("unit") or "")
            target = str(raw_unit.get("target") or "")
            dedup_key = (
                category,
                " ".join(unit.casefold().split()),
                " ".join(target.casefold().split()),
            )
            if dedup_key in seen_claims:
                normalized_duplicate_claims += 1
                continue
            seen_claims.add(dedup_key)
            normalized_units.append(
                {
                    "question_id": f"{caption_id}:u{index:04d}",
                    "category": category,
                    "unit": unit,
                    "span": str(raw_unit.get("span") or ""),
                    "target": target,
                }
            )
        if caption_id in by_caption:
            duplicate_captions += 1
        by_caption[caption_id] = {
            "request_id": str(request.get("request_id") or row.get("request_id") or ""),
            "surface": str(request.get("surface") or ""),
            "caption_id": caption_id,
            "source_row": request.get("source_row"),
            "token_budget": token_budget,
            "caption": str(request.get("caption") or ""),
            "source_caption": str(request.get("source_caption") or request.get("caption") or ""),
            "units": normalized_units,
        }

    claims: dict[str, dict[str, Any]] = {}
    duplicate_question_ids = 0
    for caption in by_caption.values():
        for unit in caption["units"]:
            question_id = unit["question_id"]
            if question_id in claims:
                duplicate_question_ids += 1
                continue
            claims[question_id] = {
                **unit,
                "extractor_request_id": caption["request_id"],
                "surface": caption["surface"],
                "caption_id": caption["caption_id"],
                "source_row": caption["source_row"],
                "token_budget": caption["token_budget"],
                "caption": caption["caption"],
                "source_caption": caption["source_caption"],
            }
    report = {
        "rows_seen": rows_seen,
        "invalid_rows": invalid_rows,
        "captions": len(by_caption),
        "claims": len(claims),
        "duplicate_captions_replaced": duplicate_captions,
        "normalized_duplicate_claims_ignored": normalized_duplicate_claims,
        "duplicate_question_ids_ignored": duplicate_question_ids,
        "mismatched_caption_ids_rejected": mismatched_caption_ids,
        "categories": dict(sorted(Counter(item["category"] for item in claims.values()).items())),
    }
    return claims, report


def _normalize_answer(value: Any) -> str | None:
    answer = str(value or "").strip().lower()
    return answer if answer in JUDGE_ANSWERS else None


def _question_map(request: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    questions = request.get("questions")
    if not isinstance(questions, list):
        return {}
    return {
        str(question["question_id"]): question
        for question in questions
        if isinstance(question, Mapping) and question.get("question_id")
    }


def load_judge_records(
    paths: Sequence[str | Path],
    *,
    token_budget: int,
    judge_name: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load latest valid image-conditioned answers keyed by CBU question ID."""

    records: dict[str, dict[str, Any]] = {}
    rows_seen = 0
    invalid_rows = 0
    invalid_answers = 0
    duplicate_answers = 0
    conflicting_answers = 0
    for row in iter_jsonl(paths):
        rows_seen += 1
        if not _is_valid_response(row):
            invalid_rows += 1
            continue
        request = _request_from_response(row)
        parsed = _parsed_from_response(row)
        if request.get("token_budget") != token_budget:
            continue
        results = parsed.get("question_results")
        if not isinstance(results, list):
            invalid_rows += 1
            continue
        questions = _question_map(request)
        for result in results:
            if not isinstance(result, Mapping):
                continue
            question_id = str(result.get("question_id") or "")
            answer = _normalize_answer(result.get("answer"))
            if not question_id or answer is None:
                invalid_answers += 1
                continue
            prior = records.get(question_id)
            if prior is not None:
                duplicate_answers += 1
                if prior["answer"] != answer:
                    conflicting_answers += 1
            question = questions.get(question_id, {})
            records[question_id] = {
                "question_id": question_id,
                "answer": answer,
                "confidence": result.get("confidence"),
                "judge": judge_name,
                "model": str(row.get("model") or ""),
                "judge_request_id": str(row.get("request_id") or request.get("request_id") or ""),
                "surface": str(request.get("surface") or ""),
                "caption_id": str(request.get("caption_id") or ""),
                "source_row": request.get("source_row"),
                "category": str(question.get("category") or ""),
                "question": str(question.get("question") or ""),
                "image_url": request.get("image_url"),
                "image_path": request.get("image_path"),
                "image_sha256": request.get("image_sha256"),
                "pair_id": request.get("pair_id"),
                "pair_key": request.get("pair_key"),
                "public_lookup_key": request.get("public_lookup_key"),
                "family": request.get("family"),
            }
    report = {
        "judge": judge_name,
        "rows_seen": rows_seen,
        "invalid_rows": invalid_rows,
        "answers": len(records),
        "invalid_answers": invalid_answers,
        "duplicate_answers_replaced": duplicate_answers,
        "conflicting_answers_replaced": conflicting_answers,
        "answers_by_label": dict(sorted(Counter(item["answer"] for item in records.values()).items())),
    }
    return records, report


def canonicalize_url(value: Any) -> str | None:
    """Return a conservative URL normalization suitable for hashing."""

    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        return raw
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


def stable_image_key(record: Mapping[str, Any]) -> tuple[str, str]:
    """Build a cross-surface image key from content hash or URL.

    Numeric img2dataset keys are intentionally not accepted as the primary
    cross-surface identity.
    """

    image_sha256 = record.get("image_sha256")
    if isinstance(image_sha256, str) and re.fullmatch(r"[0-9a-fA-F]{64}", image_sha256):
        value = image_sha256.lower()
        return f"sha256:{value}", "content_sha256"
    for field in ("public_lookup_key", "image_url"):
        url = canonicalize_url(record.get(field))
        if url:
            digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
            return f"url_sha256:{digest}", field
    raise ValueError(f"question {record.get('question_id')} has no content hash or URL identity")


def parse_historical_wds_path(value: Any) -> dict[str, str] | None:
    """Parse historical ``shard__key.ext`` cache names without trusting the cache path."""

    if not isinstance(value, str) or not value:
        return None
    name = Path(value).name
    match = _WDS_MEMBER_RE.match(name)
    if match is None:
        return None
    suffix = match.group("suffix").lower()
    if suffix not in IMAGE_SUFFIXES:
        return None
    return {"shard": match.group("shard"), "key": match.group("key"), "suffix": suffix}


def _surface_to_group(surface: str, surface_groups: Mapping[str, Sequence[str]]) -> str | None:
    matches = [group for group, surfaces in surface_groups.items() if surface in set(surfaces)]
    if len(matches) > 1:
        raise ValueError(f"surface {surface!r} appears in multiple surface groups: {matches}")
    return matches[0] if matches else None


def validate_study_design(
    surface_groups: Mapping[str, Sequence[str]],
    semantic_visual_claim_types: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Validate and return the frozen type list and surface×type cells."""

    if not surface_groups:
        raise ValueError("surface_groups must not be empty")
    normalized_groups: dict[str, list[str]] = {}
    seen_surfaces: dict[str, str] = {}
    for raw_group, raw_surfaces in surface_groups.items():
        group = str(raw_group)
        if not group or "/" in group:
            raise ValueError("surface group names must be non-empty and must not contain '/'")
        if not isinstance(raw_surfaces, Sequence) or isinstance(raw_surfaces, (str, bytes)) or not raw_surfaces:
            raise ValueError(f"surface group {group!r} must contain at least one surface")
        surfaces = [str(surface) for surface in raw_surfaces]
        if any(not surface for surface in surfaces) or len(surfaces) != len(set(surfaces)):
            raise ValueError(f"surface group {group!r} contains empty or duplicate surfaces")
        for surface in surfaces:
            previous = seen_surfaces.get(surface)
            if previous is not None:
                raise ValueError(f"surface {surface!r} appears in multiple surface groups: {previous!r}, {group!r}")
            seen_surfaces[surface] = group
        normalized_groups[group] = surfaces

    if not isinstance(semantic_visual_claim_types, Sequence) or isinstance(
        semantic_visual_claim_types,
        (str, bytes),
    ):
        raise ValueError("semantic_visual_claim_types must be a sequence")
    claim_types = [str(category) for category in semantic_visual_claim_types]
    if not claim_types or any(not category for category in claim_types):
        raise ValueError("semantic_visual_claim_types must contain non-empty values")
    if len(claim_types) != len(set(claim_types)):
        raise ValueError("semantic_visual_claim_types must not contain duplicates")
    unknown = set(claim_types) - set(SEMANTIC_VISUAL_CLAIM_TYPES)
    if unknown:
        raise ValueError("semantic_visual_claim_types contains unsupported values: " + ", ".join(sorted(unknown)))
    expected_strata = sorted(f"{group}/{category}" for group in normalized_groups for category in claim_types)
    return claim_types, expected_strata


def join_candidate_pool(
    claims: Mapping[str, Mapping[str, Any]],
    judge_records: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    surface_groups: Mapping[str, Sequence[str]],
    required_judges: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join extractor units, judge labels, and image provenance."""

    candidates: list[dict[str, Any]] = []
    missing_by_judge: Counter[str] = Counter()
    rejected_surface = 0
    locator_mismatches = 0
    for question_id, claim in claims.items():
        surface = str(claim["surface"])
        surface_group = _surface_to_group(surface, surface_groups)
        if surface_group is None:
            rejected_surface += 1
            continue
        per_judge: dict[str, Mapping[str, Any]] = {}
        missing = False
        for judge in required_judges:
            record = judge_records.get(judge, {}).get(question_id)
            if record is None:
                missing_by_judge[judge] += 1
                missing = True
                continue
            per_judge[judge] = record
        if missing:
            continue
        locator = per_judge[required_judges[0]]
        image_key, identity_basis = stable_image_key(locator)
        for judge in required_judges[1:]:
            other_key, _ = stable_image_key(per_judge[judge])
            if other_key != image_key:
                locator_mismatches += 1
                missing = True
                break
        if missing:
            continue
        candidates.append(
            {
                **dict(claim),
                "surface_group": surface_group,
                "stratum": f"{surface_group}/{claim['category']}",
                "image_key": image_key,
                "image_identity_basis": identity_basis,
                "image_source_path": locator.get("image_path"),
                "image_url": canonicalize_url(locator.get("image_url")),
                "image_sha256_expected": locator.get("image_sha256"),
                "pair_id": locator.get("pair_id"),
                "pair_key": locator.get("pair_key"),
                "public_lookup_key": locator.get("public_lookup_key"),
                "family": locator.get("family"),
                "wds_locator": parse_historical_wds_path(locator.get("image_path")),
                "judge_answers": {
                    judge: {
                        "answer": per_judge[judge]["answer"],
                        "confidence": per_judge[judge]["confidence"],
                        "model": per_judge[judge]["model"],
                    }
                    for judge in required_judges
                },
            }
        )
    report = {
        "joined_candidates": len(candidates),
        "rejected_surfaces": rejected_surface,
        "missing_by_judge": dict(sorted(missing_by_judge.items())),
        "image_locator_mismatches_rejected": locator_mismatches,
        "by_stratum": dict(sorted(Counter(item["stratum"] for item in candidates).items())),
        "unique_images": len({item["image_key"] for item in candidates}),
    }
    return candidates, report


def _deterministic_item_id(study_namespace: str, question_id: str, repeat_index: int = 0) -> str:
    payload = f"{study_namespace}\0{question_id}\0{repeat_index}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def sample_strata(
    candidates: Sequence[Mapping[str, Any]],
    *,
    claims_per_stratum: int,
    seed: int,
    study_namespace: str,
    expected_strata: Sequence[str],
    repeat_fraction: float = 0.0,
    require_full_strata: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Uniformly sample claims within each surface-group/category stratum."""

    if claims_per_stratum <= 0:
        raise ValueError("claims_per_stratum must be positive")
    if not 0.0 <= repeat_fraction < 1.0:
        raise ValueError("repeat_fraction must be in [0, 1)")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["stratum"])].append(candidate)
    if not grouped:
        raise ValueError("candidate pool is empty")
    expected = set(expected_strata)
    observed = set(grouped)
    if not expected:
        raise ValueError("expected_strata must not be empty")
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise ValueError("candidate strata do not match the configured design: " + "; ".join(details))

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    shortfalls: dict[str, dict[str, int]] = {}
    sampling_rows: list[dict[str, Any]] = []
    for stratum in sorted(grouped):
        pool = sorted(grouped[stratum], key=lambda item: str(item["question_id"]))
        rng.shuffle(pool)
        sample_size = min(claims_per_stratum, len(pool))
        if sample_size < claims_per_stratum:
            shortfalls[stratum] = {"available": len(pool), "requested": claims_per_stratum}
        selected = pool[:sample_size]
        weight = len(pool) / sample_size
        sampling_rows.append(
            {
                "stratum": stratum,
                "population_claims": len(pool),
                "sampled_claims": sample_size,
                "sampling_weight": weight,
            }
        )
        for candidate in selected:
            item = dict(candidate)
            item.update(
                {
                    "item_id": _deterministic_item_id(study_namespace, str(candidate["question_id"])),
                    "repeat_of": None,
                    "is_repeat": False,
                    "stratum_population": len(pool),
                    "stratum_sample_size": sample_size,
                    "sampling_weight": weight,
                    "asset_status": "missing",
                    "asset_relpath": None,
                }
            )
            sampled.append(item)
    if shortfalls and require_full_strata:
        details = ", ".join(
            f"{stratum}: {values['available']}/{values['requested']}" for stratum, values in shortfalls.items()
        )
        raise ValueError(f"insufficient candidates for requested strata: {details}")

    base_count = len(sampled)
    repeat_count = round(base_count * repeat_fraction)
    repeat_source = list(sampled)
    rng.shuffle(repeat_source)
    for repeat_index, source in enumerate(repeat_source[:repeat_count], start=1):
        repeat = dict(source)
        repeat["item_id"] = _deterministic_item_id(
            study_namespace,
            str(source["question_id"]),
            repeat_index=repeat_index,
        )
        repeat["repeat_of"] = source["item_id"]
        repeat["is_repeat"] = True
        repeat["sampling_weight"] = 0.0
        sampled.append(repeat)
    rng.shuffle(sampled)
    report = {
        "base_items": base_count,
        "repeat_items": repeat_count,
        "total_items": len(sampled),
        "unique_images": len({item["image_key"] for item in sampled}),
        "shortfalls": shortfalls,
        "expected_strata": sorted(expected),
        "strata": sampling_rows,
        "sample_by_surface": dict(
            sorted(Counter(item["surface"] for item in sampled if not item["is_repeat"]).items())
        ),
        "sample_by_category": dict(
            sorted(Counter(item["category"] for item in sampled if not item["is_repeat"]).items())
        ),
    }
    return sampled, report


def build_sample(
    *,
    claim_response_paths: Sequence[str | Path],
    judge_response_paths: Mapping[str, Sequence[str | Path]],
    surface_groups: Mapping[str, Sequence[str]],
    semantic_visual_claim_types: Sequence[str],
    token_budget: int,
    claims_per_stratum: int,
    seed: int,
    study_namespace: str,
    repeat_fraction: float = 0.05,
    required_judges: Sequence[str] = ("qwen", "gemma"),
    require_full_strata: bool = True,
    fingerprint_inputs: bool = True,
) -> BuildResult:
    """Build the complete human-evaluation sample and provenance report."""

    if set(required_judges) - set(judge_response_paths):
        missing = sorted(set(required_judges) - set(judge_response_paths))
        raise ValueError(f"missing judge response paths for {missing}")
    claim_types, expected_strata = validate_study_design(
        surface_groups,
        semantic_visual_claim_types,
    )
    claims, claim_report = load_claim_records(
        claim_response_paths,
        token_budget=token_budget,
        allowed_categories=claim_types,
    )
    judges: dict[str, dict[str, dict[str, Any]]] = {}
    judge_reports: dict[str, dict[str, Any]] = {}
    for judge in required_judges:
        records, report = load_judge_records(
            judge_response_paths[judge],
            token_budget=token_budget,
            judge_name=judge,
        )
        judges[judge] = records
        judge_reports[judge] = report
    candidates, join_report = join_candidate_pool(
        claims,
        judges,
        surface_groups=surface_groups,
        required_judges=required_judges,
    )
    items, sample_report = sample_strata(
        candidates,
        claims_per_stratum=claims_per_stratum,
        seed=seed,
        study_namespace=study_namespace,
        expected_strata=expected_strata,
        repeat_fraction=repeat_fraction,
        require_full_strata=require_full_strata,
    )

    paths = [Path(path) for path in claim_response_paths]
    for judge in required_judges:
        paths.extend(Path(path) for path in judge_response_paths[judge])
    report: dict[str, Any] = {
        "protocol": "human_cbu_v1",
        "study_namespace": study_namespace,
        "seed": seed,
        "token_budget": token_budget,
        "claims_per_stratum": claims_per_stratum,
        "repeat_fraction": repeat_fraction,
        "surface_groups": {key: list(value) for key, value in surface_groups.items()},
        "semantic_visual_claim_types": claim_types,
        "expected_strata": expected_strata,
        "required_judges": list(required_judges),
        "claim_load": claim_report,
        "judge_load": judge_reports,
        "join": join_report,
        "sample": sample_report,
    }
    if fingerprint_inputs:
        report["input_artifacts"] = artifact_fingerprints(paths)
    return BuildResult(items=items, report=report)
