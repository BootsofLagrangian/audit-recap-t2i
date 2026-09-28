"""Privacy-preserving SQLite persistence for blinded human CBU evaluation.

The store deliberately keeps the participant-facing projection much smaller
than the private item record.  In particular, caption tasks never expose image
or judge fields, and image tasks never expose the caption, surface, or judges.

Only pseudonymous participant identifiers are collected.  Invite codes and
bearer session tokens are generated with high entropy, returned once, and
stored only as domain-separated SHA-256 digests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import stat
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from .builder import SEMANTIC_VISUAL_CLAIM_TYPES

SCHEMA_VERSION = 3
DEFAULT_BUSY_TIMEOUT_MS = 5_000
DEFAULT_SESSION_TTL_SECONDS = 12 * 60 * 60
MAX_SESSION_TTL_SECONDS = 24 * 60 * 60
HUMAN_CBU_V1_PROTOCOL = "human_cbu_v1"
SESSION_SCOPE_FULL = "full"
SESSION_SCOPE_WITHDRAWAL_ONLY = "withdrawal_only"
ADJUDICATION_PACKET_VERSION = "human_cbu_adjudication_packet_v2"
ADJUDICATION_RESPONSE_VERSION = "human_cbu_adjudication_response_v2"
ADJUDICATION_REVIEW_VERSION = "human_cbu_adjudication_review_v1"
# Kept only so the retired shared-bundle parser below remains import-safe.
# human_cbu_v1 rejects that path before reading any artifact.
ADJUDICATION_BUNDLE_VERSION = "human_cbu_adjudication_bundle_v1_retired"
_ADJUDICATION_PACKET_ID_DOMAIN = "human-cbu:adjudication-packet-id:v2"
_ADJUDICATION_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_MAX_ADJUDICATION_PACKET_FILE_BYTES = 32 * 1024 * 1024
_MAX_ADJUDICATION_RESPONSE_BYTES = 1024 * 1024
_MAX_ADJUDICATION_REVIEW_BYTES = 4 * 1024 * 1024
_MAX_ADJUDICATION_IMAGE_BYTES = 256 * 1024 * 1024

STUDY_STATUSES = frozenset({"draft", "ready", "open", "paused", "closed", "archived"})
ASSET_STATUSES = frozenset({"pending", "fetching", "available", "unavailable", "invalid"})
PHASES = ("caption", "image")
ASSIGNMENT_ROSTER_METADATA_KEY = "assignment_roster"
PARTICIPANT_REPLACEMENTS_METADATA_KEY = "participant_replacements"
ASSIGNMENT_MODE_METADATA_KEY = "assignment_mode"
ASSIGNMENT_MODE_FIXED = "fixed_preassigned"
ASSIGNMENT_MODE_REMAINING_FIRST = "remaining_first_late_binding"
TARGET_ITEMS_PER_PARTICIPANT_METADATA_KEY = "target_items_per_participant"
OPTIONAL_EXTENSION_BATCH_SIZE_METADATA_KEY = "optional_extension_batch_size"
OPTIONAL_EXTENSION_MAX_ITEMS_METADATA_KEY = "optional_extension_max_items"
OVERLAP_TARGET_PER_STRATUM_METADATA_KEY = "overlap_target_items_per_stratum"
SELF_ENROLLMENT_LIMIT_METADATA_KEY = "self_enrollment_limit"
_DYNAMIC_REPEAT_MIN_GAP = 10
PARTICIPANT_INFORMATION_FIELDS = frozenset(
    {
        "approved_version",
        "duration",
        "compensation",
        "data_retention",
        "research_contact",
        "withdrawal_policy",
        "information_sheet_url",
    }
)
_REVIEW_NOT_REQUIRED_PREFIX = re.compile(
    r"^review[\s_-]*not[\s_-]*required(.*)$",
    flags=re.IGNORECASE,
)

CAPTION_LICENSED_LABELS = frozenset({"yes", "no", "uncertain"})
ATOMIC_VISUAL_CLAIM_LABELS = frozenset({"yes", "not_visual", "not_atomic", "uncertain"})
CATEGORY_CHECK_LABELS = frozenset({"correct", "incorrect", "uncertain", "not_applicable"})
IMAGE_SUPPORT_LABELS = frozenset(
    {
        "yes",
        "no",
        "uncertain",
        "not_visual",
        "image_unavailable",
        "prefer_not_to_answer",
    }
)


def ethics_review_basis_error(value: Any) -> str | None:
    """Return a launch-gate error for an absent or under-specified basis."""

    if (
        not isinstance(value, str)
        or not value.strip()
        or "\n" in value
        or "\r" in value
    ):
        return "ethics-review basis must be a nonempty single-line string"
    match = _REVIEW_NOT_REQUIRED_PREFIX.fullmatch(value.strip())
    if match is None:
        return None
    remainder = match.group(1).strip()
    if not remainder.startswith(":") or not remainder.removeprefix(":").strip():
        return (
            "review-not-required must include a non-personal authority or "
            "basis after ':'"
        )
    return None

CAPTION_REQUIRED_FIELDS = ("caption_licensed", "atomic_visual_claim", "category_check")
IMAGE_REQUIRED_FIELDS = ("image_support",)
COMMON_OPTIONAL_FIELDS = frozenset({"reason_tags", "note", "confidence", "elapsed_ms"})
CAPTION_OPTIONAL_FIELDS = COMMON_OPTIONAL_FIELDS | {"corrected_category"}
IMAGE_OPTIONAL_FIELDS = COMMON_OPTIONAL_FIELDS | {
    "control_usefulness",
    "salient_coverage",
}

# Intentionally coarse, categorical profile fields.  The API rejects all other
# keys rather than accepting arbitrary demographic or identifying information.
ALLOWED_PROFILE_FIELDS = frozenset(
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
PROFILE_VALUE_VOCABULARIES = {
    "author_status": frozenset({"author", "non_author"}),
    "recruitment_source": frozenset(
        {
            "lab_non_author",
            "academic_pool",
            "professional_platform",
            "other_coarse",
        }
    ),
    "age_band": frozenset(
        {
            "18-24",
            "25-34",
            "35-44",
            "45-54",
            "55-64",
            "65+",
            "prefer_not_to_say",
        }
    ),
    "primary_language_group": frozenset({"english", "korean", "multilingual", "other", "prefer_not_to_say"}),
    "english_proficiency": frozenset(
        {
            "native_or_bilingual",
            "fluent",
            "intermediate",
            "basic",
            "prefer_not_to_say",
        }
    ),
    "vision_status": frozenset({"normal_or_corrected", "low_vision", "blind", "other", "prefer_not_to_say"}),
    "color_vision": frozenset({"typical", "color_vision_deficiency", "unsure", "prefer_not_to_say"}),
    "t2i_experience": frozenset({"none", "some", "frequent", "professional", "prefer_not_to_say"}),
    "image_annotation_experience": frozenset({"none", "some", "frequent", "professional", "prefer_not_to_say"}),
    "domain_experience": frozenset({"none", "some", "expert", "prefer_not_to_say"}),
    "device_class": frozenset({"desktop", "laptop", "tablet", "mobile", "other"}),
    "screen_size_band": frozenset({"small", "medium", "large", "unknown"}),
}

_IDENTIFIER_RE = re.compile(r"^[^\x00\r\n]{1,512}$")
_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")
_PSEUDONYM_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PHONE_RE = re.compile(r"(?:\+?\d[\s().-]*){7,}")
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")


class HumanCBUStoreError(RuntimeError):
    """Base exception for the human CBU persistence layer."""


class ValidationError(HumanCBUStoreError, ValueError):
    """An input or annotation payload failed validation."""


class NotFoundError(HumanCBUStoreError, LookupError):
    """A requested study, item, participant, or assignment does not exist."""


class AuthenticationError(HumanCBUStoreError):
    """An invite code or bearer session is invalid or expired."""


class AuthorizationError(HumanCBUStoreError):
    """An authenticated participant cannot perform the requested action."""


class StudyStateError(HumanCBUStoreError):
    """The study lifecycle does not permit the requested operation."""


class ConflictError(HumanCBUStoreError):
    """The requested write conflicts with existing persisted state."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValidationError("value must be JSON serializable") from error


def _json_loads(value: str | None, default: Any = None) -> Any:
    if value is None:
        return default
    return json.loads(value)


def _require_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValidationError(f"{field} must be a non-empty string without control newlines")
    return value


def _require_nonempty_text(value: Any, field: str, *, maximum: int = 100_000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be a non-empty string")
    if len(value) > maximum:
        raise ValidationError(f"{field} exceeds the {maximum}-character limit")
    return value


def _human_cbu_v1_label_plan_error(
    *,
    protocol_version: str,
    metadata: Mapping[str, Any],
    labels_per_item: Any,
) -> str | None:
    if protocol_version != HUMAN_CBU_V1_PROTOCOL:
        return None
    required_labels = metadata.get("required_labels_per_item")
    if (
        type(required_labels) is not int
        or required_labels != 2
        or type(labels_per_item) is not int
        or labels_per_item != 2
    ):
        return "human_cbu_v1 requires exactly 2 initial labels per item"
    return None


_REPEAT_SAMPLE_INSTANCE_FIELDS = frozenset(
    {
        "is_repeat",
        "repeat_index",
        "position",
        "order",
        "order_key",
    }
)
_MISSING_REPEAT_VALUE = object()


def _frozen_semantic_visual_claim_types(
    metadata: Mapping[str, Any],
) -> tuple[str, ...]:
    values = metadata.get("semantic_visual_claim_types")
    supported = set(SEMANTIC_VISUAL_CLAIM_TYPES)
    if (
        not isinstance(values, list)
        or not values
        or any(type(value) is not str or value not in supported for value in values)
        or len(values) != len(set(values))
    ):
        raise ValidationError(
            "human_cbu_v1 must freeze a non-empty unique vocabulary of supported semantic_visual_claim_types"
        )
    return tuple(values)


def _stored_item_json(row: sqlite3.Row, column: str) -> Any:
    value = row[column]
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValidationError(f"item {row['item_id']!r} has malformed stored JSON in {column}") from error


def _repeat_payload_projection(row: sqlite3.Row) -> dict[str, Any]:
    sample = _stored_item_json(row, "sample_json")
    if isinstance(sample, Mapping):
        sample = {key: value for key, value in sample.items() if key not in _REPEAT_SAMPLE_INSTANCE_FIELDS}
    return {
        "caption": row["caption"],
        "unit": row["unit"],
        "span_text": row["span_text"],
        "span_start": row["span_start"],
        "span_end": row["span_end"],
        "target": row["target"],
        "category": row["category"],
        "surface": row["surface"],
        "item_group": row["item_group"],
        "image_locator": _stored_item_json(row, "image_locator_json"),
        "image_asset": _stored_item_json(row, "image_asset_json"),
        "asset_status": row["asset_status"],
        "asset_status_reason": row["asset_status_reason"],
        "qwen_answer": _stored_item_json(row, "qwen_answer_json"),
        "gemma_answer": _stored_item_json(row, "gemma_answer_json"),
        "stratum": row["stratum"],
        "population": _stored_item_json(row, "population_json"),
        "sample": sample,
        "population_size": row["population_size"],
    }


def _first_repeat_payload_difference(
    repeat_value: Any,
    original_value: Any,
    *,
    path: str = "",
) -> str | None:
    if isinstance(repeat_value, Mapping) and isinstance(original_value, Mapping):
        for key in sorted(set(repeat_value) | set(original_value)):
            difference = _first_repeat_payload_difference(
                repeat_value.get(key, _MISSING_REPEAT_VALUE),
                original_value.get(key, _MISSING_REPEAT_VALUE),
                path=f"{path}.{key}" if path else str(key),
            )
            if difference is not None:
                return difference
        return None
    if (
        isinstance(repeat_value, Sequence)
        and not isinstance(repeat_value, (str, bytes))
        and isinstance(original_value, Sequence)
        and not isinstance(original_value, (str, bytes))
    ):
        if len(repeat_value) != len(original_value):
            return path
        for index, (repeat_entry, original_entry) in enumerate(zip(repeat_value, original_value, strict=True)):
            difference = _first_repeat_payload_difference(
                repeat_entry,
                original_entry,
                path=f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    return path if repeat_value != original_value else None


def _human_cbu_v1_frozen_design_report(
    connection: sqlite3.Connection,
    study: sqlite3.Row,
) -> dict[str, Any]:
    if study["protocol_version"] != HUMAN_CBU_V1_PROTOCOL:
        return {
            "errors": [],
            "expected_repeat_items": None,
            "repeat_items": None,
            "repeat_payload_failures": [],
            "semantic_visual_claim_types": None,
        }
    metadata = _json_loads(study["metadata_json"], {})
    errors: list[str] = []
    semantic_types: tuple[str, ...] = ()
    if not isinstance(metadata, Mapping):
        errors.append("human_cbu_v1 study metadata is malformed")
    else:
        try:
            semantic_types = _frozen_semantic_visual_claim_types(metadata)
        except ValidationError as error:
            errors.append(str(error))
    expected_repeat_items = metadata.get("expected_repeat_items") if isinstance(metadata, Mapping) else None
    if type(expected_repeat_items) is not int or expected_repeat_items < 0:
        errors.append("human_cbu_v1 must freeze expected_repeat_items as a non-negative integer")

    rows = connection.execute(
        "SELECT * FROM items WHERE study_id = ? ORDER BY item_id",
        (study["study_id"],),
    ).fetchall()
    by_id = {row["item_id"]: row for row in rows}
    repeats = [row for row in rows if row["repeat_of"] is not None]
    if type(expected_repeat_items) is int and expected_repeat_items >= 0 and len(repeats) != expected_repeat_items:
        errors.append(
            "human_cbu_v1 hidden-repeat count does not match frozen "
            f"expected_repeat_items ({len(repeats)}/{expected_repeat_items})"
        )
    if semantic_types:
        allowed_types = set(semantic_types)
        invalid_categories = sorted({row["category"] for row in rows if row["category"] not in allowed_types})
        if invalid_categories:
            errors.append(
                "one or more item categories are outside frozen "
                "semantic_visual_claim_types: " + ", ".join(invalid_categories)
            )

    repeat_payload_failures: list[dict[str, Any]] = []
    for repeat in repeats:
        repeat_id = repeat["item_id"]
        original = by_id.get(repeat["repeat_of"])
        if original is None:
            repeat_payload_failures.append(
                {
                    "item_id": repeat_id,
                    "repeat_of": repeat["repeat_of"],
                    "field": "repeat_of",
                }
            )
            continue
        if original["repeat_of"] is not None:
            repeat_payload_failures.append(
                {
                    "item_id": repeat_id,
                    "repeat_of": repeat["repeat_of"],
                    "field": "repeat_of.repeat_of",
                }
            )
        if repeat["weight"] != 0.0:
            repeat_payload_failures.append(
                {
                    "item_id": repeat_id,
                    "repeat_of": repeat["repeat_of"],
                    "field": "weight",
                }
            )
        if repeat["sample_probability"] is not None:
            repeat_payload_failures.append(
                {
                    "item_id": repeat_id,
                    "repeat_of": repeat["repeat_of"],
                    "field": "sample_probability",
                }
            )
        try:
            difference = _first_repeat_payload_difference(
                _repeat_payload_projection(repeat),
                _repeat_payload_projection(original),
            )
        except ValidationError as error:
            errors.append(str(error))
        else:
            if difference is not None:
                repeat_payload_failures.append(
                    {
                        "item_id": repeat_id,
                        "repeat_of": repeat["repeat_of"],
                        "field": difference,
                    }
                )
    if repeat_payload_failures:
        errors.append(
            "one or more human_cbu_v1 hidden repeats differ from their frozen "
            "base payload or non-sampling repeat weight/probability"
        )
    return {
        "errors": errors,
        "expected_repeat_items": expected_repeat_items,
        "repeat_items": len(repeats),
        "repeat_payload_failures": repeat_payload_failures,
        "semantic_visual_claim_types": list(semantic_types),
    }


def _validate_human_cbu_v1_caption_correction(
    *,
    protocol_version: str,
    metadata_json: str,
    proposed_category: str,
    normalized: Mapping[str, Any],
) -> None:
    if protocol_version != HUMAN_CBU_V1_PROTOCOL or normalized["category_check"] != "incorrect":
        return
    metadata = _json_loads(metadata_json, {})
    if not isinstance(metadata, Mapping):
        raise ValidationError("human_cbu_v1 study metadata is malformed")
    allowed = set(_frozen_semantic_visual_claim_types(metadata))
    corrected = normalized.get("corrected_category")
    if corrected not in allowed:
        raise ValidationError("corrected_category must be one of the frozen semantic visual-claim types")
    if corrected == proposed_category:
        raise ValidationError("corrected_category must differ from the proposed category")


def _validate_launch_metadata(study: sqlite3.Row) -> None:
    """Fail closed before collection opens, including through the store API."""

    metadata = _json_loads(study["metadata_json"], {})
    if not isinstance(metadata, Mapping):
        raise StudyStateError("study launch metadata is malformed")
    label_plan_error = _human_cbu_v1_label_plan_error(
        protocol_version=study["protocol_version"],
        metadata=metadata,
        labels_per_item=study["labels_per_item"],
    )
    if label_plan_error is not None:
        raise StudyStateError(label_plan_error)
    determination = metadata.get("ethics_or_irb_equivalent_determination_id")
    determination_error = ethics_review_basis_error(determination)
    if determination_error is not None:
        raise StudyStateError(
            "study cannot open without an ethics-review basis: use "
            "'review-not-required:<authority-or-basis>' when review is not "
            "required, or record the institutional determination identifier; "
            f"{determination_error}"
        )
    information = metadata.get("participant_information")
    if not isinstance(information, Mapping) or set(information) != PARTICIPANT_INFORMATION_FIELDS:
        raise StudyStateError("study cannot open without the complete approved participant information")
    for field in PARTICIPANT_INFORMATION_FIELDS:
        value = information[field]
        if (
            not isinstance(value, str)
            or not value.strip()
            or any(ord(character) < 32 and character not in "\n\t" for character in value)
        ):
            raise StudyStateError(f"approved participant information field {field!r} is invalid")
    if information["approved_version"] != study["consent_version"]:
        raise StudyStateError("approved participant information does not match the study consent version")
    information_url = information["information_sheet_url"]
    if information_url != "/study-information":
        parsed = urlsplit(information_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username is not None or parsed.password is not None:
            raise StudyStateError(
                "participant information must use '/study-information' or a credential-free HTTPS URL"
            )


def _token_digest(raw_token: str, *, domain: str) -> str:
    if not isinstance(raw_token, str) or not raw_token:
        raise AuthenticationError("credential is missing")
    return hashlib.sha256(f"human-cbu:{domain}:".encode() + raw_token.encode()).hexdigest()


def _stable_digest(*parts: str) -> str:
    encoded = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json_bytes(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"{label} must be strict UTF-8 JSON") from error


def _file_signature(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


@contextmanager
def _open_stable_private_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> Iterable[dict[str, Any]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValidationError(f"{label} must be an existing non-symlink file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValidationError(f"{label} must be a private regular non-hardlinked file")
        if before.st_mode & 0o077:
            raise ValidationError(f"{label} must not grant group or world permissions")
        if before.st_size > maximum_bytes:
            raise ValidationError(f"{label} exceeds its maximum allowed size")
        raw_parts: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, min(8 * 1024 * 1024, maximum_bytes + 1 - total)):
            total += len(chunk)
            if total > maximum_bytes:
                raise ValidationError(f"{label} exceeds its maximum allowed size")
            raw_parts.append(chunk)
        raw = b"".join(raw_parts)
        after = os.fstat(descriptor)
        signature = _file_signature(after)
        if signature != _file_signature(before):
            raise ValidationError(f"{label} changed while it was being read")
        record = {
            "descriptor": descriptor,
            "path": path,
            "raw": raw,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "signature": signature,
        }
        yield record
    finally:
        os.close(descriptor)


def _assert_open_file_unchanged(
    record: Mapping[str, Any],
    *,
    label: str,
    rehash: bool = False,
) -> None:
    descriptor = int(record["descriptor"])
    expected = record["signature"]
    if _file_signature(os.fstat(descriptor)) != expected:
        raise ValidationError(f"{label} changed before adjudication was recorded")
    if rehash:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
        if _file_signature(os.fstat(descriptor)) != expected:
            raise ValidationError(f"{label} changed while it was re-verified")
        if not secrets.compare_digest(digest.hexdigest(), str(record["sha256"])):
            raise ValidationError(f"{label} bytes changed before adjudication was recorded")
    try:
        current = os.stat(record["path"], follow_symlinks=False)
    except OSError as error:
        raise ValidationError(f"{label} was replaced before adjudication was recorded") from error
    if not stat.S_ISREG(current.st_mode) or _file_signature(current) != expected:
        raise ValidationError(f"{label} was replaced before adjudication was recorded")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS studies (
    study_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    protocol_version TEXT NOT NULL,
    consent_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('draft', 'ready', 'open', 'paused', 'closed', 'archived')
    ),
    metadata_json TEXT NOT NULL,
    assignment_seed TEXT,
    assignment_participant_digest TEXT,
    adjudicator_credential_digest TEXT CHECK (
        adjudicator_credential_digest IS NULL
        OR length(adjudicator_credential_digest) = 64
    ),
    labels_per_item INTEGER CHECK (labels_per_item IS NULL OR labels_per_item > 0),
    validation_sealed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS items (
    study_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    caption TEXT NOT NULL,
    unit TEXT NOT NULL,
    span_text TEXT,
    span_start INTEGER,
    span_end INTEGER,
    target TEXT,
    category TEXT NOT NULL,
    surface TEXT NOT NULL,
    item_group TEXT NOT NULL,
    image_locator_json TEXT NOT NULL,
    image_asset_json TEXT,
    asset_status TEXT NOT NULL CHECK (
        asset_status IN ('pending', 'fetching', 'available', 'unavailable', 'invalid')
    ),
    asset_status_reason TEXT,
    qwen_answer_json TEXT,
    gemma_answer_json TEXT,
    stratum TEXT,
    population_json TEXT,
    sample_json TEXT,
    weight REAL CHECK (weight IS NULL OR weight >= 0),
    population_size INTEGER CHECK (population_size IS NULL OR population_size > 0),
    sample_probability REAL CHECK (
        sample_probability IS NULL OR (sample_probability > 0 AND sample_probability <= 1)
    ),
    repeat_of TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (study_id, item_id),
    FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE RESTRICT,
    FOREIGN KEY (study_id, repeat_of) REFERENCES items(study_id, item_id)
        DEFERRABLE INITIALLY DEFERRED,
    CHECK (
        (span_start IS NULL AND span_end IS NULL)
        OR (span_start IS NOT NULL AND span_end IS NOT NULL
            AND span_start >= 0 AND span_end >= span_start)
    )
) STRICT;

CREATE TABLE IF NOT EXISTS participants (
    participant_id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL,
    pseudonym TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('invited', 'active', 'declined', 'withdrawn')),
    consented INTEGER NOT NULL DEFAULT 0 CHECK (consented IN (0, 1)),
    consent_version TEXT,
    consented_at TEXT,
    profile_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (study_id, pseudonym),
    UNIQUE (study_id, participant_id),
    FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE RESTRICT
) STRICT;

CREATE TABLE IF NOT EXISTS participant_invites (
    invite_id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL,
    participant_id TEXT NOT NULL,
    invite_digest TEXT NOT NULL UNIQUE CHECK (length(invite_digest) = 64),
    expires_at TEXT,
    revoked_at TEXT,
    last_used_at TEXT,
    use_count INTEGER NOT NULL DEFAULT 0 CHECK (use_count >= 0),
    created_at TEXT NOT NULL,
    FOREIGN KEY (study_id, participant_id)
        REFERENCES participants(study_id, participant_id) ON DELETE RESTRICT
) STRICT;

CREATE TABLE IF NOT EXISTS bearer_sessions (
    session_id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL,
    participant_id TEXT NOT NULL,
    token_digest TEXT NOT NULL UNIQUE CHECK (length(token_digest) = 64),
    scope TEXT NOT NULL DEFAULT 'withdrawal_only' CHECK (
        scope IN ('full', 'withdrawal_only')
    ),
    expires_at TEXT NOT NULL,
    revoked_at TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (study_id, participant_id)
        REFERENCES participants(study_id, participant_id) ON DELETE RESTRICT
) STRICT;

CREATE TABLE IF NOT EXISTS consent_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    study_id TEXT NOT NULL,
    participant_id TEXT NOT NULL,
    consented INTEGER NOT NULL CHECK (consented IN (0, 1)),
    consent_version TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (study_id, participant_id)
        REFERENCES participants(study_id, participant_id) ON DELETE RESTRICT
) STRICT;

CREATE TRIGGER IF NOT EXISTS consent_events_no_update
BEFORE UPDATE ON consent_events
BEGIN
    SELECT RAISE(ABORT, 'consent_events are append-only');
END;

CREATE TRIGGER IF NOT EXISTS consent_events_no_delete
BEFORE DELETE ON consent_events
BEGIN
    SELECT RAISE(ABORT, 'consent_events are append-only');
END;

CREATE TABLE IF NOT EXISTS assignments (
    assignment_id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL,
    participant_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    phase TEXT NOT NULL CHECK (phase IN ('caption', 'image')),
    position INTEGER NOT NULL CHECK (position > 0),
    order_key TEXT NOT NULL CHECK (length(order_key) = 64),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed')),
    assigned_at TEXT NOT NULL,
    first_presented_at TEXT,
    last_presented_at TEXT,
    presentation_count INTEGER NOT NULL DEFAULT 0 CHECK (presentation_count >= 0),
    completed_at TEXT,
    UNIQUE (study_id, participant_id, item_id, phase),
    UNIQUE (assignment_id, study_id, participant_id, item_id, phase),
    FOREIGN KEY (study_id, participant_id)
        REFERENCES participants(study_id, participant_id) ON DELETE RESTRICT,
    FOREIGN KEY (study_id, item_id)
        REFERENCES items(study_id, item_id) ON DELETE RESTRICT
) STRICT;

CREATE TABLE IF NOT EXISTS annotations (
    annotation_id TEXT PRIMARY KEY,
    assignment_id TEXT NOT NULL UNIQUE,
    study_id TEXT NOT NULL,
    participant_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    phase TEXT NOT NULL CHECK (phase IN ('caption', 'image')),
    payload_json TEXT NOT NULL,
    caption_licensed TEXT,
    atomic_visual_claim TEXT,
    category_check TEXT,
    corrected_category TEXT,
    image_support TEXT,
    control_usefulness TEXT,
    reason_tags_json TEXT NOT NULL,
    note TEXT,
    confidence INTEGER CHECK (confidence IS NULL OR (confidence >= 1 AND confidence <= 5)),
    elapsed_ms INTEGER CHECK (elapsed_ms IS NULL OR (elapsed_ms >= 0 AND elapsed_ms <= 3600000)),
    revision INTEGER NOT NULL CHECK (revision > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (annotation_id, assignment_id, study_id, participant_id, item_id, phase),
    FOREIGN KEY (assignment_id) REFERENCES assignments(assignment_id) ON DELETE RESTRICT,
    FOREIGN KEY (assignment_id, study_id, participant_id, item_id, phase)
        REFERENCES assignments(assignment_id, study_id, participant_id, item_id, phase)
        ON DELETE RESTRICT,
    FOREIGN KEY (study_id, participant_id)
        REFERENCES participants(study_id, participant_id) ON DELETE RESTRICT,
    FOREIGN KEY (study_id, item_id)
        REFERENCES items(study_id, item_id) ON DELETE RESTRICT,
    CHECK (
        (phase = 'caption' AND caption_licensed IS NOT NULL
            AND atomic_visual_claim IS NOT NULL AND category_check IS NOT NULL
            AND image_support IS NULL AND control_usefulness IS NULL)
        OR
        (phase = 'image' AND image_support IS NOT NULL
            AND caption_licensed IS NULL AND atomic_visual_claim IS NULL
            AND category_check IS NULL AND corrected_category IS NULL)
    )
) STRICT;

CREATE TABLE IF NOT EXISTS annotation_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    annotation_id TEXT NOT NULL,
    assignment_id TEXT NOT NULL,
    study_id TEXT NOT NULL,
    participant_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    phase TEXT NOT NULL CHECK (phase IN ('caption', 'image')),
    payload_json TEXT NOT NULL,
    caption_licensed TEXT,
    atomic_visual_claim TEXT,
    category_check TEXT,
    corrected_category TEXT,
    image_support TEXT,
    control_usefulness TEXT,
    reason_tags_json TEXT NOT NULL,
    note TEXT,
    confidence INTEGER,
    elapsed_ms INTEGER CHECK (elapsed_ms IS NULL OR (elapsed_ms >= 0 AND elapsed_ms <= 3600000)),
    revision INTEGER NOT NULL CHECK (revision > 0),
    created_at TEXT NOT NULL,
    UNIQUE (annotation_id, revision),
    FOREIGN KEY (annotation_id) REFERENCES annotations(annotation_id) ON DELETE RESTRICT,
    FOREIGN KEY (assignment_id) REFERENCES assignments(assignment_id) ON DELETE RESTRICT,
    FOREIGN KEY (annotation_id, assignment_id, study_id, participant_id, item_id, phase)
        REFERENCES annotations(
            annotation_id, assignment_id, study_id, participant_id, item_id, phase
        ) ON DELETE RESTRICT
) STRICT;

CREATE TRIGGER IF NOT EXISTS annotation_events_no_update
BEFORE UPDATE ON annotation_events
BEGIN
    SELECT RAISE(ABORT, 'annotation_events are append-only');
END;

CREATE TRIGGER IF NOT EXISTS annotation_events_no_delete
BEFORE DELETE ON annotation_events
BEGIN
    SELECT RAISE(ABORT, 'annotation_events are append-only');
END;

CREATE TABLE IF NOT EXISTS adjudications (
    adjudication_id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    phase TEXT NOT NULL CHECK (phase IN ('caption', 'image')),
    adjudicator_pseudonym TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    packet_version TEXT,
    packet_id TEXT,
    packet_sha256 TEXT CHECK (packet_sha256 IS NULL OR length(packet_sha256) = 64),
    packet_file_sha256 TEXT CHECK (
        packet_file_sha256 IS NULL OR length(packet_file_sha256) = 64
    ),
    response_version TEXT,
    response_sha256 TEXT CHECK (response_sha256 IS NULL OR length(response_sha256) = 64),
    image_sha256 TEXT CHECK (image_sha256 IS NULL OR length(image_sha256) = 64),
    revision INTEGER NOT NULL CHECK (revision > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (study_id, item_id, phase),
    UNIQUE (study_id, packet_id),
    FOREIGN KEY (study_id, item_id)
        REFERENCES items(study_id, item_id) ON DELETE RESTRICT
) STRICT;

CREATE TABLE IF NOT EXISTS adjudication_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    adjudication_id TEXT NOT NULL,
    study_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    phase TEXT NOT NULL CHECK (phase IN ('caption', 'image')),
    adjudicator_pseudonym TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    packet_version TEXT,
    packet_id TEXT,
    packet_sha256 TEXT CHECK (packet_sha256 IS NULL OR length(packet_sha256) = 64),
    packet_file_sha256 TEXT CHECK (
        packet_file_sha256 IS NULL OR length(packet_file_sha256) = 64
    ),
    response_version TEXT,
    response_sha256 TEXT CHECK (response_sha256 IS NULL OR length(response_sha256) = 64),
    image_sha256 TEXT CHECK (image_sha256 IS NULL OR length(image_sha256) = 64),
    revision INTEGER NOT NULL CHECK (revision > 0),
    created_at TEXT NOT NULL,
    UNIQUE (adjudication_id, revision),
    FOREIGN KEY (adjudication_id) REFERENCES adjudications(adjudication_id)
        ON DELETE RESTRICT,
    FOREIGN KEY (study_id, item_id) REFERENCES items(study_id, item_id)
        ON DELETE RESTRICT
) STRICT;

CREATE TRIGGER IF NOT EXISTS adjudication_events_no_update
BEFORE UPDATE ON adjudication_events
BEGIN
    SELECT RAISE(ABORT, 'adjudication_events are append-only');
END;

CREATE TRIGGER IF NOT EXISTS adjudication_events_no_delete
BEFORE DELETE ON adjudication_events
BEGIN
    SELECT RAISE(ABORT, 'adjudication_events are append-only');
END;

CREATE INDEX IF NOT EXISTS idx_items_study_group
    ON items(study_id, item_group);
CREATE INDEX IF NOT EXISTS idx_items_study_asset
    ON items(study_id, asset_status);
CREATE INDEX IF NOT EXISTS idx_participants_study
    ON participants(study_id, status);
CREATE INDEX IF NOT EXISTS idx_sessions_digest
    ON bearer_sessions(token_digest);
CREATE INDEX IF NOT EXISTS idx_assignments_next
    ON assignments(study_id, participant_id, phase, status, position);
CREATE INDEX IF NOT EXISTS idx_annotations_item
    ON annotations(study_id, item_id, phase);
CREATE INDEX IF NOT EXISTS idx_annotation_events_item
    ON annotation_events(study_id, item_id, phase);
CREATE INDEX IF NOT EXISTS idx_adjudication_events_item
    ON adjudication_events(study_id, item_id, phase);
"""


class HumanCBUStore:
    """SQLite-backed store for a two-phase blinded human CBU study.

    Instances do not retain a live SQLite connection.  Every operation opens a
    short connection with foreign keys, a busy timeout, and an explicit
    transaction, making the object safe to share between ordinary web-server
    threads.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        clock: Any = _utc_now,
    ) -> None:
        self.path = Path(path)
        if busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be positive")
        self.busy_timeout_ms = int(busy_timeout_ms)
        self._clock = clock

    @classmethod
    def initialize(
        cls,
        path: str | Path,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        clock: Any = _utc_now,
    ) -> HumanCBUStore:
        """Create (or idempotently initialize) a database and return its store."""

        store = cls(path, busy_timeout_ms=busy_timeout_ms, clock=clock)
        store.path.parent.mkdir(parents=True, exist_ok=True)
        connection = store._connect()
        try:
            tables = {
                row["name"]
                for row in connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                    """
                ).fetchall()
            }
            if tables and "schema_metadata" not in tables:
                raise HumanCBUStoreError("refusing to initialize an existing database without schema metadata")
            if "schema_metadata" in tables:
                existing = connection.execute(
                    "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
                ).fetchone()
                if existing is None or int(existing["value"]) != SCHEMA_VERSION:
                    found = None if existing is None else existing["value"]
                    raise HumanCBUStoreError(
                        f"database schema version {found!r} is not supported (expected {SCHEMA_VERSION})"
                    )
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            if not tables:
                bootstrap = (
                    "BEGIN IMMEDIATE;\n"
                    + _SCHEMA
                    + "\n"
                    + "INSERT OR IGNORE INTO schema_metadata(key, value) "
                    + f"VALUES ('schema_version', '{SCHEMA_VERSION}');\n"
                    + f"PRAGMA user_version={SCHEMA_VERSION};\n"
                    + "COMMIT;\n"
                )
                try:
                    connection.executescript(bootstrap)
                except Exception:
                    if connection.in_transaction:
                        connection.rollback()
                    raise
            else:
                # A same-version database may have been interrupted during a
                # previous idempotent initialization.  Re-applying only
                # CREATE-IF-NOT-EXISTS statements is safe after version check.
                connection.executescript(_SCHEMA)
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            version = connection.execute("SELECT value FROM schema_metadata WHERE key = 'schema_version'").fetchone()
            if version is None or int(version["value"]) != SCHEMA_VERSION:
                raise HumanCBUStoreError("concurrent schema bootstrap did not converge")
        finally:
            connection.close()
        return store

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        clock: Any = _utc_now,
    ) -> HumanCBUStore:
        """Open an initialized database, rejecting missing/unknown schemas."""

        store = cls(path, busy_timeout_ms=busy_timeout_ms, clock=clock)
        if not store.path.is_file():
            raise FileNotFoundError(store.path)
        with store._connect() as connection:
            has_metadata = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'schema_metadata'
                """
            ).fetchone()
            if has_metadata is None:
                raise HumanCBUStoreError("database has no human CBU schema metadata")
            row = connection.execute("SELECT value FROM schema_metadata WHERE key = 'schema_version'").fetchone()
            if row is None or int(row["value"]) != SCHEMA_VERSION:
                found = None if row is None else row["value"]
                raise HumanCBUStoreError(
                    f"database schema version {found!r} is not supported (expected {SCHEMA_VERSION})"
                )
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            if str(journal_mode).lower() != "wal":
                connection.execute("PRAGMA journal_mode=WAL")
        return store

    def __enter__(self) -> HumanCBUStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Compatibility no-op; operations own and close their connections."""

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime):
            raise TypeError("clock must return datetime")
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        return connection

    @contextmanager
    def _transaction(self, *, write: bool) -> Iterable[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _study_row(connection: sqlite3.Connection, study_id: str) -> sqlite3.Row:
        study_id = _require_identifier(study_id, "study_id")
        row = connection.execute("SELECT * FROM studies WHERE study_id = ?", (study_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"study {study_id!r} does not exist")
        return row

    @staticmethod
    def _require_study_status(row: sqlite3.Row, allowed: set[str] | frozenset[str]) -> None:
        if row["status"] not in allowed:
            choices = ", ".join(sorted(allowed))
            raise StudyStateError(f"study {row['study_id']!r} has status {row['status']!r}; expected one of: {choices}")

    def create_study(
        self,
        study_id: str,
        *,
        title: str,
        protocol_version: str,
        consent_version: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a draft study."""

        study_id = _require_identifier(study_id, "study_id")
        title = _require_nonempty_text(title, "title", maximum=500)
        protocol_version = _require_nonempty_text(protocol_version, "protocol_version", maximum=100)
        consent_version = _require_nonempty_text(consent_version, "consent_version", maximum=100)
        metadata_json = _json_dumps(dict(metadata or {}))
        now = _timestamp(self._now())
        try:
            with self._transaction(write=True) as connection:
                connection.execute(
                    """
                    INSERT INTO studies(
                        study_id, title, protocol_version, consent_version, status,
                        metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'draft', ?, ?, ?)
                    """,
                    (study_id, title, protocol_version, consent_version, metadata_json, now, now),
                )
        except sqlite3.IntegrityError as error:
            raise ConflictError(f"study {study_id!r} already exists") from error
        return self.get_study(study_id)

    def get_study(self, study_id: str) -> dict[str, Any]:
        with self._transaction(write=False) as connection:
            row = self._study_row(connection, study_id)
            return self._project_study(row)

    @staticmethod
    def _project_study(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "study_id": row["study_id"],
            "title": row["title"],
            "protocol_version": row["protocol_version"],
            "consent_version": row["consent_version"],
            "status": row["status"],
            "metadata": _json_loads(row["metadata_json"], {}),
            "assignment_seed": row["assignment_seed"],
            "assignment_participant_digest": row["assignment_participant_digest"],
            "labels_per_item": row["labels_per_item"],
            "validation_sealed_at": row["validation_sealed_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def provision_adjudicator(self, study_id: str) -> dict[str, str]:
        """Provision the predeclared adjudicator credential exactly once."""

        credential = secrets.token_urlsafe(32)
        now = _timestamp(self._now())
        with self._transaction(write=True) as connection:
            study = self._study_row(connection, study_id)
            self._require_study_status(study, {"draft"})
            if study["protocol_version"] != HUMAN_CBU_V1_PROTOCOL:
                raise ValidationError("adjudicator credential provisioning is defined for human_cbu_v1")
            pseudonym = self._frozen_adjudicator_pseudonym(study)
            if study["adjudicator_credential_digest"] is not None:
                raise ConflictError("adjudicator credential has already been provisioned")
            digest = _token_digest(
                credential,
                domain=f"adjudicator:{study_id}",
            )
            connection.execute(
                """
                UPDATE studies
                SET adjudicator_credential_digest = ?, updated_at = ?
                WHERE study_id = ?
                """,
                (digest, now, study_id),
            )
        return {
            "study_id": study_id,
            "adjudicator_pseudonym": pseudonym,
            "credential": credential,
        }

    def update_study_metadata(
        self,
        study_id: str,
        *,
        metadata: Mapping[str, Any],
        protocol_version: str | None = None,
        consent_version: str | None = None,
    ) -> dict[str, Any]:
        """Update draft protocol metadata before the validation seal."""

        next_metadata = dict(metadata)
        metadata_json = _json_dumps(next_metadata)
        now = _timestamp(self._now())
        with self._transaction(write=True) as connection:
            row = self._study_row(connection, study_id)
            self._require_study_status(row, {"draft"})
            next_protocol = (
                row["protocol_version"]
                if protocol_version is None
                else _require_nonempty_text(protocol_version, "protocol_version", maximum=100)
            )
            next_consent = (
                row["consent_version"]
                if consent_version is None
                else _require_nonempty_text(consent_version, "consent_version", maximum=100)
            )
            if row["protocol_version"] == HUMAN_CBU_V1_PROTOCOL:
                current_metadata = _json_loads(row["metadata_json"], {})
                if next_protocol != HUMAN_CBU_V1_PROTOCOL:
                    raise ValidationError("human_cbu_v1 protocol_version is frozen at init-db")
                if next_metadata.get("adjudicator_pseudonym") != current_metadata.get("adjudicator_pseudonym"):
                    raise ValidationError("human_cbu_v1 adjudicator_pseudonym is frozen at init-db")
            connection.execute(
                """
                UPDATE studies
                SET metadata_json = ?, protocol_version = ?, consent_version = ?, updated_at = ?
                WHERE study_id = ?
                """,
                (metadata_json, next_protocol, next_consent, now, study_id),
            )
            updated = self._study_row(connection, study_id)
            return self._project_study(updated)

    def set_study_status(
        self,
        study_id: str,
        status: str,
        *,
        allow_incomplete: bool = False,
    ) -> dict[str, Any]:
        """Apply a valid operational lifecycle transition.

        ``seal_validation`` is the only operation that transitions ``draft`` to
        ``ready``.  Pausing is reversible; closing and archiving are not.
        """

        if status not in STUDY_STATUSES:
            raise ValidationError(f"invalid study status: {status!r}")
        transitions = {
            "draft": set(),
            "ready": {"open"},
            "open": {"paused", "closed"},
            "paused": {"open", "closed"},
            "closed": {"archived"},
            "archived": set(),
        }
        now = _timestamp(self._now())
        with self._transaction(write=True) as connection:
            row = self._study_row(connection, study_id)
            current = row["status"]
            if status == current:
                return self._project_study(row)
            if status not in transitions[current]:
                raise StudyStateError(f"invalid study status transition: {current!r} -> {status!r}")
            if status == "open":
                _validate_launch_metadata(row)
                frozen_design = _human_cbu_v1_frozen_design_report(
                    connection,
                    row,
                )
                if frozen_design["errors"]:
                    raise StudyStateError(
                        "study cannot open because frozen human_cbu_v1 design "
                        "validation failed: " + "; ".join(frozen_design["errors"])
                    )
            if status == "closed" and not allow_incomplete:
                pending = connection.execute(
                    """
                    SELECT COUNT(*) FROM assignments
                    WHERE study_id = ? AND status != 'completed'
                    """,
                    (study_id,),
                ).fetchone()[0]
                if pending:
                    raise StudyStateError(
                        f"cannot close a study with {pending} pending assignments; "
                        "pause it or explicitly set allow_incomplete=True"
                    )
                expected_labels = row["labels_per_item"]
                if not isinstance(expected_labels, int) or expected_labels < 2:
                    raise StudyStateError("cannot close a study without a valid planned initial panel")
                panel_shortfalls = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM (
                        SELECT i.item_id, phase.value AS phase,
                               COUNT(DISTINCT CASE
                                   WHEN p.status = 'active'
                                    AND p.consented = 1
                                    AND p.consent_version = ?
                                   THEN a.participant_id
                               END) AS eligible_labels
                        FROM items AS i
                        CROSS JOIN (
                            SELECT 'caption' AS value
                            UNION ALL
                            SELECT 'image'
                        ) AS phase
                        LEFT JOIN annotations AS a
                          ON a.study_id = i.study_id
                         AND a.item_id = i.item_id
                         AND a.phase = phase.value
                        LEFT JOIN participants AS p
                          ON p.study_id = a.study_id
                         AND p.participant_id = a.participant_id
                        WHERE i.study_id = ?
                        GROUP BY i.item_id, phase.value
                        HAVING eligible_labels != ?
                    )
                    """,
                    (row["consent_version"], study_id, expected_labels),
                ).fetchone()[0]
                if panel_shortfalls:
                    raise StudyStateError(
                        "cannot close: "
                        f"{panel_shortfalls} item-phase panels lack the exact "
                        "currently eligible planned initial labels; pause and "
                        "replace ineligible participants, or explicitly set "
                        "allow_incomplete=True for an emergency close"
                    )
            connection.execute(
                "UPDATE studies SET status = ?, updated_at = ? WHERE study_id = ?",
                (status, now, study_id),
            )
            return self._project_study(self._study_row(connection, study_id))

    @staticmethod
    def _normalize_span(
        caption: str,
        span: Any,
    ) -> tuple[str | None, int | None, int | None]:
        if span is None:
            return None, None, None
        if isinstance(span, str):
            return _require_nonempty_text(span, "span", maximum=10_000), None, None
        if isinstance(span, Mapping):
            start = span.get("start")
            end = span.get("end")
        elif isinstance(span, Sequence) and not isinstance(span, (str, bytes)) and len(span) == 2:
            start, end = span
        else:
            raise ValidationError("span must be null, [start, end], or {'start': ..., 'end': ...}")
        if type(start) is not int or type(end) is not int:
            raise ValidationError("span boundaries must be integers")
        if start < 0 or end < start or end > len(caption):
            raise ValidationError("span must satisfy 0 <= start <= end <= len(caption)")
        return caption[start:end], start, end

    def _normalize_item(self, study_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            "item_id",
            "caption",
            "unit",
            "span",
            "span_start",
            "span_end",
            "target",
            "category",
            "surface",
            "group",
            "item_group",
            "image_locator",
            "image_asset",
            "asset_status",
            "asset_status_reason",
            "qwen_answer",
            "gemma_answer",
            "stratum",
            "population",
            "sample",
            "weight",
            "sample_weight",
            "sampling_weight",
            "population_size",
            "sample_probability",
            "repeat_of",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValidationError(f"unknown item fields: {', '.join(sorted(unknown))}")
        item_id = _require_identifier(raw.get("item_id") or f"i_{uuid.uuid4().hex}", "item_id")
        caption = _require_nonempty_text(raw.get("caption"), "caption")
        unit = _require_nonempty_text(raw.get("unit"), "unit", maximum=10_000)
        if "span" in raw and ("span_start" in raw or "span_end" in raw):
            raise ValidationError("provide span or span_start/span_end, not both")
        span_value = raw.get("span")
        if "span_start" in raw or "span_end" in raw:
            span_value = (raw.get("span_start"), raw.get("span_end"))
        span_text, span_start, span_end = self._normalize_span(caption, span_value)
        target = raw.get("target")
        if target is not None:
            target = _require_nonempty_text(target, "target", maximum=10_000)
        category = _require_identifier(raw.get("category"), "category")
        surface = _require_identifier(raw.get("surface"), "surface")
        item_group = _require_identifier(
            raw.get("item_group") or raw.get("group") or item_id,
            "item_group",
        )
        if "image_locator" not in raw:
            raise ValidationError("image_locator is required")
        image_locator_json = _json_dumps(raw["image_locator"])
        image_asset = raw.get("image_asset")
        image_asset_json = None if image_asset is None else _json_dumps(image_asset)
        if image_asset_json is not None:
            self._asset_reference(image_asset_json)
        if isinstance(image_asset, Mapping) and "sha256" in image_asset:
            asset_sha256 = image_asset["sha256"]
            if not isinstance(asset_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", asset_sha256):
                raise ValidationError("image_asset.sha256 must be a lowercase byte SHA-256")
        default_asset_status = "available" if image_asset is not None else "pending"
        asset_status = raw.get("asset_status", default_asset_status)
        if asset_status not in ASSET_STATUSES:
            raise ValidationError(f"invalid asset_status: {asset_status!r}")
        if asset_status == "available" and image_asset is None:
            raise ValidationError("available asset_status requires image_asset")
        asset_status_reason = raw.get("asset_status_reason")
        if asset_status_reason is not None:
            asset_status_reason = _require_identifier(asset_status_reason, "asset_status_reason")
        stratum = raw.get("stratum")
        if stratum is not None:
            stratum = _require_identifier(stratum, "stratum")
        weight = raw.get(
            "weight",
            raw.get("sampling_weight", raw.get("sample_weight")),
        )
        if weight is not None and (
            isinstance(weight, bool) or not isinstance(weight, (int, float)) or float(weight) < 0
        ):
            raise ValidationError("weight must be a non-negative number")
        population_size = raw.get("population_size")
        if population_size is not None and (type(population_size) is not int or population_size <= 0):
            raise ValidationError("population_size must be a positive integer")
        sample_probability = raw.get("sample_probability")
        if sample_probability is not None and (
            isinstance(sample_probability, bool)
            or not isinstance(sample_probability, (int, float))
            or not 0 < float(sample_probability) <= 1
        ):
            raise ValidationError("sample_probability must be in (0, 1]")
        repeat_of = raw.get("repeat_of")
        if repeat_of is not None:
            repeat_of = _require_identifier(repeat_of, "repeat_of")
            if repeat_of == item_id:
                raise ValidationError("an item cannot repeat itself")
        now = _timestamp(self._now())
        return {
            "study_id": study_id,
            "item_id": item_id,
            "caption": caption,
            "unit": unit,
            "span_text": span_text,
            "span_start": span_start,
            "span_end": span_end,
            "target": target,
            "category": category,
            "surface": surface,
            "item_group": item_group,
            "image_locator_json": image_locator_json,
            "image_asset_json": image_asset_json,
            "asset_status": asset_status,
            "asset_status_reason": asset_status_reason,
            "qwen_answer_json": (None if raw.get("qwen_answer") is None else _json_dumps(raw["qwen_answer"])),
            "gemma_answer_json": (None if raw.get("gemma_answer") is None else _json_dumps(raw["gemma_answer"])),
            "stratum": stratum,
            "population_json": (None if raw.get("population") is None else _json_dumps(raw["population"])),
            "sample_json": None if raw.get("sample") is None else _json_dumps(raw["sample"]),
            "weight": None if weight is None else float(weight),
            "population_size": population_size,
            "sample_probability": (None if sample_probability is None else float(sample_probability)),
            "repeat_of": repeat_of,
            "created_at": now,
            "updated_at": now,
        }

    def add_item(self, study_id: str, **item: Any) -> str:
        """Add one private audit item to a draft study and return its item ID."""

        return self.add_items(study_id, [item])[0]

    def add_items(self, study_id: str, items: Iterable[Mapping[str, Any]]) -> list[str]:
        """Atomically add private audit items to a draft study."""

        study_id = _require_identifier(study_id, "study_id")
        normalized = [self._normalize_item(study_id, dict(item)) for item in items]
        if not normalized:
            return []
        ids = [item["item_id"] for item in normalized]
        if len(ids) != len(set(ids)):
            raise ValidationError("duplicate item_id values in input")
        columns = tuple(normalized[0])
        placeholders = ", ".join("?" for _ in columns)
        sql = f"INSERT INTO items({', '.join(columns)}) VALUES ({placeholders})"
        try:
            with self._transaction(write=True) as connection:
                study = self._study_row(connection, study_id)
                self._require_study_status(study, {"draft"})
                connection.execute("PRAGMA defer_foreign_keys=ON")
                connection.executemany(sql, [tuple(item[column] for column in columns) for item in normalized])
                missing_repeat = connection.execute(
                    """
                    SELECT child.item_id, child.repeat_of
                    FROM items AS child
                    LEFT JOIN items AS parent
                      ON parent.study_id = child.study_id AND parent.item_id = child.repeat_of
                    WHERE child.study_id = ? AND child.repeat_of IS NOT NULL
                      AND parent.item_id IS NULL
                    LIMIT 1
                    """,
                    (study_id,),
                ).fetchone()
                if missing_repeat is not None:
                    raise ValidationError(
                        f"repeat_of {missing_repeat['repeat_of']!r} for {missing_repeat['item_id']!r} does not exist"
                    )
        except sqlite3.IntegrityError as error:
            raise ConflictError(f"one or more item IDs already exist in study {study_id!r}") from error
        return ids

    def get_private_item(self, study_id: str, item_id: str) -> dict[str, Any]:
        """Return the full administrator-only item record."""

        study_id = _require_identifier(study_id, "study_id")
        item_id = _require_identifier(item_id, "item_id")
        with self._transaction(write=False) as connection:
            row = connection.execute(
                "SELECT * FROM items WHERE study_id = ? AND item_id = ?",
                (study_id, item_id),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"item {item_id!r} does not exist in study {study_id!r}")
            return self._project_private_item(row)

    @staticmethod
    def _project_private_item(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "study_id": row["study_id"],
            "item_id": row["item_id"],
            "caption": row["caption"],
            "unit": row["unit"],
            "span": row["span_text"],
            "span_offsets": (
                None if row["span_start"] is None else {"start": row["span_start"], "end": row["span_end"]}
            ),
            "target": row["target"],
            "category": row["category"],
            "surface": row["surface"],
            "item_group": row["item_group"],
            "image_locator": _json_loads(row["image_locator_json"]),
            "image_asset": _json_loads(row["image_asset_json"]),
            "asset_status": row["asset_status"],
            "asset_status_reason": row["asset_status_reason"],
            "qwen_answer": _json_loads(row["qwen_answer_json"]),
            "gemma_answer": _json_loads(row["gemma_answer_json"]),
            "stratum": row["stratum"],
            "population": _json_loads(row["population_json"]),
            "sample": _json_loads(row["sample_json"]),
            "weight": row["weight"],
            "population_size": row["population_size"],
            "sample_probability": row["sample_probability"],
            "repeat_of": row["repeat_of"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def set_asset_status(
        self,
        study_id: str,
        item_id: str,
        status: str,
        *,
        image_asset: Any = ...,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Update retrieval status for one image before the validation seal."""

        if status not in ASSET_STATUSES:
            raise ValidationError(f"invalid asset status: {status!r}")
        if reason is not None:
            reason = _require_identifier(reason, "reason")
        image_asset_json: str | None | Ellipsis
        image_asset_json = ... if image_asset is ... else (None if image_asset is None else _json_dumps(image_asset))
        if isinstance(image_asset_json, str):
            self._asset_reference(image_asset_json)
        now = _timestamp(self._now())
        with self._transaction(write=True) as connection:
            study = self._study_row(connection, study_id)
            self._require_study_status(study, {"draft"})
            row = connection.execute(
                "SELECT image_asset_json FROM items WHERE study_id = ? AND item_id = ?",
                (study_id, item_id),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"item {item_id!r} does not exist in study {study_id!r}")
            next_asset = row["image_asset_json"] if image_asset_json is ... else image_asset_json
            if status == "available" and next_asset is None:
                raise ValidationError("available status requires a resolved image_asset")
            connection.execute(
                """
                UPDATE items
                SET asset_status = ?, image_asset_json = ?, asset_status_reason = ?, updated_at = ?
                WHERE study_id = ? AND item_id = ?
                """,
                (status, next_asset, reason, now, study_id, item_id),
            )
        return self.get_asset_status(study_id, item_id)

    def get_asset_status(self, study_id: str, item_id: str) -> dict[str, Any]:
        with self._transaction(write=False) as connection:
            row = connection.execute(
                """
                SELECT item_id, asset_status, asset_status_reason, image_asset_json, updated_at
                FROM items WHERE study_id = ? AND item_id = ?
                """,
                (study_id, item_id),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"item {item_id!r} does not exist in study {study_id!r}")
            return {
                "item_id": row["item_id"],
                "status": row["asset_status"],
                "reason": row["asset_status_reason"],
                "has_asset": row["image_asset_json"] is not None,
                "updated_at": row["updated_at"],
            }

    def create_participant_invite(
        self,
        study_id: str,
        *,
        expires_in_seconds: int | None = None,
    ) -> dict[str, Any]:
        return self.create_participant_invites(
            study_id,
            count=1,
            expires_in_seconds=expires_in_seconds,
        )[0]

    def create_participant_invites(
        self,
        study_id: str,
        *,
        count: int,
        expires_in_seconds: int | None = None,
    ) -> list[dict[str, Any]]:
        """Create pseudonymous participants and return each raw invite once."""

        if type(count) is not int or count <= 0:
            raise ValidationError("count must be a positive integer")
        if expires_in_seconds is not None and (type(expires_in_seconds) is not int or expires_in_seconds <= 0):
            raise ValidationError("expires_in_seconds must be a positive integer")
        now_dt = self._now()
        now = _timestamp(now_dt)
        expires_at = None if expires_in_seconds is None else _timestamp(now_dt + timedelta(seconds=expires_in_seconds))
        created: list[dict[str, Any]] = []
        with self._transaction(write=True) as connection:
            study = self._study_row(connection, study_id)
            self._require_study_status(study, {"draft"})
            assignment_count = connection.execute(
                "SELECT COUNT(*) FROM assignments WHERE study_id = ?",
                (study_id,),
            ).fetchone()[0]
            if assignment_count:
                raise StudyStateError("participant invitations are frozen after assignment planning")
            for _ in range(count):
                participant_id = f"p_{uuid.uuid4().hex}"
                pseudonym = f"r_{secrets.token_hex(8)}"
                raw_invite = f"hcu_{secrets.token_urlsafe(32)}"
                invite_id = f"v_{uuid.uuid4().hex}"
                invite_digest = _token_digest(raw_invite, domain="invite")
                connection.execute(
                    """
                    INSERT INTO participants(
                        participant_id, study_id, pseudonym, status, profile_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, 'invited', '{}', ?, ?)
                    """,
                    (participant_id, study_id, pseudonym, now, now),
                )
                connection.execute(
                    """
                    INSERT INTO participant_invites(
                        invite_id, study_id, participant_id, invite_digest,
                        expires_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (invite_id, study_id, participant_id, invite_digest, expires_at, now),
                )
                created.append(
                    {
                        "participant_id": participant_id,
                        "pseudonym": pseudonym,
                        "invite_code": raw_invite,
                        "expires_at": expires_at,
                    }
                )
        return created

    def _invite_context(
        self,
        connection: sqlite3.Connection,
        invite_code: str,
        *,
        study_id: str | None = None,
        mark_used: bool,
    ) -> sqlite3.Row:
        digest = _token_digest(invite_code, domain="invite")
        row = connection.execute(
            """
            SELECT i.*, p.pseudonym, p.status AS participant_status,
                   p.consented, p.consent_version AS participant_consent_version,
                   s.status AS study_status,
                   s.consent_version AS required_consent_version
            FROM participant_invites AS i
            JOIN participants AS p
              ON p.study_id = i.study_id AND p.participant_id = i.participant_id
            JOIN studies AS s ON s.study_id = i.study_id
            WHERE i.invite_digest = ?
            """,
            (digest,),
        ).fetchone()
        if row is None or not secrets.compare_digest(row["invite_digest"], digest):
            raise AuthenticationError("invalid invite code")
        if study_id is not None and row["study_id"] != study_id:
            raise AuthenticationError("invite code is not valid for this study")
        now_dt = self._now()
        if row["revoked_at"] is not None:
            raise AuthenticationError("invite code has been revoked")
        if row["expires_at"] is not None and _parse_timestamp(row["expires_at"]) <= now_dt:
            raise AuthenticationError("invite code has expired")
        if row["participant_status"] in {"declined", "withdrawn"}:
            raise AuthorizationError("participant is not eligible to start a session")
        if mark_used:
            connection.execute(
                """
                UPDATE participant_invites
                SET last_used_at = ?, use_count = use_count + 1
                WHERE invite_id = ?
                """,
                (_timestamp(now_dt), row["invite_id"]),
            )
        return row

    def authenticate_invite(
        self,
        invite_code: str,
        *,
        study_id: str | None = None,
    ) -> dict[str, Any]:
        """Validate an invite without returning or persisting the raw code."""

        with self._transaction(write=False) as connection:
            row = self._invite_context(
                connection,
                invite_code,
                study_id=study_id,
                mark_used=False,
            )
            return {
                "study_id": row["study_id"],
                "participant_id": row["participant_id"],
                "pseudonym": row["pseudonym"],
                "participant_status": row["participant_status"],
                "study_status": row["study_status"],
            }

    def create_session(
        self,
        invite_code: str,
        *,
        study_id: str | None = None,
        ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
    ) -> dict[str, Any]:
        """Exchange an invite for a short-lived bearer token, returned once."""

        if type(ttl_seconds) is not int or not 60 <= ttl_seconds <= MAX_SESSION_TTL_SECONDS:
            raise ValidationError(f"ttl_seconds must be between 60 and {MAX_SESSION_TTL_SECONDS}")
        now_dt = self._now()
        now = _timestamp(now_dt)
        expires_at = _timestamp(now_dt + timedelta(seconds=ttl_seconds))
        raw_token = f"hcs_{secrets.token_urlsafe(32)}"
        token_digest = _token_digest(raw_token, domain="session")
        session_id = f"s_{uuid.uuid4().hex}"
        with self._transaction(write=True) as connection:
            invite = self._invite_context(
                connection,
                invite_code,
                study_id=study_id,
                mark_used=True,
            )
            if invite["study_status"] == "open":
                session_scope = SESSION_SCOPE_FULL
            else:
                withdrawal_reauthentication = (
                    invite["study_status"] in {"paused", "closed"}
                    and invite["participant_status"] == "active"
                    and bool(invite["consented"])
                    and invite["participant_consent_version"] == invite["required_consent_version"]
                )
                if not withdrawal_reauthentication:
                    raise StudyStateError(
                        "new participant sessions require an open study; "
                        "paused/closed re-authentication is limited to active "
                        "current-consent participants so they can withdraw"
                    )
                session_scope = SESSION_SCOPE_WITHDRAWAL_ONLY
            connection.execute(
                """
                INSERT INTO bearer_sessions(
                    session_id, study_id, participant_id, token_digest,
                    scope, expires_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    invite["study_id"],
                    invite["participant_id"],
                    token_digest,
                    session_scope,
                    expires_at,
                    now,
                ),
            )
        return {
            "session_token": raw_token,
            "expires_at": expires_at,
            "study_id": invite["study_id"],
            "participant_id": invite["participant_id"],
            "pseudonym": invite["pseudonym"],
            "session_scope": session_scope,
        }

    def claim_preprovisioned_participant_session(
        self,
        study_id: str,
        invite_codes: Sequence[str],
        *,
        ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
    ) -> dict[str, Any]:
        """Atomically issue one unused preprovisioned code and live session.

        Raw participant codes remain outside SQLite. The production server
        loads the mode-0600 invite artifact and supplies the finite frozen pool
        here. SQLite stores only digests and atomically marks one unused code
        as claimed, so concurrent visitors cannot receive the same credential
        or exceed the frozen enrollment limit.
        """

        if type(ttl_seconds) is not int or not 60 <= ttl_seconds <= MAX_SESSION_TTL_SECONDS:
            raise ValidationError(f"ttl_seconds must be between 60 and {MAX_SESSION_TTL_SECONDS}")
        normalized_codes = [
            _require_nonempty_text(code, "participant code", maximum=500)
            for code in invite_codes
        ]
        if not normalized_codes or len(normalized_codes) != len(set(normalized_codes)):
            raise ValidationError("self-enrollment participant codes must be nonempty and unique")
        code_by_digest = {
            _token_digest(code, domain="invite"): code for code in normalized_codes
        }
        now_dt = self._now()
        now = _timestamp(now_dt)
        expires_at = _timestamp(now_dt + timedelta(seconds=ttl_seconds))
        with self._transaction(write=True) as connection:
            study = self._study_row(connection, study_id)
            self._require_study_status(study, {"open"})
            metadata = _json_loads(study["metadata_json"], {})
            enrollment_limit = metadata.get(SELF_ENROLLMENT_LIMIT_METADATA_KEY)
            if type(enrollment_limit) is not int or enrollment_limit <= 0:
                raise StudyStateError("study does not enable bounded self-enrollment")
            if len(normalized_codes) != enrollment_limit:
                raise StudyStateError(
                    "self-enrollment code artifact does not match the frozen enrollment limit"
                )
            placeholders = ", ".join("?" for _ in code_by_digest)
            matching = connection.execute(
                f"""
                SELECT i.invite_id, i.invite_digest, i.participant_id,
                       p.pseudonym
                FROM participant_invites AS i
                JOIN participants AS p
                  ON p.study_id = i.study_id
                 AND p.participant_id = i.participant_id
                WHERE i.study_id = ?
                  AND i.invite_digest IN ({placeholders})
                  AND i.revoked_at IS NULL
                  AND i.use_count = 0
                  AND p.status = 'invited'
                ORDER BY i.created_at, i.invite_id
                LIMIT 1
                """,
                (study_id, *code_by_digest),
            ).fetchone()
            if matching is None:
                raise ConflictError("no self-enrollment slots remain")
            claimed = connection.execute(
                """
                UPDATE participant_invites
                SET last_used_at = ?, use_count = 1
                WHERE invite_id = ? AND use_count = 0 AND revoked_at IS NULL
                """,
                (now, matching["invite_id"]),
            )
            if claimed.rowcount != 1:
                raise ConflictError("self-enrollment slot was claimed concurrently")
            raw_token = f"hcs_{secrets.token_urlsafe(32)}"
            token_digest = _token_digest(raw_token, domain="session")
            session_id = f"s_{uuid.uuid4().hex}"
            connection.execute(
                """
                INSERT INTO bearer_sessions(
                    session_id, study_id, participant_id, token_digest,
                    scope, expires_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    study_id,
                    matching["participant_id"],
                    token_digest,
                    SESSION_SCOPE_FULL,
                    expires_at,
                    now,
                ),
            )
            return {
                "participant_code": code_by_digest[matching["invite_digest"]],
                "session_token": raw_token,
                "expires_at": expires_at,
                "study_id": study_id,
                "participant_id": matching["participant_id"],
                "pseudonym": matching["pseudonym"],
                "session_scope": SESSION_SCOPE_FULL,
            }

    def _session_context(
        self,
        connection: sqlite3.Connection,
        session_token: str,
        *,
        study_id: str | None = None,
        require_open: bool,
        require_full_scope: bool = False,
    ) -> sqlite3.Row:
        digest = _token_digest(session_token, domain="session")
        row = connection.execute(
            """
            SELECT bs.*, p.pseudonym, p.status AS participant_status,
                   p.consented, p.consent_version AS participant_consent_version,
                   p.profile_json, s.status AS study_status,
                   s.consent_version AS required_consent_version,
                   s.protocol_version AS study_protocol_version,
                   s.metadata_json AS study_metadata_json
            FROM bearer_sessions AS bs
            JOIN participants AS p
              ON p.study_id = bs.study_id AND p.participant_id = bs.participant_id
            JOIN studies AS s ON s.study_id = bs.study_id
            WHERE bs.token_digest = ?
            """,
            (digest,),
        ).fetchone()
        if row is None or not secrets.compare_digest(row["token_digest"], digest):
            raise AuthenticationError("invalid bearer session")
        if study_id is not None and row["study_id"] != study_id:
            raise AuthenticationError("session is not valid for this study")
        if row["revoked_at"] is not None:
            raise AuthenticationError("bearer session has been revoked")
        if _parse_timestamp(row["expires_at"]) <= self._now():
            raise AuthenticationError("bearer session has expired")
        if require_full_scope and row["scope"] != SESSION_SCOPE_FULL:
            raise AuthorizationError("bearer session is limited to withdrawal")
        if require_open and row["study_status"] != "open":
            raise StudyStateError("the study is not open for annotation")
        return row

    def authenticate_session(
        self,
        session_token: str,
        *,
        study_id: str | None = None,
    ) -> dict[str, Any]:
        with self._transaction(write=False) as connection:
            row = self._session_context(
                connection,
                session_token,
                study_id=study_id,
                require_open=False,
            )
            return {
                "session_id": row["session_id"],
                "study_id": row["study_id"],
                "participant_id": row["participant_id"],
                "pseudonym": row["pseudonym"],
                "participant_status": row["participant_status"],
                "consented": bool(row["consented"]),
                "consent_version": row["participant_consent_version"],
                "study_status": row["study_status"],
                "expires_at": row["expires_at"],
                "session_scope": row["scope"],
            }

    def revoke_session(
        self,
        session_token: str,
        *,
        study_id: str | None = None,
    ) -> None:
        with self._transaction(write=True) as connection:
            row = self._session_context(
                connection,
                session_token,
                study_id=study_id,
                require_open=False,
            )
            connection.execute(
                "UPDATE bearer_sessions SET revoked_at = ? WHERE session_id = ?",
                (_timestamp(self._now()), row["session_id"]),
            )

    @staticmethod
    def _validate_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
        unknown = set(profile) - ALLOWED_PROFILE_FIELDS
        if unknown:
            raise ValidationError("profile contains non-coarse or unsupported fields: " + ", ".join(sorted(unknown)))
        normalized: dict[str, Any] = {}
        for key, value in profile.items():
            vocabulary = PROFILE_VALUE_VOCABULARIES[key]
            if value is None:
                normalized[key] = value
            elif isinstance(value, str):
                if value not in vocabulary:
                    raise ValidationError(f"profile field {key!r} is not an allowed coarse category")
                normalized[key] = value
            elif isinstance(value, list):
                if (
                    not value
                    or len(value) > 8
                    or any(not isinstance(entry, str) or entry not in vocabulary for entry in value)
                ):
                    raise ValidationError(f"profile field {key!r} must contain allowed coarse categories")
                normalized[key] = list(dict.fromkeys(value))
            else:
                raise ValidationError(f"profile field {key!r} must be categorical")
        return normalized

    def record_consent_profile(
        self,
        session_token: str,
        *,
        study_id: str | None = None,
        consented: bool,
        consent_version: str,
        profile: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record current consent and an allow-listed coarse profile."""

        if type(consented) is not bool:
            raise ValidationError("consented must be boolean")
        consent_version = _require_nonempty_text(consent_version, "consent_version", maximum=100)
        normalized_profile = self._validate_profile(profile or {})
        profile_json = _json_dumps(normalized_profile)
        now = _timestamp(self._now())
        with self._transaction(write=True) as connection:
            session = self._session_context(
                connection,
                session_token,
                study_id=study_id,
                require_open=consented,
                require_full_scope=consented,
            )
            if not consented and session["study_status"] not in {
                "open",
                "paused",
                "closed",
            }:
                raise StudyStateError(
                    "declining or withdrawing through the study UI is available only "
                    "while the study is open, paused, or closed"
                )
            if consent_version != session["required_consent_version"]:
                raise ValidationError("consent_version does not match the study's current form")
            previous_status = session["participant_status"]
            if consented and previous_status in {"declined", "withdrawn"}:
                raise AuthorizationError("declined or withdrawn participation cannot be reactivated")
            if consented:
                next_status = "active"
            elif previous_status == "withdrawn":
                next_status = "withdrawn"
            elif previous_status == "active":
                next_status = "withdrawn"
            else:
                next_status = "declined"
            connection.execute(
                """
                UPDATE participants
                SET status = ?, consented = ?, consent_version = ?,
                    consented_at = ?, profile_json = ?, updated_at = ?
                WHERE study_id = ? AND participant_id = ?
                """,
                (
                    next_status,
                    int(consented),
                    consent_version,
                    now if consented else None,
                    profile_json,
                    now,
                    session["study_id"],
                    session["participant_id"],
                ),
            )
            connection.execute(
                """
                INSERT INTO consent_events(
                    study_id, participant_id, consented, consent_version,
                    profile_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session["study_id"],
                    session["participant_id"],
                    int(consented),
                    consent_version,
                    profile_json,
                    now,
                ),
            )
            return {
                "participant_id": session["participant_id"],
                "pseudonym": session["pseudonym"],
                "consented": consented,
                "consent_version": consent_version,
                "profile": normalized_profile,
                "status": next_status,
            }

    @staticmethod
    def _phase_item_order(
        item_ids: Sequence[str],
        *,
        repeats: Mapping[str, str | None],
        seed: str,
        phase: str,
        participant_id: str,
    ) -> list[str]:
        """Deterministically order tasks while separating hidden repeats."""

        selected = set(item_ids)
        final_size = len(item_ids)
        desired_gap = max(10, (final_size + 9) // 10)
        order = sorted(
            (item_id for item_id in item_ids if repeats[item_id] is None),
            key=lambda item_id: (
                _stable_digest(seed, "order", phase, participant_id, item_id),
                item_id,
            ),
        )
        pending = {item_id for item_id in item_ids if repeats[item_id] is not None}
        while pending:
            ready = sorted(
                (item_id for item_id in pending if repeats[item_id] in selected and repeats[item_id] in order),
                key=lambda item_id: (
                    _stable_digest(seed, "repeat", phase, participant_id, item_id),
                    item_id,
                ),
            )
            if not ready:
                raise ValidationError("repeat assignments contain a cycle or missing original")
            for item_id in ready:
                parent = repeats[item_id]
                assert parent is not None
                parent_index = order.index(parent)
                candidates: list[tuple[int, int]] = []
                for slot in range(len(order) + 1):
                    shifted_parent = parent_index + 1 if slot <= parent_index else parent_index
                    candidates.append((slot, abs(slot - shifted_parent)))
                feasible = [candidate for candidate in candidates if candidate[1] >= desired_gap]
                if feasible:
                    possible = feasible
                else:
                    farthest = max(distance for _, distance in candidates)
                    possible = [candidate for candidate in candidates if candidate[1] == farthest]
                slot, _ = min(
                    possible,
                    key=lambda candidate: (
                        _stable_digest(
                            seed,
                            "repeat-slot",
                            phase,
                            participant_id,
                            item_id,
                            str(candidate[0]),
                        ),
                        candidate[0],
                    ),
                )
                order.insert(slot, item_id)
                pending.remove(item_id)
        if len(order) != final_size or set(order) != selected:
            raise HumanCBUStoreError("deterministic assignment ordering lost an item")
        return order

    def assign_items(
        self,
        study_id: str,
        *,
        labels_per_item: int,
        participant_ids: Sequence[str] | None = None,
        primary_participant_count: int | None = None,
        reserve_participant_count: int = 0,
        seed: str = "human-cbu-v1",
    ) -> dict[str, Any]:
        """Create deterministic caption and image assignments.

        Participants for each original item are chosen using deterministic
        rendezvous scores.  A repeat item is assigned to the same participants
        as its referenced original.  Caption and image task orderings use
        separate deterministic hashes.  Participants outside the primary
        roster are frozen as reserves and receive no assignments unless an
        eligible primary participant is later replaced.
        """

        if type(labels_per_item) is not int or labels_per_item <= 0:
            raise ValidationError("labels_per_item must be a positive integer")
        if primary_participant_count is not None and (
            type(primary_participant_count) is not int or primary_participant_count <= 0
        ):
            raise ValidationError("primary_participant_count must be a positive integer")
        if type(reserve_participant_count) is not int or reserve_participant_count < 0:
            raise ValidationError("reserve_participant_count must be a non-negative integer")
        seed = _require_nonempty_text(seed, "seed", maximum=500)
        with self._transaction(write=True) as connection:
            study = self._study_row(connection, study_id)
            self._require_study_status(study, {"draft"})
            study_metadata = _json_loads(study["metadata_json"], {})
            label_plan_error = _human_cbu_v1_label_plan_error(
                protocol_version=study["protocol_version"],
                metadata=study_metadata,
                labels_per_item=labels_per_item,
            )
            if label_plan_error is not None:
                raise ValidationError(label_plan_error)
            frozen_assignment_seed = study_metadata.get("assignment_seed")
            if frozen_assignment_seed is not None:
                if not isinstance(frozen_assignment_seed, str) or not frozen_assignment_seed:
                    raise ValidationError("study metadata contains an invalid assignment_seed")
                if seed != frozen_assignment_seed:
                    raise ValidationError("seed does not match frozen assignment_seed")
            frozen_labels_per_item = study_metadata.get("required_labels_per_item")
            if frozen_labels_per_item is not None:
                if (
                    isinstance(frozen_labels_per_item, bool)
                    or not isinstance(frozen_labels_per_item, int)
                    or frozen_labels_per_item < 2
                ):
                    raise ValidationError("study metadata must freeze required_labels_per_item >= 2")
                if labels_per_item != frozen_labels_per_item:
                    raise ValidationError("labels_per_item does not match frozen required_labels_per_item")
            participant_rows = connection.execute(
                """
                SELECT p.participant_id
                FROM participants AS p
                WHERE p.study_id = ?
                ORDER BY p.rowid
                """,
                (study_id,),
            ).fetchall()
            all_participants = [row["participant_id"] for row in participant_rows]
            if participant_ids is None:
                if reserve_participant_count >= len(all_participants) and all_participants:
                    raise ValidationError("reserve_participant_count must leave at least one primary participant")
                split = len(all_participants) - reserve_participant_count
                selected_participants = all_participants[:split]
                reserve_participants = all_participants[split:]
            else:
                if reserve_participant_count:
                    raise ValidationError(
                        "reserve_participant_count cannot be combined with explicit participant_ids; "
                        "unselected invited participants are the reserve roster"
                    )
                selected_participants = [_require_identifier(value, "participant_id") for value in participant_ids]
                if len(selected_participants) != len(set(selected_participants)):
                    raise ValidationError("participant_ids contains duplicates")
                if not set(selected_participants).issubset(all_participants):
                    raise NotFoundError("one or more participants do not belong to the study")
                selected_set = set(selected_participants)
                reserve_participants = [
                    participant_id for participant_id in all_participants if participant_id not in selected_set
                ]
            if primary_participant_count is not None and len(selected_participants) != primary_participant_count:
                raise ValidationError("the frozen invite roster does not match primary_participant_count")
            if labels_per_item > len(selected_participants):
                raise ValidationError("labels_per_item exceeds the number of participants")
            participant_plan_digest = _stable_digest(
                "participant-plan",
                *sorted(selected_participants),
            )
            roster = {
                "version": 1,
                "initial_primary_participant_ids": sorted(selected_participants),
                "current_primary_participant_ids": sorted(selected_participants),
                "reserve_participant_ids": sorted(reserve_participants),
            }
            item_rows = connection.execute(
                "SELECT item_id, repeat_of FROM items WHERE study_id = ? ORDER BY item_id",
                (study_id,),
            ).fetchall()
            if not item_rows:
                raise ValidationError("cannot assign an empty study")
            existing_count = connection.execute(
                "SELECT COUNT(*) FROM assignments WHERE study_id = ?",
                (study_id,),
            ).fetchone()[0]
            if existing_count:
                if study["labels_per_item"] == labels_per_item and study["assignment_seed"] == seed:
                    if study["assignment_participant_digest"] == participant_plan_digest:
                        metadata = _json_loads(study["metadata_json"], {})
                        existing_roster = metadata.get(ASSIGNMENT_ROSTER_METADATA_KEY)
                        if existing_roster is not None and existing_roster != roster:
                            raise ConflictError("study assignments already exist with a different reserve roster")
                        if existing_roster is None:
                            metadata[ASSIGNMENT_ROSTER_METADATA_KEY] = roster
                            metadata.setdefault(PARTICIPANT_REPLACEMENTS_METADATA_KEY, [])
                            connection.execute(
                                """
                                UPDATE studies SET metadata_json = ?, updated_at = ?
                                WHERE study_id = ?
                                """,
                                (_json_dumps(metadata), _timestamp(self._now()), study_id),
                            )
                        return self._assignment_summary(connection, study_id)
                raise ConflictError("study assignments already exist with a different plan")

            repeats = {row["item_id"]: row["repeat_of"] for row in item_rows}
            selected_by_item: dict[str, list[str]] = {}

            def choose(item_id: str, trail: frozenset[str] = frozenset()) -> list[str]:
                if item_id in selected_by_item:
                    return selected_by_item[item_id]
                if item_id in trail:
                    raise ValidationError("repeat_of contains a cycle")
                repeat_of = repeats[item_id]
                if repeat_of is not None:
                    if repeat_of not in repeats:
                        raise ValidationError(f"repeat_of target {repeat_of!r} is not in the study")
                    result = list(choose(repeat_of, trail | {item_id}))
                else:
                    result = sorted(
                        selected_participants,
                        key=lambda participant_id: (
                            _stable_digest(seed, "select", item_id, participant_id),
                            participant_id,
                        ),
                    )[:labels_per_item]
                selected_by_item[item_id] = result
                return result

            for item_id in repeats:
                choose(item_id)

            participant_items: dict[str, list[str]] = {participant_id: [] for participant_id in selected_participants}
            for item_id, assigned_participants in selected_by_item.items():
                for participant_id in assigned_participants:
                    participant_items[participant_id].append(item_id)

            now = _timestamp(self._now())
            for participant_id, item_ids in participant_items.items():
                for phase in PHASES:
                    ordered = self._phase_item_order(
                        item_ids,
                        repeats=repeats,
                        seed=seed,
                        phase=phase,
                        participant_id=participant_id,
                    )
                    for position, item_id in enumerate(ordered, start=1):
                        assignment_id = "a_" + _stable_digest(study_id, participant_id, item_id, phase)[:32]
                        order_key = _stable_digest(seed, "order", phase, participant_id, item_id)
                        connection.execute(
                            """
                            INSERT INTO assignments(
                                assignment_id, study_id, participant_id, item_id,
                                phase, position, order_key, assigned_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                assignment_id,
                                study_id,
                                participant_id,
                                item_id,
                                phase,
                                position,
                                order_key,
                                now,
                            ),
                        )
            metadata = _json_loads(study["metadata_json"], {})
            existing_mode = metadata.get(ASSIGNMENT_MODE_METADATA_KEY)
            if existing_mode not in {None, ASSIGNMENT_MODE_FIXED}:
                raise ConflictError("study already freezes a different assignment mode")
            metadata[ASSIGNMENT_MODE_METADATA_KEY] = ASSIGNMENT_MODE_FIXED
            metadata[ASSIGNMENT_ROSTER_METADATA_KEY] = roster
            metadata.setdefault(PARTICIPANT_REPLACEMENTS_METADATA_KEY, [])
            connection.execute(
                """
                UPDATE studies
                SET assignment_seed = ?, assignment_participant_digest = ?,
                    labels_per_item = ?, metadata_json = ?, updated_at = ?
                WHERE study_id = ?
                """,
                (
                    seed,
                    participant_plan_digest,
                    labels_per_item,
                    _json_dumps(metadata),
                    now,
                    study_id,
                ),
            )
            return self._assignment_summary(connection, study_id)

    def plan_remaining_first_assignments(
        self,
        study_id: str,
        *,
        labels_per_item: int,
        primary_participant_count: int,
        reserve_participant_count: int = 0,
        seed: str = "human-cbu-v1",
    ) -> dict[str, Any]:
        """Freeze a roster while leaving item ownership to the live work queue.

        No task is owned merely because an invite exists.  After consent, an
        active primary atomically claims one still-underfilled base item at a
        time.  The paired image task and any hidden repeat are bound to the
        same participant in that transaction.  This prevents early or absent
        invitees from hoarding work while preserving two distinct raters per
        item and caption-before-image blinding.
        """

        if type(labels_per_item) is not int or labels_per_item <= 0:
            raise ValidationError("labels_per_item must be a positive integer")
        if type(primary_participant_count) is not int or primary_participant_count <= 0:
            raise ValidationError("primary_participant_count must be a positive integer")
        if type(reserve_participant_count) is not int or reserve_participant_count < 0:
            raise ValidationError("reserve_participant_count must be a non-negative integer")
        seed = _require_nonempty_text(seed, "seed", maximum=500)
        with self._transaction(write=True) as connection:
            study = self._study_row(connection, study_id)
            self._require_study_status(study, {"draft"})
            metadata = _json_loads(study["metadata_json"], {})
            label_plan_error = _human_cbu_v1_label_plan_error(
                protocol_version=study["protocol_version"],
                metadata=metadata,
                labels_per_item=labels_per_item,
            )
            if label_plan_error is not None:
                raise ValidationError(label_plan_error)
            frozen_seed = metadata.get("assignment_seed")
            if frozen_seed is not None and frozen_seed != seed:
                raise ValidationError("seed does not match frozen assignment_seed")
            frozen_labels = metadata.get("required_labels_per_item")
            if frozen_labels is not None and frozen_labels != labels_per_item:
                raise ValidationError("labels_per_item does not match frozen required_labels_per_item")
            participant_rows = connection.execute(
                """
                SELECT participant_id FROM participants
                WHERE study_id = ? ORDER BY rowid
                """,
                (study_id,),
            ).fetchall()
            all_participants = [row["participant_id"] for row in participant_rows]
            expected = primary_participant_count + reserve_participant_count
            if len(all_participants) != expected:
                raise ValidationError(
                    "the invite roster does not match primary_participant_count + "
                    "reserve_participant_count"
                )
            if labels_per_item > primary_participant_count:
                raise ValidationError("labels_per_item exceeds the number of primary participants")
            if connection.execute(
                "SELECT COUNT(*) FROM assignments WHERE study_id = ?",
                (study_id,),
            ).fetchone()[0]:
                raise ConflictError("remaining-first planning requires an empty assignment table")
            selected = all_participants[:primary_participant_count]
            reserves = all_participants[primary_participant_count:]
            participant_plan_digest = _stable_digest("participant-plan", *sorted(selected))
            roster = {
                "version": 2,
                "initial_primary_participant_ids": sorted(selected),
                "current_primary_participant_ids": sorted(selected),
                "reserve_participant_ids": sorted(reserves),
            }
            existing_mode = metadata.get(ASSIGNMENT_MODE_METADATA_KEY)
            if existing_mode not in {None, ASSIGNMENT_MODE_REMAINING_FIRST}:
                raise ConflictError("study already freezes a different assignment mode")
            existing_roster = metadata.get(ASSIGNMENT_ROSTER_METADATA_KEY)
            if existing_roster is not None and existing_roster != roster:
                raise ConflictError("study already freezes a different assignment roster")
            metadata[ASSIGNMENT_MODE_METADATA_KEY] = ASSIGNMENT_MODE_REMAINING_FIRST
            metadata[ASSIGNMENT_ROSTER_METADATA_KEY] = roster
            metadata.setdefault(PARTICIPANT_REPLACEMENTS_METADATA_KEY, [])
            now = _timestamp(self._now())
            connection.execute(
                """
                UPDATE studies
                SET assignment_seed = ?, assignment_participant_digest = ?,
                    labels_per_item = ?, metadata_json = ?, updated_at = ?
                WHERE study_id = ?
                """,
                (
                    seed,
                    participant_plan_digest,
                    labels_per_item,
                    _json_dumps(metadata),
                    now,
                    study_id,
                ),
            )
            summary = self._assignment_summary(connection, study_id)
            summary["assignment_mode"] = ASSIGNMENT_MODE_REMAINING_FIRST
            summary["late_bound"] = True
            return summary

    @staticmethod
    def _assignment_summary(connection: sqlite3.Connection, study_id: str) -> dict[str, Any]:
        stored_rows = connection.execute(
            """
            SELECT phase, COUNT(*) AS count
            FROM assignments WHERE study_id = ? GROUP BY phase
            """,
            (study_id,),
        ).fetchall()
        rows = connection.execute(
            """
            SELECT a.phase, COUNT(*) AS count
            FROM assignments AS a
            JOIN participants AS p
              ON p.study_id = a.study_id AND p.participant_id = a.participant_id
            WHERE a.study_id = ? AND p.status NOT IN ('declined', 'withdrawn')
            GROUP BY a.phase
            """,
            (study_id,),
        ).fetchall()
        by_phase = {phase: 0 for phase in PHASES}
        by_phase.update({row["phase"]: row["count"] for row in rows})
        stored_by_phase = {phase: 0 for phase in PHASES}
        stored_by_phase.update({row["phase"]: row["count"] for row in stored_rows})
        study = connection.execute(
            "SELECT metadata_json FROM studies WHERE study_id = ?",
            (study_id,),
        ).fetchone()
        metadata = _json_loads(study["metadata_json"], {}) if study is not None else {}
        roster = metadata.get(ASSIGNMENT_ROSTER_METADATA_KEY, {})
        return {
            "study_id": study_id,
            "participants": connection.execute(
                """
                SELECT COUNT(DISTINCT a.participant_id)
                FROM assignments AS a
                JOIN participants AS p
                  ON p.study_id = a.study_id AND p.participant_id = a.participant_id
                WHERE a.study_id = ? AND p.status NOT IN ('declined', 'withdrawn')
                """,
                (study_id,),
            ).fetchone()[0],
            "stored_participants": connection.execute(
                """
                SELECT COUNT(DISTINCT participant_id)
                FROM assignments WHERE study_id = ?
                """,
                (study_id,),
            ).fetchone()[0],
            "planned_primary_participants": len(roster.get("current_primary_participant_ids", ())),
            "reserve_participants": len(roster.get("reserve_participant_ids", ())),
            "participant_replacements": len(metadata.get(PARTICIPANT_REPLACEMENTS_METADATA_KEY, ())),
            "items": connection.execute(
                "SELECT COUNT(*) FROM items WHERE study_id = ?",
                (study_id,),
            ).fetchone()[0],
            "assignments": by_phase,
            "stored_assignments": stored_by_phase,
        }

    def replace_participant(
        self,
        study_id: str,
        *,
        source_participant_id: str,
        replacement_participant_id: str,
    ) -> dict[str, Any]:
        """Replace an ineligible primary with one predeclared reserve.

        The operation requires a paused study.  Completed assignments and their
        append-only annotations remain attached to the withdrawn/declined
        participant for auditability, while fresh pending assignments are
        created for the replacement.  Pending, unannotated assignments are
        atomically re-keyed to the replacement and all presentation state is
        cleared.  Consequently the reserve independently rates the source
        participant's complete frozen task set, while ineligible source labels
        remain stored but are excluded by the ordinary eligibility filters.
        """

        source_participant_id = _require_identifier(
            source_participant_id,
            "source_participant_id",
        )
        replacement_participant_id = _require_identifier(
            replacement_participant_id,
            "replacement_participant_id",
        )
        if source_participant_id == replacement_participant_id:
            raise ValidationError("source and replacement participants must differ")

        with self._transaction(write=True) as connection:
            study = self._study_row(connection, study_id)
            metadata = _json_loads(study["metadata_json"], {})
            roster = metadata.get(ASSIGNMENT_ROSTER_METADATA_KEY)
            history = metadata.get(PARTICIPANT_REPLACEMENTS_METADATA_KEY, [])
            if not isinstance(roster, Mapping) or roster.get("version") != 1:
                raise StudyStateError("the frozen assignment plan has no reserve-participant roster")
            if not isinstance(history, list) or any(not isinstance(event, Mapping) for event in history):
                raise HumanCBUStoreError("participant replacement history is malformed")

            prior_for_source = [
                event for event in history if event.get("source_participant_id") == source_participant_id
            ]
            if prior_for_source:
                previous = dict(prior_for_source[-1])
                if previous.get("replacement_participant_id") != replacement_participant_id:
                    raise ConflictError("source participant was already replaced by a different reserve")
                return {
                    "study_id": study_id,
                    "idempotent": True,
                    "replacement": previous,
                    "assignment_plan": self._assignment_summary(connection, study_id),
                }

            self._require_study_status(study, {"paused"})
            current_primary = roster.get("current_primary_participant_ids")
            reserves = roster.get("reserve_participant_ids")
            if (
                not isinstance(current_primary, list)
                or not all(isinstance(value, str) for value in current_primary)
                or not isinstance(reserves, list)
                or not all(isinstance(value, str) for value in reserves)
            ):
                raise HumanCBUStoreError("assignment roster is malformed")
            if source_participant_id not in current_primary:
                raise ValidationError("source participant is not in the current primary roster")
            if replacement_participant_id not in reserves:
                raise ValidationError("replacement participant is not an unused predeclared reserve")

            participant_rows = connection.execute(
                """
                SELECT participant_id, status, consented, consent_version
                FROM participants
                WHERE study_id = ? AND participant_id IN (?, ?)
                """,
                (study_id, source_participant_id, replacement_participant_id),
            ).fetchall()
            participants = {row["participant_id"]: row for row in participant_rows}
            if len(participants) != 2:
                raise NotFoundError("source and replacement participants must belong to the study")
            source = participants[source_participant_id]
            replacement = participants[replacement_participant_id]
            if source["status"] not in {"declined", "withdrawn"}:
                raise ValidationError("source participant must have declined or withdrawn before replacement")
            if not (
                replacement["status"] == "active"
                and bool(replacement["consented"])
                and replacement["consent_version"] == study["consent_version"]
            ):
                raise ValidationError("replacement reserve must be active with the study's current consent")
            replacement_assignment_count = connection.execute(
                """
                SELECT COUNT(*) FROM assignments
                WHERE study_id = ? AND participant_id = ?
                """,
                (study_id, replacement_participant_id),
            ).fetchone()[0]
            replacement_annotation_count = connection.execute(
                """
                SELECT COUNT(*) FROM annotations
                WHERE study_id = ? AND participant_id = ?
                """,
                (study_id, replacement_participant_id),
            ).fetchone()[0]
            if replacement_assignment_count or replacement_annotation_count:
                raise ConflictError("replacement reserve already has assignments or annotations")

            source_rows = connection.execute(
                """
                SELECT a.*, n.annotation_id
                FROM assignments AS a
                LEFT JOIN annotations AS n ON n.assignment_id = a.assignment_id
                WHERE a.study_id = ? AND a.participant_id = ?
                ORDER BY a.phase, a.position, a.assignment_id
                """,
                (study_id, source_participant_id),
            ).fetchall()
            if not source_rows:
                raise ValidationError("source participant has no frozen assignments")
            for row in source_rows:
                if (row["status"] == "completed") != (row["annotation_id"] is not None):
                    raise HumanCBUStoreError("source assignment status and latest annotation are inconsistent")
            phase_items = {phase: [row["item_id"] for row in source_rows if row["phase"] == phase] for phase in PHASES}
            if set(phase_items["caption"]) != set(phase_items["image"]):
                raise HumanCBUStoreError("source caption and image assignment sets are inconsistent")
            repeat_rows = connection.execute(
                """
                SELECT item_id, repeat_of FROM items
                WHERE study_id = ? AND item_id IN (
                    SELECT item_id FROM assignments
                    WHERE study_id = ? AND participant_id = ?
                )
                """,
                (study_id, study_id, source_participant_id),
            ).fetchall()
            repeats = {row["item_id"]: row["repeat_of"] for row in repeat_rows}
            if set(repeats) != set(phase_items["caption"]):
                raise HumanCBUStoreError("source assignment items are missing from the frozen item set")
            seed = study["assignment_seed"]
            if not isinstance(seed, str) or not seed:
                raise HumanCBUStoreError("study assignment seed is missing")

            replacement_positions: dict[tuple[str, str], int] = {}
            for phase in PHASES:
                ordered = self._phase_item_order(
                    phase_items[phase],
                    repeats=repeats,
                    seed=seed,
                    phase=phase,
                    participant_id=replacement_participant_id,
                )
                replacement_positions.update(
                    {(phase, item_id): position for position, item_id in enumerate(ordered, start=1)}
                )

            now = _timestamp(self._now())
            transferred_pending = 0
            fresh_reratings = 0
            source_assignment_ids: list[str] = []
            replacement_assignment_ids: list[str] = []
            for row in source_rows:
                source_assignment_ids.append(row["assignment_id"])
                phase = row["phase"]
                item_id = row["item_id"]
                replacement_assignment_id = (
                    "a_"
                    + _stable_digest(
                        study_id,
                        replacement_participant_id,
                        item_id,
                        phase,
                    )[:32]
                )
                replacement_assignment_ids.append(replacement_assignment_id)
                position = replacement_positions[(phase, item_id)]
                order_key = _stable_digest(
                    seed,
                    "order",
                    phase,
                    replacement_participant_id,
                    item_id,
                )
                if row["annotation_id"] is None:
                    cursor = connection.execute(
                        """
                        UPDATE assignments
                        SET assignment_id = ?, participant_id = ?, position = ?,
                            order_key = ?, status = 'pending', assigned_at = ?,
                            first_presented_at = NULL, last_presented_at = NULL,
                            presentation_count = 0, completed_at = NULL
                        WHERE assignment_id = ? AND status = 'pending'
                        """,
                        (
                            replacement_assignment_id,
                            replacement_participant_id,
                            position,
                            order_key,
                            now,
                            row["assignment_id"],
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise HumanCBUStoreError("pending source assignment changed during replacement")
                    transferred_pending += 1
                else:
                    connection.execute(
                        """
                        INSERT INTO assignments(
                            assignment_id, study_id, participant_id, item_id,
                            phase, position, order_key, assigned_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            replacement_assignment_id,
                            study_id,
                            replacement_participant_id,
                            item_id,
                            phase,
                            position,
                            order_key,
                            now,
                        ),
                    )
                    fresh_reratings += 1

            next_primary = sorted(
                replacement_participant_id if participant_id == source_participant_id else participant_id
                for participant_id in current_primary
            )
            next_reserves = sorted(
                participant_id for participant_id in reserves if participant_id != replacement_participant_id
            )
            next_roster = dict(roster)
            next_roster["current_primary_participant_ids"] = next_primary
            next_roster["reserve_participant_ids"] = next_reserves

            source_assignment_digest = _stable_digest(
                "source-assignments",
                *sorted(source_assignment_ids),
            )
            replacement_assignment_digest = _stable_digest(
                "replacement-assignments",
                *sorted(replacement_assignment_ids),
            )
            previous_event_digest = history[-1].get("event_digest") if history else None
            event = {
                "sequence": len(history) + 1,
                "source_participant_id": source_participant_id,
                "replacement_participant_id": replacement_participant_id,
                "source_status": source["status"],
                "source_assignment_count": len(source_rows),
                "source_completed_assignment_count": fresh_reratings,
                "transferred_pending_assignment_count": transferred_pending,
                "fresh_rerating_assignment_count": fresh_reratings,
                "source_assignment_digest": source_assignment_digest,
                "replacement_assignment_digest": replacement_assignment_digest,
                "remaining_reserve_participants": len(next_reserves),
                "previous_event_digest": previous_event_digest,
                "created_at": now,
            }
            event["event_digest"] = hashlib.sha256(_json_dumps(event).encode("utf-8")).hexdigest()
            history.append(event)
            metadata[ASSIGNMENT_ROSTER_METADATA_KEY] = next_roster
            metadata[PARTICIPANT_REPLACEMENTS_METADATA_KEY] = history
            participant_plan_digest = _stable_digest(
                "participant-plan",
                *next_primary,
            )
            connection.execute(
                """
                UPDATE studies
                SET metadata_json = ?, assignment_participant_digest = ?,
                    updated_at = ?
                WHERE study_id = ?
                """,
                (
                    _json_dumps(metadata),
                    participant_plan_digest,
                    now,
                    study_id,
                ),
            )
            return {
                "study_id": study_id,
                "idempotent": False,
                "replacement": event,
                "assignment_plan": self._assignment_summary(connection, study_id),
            }

    def validation_report(self, study_id: str, *, require_assets: bool = True) -> dict[str, Any]:
        """Return pre-launch validation failures without changing study state."""

        with self._transaction(write=False) as connection:
            self._study_row(connection, study_id)
            return self._validation_report(connection, study_id, require_assets=require_assets)

    @staticmethod
    def _validation_report(
        connection: sqlite3.Connection,
        study_id: str,
        *,
        require_assets: bool,
    ) -> dict[str, Any]:
        study = connection.execute(
            """
            SELECT study_id, labels_per_item, metadata_json, protocol_version,
                   adjudicator_credential_digest
            FROM studies
            WHERE study_id = ?
            """,
            (study_id,),
        ).fetchone()
        item_count = connection.execute(
            "SELECT COUNT(*) FROM items WHERE study_id = ?",
            (study_id,),
        ).fetchone()[0]
        participant_count = connection.execute(
            "SELECT COUNT(*) FROM participants WHERE study_id = ?",
            (study_id,),
        ).fetchone()[0]
        asset_failures = connection.execute(
            """
            SELECT item_id, asset_status
            FROM items
            WHERE study_id = ? AND (asset_status != 'available' OR image_asset_json IS NULL)
            ORDER BY item_id
            """,
            (study_id,),
        ).fetchall()
        assignment_failures: list[dict[str, Any]] = []
        labels_per_item = study["labels_per_item"]
        stored_metadata = _json_loads(study["metadata_json"], {})
        metadata = stored_metadata if isinstance(stored_metadata, Mapping) else {}
        assignment_mode = metadata.get(ASSIGNMENT_MODE_METADATA_KEY, ASSIGNMENT_MODE_FIXED)
        if assignment_mode not in {ASSIGNMENT_MODE_FIXED, ASSIGNMENT_MODE_REMAINING_FIRST}:
            assignment_mode = "invalid"
        required_labels_per_item = metadata.get("required_labels_per_item")
        frozen_design = _human_cbu_v1_frozen_design_report(connection, study)
        if labels_per_item is not None:
            rows = connection.execute(
                """
                SELECT i.item_id, p.phase,
                       COUNT(DISTINCT CASE
                           WHEN r.status NOT IN ('declined', 'withdrawn')
                           THEN a.participant_id
                       END) AS labels
                FROM items AS i
                CROSS JOIN (SELECT 'caption' AS phase UNION ALL SELECT 'image') AS p
                LEFT JOIN assignments AS a
                  ON a.study_id = i.study_id AND a.item_id = i.item_id AND a.phase = p.phase
                LEFT JOIN participants AS r
                  ON r.study_id = a.study_id AND r.participant_id = a.participant_id
                WHERE i.study_id = ?
                GROUP BY i.item_id, p.phase
                HAVING labels != ?
                ORDER BY i.item_id, p.phase
                """,
                (study_id, labels_per_item),
            ).fetchall()
            assignment_failures = [
                {"item_id": row["item_id"], "phase": row["phase"], "labels": row["labels"]} for row in rows
            ]
        repeat_rows = connection.execute(
            """
            SELECT child.participant_id, child.phase, child.item_id,
                   repeated.repeat_of, child.position AS repeat_position,
                   original.position AS original_position,
                   (
                       SELECT COUNT(*) FROM assignments AS phase_block
                       WHERE phase_block.study_id = child.study_id
                         AND phase_block.participant_id = child.participant_id
                         AND phase_block.phase = child.phase
                   ) AS block_size
            FROM items AS repeated
            JOIN assignments AS child
              ON child.study_id = repeated.study_id
             AND child.item_id = repeated.item_id
            JOIN participants AS participant
              ON participant.study_id = child.study_id
             AND participant.participant_id = child.participant_id
            JOIN assignments AS original
              ON original.study_id = child.study_id
             AND original.participant_id = child.participant_id
             AND original.phase = child.phase
             AND original.item_id = repeated.repeat_of
            WHERE repeated.study_id = ? AND repeated.repeat_of IS NOT NULL
              AND participant.status NOT IN ('declined', 'withdrawn')
            ORDER BY child.participant_id, child.phase, child.item_id
            """,
            (study_id,),
        ).fetchall()
        repeat_spacing = []
        repeat_spacing_failures = []
        for row in repeat_rows:
            desired_gap = max(10, (row["block_size"] + 9) // 10)
            distance = abs(row["repeat_position"] - row["original_position"])
            result = {
                "participant_id": row["participant_id"],
                "phase": row["phase"],
                "item_id": row["item_id"],
                "repeat_of": row["repeat_of"],
                "block_size": row["block_size"],
                "position_distance": distance,
                "desired_distance": desired_gap,
                "target_met": distance >= desired_gap,
            }
            repeat_spacing.append(result)
            # With this many positions the target is achievable regardless of
            # the original's position, so a miss indicates an ordering bug.
            if not result["target_met"] and row["block_size"] >= 2 * desired_gap + 1:
                repeat_spacing_failures.append(result)
        errors: list[str] = []
        if item_count == 0:
            errors.append("study has no items")
        if participant_count == 0:
            errors.append("study has no participants")
        if labels_per_item is None:
            errors.append("assignment plan is missing")
        elif labels_per_item < 2:
            errors.append("labels_per_item must be at least 2")
        if (
            isinstance(required_labels_per_item, bool)
            or not isinstance(required_labels_per_item, int)
            or required_labels_per_item < 2
        ):
            errors.append("study metadata must freeze required_labels_per_item >= 2")
        elif labels_per_item is not None and labels_per_item != required_labels_per_item:
            errors.append("assignment labels_per_item does not match frozen required_labels_per_item")
        label_plan_error = _human_cbu_v1_label_plan_error(
            protocol_version=study["protocol_version"],
            metadata=metadata,
            labels_per_item=labels_per_item,
        )
        if label_plan_error is not None:
            errors.append(label_plan_error)
        if study["protocol_version"] == HUMAN_CBU_V1_PROTOCOL and study["adjudicator_credential_digest"] is None:
            errors.append("predeclared adjudicator credential has not been provisioned")
        errors.extend(frozen_design["errors"])
        if assignment_mode == "invalid":
            errors.append("study metadata contains an invalid assignment mode")
        if assignment_failures and assignment_mode != ASSIGNMENT_MODE_REMAINING_FIRST:
            errors.append("one or more items do not have the planned labels_per_item")
        if repeat_spacing_failures:
            errors.append("one or more hidden repeats are insufficiently separated")
        if require_assets and asset_failures:
            errors.append("one or more image assets are not available")
        return {
            "study_id": study_id,
            "valid": not errors,
            "errors": errors,
            "item_count": item_count,
            "participant_count": participant_count,
            "labels_per_item": labels_per_item,
            "assignment_mode": assignment_mode,
            "late_binding_incomplete_panels": (
                len(assignment_failures)
                if assignment_mode == ASSIGNMENT_MODE_REMAINING_FIRST
                else 0
            ),
            "required_labels_per_item": required_labels_per_item,
            "asset_failures": [{"item_id": row["item_id"], "status": row["asset_status"]} for row in asset_failures],
            "assignment_failures": assignment_failures,
            "repeat_spacing": repeat_spacing,
            "repeat_spacing_failures": repeat_spacing_failures,
            "expected_repeat_items": frozen_design["expected_repeat_items"],
            "repeat_items": frozen_design["repeat_items"],
            "repeat_payload_failures": frozen_design["repeat_payload_failures"],
            "semantic_visual_claim_types": frozen_design["semantic_visual_claim_types"],
        }

    def seal_validation(
        self,
        study_id: str,
        *,
        require_assets: bool = True,
    ) -> dict[str, Any]:
        """Freeze the draft sample/protocol/assets and mark it ready to open."""

        if not require_assets:
            raise ValidationError("validation cannot be sealed before every selected image asset is available")
        now = _timestamp(self._now())
        with self._transaction(write=True) as connection:
            study = self._study_row(connection, study_id)
            if study["status"] == "ready" and study["validation_sealed_at"] is not None:
                report = self._validation_report(connection, study_id, require_assets=require_assets)
                report["sealed_at"] = study["validation_sealed_at"]
                return report
            self._require_study_status(study, {"draft"})
            report = self._validation_report(connection, study_id, require_assets=require_assets)
            if not report["valid"]:
                raise ValidationError("study validation failed: " + "; ".join(report["errors"]))
            connection.execute(
                """
                UPDATE studies
                SET status = 'ready', validation_sealed_at = ?, updated_at = ?
                WHERE study_id = ?
                """,
                (now, now, study_id),
            )
            report["sealed_at"] = now
            return report

    @staticmethod
    def _has_current_consent(session: sqlite3.Row) -> bool:
        return bool(session["consented"]) and (
            session["participant_status"] == "active"
            and session["participant_consent_version"] == session["required_consent_version"]
        )

    @staticmethod
    def _progress_for(
        connection: sqlite3.Connection,
        study_id: str,
        participant_id: str,
    ) -> dict[str, Any]:
        rows = connection.execute(
            """
            SELECT phase, COUNT(*) AS total,
                   SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed
            FROM assignments
            WHERE study_id = ? AND participant_id = ?
            GROUP BY phase
            """,
            (study_id, participant_id),
        ).fetchall()
        phases: dict[str, dict[str, int]] = {phase: {"total": 0, "completed": 0, "remaining": 0} for phase in PHASES}
        for row in rows:
            total = int(row["total"])
            completed = int(row["completed"] or 0)
            phases[row["phase"]] = {
                "total": total,
                "completed": completed,
                "remaining": total - completed,
            }
        study = connection.execute(
            "SELECT labels_per_item, metadata_json FROM studies WHERE study_id = ?",
            (study_id,),
        ).fetchone()
        if study is None:
            raise NotFoundError(f"study not found: {study_id}")
        metadata = _json_loads(study["metadata_json"], {})
        target = metadata.get(TARGET_ITEMS_PER_PARTICIPANT_METADATA_KEY)
        remaining_first = (
            metadata.get(ASSIGNMENT_MODE_METADATA_KEY)
            == ASSIGNMENT_MODE_REMAINING_FIRST
        )
        if (
            remaining_first
            and type(target) is int
            and target > 0
            and type(study["labels_per_item"]) is int
            and HumanCBUStore._remaining_first_pool_incomplete(
                connection,
                study_id,
                int(study["labels_per_item"]),
            )
        ):
            for phase in PHASES:
                visible_total = max(phases[phase]["total"], target)
                phases[phase]["total"] = visible_total
                phases[phase]["remaining"] = max(
                    0,
                    visible_total - phases[phase]["completed"],
                )
        if phases["caption"]["remaining"] > 0:
            current_phase = "caption"
        elif phases["image"]["remaining"] > 0:
            current_phase = "image"
        else:
            current_phase = "complete"
        return {
            "caption": phases["caption"],
            "image": phases["image"],
            "current_phase": current_phase,
        }

    def get_progress(
        self,
        session_token: str,
        *,
        study_id: str | None = None,
    ) -> dict[str, Any]:
        with self._transaction(write=False) as connection:
            session = self._session_context(
                connection,
                session_token,
                study_id=study_id,
                require_open=False,
            )
            progress = self._progress_for(connection, session["study_id"], session["participant_id"])
            progress.update(
                {
                    "study_id": session["study_id"],
                    "participant_id": session["participant_id"],
                    "consent_current": self._has_current_consent(session),
                }
            )
            return progress

    def progress(
        self,
        session_token: str,
        *,
        study_id: str | None = None,
    ) -> dict[str, Any]:
        return self.get_progress(session_token, study_id=study_id)

    @staticmethod
    def _caption_task(row: sqlite3.Row) -> dict[str, Any]:
        # Explicit allow-list: no image, surface, group, sampling, or judge keys.
        return {
            "assignment_id": row["assignment_id"],
            "phase": "caption",
            "position": row["position"],
            "caption": row["caption"],
            "unit": row["unit"],
            "span": row["span_text"],
            "target": row["target"],
            "category": row["category"],
        }

    @staticmethod
    def _image_task(row: sqlite3.Row) -> dict[str, Any]:
        # Explicit allow-list: no item ID, asset path, caption, span, surface,
        # group, sampling metadata, or judges.
        return {
            "assignment_id": row["assignment_id"],
            "phase": "image",
            "position": row["position"],
            "unit": row["unit"],
            "target": row["target"],
            "category": row["category"],
            "image_url": f"/api/image/{row['assignment_id']}",
        }

    @staticmethod
    def _asset_reference(image_asset_json: str) -> str:
        asset = _json_loads(image_asset_json)
        if isinstance(asset, str):
            reference = asset
        elif isinstance(asset, Mapping):
            reference = next(
                (
                    asset[key]
                    for key in ("asset_relpath", "route", "path", "asset_id")
                    if isinstance(asset.get(key), str) and asset[key]
                ),
                None,
            )
        else:
            reference = None
        if not isinstance(reference, str) or not reference or "\x00" in reference:
            raise ValidationError(
                "image_asset must be a path/reference string or contain asset_relpath, route, path, or asset_id"
            )
        reference_path = Path(reference)
        if (
            reference_path.is_absolute()
            or ".." in reference_path.parts
            or "://" in reference
            or reference.startswith("\\")
        ):
            raise ValidationError("image_asset reference must be a safe relative asset path")
        return reference

    def authorized_image_asset(
        self,
        session_token: str,
        assignment_id: str,
        *,
        study_id: str | None = None,
    ) -> dict[str, Any]:
        """Return one resolved asset after bearer, ownership, and phase checks.

        The browser receives an image assignment ID from ``fetch_next_task``.
        It cannot use an item ID to enumerate assets, and it cannot retrieve an
        image before the matching caption judgment is complete and that image
        assignment has actually been presented.
        """

        assignment_id = _require_identifier(assignment_id, "assignment_id")
        with self._transaction(write=False) as connection:
            session = self._session_context(
                connection,
                session_token,
                study_id=study_id,
                require_open=True,
                require_full_scope=True,
            )
            if not self._has_current_consent(session):
                raise AuthorizationError("current consent is required before image retrieval")
            row = connection.execute(
                """
                SELECT a.assignment_id, a.item_id, a.phase, a.first_presented_at,
                       i.image_asset_json, i.asset_status
                FROM assignments AS a
                JOIN items AS i
                  ON i.study_id = a.study_id AND i.item_id = a.item_id
                WHERE a.assignment_id = ? AND a.study_id = ? AND a.participant_id = ?
                """,
                (assignment_id, session["study_id"], session["participant_id"]),
            ).fetchone()
            if row is None or row["phase"] != "image":
                raise AuthorizationError("image assignment does not belong to this participant")
            matching_caption_complete = connection.execute(
                """
                SELECT 1 FROM assignments
                WHERE study_id = ? AND participant_id = ?
                  AND item_id = ? AND phase = 'caption' AND status = 'completed'
                """,
                (session["study_id"], session["participant_id"], row["item_id"]),
            ).fetchone()
            if matching_caption_complete is None:
                raise AuthorizationError("the matching caption judgment must be completed before image retrieval")
            if row["first_presented_at"] is None:
                raise AuthorizationError("image assignment has not been presented")
            if row["asset_status"] != "available" or row["image_asset_json"] is None:
                raise NotFoundError("image asset is not available")
            return {
                "assignment_id": row["assignment_id"],
                "asset_ref": self._asset_reference(row["image_asset_json"]),
            }

    def authorized_coverage_caption(
        self,
        session_token: str,
        assignment_id: str,
        *,
        study_id: str | None = None,
    ) -> dict[str, str]:
        """Reveal the paired caption only for a presented image assignment.

        The participant UI calls this after locally locking the image-support
        answer.  The endpoint is intentionally separate from the image-task
        projection so the caption is not delivered with the initial
        image-only judgment screen.
        """

        assignment_id = _require_identifier(assignment_id, "assignment_id")
        with self._transaction(write=False) as connection:
            session = self._session_context(
                connection,
                session_token,
                study_id=study_id,
                require_open=True,
                require_full_scope=True,
            )
            if not self._has_current_consent(session):
                raise AuthorizationError("current consent is required before caption retrieval")
            row = connection.execute(
                """
                SELECT a.assignment_id, a.item_id, a.phase, a.first_presented_at,
                       i.caption
                FROM assignments AS a
                JOIN items AS i
                  ON i.study_id = a.study_id AND i.item_id = a.item_id
                WHERE a.assignment_id = ? AND a.study_id = ? AND a.participant_id = ?
                """,
                (assignment_id, session["study_id"], session["participant_id"]),
            ).fetchone()
            if row is None or row["phase"] != "image":
                raise AuthorizationError("image assignment does not belong to this participant")
            matching_caption_complete = connection.execute(
                """
                SELECT 1 FROM assignments
                WHERE study_id = ? AND participant_id = ?
                  AND item_id = ? AND phase = 'caption' AND status = 'completed'
                """,
                (session["study_id"], session["participant_id"], row["item_id"]),
            ).fetchone()
            if matching_caption_complete is None:
                raise AuthorizationError("the matching caption judgment must be completed before caption retrieval")
            if row["first_presented_at"] is None:
                raise AuthorizationError("image assignment has not been presented")
            return {
                "assignment_id": row["assignment_id"],
                "caption": row["caption"],
            }

    @staticmethod
    def _remaining_first_pool_incomplete(
        connection: sqlite3.Connection,
        study_id: str,
        labels_per_item: int,
    ) -> bool:
        row = connection.execute(
            """
            SELECT 1
            FROM items AS i
            LEFT JOIN assignments AS a
              ON a.study_id = i.study_id
             AND a.item_id = i.item_id
             AND a.phase = 'caption'
            LEFT JOIN participants AS p
              ON p.study_id = a.study_id AND p.participant_id = a.participant_id
            WHERE i.study_id = ? AND i.repeat_of IS NULL
            GROUP BY i.item_id
            HAVING COUNT(DISTINCT CASE
                WHEN p.status NOT IN ('declined', 'withdrawn')
                THEN a.participant_id
            END) < ?
            LIMIT 1
            """,
            (study_id, labels_per_item),
        ).fetchone()
        return row is not None

    def _claim_remaining_first_item(
        self,
        connection: sqlite3.Connection,
        *,
        study: sqlite3.Row,
        participant_id: str,
        enforce_target: bool = True,
        require_no_pending: bool = True,
    ) -> bool:
        """Atomically bind one least-covered base item and its paired tasks."""

        metadata = _json_loads(study["metadata_json"], {})
        if metadata.get(ASSIGNMENT_MODE_METADATA_KEY) != ASSIGNMENT_MODE_REMAINING_FIRST:
            return False
        roster = metadata.get(ASSIGNMENT_ROSTER_METADATA_KEY, {})
        current_primary = roster.get("current_primary_participant_ids", [])
        if participant_id not in current_primary:
            return False
        labels_per_item = study["labels_per_item"]
        seed = study["assignment_seed"]
        if not isinstance(labels_per_item, int) or labels_per_item < 2:
            raise HumanCBUStoreError("remaining-first study has no valid label target")
        if not isinstance(seed, str) or not seed:
            raise HumanCBUStoreError("remaining-first study has no assignment seed")
        target = metadata.get(TARGET_ITEMS_PER_PARTICIPANT_METADATA_KEY)
        if target is not None and (type(target) is not int or target <= 0):
            raise HumanCBUStoreError(
                "remaining-first study has an invalid participant item target"
            )
        overlap_target = metadata.get(OVERLAP_TARGET_PER_STRATUM_METADATA_KEY, 0)
        if type(overlap_target) is not int or overlap_target < 0:
            raise HumanCBUStoreError(
                "remaining-first study has an invalid overlap target per stratum"
            )
        if enforce_target and target is not None:
            assigned_items = connection.execute(
                """
                SELECT COUNT(*)
                FROM assignments AS a
                JOIN items AS i
                  ON i.study_id = a.study_id AND i.item_id = a.item_id
                WHERE a.study_id = ? AND a.participant_id = ?
                  AND a.phase = 'caption' AND i.repeat_of IS NULL
                """,
                (study["study_id"], participant_id),
            ).fetchone()[0]
            if int(assigned_items) >= target:
                return False
        pending_normal = connection.execute(
            """
            SELECT 1
            FROM assignments AS a
            JOIN items AS i
              ON i.study_id = a.study_id AND i.item_id = a.item_id
            WHERE a.study_id = ? AND a.participant_id = ?
              AND a.phase IN ('caption', 'image') AND a.status = 'pending'
              AND i.repeat_of IS NULL
            LIMIT 1
            """,
            (study["study_id"], participant_id),
        ).fetchone()
        if require_no_pending and pending_normal is not None:
            return False
        candidates = connection.execute(
            """
            WITH item_stats AS (
                SELECT base.item_id, base.stratum,
                       (
                           SELECT COUNT(DISTINCT assigned.participant_id)
                           FROM assignments AS assigned
                           JOIN participants AS assigned_participant
                             ON assigned_participant.study_id = assigned.study_id
                            AND assigned_participant.participant_id = assigned.participant_id
                           WHERE assigned.study_id = base.study_id
                             AND assigned.item_id = base.item_id
                             AND assigned.phase = 'caption'
                             AND assigned_participant.status NOT IN ('declined', 'withdrawn')
                       ) AS assigned_labels,
                       (
                           SELECT COUNT(DISTINCT completed.participant_id)
                           FROM assignments AS completed
                           JOIN participants AS completed_participant
                             ON completed_participant.study_id = completed.study_id
                            AND completed_participant.participant_id = completed.participant_id
                           WHERE completed.study_id = base.study_id
                             AND completed.item_id = base.item_id
                             AND completed.phase = 'image'
                             AND completed.status = 'completed'
                             AND completed_participant.status NOT IN ('declined', 'withdrawn')
                       ) AS completed_labels
                FROM items AS base
                WHERE base.study_id = ? AND base.repeat_of IS NULL
            ),
            stratum_stats AS (
                SELECT stratum,
                       SUM(CASE WHEN completed_labels >= 2 THEN 1 ELSE 0 END)
                           AS completed_overlap_items,
                       SUM(assigned_labels) AS assigned_slots
                FROM item_stats
                GROUP BY stratum
            )
            SELECT stats.item_id, stats.stratum, stats.assigned_labels,
                   stats.completed_labels,
                   COALESCE(strata.completed_overlap_items, 0)
                       AS stratum_completed_overlap_items,
                   COALESCE(strata.assigned_slots, 0) AS stratum_assigned_slots
            FROM item_stats AS stats
            LEFT JOIN stratum_stats AS strata ON strata.stratum = stats.stratum
            WHERE stats.assigned_labels < ?
              AND NOT EXISTS (
                  SELECT 1 FROM assignments AS own
                  WHERE own.study_id = ?
                    AND own.item_id = stats.item_id
                    AND own.phase = 'caption'
                    AND own.participant_id = ?
              )
            """,
            (study["study_id"], labels_per_item, study["study_id"], participant_id),
        ).fetchall()
        if not candidates:
            return False
        selected = min(
            candidates,
            key=lambda row: (
                0
                if overlap_target > 0
                and row["completed_labels"] == 1
                and row["stratum_completed_overlap_items"] < overlap_target
                else 1,
                row["stratum_completed_overlap_items"]
                if overlap_target > 0
                and row["completed_labels"] == 1
                and row["stratum_completed_overlap_items"] < overlap_target
                else row["stratum_assigned_slots"],
                row["assigned_labels"],
                _stable_digest(
                    seed,
                    "remaining-first",
                    str(row["assigned_labels"]),
                    participant_id,
                    row["item_id"],
                ),
                row["item_id"],
            ),
        )
        item_rows = connection.execute(
            """
            SELECT item_id, repeat_of
            FROM items
            WHERE study_id = ? AND (item_id = ? OR repeat_of = ?)
            ORDER BY CASE WHEN repeat_of IS NULL THEN 0 ELSE 1 END, item_id
            """,
            (study["study_id"], selected["item_id"], selected["item_id"]),
        ).fetchall()
        if not item_rows or item_rows[0]["item_id"] != selected["item_id"]:
            raise HumanCBUStoreError("remaining-first selected item disappeared")
        now = _timestamp(self._now())
        for phase in PHASES:
            normal_position = (
                connection.execute(
                    """
                    SELECT COALESCE(MAX(a.position), 0)
                    FROM assignments AS a
                    JOIN items AS i
                      ON i.study_id = a.study_id AND i.item_id = a.item_id
                    WHERE a.study_id = ? AND a.participant_id = ?
                      AND a.phase = ? AND i.repeat_of IS NULL
                    """,
                    (study["study_id"], participant_id, phase),
                ).fetchone()[0]
                + 1
            )
            extension_max = metadata.get(OPTIONAL_EXTENSION_MAX_ITEMS_METADATA_KEY)
            repeat_window_end = extension_max if not enforce_target else target
            for item in item_rows:
                repeat_ordinal = sum(
                    1
                    for candidate in item_rows
                    if candidate["repeat_of"] is not None
                    and candidate["item_id"] < item["item_id"]
                )
                position = (
                    normal_position
                    if item["repeat_of"] is None
                    else normal_position + _DYNAMIC_REPEAT_MIN_GAP + repeat_ordinal
                )
                if (
                    item["repeat_of"] is not None
                    and type(repeat_window_end) is int
                    and position > repeat_window_end
                ):
                    # A late repeat cannot satisfy the minimum memory gap before
                    # this participant's current work window ends. Omitting it is
                    # less biasing than exposing an obvious repeat block at the end.
                    continue
                assignment_id = "a_" + _stable_digest(
                    study["study_id"],
                    participant_id,
                    item["item_id"],
                    phase,
                )[:32]
                connection.execute(
                    """
                    INSERT INTO assignments(
                        assignment_id, study_id, participant_id, item_id,
                        phase, position, order_key, assigned_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        assignment_id,
                        study["study_id"],
                        participant_id,
                        item["item_id"],
                        phase,
                        position,
                        _stable_digest(seed, "order", phase, participant_id, item["item_id"]),
                        now,
                    ),
                )
        return True

    @staticmethod
    def _remaining_first_participant_at_target(
        connection: sqlite3.Connection,
        *,
        study: sqlite3.Row,
        participant_id: str,
    ) -> bool:
        metadata = _json_loads(study["metadata_json"], {})
        target = metadata.get(TARGET_ITEMS_PER_PARTICIPANT_METADATA_KEY)
        if target is None:
            return False
        if type(target) is not int or target <= 0:
            raise HumanCBUStoreError(
                "remaining-first study has an invalid participant item target"
            )
        assigned_items = connection.execute(
            """
            SELECT COUNT(*)
            FROM assignments AS a
            JOIN items AS i
              ON i.study_id = a.study_id AND i.item_id = a.item_id
            WHERE a.study_id = ? AND a.participant_id = ?
              AND a.phase = 'caption' AND i.repeat_of IS NULL
            """,
            (study["study_id"], participant_id),
        ).fetchone()[0]
        return int(assigned_items) >= target

    @staticmethod
    def _remaining_first_extension_status(
        connection: sqlite3.Connection,
        *,
        study: sqlite3.Row,
        participant_id: str,
    ) -> dict[str, Any] | None:
        metadata = _json_loads(study["metadata_json"], {})
        batch_size = metadata.get(OPTIONAL_EXTENSION_BATCH_SIZE_METADATA_KEY)
        max_items = metadata.get(OPTIONAL_EXTENSION_MAX_ITEMS_METADATA_KEY)
        target = metadata.get(TARGET_ITEMS_PER_PARTICIPANT_METADATA_KEY)
        if batch_size is None and max_items is None:
            return None
        if type(batch_size) is not int or batch_size <= 0:
            raise HumanCBUStoreError("remaining-first study has an invalid extension batch size")
        if type(max_items) is not int or max_items <= 0:
            raise HumanCBUStoreError("remaining-first study has an invalid extension maximum")
        if type(target) is not int or target <= 0 or max_items < target:
            raise HumanCBUStoreError("remaining-first study has an invalid extension range")
        assigned_items = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM assignments AS a
                JOIN items AS i
                  ON i.study_id = a.study_id AND i.item_id = a.item_id
                WHERE a.study_id = ? AND a.participant_id = ?
                  AND a.phase = 'caption' AND i.repeat_of IS NULL
                """,
                (study["study_id"], participant_id),
            ).fetchone()[0]
        )
        pending_items = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM assignments AS a
                JOIN items AS i
                  ON i.study_id = a.study_id AND i.item_id = a.item_id
                WHERE a.study_id = ? AND a.participant_id = ?
                  AND a.phase IN ('caption', 'image') AND a.status = 'pending'
                """,
                (study["study_id"], participant_id),
            ).fetchone()[0]
        )
        return {
            "batch_size": min(batch_size, max(0, max_items - assigned_items)),
            "max_items": max_items,
            "assigned_items": assigned_items,
            "can_extend": assigned_items >= target
            and assigned_items < max_items
            and pending_items == 0
            and HumanCBUStore._remaining_first_pool_incomplete(
                connection,
                study["study_id"],
                int(study["labels_per_item"]),
            ),
        }

    def extend_remaining_first_work(
        self,
        session_token: str,
        *,
        study_id: str | None = None,
    ) -> dict[str, Any]:
        """Opt an eligible participant into one bounded batch of extra base items."""

        with self._transaction(write=True) as connection:
            session = self._session_context(
                connection,
                session_token,
                study_id=study_id,
                require_open=True,
                require_full_scope=True,
            )
            if not self._has_current_consent(session):
                raise AuthorizationError("current consent is required to extend work")
            study = self._study_row(connection, session["study_id"])
            metadata = _json_loads(study["metadata_json"], {})
            if metadata.get(ASSIGNMENT_MODE_METADATA_KEY) != ASSIGNMENT_MODE_REMAINING_FIRST:
                raise StudyStateError("optional work is unavailable for this assignment mode")
            extension = self._remaining_first_extension_status(
                connection,
                study=study,
                participant_id=session["participant_id"],
            )
            if extension is None or not extension["can_extend"]:
                raise ConflictError("optional work is not currently available")
            granted = 0
            for _ in range(extension["batch_size"]):
                if not self._claim_remaining_first_item(
                    connection,
                    study=study,
                    participant_id=session["participant_id"],
                    enforce_target=False,
                    require_no_pending=False,
                ):
                    break
                granted += 1
            if granted == 0:
                raise ConflictError("no optional work remains")
            updated = self._remaining_first_extension_status(
                connection,
                study=study,
                participant_id=session["participant_id"],
            )
            return {"granted_items": granted, "work_extension": updated}

    def fetch_next_task(
        self,
        session_token: str,
        *,
        study_id: str | None = None,
    ) -> dict[str, Any]:
        """Return the next safe task projection or a non-task status."""

        with self._transaction(write=True) as connection:
            session = self._session_context(
                connection,
                session_token,
                study_id=study_id,
                require_open=True,
                require_full_scope=True,
            )
            progress = self._progress_for(connection, session["study_id"], session["participant_id"])
            if not self._has_current_consent(session):
                return {
                    "status": "consent_required",
                    "required_consent_version": session["required_consent_version"],
                    "progress": progress,
                }
            study = self._study_row(connection, session["study_id"])
            metadata = _json_loads(study["metadata_json"], {})
            remaining_first = (
                metadata.get(ASSIGNMENT_MODE_METADATA_KEY)
                == ASSIGNMENT_MODE_REMAINING_FIRST
            )
            if remaining_first:
                self._claim_remaining_first_item(
                    connection,
                    study=study,
                    participant_id=session["participant_id"],
                )
                progress = self._progress_for(
                    connection,
                    session["study_id"],
                    session["participant_id"],
                )
            if remaining_first:
                row = connection.execute(
                    """
                    SELECT a.*, i.caption, i.unit, i.span_text, i.span_start, i.span_end,
                           i.target, i.category, i.image_asset_json, i.asset_status,
                           i.repeat_of
                    FROM assignments AS a
                    JOIN items AS i
                      ON i.study_id = a.study_id AND i.item_id = a.item_id
                    LEFT JOIN assignments AS caption_assignment
                      ON caption_assignment.study_id = a.study_id
                     AND caption_assignment.participant_id = a.participant_id
                     AND caption_assignment.item_id = a.item_id
                     AND caption_assignment.phase = 'caption'
                    WHERE a.study_id = ? AND a.participant_id = ?
                      AND a.status = 'pending'
                      AND (
                          a.phase = 'caption'
                          OR (
                              a.phase = 'image'
                              AND caption_assignment.status = 'completed'
                          )
                      )
                    ORDER BY
                      CASE WHEN a.phase = 'image' THEN 0 ELSE 1 END,
                      a.position,
                      a.assignment_id
                    LIMIT 1
                    """,
                    (session["study_id"], session["participant_id"]),
                ).fetchone()
                if row is None:
                    if self._remaining_first_participant_at_target(
                        connection,
                        study=study,
                        participant_id=session["participant_id"],
                    ):
                        return {
                            "status": "complete",
                            "progress": progress,
                            "work_extension": self._remaining_first_extension_status(
                                connection,
                                study=study,
                                participant_id=session["participant_id"],
                            ),
                        }
                    if self._remaining_first_pool_incomplete(
                        connection,
                        session["study_id"],
                        int(study["labels_per_item"]),
                    ):
                        return {
                            "status": "waiting_for_peer_labels",
                            "progress": progress,
                        }
                    return {
                        "status": "complete",
                        "progress": progress,
                        "work_extension": self._remaining_first_extension_status(
                            connection,
                            study=study,
                            participant_id=session["participant_id"],
                        ),
                    }
                phase = row["phase"]
                progress["current_phase"] = phase
            else:
                phase = progress["current_phase"]
                if phase == "complete":
                    return {"status": "complete", "progress": progress}
                row = connection.execute(
                    """
                    SELECT a.*, i.caption, i.unit, i.span_text, i.span_start, i.span_end, i.target,
                           i.category, i.image_asset_json, i.asset_status
                    FROM assignments AS a
                    JOIN items AS i
                      ON i.study_id = a.study_id AND i.item_id = a.item_id
                    WHERE a.study_id = ? AND a.participant_id = ?
                      AND a.phase = ? AND a.status = 'pending'
                    ORDER BY a.position, a.assignment_id
                    LIMIT 1
                    """,
                    (session["study_id"], session["participant_id"], phase),
                ).fetchone()
            if row is None:
                raise HumanCBUStoreError("assignment progress is inconsistent")
            if phase == "image" and (row["asset_status"] != "available" or row["image_asset_json"] is None):
                return {"status": "waiting_for_assets", "progress": progress}
            presented_at = _timestamp(self._now())
            connection.execute(
                """
                UPDATE assignments
                SET first_presented_at = COALESCE(first_presented_at, ?),
                    last_presented_at = ?,
                    presentation_count = presentation_count + 1
                WHERE assignment_id = ?
                """,
                (presented_at, presented_at, row["assignment_id"]),
            )
            task = self._caption_task(row) if phase == "caption" else self._image_task(row)
            return {"status": "task", "phase": phase, "task": task, "progress": progress}

    def next_task(
        self,
        session_token: str,
        *,
        study_id: str | None = None,
    ) -> dict[str, Any]:
        return self.fetch_next_task(session_token, study_id=study_id)

    @staticmethod
    def _validate_common_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        reason_tags = payload.get("reason_tags", [])
        if not isinstance(reason_tags, list) or len(reason_tags) > 16:
            raise ValidationError("reason_tags must be a list of at most 16 tags")
        normalized_tags: list[str] = []
        for tag in reason_tags:
            if not isinstance(tag, str) or not _TAG_RE.fullmatch(tag):
                raise ValidationError("reason_tags must use short lowercase tag identifiers")
            if tag not in normalized_tags:
                normalized_tags.append(tag)
        note = payload.get("note")
        if note is not None:
            if not isinstance(note, str) or len(note) > 1_000:
                raise ValidationError("note must be a string of at most 1000 characters")
            if _EMAIL_RE.search(note) or _IPV4_RE.search(note) or _PHONE_RE.search(note) or _URL_RE.search(note):
                raise ValidationError("note must not contain contact or network identifiers")
        confidence = payload.get("confidence")
        if confidence is not None and (type(confidence) is not int or not 1 <= confidence <= 5):
            raise ValidationError("confidence must be an integer from 1 to 5")
        elapsed_ms = payload.get("elapsed_ms")
        if elapsed_ms is not None and (type(elapsed_ms) is not int or not 0 <= elapsed_ms <= 3_600_000):
            raise ValidationError("elapsed_ms must be an integer from 0 to 3600000")
        return {
            "reason_tags": normalized_tags,
            "note": note,
            "confidence": confidence,
            "elapsed_ms": elapsed_ms,
        }

    @classmethod
    def validate_annotation_payload(
        cls,
        phase: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate and canonicalize a phase-specific annotation payload."""

        if phase not in PHASES:
            raise ValidationError(f"invalid annotation phase: {phase!r}")
        if not isinstance(payload, Mapping):
            raise ValidationError("annotation payload must be an object")
        allowed = (
            set(CAPTION_REQUIRED_FIELDS) | set(CAPTION_OPTIONAL_FIELDS)
            if phase == "caption"
            else set(IMAGE_REQUIRED_FIELDS) | set(IMAGE_OPTIONAL_FIELDS)
        )
        unknown = set(payload) - allowed
        if unknown:
            raise ValidationError(f"fields are not allowed for {phase} phase: {', '.join(sorted(unknown))}")
        common = cls._validate_common_payload(payload)
        if phase == "caption":
            missing = set(CAPTION_REQUIRED_FIELDS) - set(payload)
            if missing:
                raise ValidationError("missing caption labels: " + ", ".join(sorted(missing)))
            if payload["caption_licensed"] not in CAPTION_LICENSED_LABELS:
                raise ValidationError("invalid caption_licensed label")
            if payload["atomic_visual_claim"] not in ATOMIC_VISUAL_CLAIM_LABELS:
                raise ValidationError("invalid atomic_visual_claim label")
            if payload["category_check"] not in CATEGORY_CHECK_LABELS:
                raise ValidationError("invalid category_check label")
            atomic_label = payload["atomic_visual_claim"]
            category_label = payload["category_check"]
            if atomic_label in {"not_visual", "not_atomic"} and category_label != "not_applicable":
                raise ValidationError("category_check must be 'not_applicable' for non-visual/non-atomic units")
            if atomic_label == "yes" and category_label == "not_applicable":
                raise ValidationError("category_check cannot be 'not_applicable' for an atomic visual claim")
            corrected_category_present = "corrected_category" in payload
            corrected_category = payload.get("corrected_category")
            if corrected_category is not None:
                corrected_category = _require_identifier(corrected_category, "corrected_category")
            if category_label == "incorrect" and corrected_category is None:
                raise ValidationError("corrected_category is required when category_check is 'incorrect'")
            if category_label != "incorrect" and corrected_category_present:
                raise ValidationError("corrected_category is only allowed when category_check is 'incorrect'")
            return {
                "caption_licensed": payload["caption_licensed"],
                "atomic_visual_claim": payload["atomic_visual_claim"],
                "category_check": payload["category_check"],
                "corrected_category": corrected_category,
                **common,
            }
        missing = set(IMAGE_REQUIRED_FIELDS) - set(payload)
        if missing:
            raise ValidationError("missing image labels: " + ", ".join(sorted(missing)))
        if payload["image_support"] not in IMAGE_SUPPORT_LABELS:
            raise ValidationError("invalid image_support label")
        control_usefulness = payload.get("control_usefulness")
        if control_usefulness is not None and control_usefulness != "cannot_judge":
            if type(control_usefulness) is not int or not 1 <= control_usefulness <= 5:
                raise ValidationError("control_usefulness must be 1..5, null, or 'cannot_judge'")
        if payload["image_support"] != "yes" and control_usefulness is not None:
            raise ValidationError("control_usefulness must be null unless image_support is 'yes'")
        salient_coverage = payload.get("salient_coverage")
        if salient_coverage is not None:
            if type(salient_coverage) is not int or not 1 <= salient_coverage <= 5:
                raise ValidationError("salient_coverage must be an integer from 1 to 5")
        coverage_allowed = payload["image_support"] not in {
            "image_unavailable",
            "prefer_not_to_answer",
        }
        if not coverage_allowed and salient_coverage is not None:
            raise ValidationError(
                "salient_coverage must be null for unavailable or skipped images"
            )
        return {
            "image_support": payload["image_support"],
            "control_usefulness": control_usefulness,
            "salient_coverage": salient_coverage,
            **common,
        }

    @staticmethod
    def _annotation_columns(phase: str, payload: Mapping[str, Any]) -> tuple[Any, ...]:
        if phase == "caption":
            caption_licensed = payload["caption_licensed"]
            atomic_visual_claim = payload["atomic_visual_claim"]
            category_check = payload["category_check"]
            corrected_category = payload.get("corrected_category")
            image_support = None
            control_usefulness = None
        else:
            caption_licensed = None
            atomic_visual_claim = None
            category_check = None
            corrected_category = None
            image_support = payload["image_support"]
            usefulness = payload.get("control_usefulness")
            control_usefulness = None if usefulness is None else str(usefulness)
        return (
            _json_dumps(payload),
            caption_licensed,
            atomic_visual_claim,
            category_check,
            corrected_category,
            image_support,
            control_usefulness,
            _json_dumps(payload.get("reason_tags", [])),
            payload.get("note"),
            payload.get("confidence"),
            payload.get("elapsed_ms"),
        )

    def save_annotation(
        self,
        session_token: str,
        assignment_id: str,
        payload: Mapping[str, Any],
        *,
        study_id: str | None = None,
        elapsed_ms: int | None = None,
        duration_ms: int | None = None,
    ) -> dict[str, Any]:
        """Save the latest annotation and append an immutable revision event."""

        assignment_id = _require_identifier(assignment_id, "assignment_id")
        if elapsed_ms is not None and duration_ms is not None:
            raise ValidationError("provide elapsed_ms or duration_ms, not both")
        normalized_input = dict(payload)
        external_elapsed = elapsed_ms if elapsed_ms is not None else duration_ms
        if external_elapsed is not None:
            if "elapsed_ms" in normalized_input:
                raise ValidationError("elapsed_ms was provided both in payload and as an argument")
            normalized_input["elapsed_ms"] = external_elapsed
        now = _timestamp(self._now())
        with self._transaction(write=True) as connection:
            session = self._session_context(
                connection,
                session_token,
                study_id=study_id,
                require_open=True,
                require_full_scope=True,
            )
            if not self._has_current_consent(session):
                raise AuthorizationError("current consent is required before annotation")
            assignment = connection.execute(
                """
                SELECT a.*, i.asset_status, i.image_asset_json, i.category
                FROM assignments AS a
                JOIN items AS i
                  ON i.study_id = a.study_id AND i.item_id = a.item_id
                WHERE a.assignment_id = ? AND a.study_id = ? AND a.participant_id = ?
                """,
                (assignment_id, session["study_id"], session["participant_id"]),
            ).fetchone()
            if assignment is None:
                raise AuthorizationError("assignment does not belong to this participant")
            phase = assignment["phase"]
            if phase == "image":
                study_metadata = _json_loads(session["study_metadata_json"], {})
                remaining_first = (
                    study_metadata.get(ASSIGNMENT_MODE_METADATA_KEY)
                    == ASSIGNMENT_MODE_REMAINING_FIRST
                )
                if remaining_first:
                    matching_caption = connection.execute(
                        """
                        SELECT status FROM assignments
                        WHERE study_id = ? AND participant_id = ?
                          AND item_id = ? AND phase = 'caption'
                        """,
                        (
                            session["study_id"],
                            session["participant_id"],
                            assignment["item_id"],
                        ),
                    ).fetchone()
                    if matching_caption is None or matching_caption["status"] != "completed":
                        raise AuthorizationError(
                            "the matching caption judgment must be completed before image labels"
                        )
                else:
                    incomplete_caption_count = connection.execute(
                        """
                        SELECT COUNT(*) FROM assignments
                        WHERE study_id = ? AND participant_id = ?
                          AND phase = 'caption' AND status != 'completed'
                        """,
                        (session["study_id"], session["participant_id"]),
                    ).fetchone()[0]
                    if incomplete_caption_count:
                        raise AuthorizationError(
                            "all caption-phase assignments must be completed before image labels"
                        )
                if assignment["first_presented_at"] is None:
                    raise AuthorizationError("image assignment has not been presented")
                if assignment["asset_status"] != "available" or assignment["image_asset_json"] is None:
                    raise AuthorizationError("image asset is not available")
            normalized = self.validate_annotation_payload(phase, normalized_input)
            if phase == "caption":
                _validate_human_cbu_v1_caption_correction(
                    protocol_version=session["study_protocol_version"],
                    metadata_json=session["study_metadata_json"],
                    proposed_category=assignment["category"],
                    normalized=normalized,
                )
            existing = connection.execute(
                "SELECT * FROM annotations WHERE assignment_id = ?",
                (assignment_id,),
            ).fetchone()
            if existing is None and assignment["status"] != "pending":
                raise HumanCBUStoreError("completed assignment has no latest annotation")
            if existing is not None and phase == "caption":
                image_phase_started = connection.execute(
                    """
                    SELECT 1 FROM assignments
                    WHERE study_id = ? AND participant_id = ? AND phase = 'image'
                      AND first_presented_at IS NOT NULL
                    LIMIT 1
                    """,
                    (session["study_id"], session["participant_id"]),
                ).fetchone()
                if image_phase_started is not None:
                    raise AuthorizationError("caption annotations are frozen after the image phase is presented")
            if existing is None:
                if phase == "image" and remaining_first:
                    next_pending = connection.execute(
                        """
                        SELECT pending.assignment_id
                        FROM assignments AS pending
                        JOIN assignments AS matching_caption
                          ON matching_caption.study_id = pending.study_id
                         AND matching_caption.participant_id = pending.participant_id
                         AND matching_caption.item_id = pending.item_id
                         AND matching_caption.phase = 'caption'
                        WHERE pending.study_id = ? AND pending.participant_id = ?
                          AND pending.phase = 'image' AND pending.status = 'pending'
                          AND matching_caption.status = 'completed'
                        ORDER BY pending.position, pending.assignment_id
                        LIMIT 1
                        """,
                        (session["study_id"], session["participant_id"]),
                    ).fetchone()
                else:
                    next_pending = connection.execute(
                        """
                        SELECT assignment_id FROM assignments
                        WHERE study_id = ? AND participant_id = ? AND phase = ?
                          AND status = 'pending'
                        ORDER BY position, assignment_id LIMIT 1
                        """,
                        (session["study_id"], session["participant_id"], phase),
                    ).fetchone()
                if next_pending is None or next_pending["assignment_id"] != assignment_id:
                    raise AuthorizationError("annotations must follow the assigned deterministic order")
            annotation_id = f"n_{uuid.uuid4().hex}" if existing is None else existing["annotation_id"]
            revision = 1 if existing is None else int(existing["revision"]) + 1
            values = self._annotation_columns(phase, normalized)
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO annotations(
                        annotation_id, assignment_id, study_id, participant_id,
                        item_id, phase, payload_json, caption_licensed,
                        atomic_visual_claim, category_check, corrected_category,
                        image_support, control_usefulness, reason_tags_json,
                        note, confidence, elapsed_ms, revision, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        annotation_id,
                        assignment_id,
                        session["study_id"],
                        session["participant_id"],
                        assignment["item_id"],
                        phase,
                        *values,
                        revision,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """
                    UPDATE assignments
                    SET status = 'completed', completed_at = ?
                    WHERE assignment_id = ?
                    """,
                    (now, assignment_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE annotations
                    SET payload_json = ?, caption_licensed = ?,
                        atomic_visual_claim = ?, category_check = ?,
                        corrected_category = ?, image_support = ?,
                        control_usefulness = ?, reason_tags_json = ?,
                        note = ?, confidence = ?, elapsed_ms = ?,
                        revision = ?, updated_at = ?
                    WHERE annotation_id = ?
                    """,
                    (*values, revision, now, annotation_id),
                )
            connection.execute(
                """
                INSERT INTO annotation_events(
                    annotation_id, assignment_id, study_id, participant_id,
                    item_id, phase, payload_json, caption_licensed,
                    atomic_visual_claim, category_check, corrected_category,
                    image_support, control_usefulness, reason_tags_json,
                    note, confidence, elapsed_ms, revision, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    annotation_id,
                    assignment_id,
                    session["study_id"],
                    session["participant_id"],
                    assignment["item_id"],
                    phase,
                    *values,
                    revision,
                    now,
                ),
            )
            return {
                "annotation_id": annotation_id,
                "assignment_id": assignment_id,
                "item_id": assignment["item_id"],
                "phase": phase,
                "payload": normalized,
                "revision": revision,
                "created_at": now if existing is None else existing["created_at"],
                "updated_at": now,
            }

    def annotation_events(
        self,
        study_id: str,
        *,
        assignment_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return immutable revision snapshots for administrative export/tests."""

        parameters: list[Any] = [study_id]
        condition = ""
        if assignment_id is not None:
            condition = " AND assignment_id = ?"
            parameters.append(assignment_id)
        with self._transaction(write=False) as connection:
            self._study_row(connection, study_id)
            rows = connection.execute(
                f"""
                SELECT event_id, annotation_id, assignment_id, item_id, phase,
                       payload_json, revision, created_at
                FROM annotation_events
                WHERE study_id = ? {condition}
                ORDER BY event_id
                """,
                parameters,
            ).fetchall()
            return [
                {
                    "event_id": row["event_id"],
                    "annotation_id": row["annotation_id"],
                    "assignment_id": row["assignment_id"],
                    "item_id": row["item_id"],
                    "phase": row["phase"],
                    "payload": _json_loads(row["payload_json"]),
                    "revision": row["revision"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    @staticmethod
    def _agreement_key(phase: str, payload: Mapping[str, Any]) -> tuple[Any, ...]:
        if phase == "caption":
            corrected_category = (
                payload.get("corrected_category") if payload.get("category_check") == "incorrect" else None
            )
            return (
                *(payload.get(field) for field in CAPTION_REQUIRED_FIELDS),
                corrected_category,
            )
        # Control usefulness is optional, exploratory ordinal data.  It is
        # analyzed as a raw-rating distribution and must not create a primary
        # support disagreement or force adjudication.
        return tuple(payload.get(field) for field in IMAGE_REQUIRED_FIELDS)

    @staticmethod
    def _frozen_adjudicator_pseudonym(study: sqlite3.Row) -> str:
        metadata = _json_loads(study["metadata_json"], {})
        pseudonym = metadata.get("adjudicator_pseudonym")
        if not isinstance(pseudonym, str) or not _PSEUDONYM_RE.fullmatch(pseudonym):
            raise ValidationError("human_cbu_v1 study metadata must freeze a valid adjudicator_pseudonym")
        return pseudonym

    @staticmethod
    def _require_adjudicator_provisioned(study: sqlite3.Row) -> str:
        digest = study["adjudicator_credential_digest"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValidationError("human_cbu_v1 adjudication requires a provisioned adjudicator credential")
        return digest

    @classmethod
    def _verify_adjudicator_credential(
        cls,
        study: sqlite3.Row,
        credential: str | None,
    ) -> None:
        expected = cls._require_adjudicator_provisioned(study)
        if not isinstance(credential, str) or not credential:
            raise AuthenticationError("adjudicator credential is required")
        observed = _token_digest(
            credential,
            domain=f"adjudicator:{study['study_id']}",
        )
        if not secrets.compare_digest(expected, observed):
            raise AuthenticationError("adjudicator credential is invalid")

    @classmethod
    def _adjudication_freeze_digest(
        cls,
        study: sqlite3.Row,
    ) -> str:
        metadata = _json_loads(study["metadata_json"], {})
        sample_manifest_sha256 = metadata.get("sample_manifest_sha256")
        if not isinstance(sample_manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sample_manifest_sha256):
            raise ValidationError("human_cbu_v1 adjudication requires the frozen sample manifest SHA-256")
        if not study["validation_sealed_at"] or not study["assignment_participant_digest"]:
            raise ValidationError("human_cbu_v1 adjudication requires a sealed assignment and sample freeze")
        frozen_projection = {
            "adjudicator_credential_digest": cls._require_adjudicator_provisioned(study),
            "assignment_participant_digest": study["assignment_participant_digest"],
            "assignment_seed": study["assignment_seed"],
            "consent_version": study["consent_version"],
            "labels_per_item": study["labels_per_item"],
            "protocol_version": study["protocol_version"],
            "sample_manifest_sha256": sample_manifest_sha256,
            "study_id": study["study_id"],
            "validation_sealed_at": study["validation_sealed_at"],
        }
        return hashlib.sha256(_json_dumps(frozen_projection).encode("utf-8")).hexdigest()

    def _adjudication_packet_candidates(
        self,
        connection: sqlite3.Connection,
        study: sqlite3.Row,
    ) -> list[dict[str, Any]]:
        """Build current unresolved, phase-safe packet projections.

        The database identifiers and initial labels stay in the internal
        candidate record.  Only ``projection`` plus its SHA-256 may leave this
        method through :meth:`adjudication_packets`.
        """

        expected_labels = study["labels_per_item"]
        if not isinstance(expected_labels, int) or expected_labels < 2:
            raise ValidationError("adjudication requires a valid planned initial panel")
        freeze_digest = self._adjudication_freeze_digest(study)
        rows = connection.execute(
            """
            SELECT i.item_id, i.caption, i.unit, i.span_text, i.target, i.category,
                   i.image_asset_json, i.asset_status,
                   a.phase, a.payload_json,
                   p.pseudonym AS participant_pseudonym
            FROM items AS i
            JOIN annotations AS a
              ON a.study_id = i.study_id AND a.item_id = i.item_id
            JOIN participants AS p
              ON p.study_id = a.study_id AND p.participant_id = a.participant_id
            LEFT JOIN adjudications AS d
              ON d.study_id = a.study_id
             AND d.item_id = a.item_id
             AND d.phase = a.phase
            WHERE i.study_id = ?
              AND p.status = 'active'
              AND p.consented = 1
              AND p.consent_version = ?
              AND d.adjudication_id IS NULL
            ORDER BY i.item_id, a.phase, p.pseudonym
            """,
            (study["study_id"], study["consent_version"]),
        ).fetchall()
        panels: dict[tuple[str, str], list[sqlite3.Row]] = {}
        for row in rows:
            panels.setdefault((row["item_id"], row["phase"]), []).append(row)

        candidates: list[dict[str, Any]] = []
        for (item_id, phase), panel in panels.items():
            if len(panel) != expected_labels:
                continue
            keys = {self._agreement_key(phase, _json_loads(row["payload_json"])) for row in panel}
            if len(keys) < 2:
                continue
            item = panel[0]
            packet_id = "ap_" + _stable_digest(
                _ADJUDICATION_PACKET_ID_DOMAIN,
                freeze_digest,
                item_id,
                phase,
            )
            if phase == "caption":
                evidence = {
                    "unit": item["unit"],
                    "target": item["target"],
                    "category": item["category"],
                    "caption": item["caption"],
                    "span": item["span_text"],
                }
            else:
                if item["asset_status"] != "available" or item["image_asset_json"] is None:
                    raise ValidationError("image adjudication packet requires an available authorized image asset")
                image_asset = _json_loads(item["image_asset_json"])
                if not isinstance(image_asset, Mapping):
                    raise ValidationError("image adjudication packet requires frozen image asset metadata")
                image_sha256 = image_asset.get("sha256")
                if not isinstance(image_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", image_sha256):
                    raise ValidationError("image adjudication packet requires the frozen materialization SHA-256")
                image_asset_ref = self._asset_reference(item["image_asset_json"])
                image_suffix = PurePosixPath(image_asset_ref).suffix.lower()
                if image_suffix not in _ADJUDICATION_IMAGE_SUFFIXES:
                    raise ValidationError("image adjudication packet requires a supported frozen image suffix")
                image_file = f"image{image_suffix}"
                evidence = {
                    "unit": item["unit"],
                    "target": item["target"],
                    "category": item["category"],
                    "image_file": image_file,
                    "image_sha256": image_sha256,
                }
            projection = {
                "packet_version": ADJUDICATION_PACKET_VERSION,
                "packet_id": packet_id,
                "phase": phase,
                "evidence": evidence,
            }
            candidates.append(
                {
                    "item_id": item_id,
                    "phase": phase,
                    "projection": projection,
                    "packet_sha256": hashlib.sha256(_json_dumps(projection).encode("utf-8")).hexdigest(),
                    "initial_rater_pseudonyms": frozenset(row["participant_pseudonym"].casefold() for row in panel),
                    "image_source_ref": (image_asset_ref if phase == "image" else None),
                }
            )
        phase_order = {value: index for index, value in enumerate(PHASES)}
        return sorted(
            candidates,
            key=lambda candidate: (
                phase_order[candidate["phase"]],
                candidate["projection"]["packet_id"],
            ),
        )

    def adjudication_packets(
        self,
        study_id: str,
        phase: str,
    ) -> list[dict[str, Any]]:
        """Return deterministic blinded packets for one adjudication phase."""

        return self.adjudication_packet_bundle_inputs(study_id, phase)["packets"]

    def adjudication_packet_bundle_inputs(
        self,
        study_id: str,
        phase: str,
    ) -> dict[str, Any]:
        """Return phase-safe packets plus an operator-only image source map.

        ``image_sources`` is private local input for byte-copying and must
        never be serialized into the adjudicator-facing packet JSON.
        """

        if phase not in PHASES:
            raise ValidationError(f"invalid adjudication phase: {phase!r}")
        with self._transaction(write=False) as connection:
            study = self._study_row(connection, study_id)
            self._require_study_status(study, {"closed"})
            if study["protocol_version"] != HUMAN_CBU_V1_PROTOCOL:
                raise ValidationError("blinded adjudication packets are defined for human_cbu_v1 studies")
            self._frozen_adjudicator_pseudonym(study)
            self._require_adjudicator_provisioned(study)
            all_candidates = self._adjudication_packet_candidates(connection, study)
            if phase == "image" and any(candidate["phase"] == "caption" for candidate in all_candidates):
                raise StudyStateError(
                    "image adjudication packets remain blocked until every caption disagreement is adjudicated"
                )
            candidates = [candidate for candidate in all_candidates if candidate["phase"] == phase]
            packets = [
                {
                    **candidate["projection"],
                    "packet_sha256": candidate["packet_sha256"],
                }
                for candidate in candidates
            ]
            image_sources = {
                candidate["projection"]["packet_id"]: candidate["image_source_ref"]
                for candidate in candidates
                if candidate["phase"] == "image"
            }
            return {
                "packets": packets,
                "image_sources": image_sources,
            }

    @staticmethod
    def _validate_reviewed_packet(
        packet: Any,
        *,
        phase: str,
    ) -> tuple[dict[str, Any], str]:
        packet_keys = {
            "packet_version",
            "packet_id",
            "packet_sha256",
            "phase",
            "evidence",
        }
        if not isinstance(packet, Mapping) or set(packet) != packet_keys:
            raise ValidationError("reviewed packet violates the exact top-level allowlist")
        packet_id = packet["packet_id"]
        if not isinstance(packet_id, str) or not re.fullmatch(r"ap_[0-9a-f]{64}", packet_id):
            raise ValidationError("reviewed packet_id is invalid")
        if packet["packet_version"] != ADJUDICATION_PACKET_VERSION:
            raise ValidationError("reviewed packet version mismatch")
        if packet["phase"] != phase:
            raise ValidationError("reviewed packet phase does not match the response")
        evidence = packet["evidence"]
        if not isinstance(evidence, Mapping):
            raise ValidationError("reviewed packet evidence must be an object")
        caption_keys = {"unit", "target", "category", "caption", "span"}
        image_keys = {"unit", "target", "category", "image_file", "image_sha256"}
        if phase == "caption":
            if set(evidence) != caption_keys:
                raise ValidationError("caption packet violates the evidence allowlist")
        else:
            if set(evidence) != image_keys:
                raise ValidationError("image packet violates the evidence allowlist")
            image_sha256 = evidence["image_sha256"]
            if not isinstance(image_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", image_sha256):
                raise ValidationError("image packet has an invalid frozen SHA-256")
            image_file = evidence["image_file"]
            if not isinstance(image_file, str) or "\\" in image_file:
                raise ValidationError("image packet has an unsafe packet-local filename")
            image_path = PurePosixPath(image_file)
            if (
                image_path.is_absolute()
                or len(image_path.parts) != 1
                or image_path.name != f"image{image_path.suffix}"
                or image_path.suffix.lower() not in _ADJUDICATION_IMAGE_SUFFIXES
            ):
                raise ValidationError("image packet has an unsafe packet-local filename")
        projection = {key: packet[key] for key in ("packet_version", "packet_id", "phase", "evidence")}
        packet_sha256 = hashlib.sha256(_json_dumps(projection).encode("utf-8")).hexdigest()
        if not secrets.compare_digest(packet["packet_sha256"], packet_sha256):
            raise ValidationError("reviewed packet SHA-256 does not match its evidence")
        return dict(projection), packet_sha256

    @staticmethod
    def _validate_adjudication_correction(
        study: sqlite3.Row,
        packet: Mapping[str, Any],
        normalized: Mapping[str, Any],
    ) -> None:
        if packet["phase"] != "caption":
            return
        _validate_human_cbu_v1_caption_correction(
            protocol_version=study["protocol_version"],
            metadata_json=study["metadata_json"],
            proposed_category=packet["evidence"]["category"],
            normalized=normalized,
        )

    @staticmethod
    def _private_bundle_directory(path: str | Path) -> Path:
        raw = Path(path)
        try:
            metadata = raw.lstat()
        except OSError as error:
            raise ValidationError("adjudication bundle directory does not exist") from error
        if not stat.S_ISDIR(metadata.st_mode) or raw.is_symlink():
            raise ValidationError("adjudication bundle must be a non-symlink directory")
        if metadata.st_mode & 0o077:
            raise ValidationError("adjudication bundle must not grant group or world permissions")
        return raw.resolve(strict=True)

    @staticmethod
    def _private_bundle_child_directory(
        bundle: Path,
        name: str,
        *,
        label: str,
    ) -> Path:
        child = bundle / name
        try:
            metadata = child.lstat()
            resolved = child.resolve(strict=True)
        except OSError as error:
            raise ValidationError(f"adjudication bundle has no private {label} directory") from error
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or child.is_symlink()
            or metadata.st_mode & 0o077
            or resolved != child
            or resolved.parent != bundle
        ):
            raise ValidationError(f"adjudication {label} directory is not a private packet-local directory")
        return resolved

    def adjudicate_bundle_response(
        self,
        study_id: str,
        phase: str,
        *,
        bundle_dir: str | Path,
        response_path: str | Path,
        adjudicator_credential: str,
    ) -> dict[str, Any]:
        """Reject the retired shared ``packets.json`` adjudication protocol.

        Keeping the public method as an explicit rejection prevents an old
        operator script from silently downgrading a v2 packet-scoped study.
        """

        raise ValidationError(
            "shared adjudication bundles are retired; use adjudicate_packet_response with one private packet directory"
        )

        # The former v1 implementation intentionally remains unreachable in
        # this pre-release branch until the packet-scoped migration is fully
        # reviewed, making accidental downgrade impossible.
        if phase not in PHASES:
            raise ValidationError(f"invalid adjudication phase: {phase!r}")
        bundle = self._private_bundle_directory(bundle_dir)
        responses_dir = self._private_bundle_child_directory(
            bundle,
            "responses",
            label="responses",
        )
        supplied_response = Path(response_path)
        if not supplied_response.is_absolute():
            supplied_response = Path.cwd() / supplied_response
        supplied_response = supplied_response.absolute()
        if supplied_response.parent != responses_dir:
            raise ValidationError("adjudication response must be directly inside bundle/responses")
        packet_file = bundle / "packets.json"
        now = _timestamp(self._now())
        with self._transaction(write=True) as connection, ExitStack() as files:
            study = self._study_row(connection, study_id)
            self._require_study_status(study, {"closed"})
            if study["protocol_version"] != HUMAN_CBU_V1_PROTOCOL:
                raise ValidationError("bundle-bound adjudication is defined for human_cbu_v1 studies")
            self._verify_adjudicator_credential(study, adjudicator_credential)
            frozen_adjudicator = self._frozen_adjudicator_pseudonym(study)

            response_file = files.enter_context(
                _open_stable_private_file(
                    supplied_response,
                    label="adjudication response",
                    maximum_bytes=_MAX_ADJUDICATION_RESPONSE_BYTES,
                )
            )
            response = _strict_json_bytes(
                response_file["raw"],
                label="adjudication response",
            )
            response_keys = {
                "response_version",
                "phase",
                "packet_file_sha256",
                "packet_id",
                "packet_sha256",
                "decision",
            }
            if not isinstance(response, Mapping) or set(response) != response_keys:
                raise ValidationError("adjudication response violates the exact envelope allowlist")
            if response["response_version"] != ADJUDICATION_RESPONSE_VERSION:
                raise ValidationError("adjudication response version mismatch")
            if response["phase"] != phase:
                raise ValidationError("adjudication response phase mismatch")
            packet_id = response["packet_id"]
            packet_sha256 = response["packet_sha256"]
            packet_file_sha256 = response["packet_file_sha256"]
            for label, value in (
                ("packet_id", packet_id),
                ("packet_sha256", packet_sha256),
                ("packet_file_sha256", packet_file_sha256),
            ):
                pattern = r"ap_[0-9a-f]{64}" if label == "packet_id" else r"[0-9a-f]{64}"
                if not isinstance(value, str) or not re.fullmatch(pattern, value):
                    raise ValidationError(f"adjudication response {label} is invalid")
            expected_response_name = f"{packet_id}.response.json"
            if supplied_response.name != expected_response_name:
                raise ValidationError("adjudication response filename does not match packet_id")

            packet_file_record = files.enter_context(
                _open_stable_private_file(
                    packet_file,
                    label="adjudication packet file",
                    maximum_bytes=_MAX_ADJUDICATION_PACKET_FILE_BYTES,
                )
            )
            if not secrets.compare_digest(
                packet_file_record["sha256"],
                packet_file_sha256,
            ):
                raise ValidationError("adjudication packet file SHA-256 does not match the reviewed response")
            bundle_payload = _strict_json_bytes(
                packet_file_record["raw"],
                label="adjudication packet file",
            )
            bundle_keys = {
                "bundle_version",
                "packet_version",
                "phase",
                "packets",
            }
            if not isinstance(bundle_payload, Mapping) or set(bundle_payload) != bundle_keys:
                raise ValidationError("adjudication packet file violates the exact bundle allowlist")
            if bundle_payload["bundle_version"] != ADJUDICATION_BUNDLE_VERSION:
                raise ValidationError("adjudication bundle version mismatch")
            if bundle_payload["packet_version"] != ADJUDICATION_PACKET_VERSION:
                raise ValidationError("adjudication bundle packet version mismatch")
            if bundle_payload["phase"] != phase:
                raise ValidationError("adjudication bundle phase mismatch")
            packets = bundle_payload["packets"]
            if not isinstance(packets, list):
                raise ValidationError("adjudication bundle packets must be a list")
            matching_packets = [
                packet for packet in packets if isinstance(packet, Mapping) and packet.get("packet_id") == packet_id
            ]
            if len(matching_packets) != 1:
                raise ValidationError("adjudication response packet_id must occur exactly once in the bundle")
            selected_packet = matching_packets[0]
            projection, observed_packet_sha256 = self._validate_reviewed_packet(
                selected_packet,
                phase=phase,
            )
            if not secrets.compare_digest(packet_sha256, observed_packet_sha256):
                raise ValidationError("adjudication response packet SHA-256 does not match the reviewed packet")

            image_file_record: Mapping[str, Any] | None = None
            observed_image_sha256: str | None = None
            if phase == "image":
                self._private_bundle_child_directory(
                    bundle,
                    "images",
                    label="images",
                )
                image_relpath = PurePosixPath(projection["evidence"]["image_file"])
                image_path = bundle / Path(*image_relpath.parts)
                image_file_record = files.enter_context(
                    _open_stable_private_file(
                        image_path,
                        label="adjudication packet image",
                        maximum_bytes=_MAX_ADJUDICATION_IMAGE_BYTES,
                    )
                )
                observed_image_sha256 = image_file_record["sha256"]
                if not secrets.compare_digest(
                    observed_image_sha256,
                    projection["evidence"]["image_sha256"],
                ):
                    raise ValidationError("adjudication packet image SHA-256 does not match reviewed evidence")

            all_candidates = self._adjudication_packet_candidates(connection, study)
            if phase == "image" and any(candidate["phase"] == "caption" for candidate in all_candidates):
                raise StudyStateError(
                    "image adjudication remains blocked until every caption disagreement is adjudicated"
                )
            candidate = next(
                (
                    value
                    for value in all_candidates
                    if secrets.compare_digest(
                        value["projection"]["packet_id"],
                        packet_id,
                    )
                ),
                None,
            )
            existing = connection.execute(
                """
                SELECT * FROM adjudications
                WHERE study_id = ? AND packet_id = ?
                """,
                (study_id, packet_id),
            ).fetchone()
            expected_packet = (
                None
                if candidate is None
                else {
                    **candidate["projection"],
                    "packet_sha256": candidate["packet_sha256"],
                }
            )
            if candidate is not None and selected_packet != expected_packet:
                raise ValidationError("reviewed packet does not match the frozen database projection")
            response_sha256 = response_file["sha256"]

            if existing is not None:
                exact_binding = (
                    existing["phase"] == phase
                    and existing["packet_version"] == ADJUDICATION_PACKET_VERSION
                    and secrets.compare_digest(existing["packet_sha256"], packet_sha256)
                    and secrets.compare_digest(
                        existing["packet_file_sha256"],
                        packet_file_sha256,
                    )
                    and existing["response_version"] == ADJUDICATION_RESPONSE_VERSION
                    and secrets.compare_digest(
                        existing["response_sha256"],
                        response_sha256,
                    )
                    and existing["image_sha256"] == observed_image_sha256
                )
                if not exact_binding:
                    raise ConflictError("packet already has a different adjudication response")

            decision = response["decision"]
            if phase == "image" and isinstance(decision, Mapping) and "control_usefulness" in decision:
                raise ValidationError("control_usefulness is not allowed in image adjudication")
            normalized = self.validate_annotation_payload(phase, decision)
            self._validate_adjudication_correction(study, selected_packet, normalized)

            if existing is not None:
                if existing["payload_json"] != _json_dumps(normalized):
                    raise ConflictError("packet already has a different adjudication response")
                for record, label in (
                    (response_file, "adjudication response"),
                    (packet_file_record, "adjudication packet file"),
                    (image_file_record, "adjudication packet image"),
                ):
                    if record is not None:
                        _assert_open_file_unchanged(
                            record,
                            label=label,
                            rehash=True,
                        )
                return {
                    "adjudication_id": existing["adjudication_id"],
                    "study_id": study_id,
                    "phase": phase,
                    "adjudicator_pseudonym": frozen_adjudicator,
                    "payload": normalized,
                    "packet_id": packet_id,
                    "packet_sha256": packet_sha256,
                    "response_sha256": response_sha256,
                    "revision": existing["revision"],
                    "created_at": existing["created_at"],
                    "updated_at": existing["updated_at"],
                    "idempotent_replay": True,
                }
            if candidate is None:
                raise ValidationError("adjudication packet is stale, unknown, or no longer an unresolved disagreement")
            if frozen_adjudicator.casefold() in candidate["initial_rater_pseudonyms"]:
                raise ValidationError("predeclared adjudicator must differ from every eligible initial rater")
            for record, label in (
                (response_file, "adjudication response"),
                (packet_file_record, "adjudication packet file"),
                (image_file_record, "adjudication packet image"),
            ):
                if record is not None:
                    _assert_open_file_unchanged(
                        record,
                        label=label,
                        rehash=True,
                    )

            adjudication_id = f"d_{uuid.uuid4().hex}"
            payload_json = _json_dumps(normalized)
            audit_values = (
                ADJUDICATION_PACKET_VERSION,
                packet_id,
                packet_sha256,
                packet_file_sha256,
                ADJUDICATION_RESPONSE_VERSION,
                response_sha256,
                observed_image_sha256,
            )
            connection.execute(
                """
                INSERT INTO adjudications(
                    adjudication_id, study_id, item_id, phase,
                    adjudicator_pseudonym, payload_json,
                    packet_version, packet_id, packet_sha256,
                    packet_file_sha256, response_version, response_sha256,
                    image_sha256, revision, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    adjudication_id,
                    study_id,
                    candidate["item_id"],
                    phase,
                    frozen_adjudicator,
                    payload_json,
                    *audit_values,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO adjudication_events(
                    adjudication_id, study_id, item_id, phase,
                    adjudicator_pseudonym, payload_json,
                    packet_version, packet_id, packet_sha256,
                    packet_file_sha256, response_version, response_sha256,
                    image_sha256, revision, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    adjudication_id,
                    study_id,
                    candidate["item_id"],
                    phase,
                    frozen_adjudicator,
                    payload_json,
                    *audit_values,
                    now,
                ),
            )
            # Keep every reviewed artifact open and re-hash it again after the
            # writes but before the transaction can commit.  A concurrent
            # replacement or in-place edit therefore rolls both inserts back.
            for record, label in (
                (response_file, "adjudication response"),
                (packet_file_record, "adjudication packet file"),
                (image_file_record, "adjudication packet image"),
            ):
                if record is not None:
                    _assert_open_file_unchanged(
                        record,
                        label=label,
                        rehash=True,
                    )
            return {
                "adjudication_id": adjudication_id,
                "study_id": study_id,
                "phase": phase,
                "adjudicator_pseudonym": frozen_adjudicator,
                "payload": normalized,
                "packet_id": packet_id,
                "packet_sha256": packet_sha256,
                "packet_file_sha256": packet_file_sha256,
                "response_sha256": response_sha256,
                "image_sha256": observed_image_sha256,
                "revision": 1,
                "created_at": now,
                "updated_at": now,
                "idempotent_replay": False,
            }

    @staticmethod
    def _private_adjudication_packet_directory(path: str | Path) -> Path:
        raw = Path(path)
        if not raw.is_absolute():
            raw = Path.cwd() / raw
        raw = raw.absolute()
        packets_directory = raw.parent
        phase_root = packets_directory.parent
        if packets_directory.name != "packets":
            raise ValidationError("adjudication packet directory must be directly inside <phase-root>/packets")
        for candidate, label in (
            (phase_root, "phase root"),
            (packets_directory, "packets directory"),
            (raw, "packet directory"),
        ):
            try:
                metadata = candidate.lstat()
                resolved = candidate.resolve(strict=True)
            except OSError as error:
                raise ValidationError(f"adjudication {label} does not exist") from error
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or candidate.is_symlink()
                or metadata.st_mode & 0o077
                or resolved != candidate
            ):
                raise ValidationError(f"adjudication {label} must be a private canonical non-symlink directory")
        if not re.fullmatch(r"ap_[0-9a-f]{64}", raw.name):
            raise ValidationError("adjudication packet directory name is not a valid packet_id")
        return raw

    def adjudicate_packet_response(
        self,
        study_id: str,
        phase: str,
        *,
        packet_dir: str | Path,
        adjudicator_credential: str,
    ) -> dict[str, Any]:
        """Record one packet-scoped response with its exact visible context.

        The response embeds the full packet reviewed by the adjudicator, while
        the packet directory basename, sidecar, deterministic review page,
        image bytes, and current database projection must all agree.  This
        removes independent evidence and response selectors.  It does not
        prove cognitive attention or detect a person deliberately copying only
        a label from another packet while viewing the correct current page.
        """

        if phase not in PHASES:
            raise ValidationError(f"invalid adjudication phase: {phase!r}")
        packet_directory = self._private_adjudication_packet_directory(packet_dir)
        packet_file = packet_directory / "packet.json"
        review_file = packet_directory / "review.html"
        response_path = packet_directory / "response.json"
        now = _timestamp(self._now())

        with self._transaction(write=True) as connection, ExitStack() as files:
            study = self._study_row(connection, study_id)
            self._require_study_status(study, {"closed"})
            if study["protocol_version"] != HUMAN_CBU_V1_PROTOCOL:
                raise ValidationError("packet-scoped adjudication is defined for human_cbu_v1 studies")
            self._verify_adjudicator_credential(study, adjudicator_credential)
            frozen_adjudicator = self._frozen_adjudicator_pseudonym(study)

            packet_file_record = files.enter_context(
                _open_stable_private_file(
                    packet_file,
                    label="adjudication packet sidecar",
                    maximum_bytes=_MAX_ADJUDICATION_PACKET_FILE_BYTES,
                )
            )
            selected_packet = _strict_json_bytes(
                packet_file_record["raw"],
                label="adjudication packet sidecar",
            )
            projection, packet_sha256 = self._validate_reviewed_packet(
                selected_packet,
                phase=phase,
            )
            packet_id = projection["packet_id"]
            if packet_directory.name != packet_id:
                raise ValidationError("adjudication packet directory name does not match packet_id")
            packet_file_sha256 = packet_file_record["sha256"]

            response_file = files.enter_context(
                _open_stable_private_file(
                    response_path,
                    label="adjudication response",
                    maximum_bytes=_MAX_ADJUDICATION_RESPONSE_BYTES,
                )
            )
            response = _strict_json_bytes(
                response_file["raw"],
                label="adjudication response",
            )
            response_keys = {
                "response_version",
                "review_version",
                "packet_file_sha256",
                "packet",
                "decision",
            }
            if not isinstance(response, Mapping) or set(response) != response_keys:
                raise ValidationError("adjudication response violates the exact envelope allowlist")
            if response["response_version"] != ADJUDICATION_RESPONSE_VERSION:
                raise ValidationError("adjudication response version mismatch")
            if response["review_version"] != ADJUDICATION_REVIEW_VERSION:
                raise ValidationError("adjudication review version mismatch")
            supplied_packet_file_sha256 = response["packet_file_sha256"]
            if (
                not isinstance(supplied_packet_file_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", supplied_packet_file_sha256)
                or not secrets.compare_digest(
                    supplied_packet_file_sha256,
                    packet_file_sha256,
                )
            ):
                raise ValidationError("adjudication response does not match the exact packet sidecar")
            if response["packet"] != selected_packet:
                raise ValidationError("adjudication response packet does not match its packet sidecar")

            metadata = _json_loads(study["metadata_json"], {})
            if not isinstance(metadata, Mapping):
                raise ValidationError("human_cbu_v1 study metadata is malformed")
            semantic_categories = _frozen_semantic_visual_claim_types(metadata)

            review_file_record = files.enter_context(
                _open_stable_private_file(
                    review_file,
                    label="adjudication review page",
                    maximum_bytes=_MAX_ADJUDICATION_REVIEW_BYTES,
                )
            )
            from audit_recap_t2i.human_cbu.adjudication import (
                render_adjudication_html,
            )

            expected_review = render_adjudication_html(
                selected_packet,
                semantic_categories=semantic_categories,
            ).encode("utf-8")
            if not secrets.compare_digest(
                review_file_record["raw"],
                expected_review,
            ):
                raise ValidationError("adjudication review page does not deterministically match its packet sidecar")

            image_file_record: Mapping[str, Any] | None = None
            observed_image_sha256: str | None = None
            allowed_entries = {"packet.json", "review.html", "response.json"}
            if phase == "image":
                image_file = projection["evidence"]["image_file"]
                allowed_entries.add(image_file)
                image_file_record = files.enter_context(
                    _open_stable_private_file(
                        packet_directory / image_file,
                        label="adjudication packet image",
                        maximum_bytes=_MAX_ADJUDICATION_IMAGE_BYTES,
                    )
                )
                observed_image_sha256 = image_file_record["sha256"]
                if not secrets.compare_digest(
                    observed_image_sha256,
                    projection["evidence"]["image_sha256"],
                ):
                    raise ValidationError("adjudication packet image SHA-256 does not match reviewed evidence")
            observed_entries = {entry.name for entry in packet_directory.iterdir()}
            if observed_entries != allowed_entries:
                raise ValidationError("adjudication packet directory contains missing or unexpected review artifacts")

            all_candidates = self._adjudication_packet_candidates(connection, study)
            if phase == "image" and any(candidate["phase"] == "caption" for candidate in all_candidates):
                raise StudyStateError(
                    "image adjudication remains blocked until every caption disagreement is adjudicated"
                )
            candidate = next(
                (
                    value
                    for value in all_candidates
                    if secrets.compare_digest(
                        value["projection"]["packet_id"],
                        packet_id,
                    )
                ),
                None,
            )
            existing = connection.execute(
                """
                SELECT * FROM adjudications
                WHERE study_id = ? AND packet_id = ?
                """,
                (study_id, packet_id),
            ).fetchone()
            expected_packet = (
                None
                if candidate is None
                else {
                    **candidate["projection"],
                    "packet_sha256": candidate["packet_sha256"],
                }
            )
            if candidate is not None and selected_packet != expected_packet:
                raise ValidationError("reviewed packet does not match the frozen database projection")
            response_sha256 = response_file["sha256"]

            if existing is not None:
                exact_binding = (
                    existing["phase"] == phase
                    and existing["packet_version"] == ADJUDICATION_PACKET_VERSION
                    and secrets.compare_digest(
                        existing["packet_sha256"],
                        packet_sha256,
                    )
                    and secrets.compare_digest(
                        existing["packet_file_sha256"],
                        packet_file_sha256,
                    )
                    and existing["response_version"] == ADJUDICATION_RESPONSE_VERSION
                    and secrets.compare_digest(
                        existing["response_sha256"],
                        response_sha256,
                    )
                    and existing["image_sha256"] == observed_image_sha256
                )
                if not exact_binding:
                    raise ConflictError("packet already has a different adjudication response")

            decision = response["decision"]
            if phase == "image" and isinstance(decision, Mapping) and "control_usefulness" in decision:
                raise ValidationError("control_usefulness is not allowed in image adjudication")
            normalized = self.validate_annotation_payload(phase, decision)
            self._validate_adjudication_correction(
                study,
                selected_packet,
                normalized,
            )

            artifact_records = (
                (response_file, "adjudication response"),
                (packet_file_record, "adjudication packet sidecar"),
                (review_file_record, "adjudication review page"),
                (image_file_record, "adjudication packet image"),
            )
            if existing is not None:
                if existing["payload_json"] != _json_dumps(normalized):
                    raise ConflictError("packet already has a different adjudication response")
                for record, label in artifact_records:
                    if record is not None:
                        _assert_open_file_unchanged(
                            record,
                            label=label,
                            rehash=True,
                        )
                return {
                    "adjudication_id": existing["adjudication_id"],
                    "study_id": study_id,
                    "phase": phase,
                    "adjudicator_pseudonym": frozen_adjudicator,
                    "payload": normalized,
                    "packet_id": packet_id,
                    "packet_sha256": packet_sha256,
                    "packet_file_sha256": packet_file_sha256,
                    "response_sha256": response_sha256,
                    "revision": existing["revision"],
                    "created_at": existing["created_at"],
                    "updated_at": existing["updated_at"],
                    "idempotent_replay": True,
                }
            if candidate is None:
                raise ValidationError("adjudication packet is stale, unknown, or no longer an unresolved disagreement")
            if frozen_adjudicator.casefold() in candidate["initial_rater_pseudonyms"]:
                raise ValidationError("predeclared adjudicator must differ from every eligible initial rater")
            for record, label in artifact_records:
                if record is not None:
                    _assert_open_file_unchanged(
                        record,
                        label=label,
                        rehash=True,
                    )

            adjudication_id = f"d_{uuid.uuid4().hex}"
            payload_json = _json_dumps(normalized)
            audit_values = (
                ADJUDICATION_PACKET_VERSION,
                packet_id,
                packet_sha256,
                packet_file_sha256,
                ADJUDICATION_RESPONSE_VERSION,
                response_sha256,
                observed_image_sha256,
            )
            connection.execute(
                """
                INSERT INTO adjudications(
                    adjudication_id, study_id, item_id, phase,
                    adjudicator_pseudonym, payload_json,
                    packet_version, packet_id, packet_sha256,
                    packet_file_sha256, response_version, response_sha256,
                    image_sha256, revision, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    adjudication_id,
                    study_id,
                    candidate["item_id"],
                    phase,
                    frozen_adjudicator,
                    payload_json,
                    *audit_values,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO adjudication_events(
                    adjudication_id, study_id, item_id, phase,
                    adjudicator_pseudonym, payload_json,
                    packet_version, packet_id, packet_sha256,
                    packet_file_sha256, response_version, response_sha256,
                    image_sha256, revision, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    adjudication_id,
                    study_id,
                    candidate["item_id"],
                    phase,
                    frozen_adjudicator,
                    payload_json,
                    *audit_values,
                    now,
                ),
            )
            for record, label in artifact_records:
                if record is not None:
                    _assert_open_file_unchanged(
                        record,
                        label=label,
                        rehash=True,
                    )
            return {
                "adjudication_id": adjudication_id,
                "study_id": study_id,
                "phase": phase,
                "adjudicator_pseudonym": frozen_adjudicator,
                "payload": normalized,
                "packet_id": packet_id,
                "packet_sha256": packet_sha256,
                "packet_file_sha256": packet_file_sha256,
                "response_sha256": response_sha256,
                "image_sha256": observed_image_sha256,
                "revision": 1,
                "created_at": now,
                "updated_at": now,
                "idempotent_replay": False,
            }

    def adjudicate(
        self,
        study_id: str,
        item_id: str | None,
        phase: str | None,
        payload: Mapping[str, Any],
        *,
        adjudicator_pseudonym: str = "adjudicator",
        packet_id: str | None = None,
        packet_sha256: str | None = None,
        adjudicator_credential: str | None = None,
    ) -> dict[str, Any]:
        """Record a pseudonymous adjudication for a two-or-more-rater disagreement."""

        if not _PSEUDONYM_RE.fullmatch(adjudicator_pseudonym):
            raise ValidationError("adjudicator_pseudonym must be a short pseudonymous token")
        now = _timestamp(self._now())
        with self._transaction(write=True) as connection:
            study = self._study_row(connection, study_id)
            self._require_study_status(study, {"closed"})
            if study["protocol_version"] == HUMAN_CBU_V1_PROTOCOL:
                raise ValidationError("human_cbu_v1 requires packet-scoped adjudication via adjudicate_packet_response")
            if adjudicator_credential is not None:
                raise ValidationError("adjudicator credentials are only accepted for human_cbu_v1 studies")
            if packet_id is not None or packet_sha256 is not None:
                raise ValidationError("packet credentials are only accepted for human_cbu_v1 studies")
            if item_id is None or phase is None:
                raise ValidationError("legacy adjudication requires item_id and phase")
            assert item_id is not None
            assert phase is not None
            normalized = self.validate_annotation_payload(phase, payload)
            item = connection.execute(
                "SELECT 1 FROM items WHERE study_id = ? AND item_id = ?",
                (study_id, item_id),
            ).fetchone()
            if item is None:
                raise NotFoundError(f"item {item_id!r} does not exist in study {study_id!r}")
            annotations = connection.execute(
                """
                SELECT a.payload_json,
                       p.pseudonym AS participant_pseudonym
                FROM annotations AS a
                JOIN participants AS p
                  ON p.study_id = a.study_id AND p.participant_id = a.participant_id
                JOIN studies AS s ON s.study_id = a.study_id
                WHERE a.study_id = ? AND a.item_id = ? AND a.phase = ?
                  AND p.status = 'active' AND p.consented = 1
                  AND p.consent_version = s.consent_version
                """,
                (study_id, item_id, phase),
            ).fetchall()
            expected_labels = study["labels_per_item"]
            if expected_labels is None or len(annotations) != expected_labels:
                raise ValidationError(
                    "adjudication requires the complete planned annotation panel "
                    f"({len(annotations)}/{expected_labels or 0} labels available)"
                )
            if len(annotations) < 2:
                raise ValidationError("adjudication requires at least two independent annotations")
            initial_rater_pseudonyms = {row["participant_pseudonym"].casefold() for row in annotations}
            if adjudicator_pseudonym.casefold() in initial_rater_pseudonyms:
                raise ValidationError(
                    "adjudicator_pseudonym must differ from every eligible "
                    "initial rater pseudonym for this item and phase"
                )
            keys = {self._agreement_key(phase, _json_loads(row["payload_json"])) for row in annotations}
            if len(keys) < 2:
                raise ValidationError("adjudication is only accepted for a disagreement")
            existing = connection.execute(
                """
                SELECT * FROM adjudications
                WHERE study_id = ? AND item_id = ? AND phase = ?
                """,
                (study_id, item_id, phase),
            ).fetchone()
            if existing is None:
                adjudication_id = f"d_{uuid.uuid4().hex}"
                revision = 1
                created_at = now
                connection.execute(
                    """
                    INSERT INTO adjudications(
                        adjudication_id, study_id, item_id, phase,
                        adjudicator_pseudonym, payload_json, revision,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        adjudication_id,
                        study_id,
                        item_id,
                        phase,
                        adjudicator_pseudonym,
                        _json_dumps(normalized),
                        revision,
                        now,
                        now,
                    ),
                )
            else:
                adjudication_id = existing["adjudication_id"]
                revision = int(existing["revision"]) + 1
                created_at = existing["created_at"]
                connection.execute(
                    """
                    UPDATE adjudications
                    SET adjudicator_pseudonym = ?, payload_json = ?,
                        revision = ?, updated_at = ?
                    WHERE adjudication_id = ?
                    """,
                    (
                        adjudicator_pseudonym,
                        _json_dumps(normalized),
                        revision,
                        now,
                        adjudication_id,
                    ),
                )
            connection.execute(
                """
                INSERT INTO adjudication_events(
                    adjudication_id, study_id, item_id, phase,
                    adjudicator_pseudonym, payload_json, revision, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    adjudication_id,
                    study_id,
                    item_id,
                    phase,
                    adjudicator_pseudonym,
                    _json_dumps(normalized),
                    revision,
                    now,
                ),
            )
            return {
                "adjudication_id": adjudication_id,
                "study_id": study_id,
                "phase": phase,
                "adjudicator_pseudonym": adjudicator_pseudonym,
                "payload": normalized,
                "revision": revision,
                "created_at": created_at,
                "updated_at": now,
                "item_id": item_id,
            }

    def adjudication_events(
        self,
        study_id: str,
        *,
        item_id: str | None = None,
        phase: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return append-only adjudication revision snapshots."""

        if phase is not None and phase not in PHASES:
            raise ValidationError(f"invalid annotation phase: {phase!r}")
        conditions = ["study_id = ?"]
        parameters: list[Any] = [study_id]
        if item_id is not None:
            conditions.append("item_id = ?")
            parameters.append(_require_identifier(item_id, "item_id"))
        if phase is not None:
            conditions.append("phase = ?")
            parameters.append(phase)
        with self._transaction(write=False) as connection:
            self._study_row(connection, study_id)
            rows = connection.execute(
                f"""
                SELECT event_id, adjudication_id, item_id, phase,
                       adjudicator_pseudonym, payload_json,
                       packet_version, packet_id, packet_sha256,
                       packet_file_sha256, response_version, response_sha256,
                       image_sha256, revision, created_at
                FROM adjudication_events
                WHERE {" AND ".join(conditions)}
                ORDER BY event_id
                """,
                parameters,
            ).fetchall()
            return [
                {
                    "event_id": row["event_id"],
                    "adjudication_id": row["adjudication_id"],
                    "item_id": row["item_id"],
                    "phase": row["phase"],
                    "adjudicator_pseudonym": row["adjudicator_pseudonym"],
                    "payload": _json_loads(row["payload_json"]),
                    "packet_version": row["packet_version"],
                    "packet_id": row["packet_id"],
                    "packet_sha256": row["packet_sha256"],
                    "packet_file_sha256": row["packet_file_sha256"],
                    "response_version": row["response_version"],
                    "response_sha256": row["response_sha256"],
                    "image_sha256": row["image_sha256"],
                    "revision": row["revision"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    def status_summary(
        self,
        study_id: str,
        *,
        include_analysis: bool = False,
    ) -> dict[str, Any]:
        """Return operational counts, redacting labels until collection closes."""

        with self._transaction(write=False) as connection:
            study = self._study_row(connection, study_id)
            participant_rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM participants WHERE study_id = ? GROUP BY status
                """,
                (study_id,),
            ).fetchall()
            asset_rows = connection.execute(
                """
                SELECT asset_status, COUNT(*) AS count
                FROM items WHERE study_id = ? GROUP BY asset_status
                """,
                (study_id,),
            ).fetchall()
            assignment_rows = connection.execute(
                """
                SELECT a.phase, a.status, COUNT(*) AS count
                FROM assignments AS a
                JOIN participants AS p
                  ON p.study_id = a.study_id AND p.participant_id = a.participant_id
                WHERE a.study_id = ? AND p.status NOT IN ('declined', 'withdrawn')
                GROUP BY a.phase, a.status
                """,
                (study_id,),
            ).fetchall()
            stored_assignment_rows = connection.execute(
                """
                SELECT phase, status, COUNT(*) AS count
                FROM assignments WHERE study_id = ? GROUP BY phase, status
                """,
                (study_id,),
            ).fetchall()
            participants = {status: 0 for status in ("invited", "active", "declined", "withdrawn")}
            participants.update({row["status"]: row["count"] for row in participant_rows})
            assets = {status: 0 for status in ASSET_STATUSES}
            assets.update({row["asset_status"]: row["count"] for row in asset_rows})
            assignments = {phase: {"pending": 0, "completed": 0, "total": 0} for phase in PHASES}
            for row in assignment_rows:
                assignments[row["phase"]][row["status"]] = row["count"]
                assignments[row["phase"]]["total"] += row["count"]
            stored_assignments = {phase: {"pending": 0, "completed": 0, "total": 0} for phase in PHASES}
            for row in stored_assignment_rows:
                stored_assignments[row["phase"]][row["status"]] = row["count"]
                stored_assignments[row["phase"]]["total"] += row["count"]

            latest_annotations = connection.execute(
                """
                SELECT COUNT(*)
                FROM annotations AS a
                JOIN participants AS p
                  ON p.study_id = a.study_id AND p.participant_id = a.participant_id
                WHERE a.study_id = ? AND p.status = 'active' AND p.consented = 1
                  AND p.consent_version = ?
                """,
                (study_id, study["consent_version"]),
            ).fetchone()[0]
            stored_latest_annotations = connection.execute(
                "SELECT COUNT(*) FROM annotations WHERE study_id = ?",
                (study_id,),
            ).fetchone()[0]
            metadata = _json_loads(study["metadata_json"], {})
            assignment_mode = metadata.get(
                ASSIGNMENT_MODE_METADATA_KEY,
                ASSIGNMENT_MODE_FIXED,
            )
            self_enrollment_limit = metadata.get(SELF_ENROLLMENT_LIMIT_METADATA_KEY)
            self_enrollment = None
            if type(self_enrollment_limit) is int and self_enrollment_limit > 0:
                enrollment_rows = connection.execute(
                    """
                    SELECT
                        SUM(CASE WHEN revoked_at IS NULL AND use_count = 0 THEN 1 ELSE 0 END)
                            AS available,
                        SUM(CASE WHEN use_count > 0 THEN 1 ELSE 0 END) AS issued,
                        SUM(CASE WHEN revoked_at IS NOT NULL THEN 1 ELSE 0 END) AS revoked
                    FROM participant_invites
                    WHERE study_id = ?
                    """,
                    (study_id,),
                ).fetchone()
                self_enrollment = {
                    "limit": self_enrollment_limit,
                    "available": int(enrollment_rows["available"] or 0),
                    "issued": int(enrollment_rows["issued"] or 0),
                    "revoked": int(enrollment_rows["revoked"] or 0),
                }
            participant_progress_rows = connection.execute(
                """
                SELECT p.pseudonym, p.status, p.consented,
                       SUM(CASE WHEN a.phase = 'caption' THEN 1 ELSE 0 END) AS caption_total,
                       SUM(CASE WHEN a.phase = 'caption' AND a.status = 'completed' THEN 1 ELSE 0 END)
                           AS caption_completed,
                       SUM(CASE WHEN a.phase = 'image' THEN 1 ELSE 0 END) AS image_total,
                       SUM(CASE WHEN a.phase = 'image' AND a.status = 'completed' THEN 1 ELSE 0 END)
                           AS image_completed
                FROM participants AS p
                LEFT JOIN assignments AS a
                  ON a.study_id = p.study_id AND a.participant_id = p.participant_id
                WHERE p.study_id = ?
                GROUP BY p.participant_id
                ORDER BY p.pseudonym
                """,
                (study_id,),
            ).fetchall()
            result = {
                "study": self._project_study(study),
                "assignment_mode": assignment_mode,
                "items": connection.execute(
                    "SELECT COUNT(*) FROM items WHERE study_id = ?",
                    (study_id,),
                ).fetchone()[0],
                "participants": participants,
                "assets": assets,
                "assignments": assignments,
                "stored_assignments": stored_assignments,
                "latest_annotations": latest_annotations,
                "stored_latest_annotations": stored_latest_annotations,
                "annotation_events": connection.execute(
                    "SELECT COUNT(*) FROM annotation_events WHERE study_id = ?",
                    (study_id,),
                ).fetchone()[0],
                "adjudication_events": connection.execute(
                    "SELECT COUNT(*) FROM adjudication_events WHERE study_id = ?",
                    (study_id,),
                ).fetchone()[0],
                "participant_progress": [
                    {
                        "pseudonym": row["pseudonym"],
                        "status": row["status"],
                        "consented": bool(row["consented"]),
                        "caption": {
                            "completed": row["caption_completed"],
                            "total": row["caption_total"],
                        },
                        "image": {
                            "completed": row["image_completed"],
                            "total": row["image_total"],
                        },
                    }
                    for row in participant_progress_rows
                ],
            }
            if self_enrollment is not None:
                result["self_enrollment"] = self_enrollment
            if assignment_mode == ASSIGNMENT_MODE_REMAINING_FIRST:
                base_items = connection.execute(
                    "SELECT COUNT(*) FROM items WHERE study_id = ? AND repeat_of IS NULL",
                    (study_id,),
                ).fetchone()[0]
                labels_per_item = int(study["labels_per_item"])
                claimed_base_slots = connection.execute(
                    """
                    SELECT COUNT(*) FROM assignments AS a
                    JOIN items AS i
                      ON i.study_id = a.study_id AND i.item_id = a.item_id
                    JOIN participants AS p
                      ON p.study_id = a.study_id AND p.participant_id = a.participant_id
                    WHERE a.study_id = ? AND a.phase = 'caption'
                      AND i.repeat_of IS NULL
                      AND p.status NOT IN ('declined', 'withdrawn')
                    """,
                    (study_id,),
                ).fetchone()[0]
                required_base_slots = base_items * labels_per_item
                result["work_queue"] = {
                    "base_items": base_items,
                    "labels_per_item": labels_per_item,
                    "required_base_label_slots": required_base_slots,
                    "claimed_base_label_slots": claimed_base_slots,
                    "remaining_base_label_slots": max(
                        0,
                        required_base_slots - claimed_base_slots,
                    ),
                    "pool_complete": claimed_base_slots == required_base_slots,
                }
            if not include_analysis:
                result["analysis_redacted"] = True
                return result
            if study["status"] not in {"closed", "archived"}:
                raise StudyStateError("label counts and agreement are unavailable until the study is closed")

            annotation_rows = connection.execute(
                """
                SELECT a.item_id, a.phase, a.payload_json
                FROM annotations AS a
                JOIN participants AS p
                  ON p.study_id = a.study_id AND p.participant_id = a.participant_id
                JOIN studies AS s ON s.study_id = a.study_id
                WHERE a.study_id = ? AND p.status = 'active' AND p.consented = 1
                  AND p.consent_version = s.consent_version
                ORDER BY a.item_id, a.phase
                """,
                (study_id,),
            ).fetchall()
            label_counts = {
                "caption_licensed": {label: 0 for label in sorted(CAPTION_LICENSED_LABELS)},
                "image_support": {label: 0 for label in sorted(IMAGE_SUPPORT_LABELS)},
            }
            grouped: dict[tuple[str, str], list[tuple[Any, ...]]] = {}
            for row in annotation_rows:
                payload = _json_loads(row["payload_json"])
                if row["phase"] == "caption":
                    label_counts["caption_licensed"][payload["caption_licensed"]] += 1
                else:
                    label_counts["image_support"][payload["image_support"]] += 1
                grouped.setdefault((row["item_id"], row["phase"]), []).append(
                    self._agreement_key(row["phase"], payload)
                )
            adjudicated = {
                (row["item_id"], row["phase"])
                for row in connection.execute(
                    "SELECT item_id, phase FROM adjudications WHERE study_id = ?",
                    (study_id,),
                ).fetchall()
            }
            agreement = {
                phase: {
                    "items_with_two_or_more_labels": 0,
                    "disagreements": 0,
                    "adjudicated": 0,
                    "unresolved_disagreements": 0,
                    "eligible_panel_shortfalls": 0,
                    "eligible_panels_complete": 0,
                    "required_labels_per_item": int(study["labels_per_item"] or 0),
                }
                for phase in PHASES
            }
            for key, labels in grouped.items():
                item_id, phase = key
                if len(labels) < 2:
                    continue
                agreement[phase]["items_with_two_or_more_labels"] += 1
                disagreement = len(set(labels)) > 1
                if disagreement:
                    agreement[phase]["disagreements"] += 1
                    if key in adjudicated:
                        agreement[phase]["adjudicated"] += 1
                    else:
                        agreement[phase]["unresolved_disagreements"] += 1
            item_ids = [
                row["item_id"]
                for row in connection.execute(
                    "SELECT item_id FROM items WHERE study_id = ?",
                    (study_id,),
                ).fetchall()
            ]
            required_labels = int(study["labels_per_item"] or 0)
            for phase in PHASES:
                for item_id in item_ids:
                    panel_size = len(grouped.get((item_id, phase), ()))
                    if required_labels and panel_size == required_labels:
                        agreement[phase]["eligible_panels_complete"] += 1
                    else:
                        agreement[phase]["eligible_panel_shortfalls"] += 1
            result["analysis_redacted"] = False
            result["label_counts"] = label_counts
            result["agreement"] = agreement
            return result

    def participant_summary(self, study_id: str) -> dict[str, Any]:
        """Return privacy-reviewed participant flow and coarse marginals."""

        with self._transaction(write=False) as connection:
            study = self._study_row(connection, study_id)
            status_counts = {status: 0 for status in ("invited", "active", "declined", "withdrawn")}
            status_counts.update(
                {
                    row["status"]: row["count"]
                    for row in connection.execute(
                        """
                        SELECT status, COUNT(*) AS count
                        FROM participants WHERE study_id = ? GROUP BY status
                        """,
                        (study_id,),
                    ).fetchall()
                }
            )
            invited_total = connection.execute(
                "SELECT COUNT(*) FROM participant_invites WHERE study_id = ?",
                (study_id,),
            ).fetchone()[0]
            completed = connection.execute(
                """
                SELECT COUNT(*) FROM participants AS p
                WHERE p.study_id = ? AND p.status = 'active' AND p.consented = 1
                  AND p.consent_version = ?
                  AND EXISTS (
                    SELECT 1 FROM assignments AS a
                    WHERE a.study_id = p.study_id AND a.participant_id = p.participant_id
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM assignments AS a
                    WHERE a.study_id = p.study_id AND a.participant_id = p.participant_id
                      AND a.status != 'completed'
                  )
                """,
                (study_id, study["consent_version"]),
            ).fetchone()[0]
            procedurally_completed_rows = connection.execute(
                """
                SELECT p.participant_id FROM participants AS p
                WHERE p.study_id = ?
                  AND EXISTS (
                    SELECT 1 FROM assignments AS a
                    WHERE a.study_id = p.study_id AND a.participant_id = p.participant_id
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM assignments AS a
                    WHERE a.study_id = p.study_id AND a.participant_id = p.participant_id
                      AND a.status != 'completed'
                  )
                """,
                (study_id,),
            ).fetchall()
            replacements = _json_loads(
                study["metadata_json"],
                {},
            ).get(PARTICIPANT_REPLACEMENTS_METADATA_KEY, [])
            partially_completed_sources = {
                event.get("source_participant_id")
                for event in replacements
                if isinstance(event, Mapping)
                and event.get("source_completed_assignment_count", 0) < event.get("source_assignment_count", 0)
            }
            procedurally_completed_total = sum(
                row["participant_id"] not in partially_completed_sources for row in procedurally_completed_rows
            )
            consented_current = connection.execute(
                """
                SELECT COUNT(*) FROM participants AS p
                WHERE p.study_id = ? AND p.status = 'active' AND p.consented = 1
                  AND p.consent_version = ?
                """,
                (study_id, study["consent_version"]),
            ).fetchone()[0]
            analyzed_rows = connection.execute(
                """
                SELECT p.profile_json
                FROM participants AS p
                WHERE p.study_id = ? AND p.status = 'active' AND p.consented = 1
                  AND p.consent_version = ?
                  AND EXISTS (
                    SELECT 1 FROM annotations AS a
                    WHERE a.study_id = p.study_id AND a.participant_id = p.participant_id
                  )
                ORDER BY p.participant_id
                """,
                (study_id, study["consent_version"]),
            ).fetchall()
            profiles = [_json_loads(row["profile_json"], {}) for row in analyzed_rows]
            fields = sorted({field for profile in profiles for field in profile})
            marginals: dict[str, Any] = {}
            for field in fields:
                counts: dict[str, int] = {}
                missing = 0
                multi_select = False
                for profile in profiles:
                    value = profile.get(field)
                    if value is None:
                        missing += 1
                    elif isinstance(value, list):
                        multi_select = True
                        for entry in value:
                            counts[str(entry)] = counts.get(str(entry), 0) + 1
                    else:
                        counts[str(value)] = counts.get(str(value), 0) + 1
                marginals[field] = {
                    "counts": [{"value": value, "count": count} for value, count in sorted(counts.items())],
                    "missing": missing,
                    "multi_select": multi_select,
                }
            return {
                "identity": "study-local pseudonyms",
                "direct_identifiers_collected": False,
                "flow": {
                    "invited_total": invited_total,
                    "current_status": status_counts,
                    "consented_current": consented_current,
                    "completed": completed,
                    "procedurally_completed_total": procedurally_completed_total,
                    "analyzed": len(analyzed_rows),
                },
                "profile_marginals": {
                    "n_participants": len(analyzed_rows),
                    "joint_profiles_released": False,
                    "fields": marginals,
                },
            }

    def export_rows(
        self,
        study_id: str,
        *,
        include_exact_timestamps: bool = False,
        include_notes: bool = False,
    ) -> list[dict[str, Any]]:
        """Export analysis-ready latest labels without image/caption payloads.

        This administrator API includes private audit dimensions (surface and
        judge answers), but excludes captions, asset locators, asset paths,
        invite/session digests, and all identity-adjacent request metadata.
        """

        with self._transaction(write=False) as connection:
            study = self._study_row(connection, study_id)
            rows = connection.execute(
                """
                SELECT a.annotation_id, a.assignment_id, a.item_id, a.phase,
                       a.payload_json, a.revision, a.created_at, a.updated_at,
                       p.pseudonym AS participant_pseudonym,
                       i.item_group, i.category, i.surface, i.qwen_answer_json,
                       i.gemma_answer_json, i.stratum, i.population_json,
                       i.sample_json, i.weight, i.population_size,
                       i.sample_probability, i.repeat_of,
                       d.payload_json AS adjudication_json,
                       d.adjudicator_pseudonym, d.revision AS adjudication_revision,
                       d.created_at AS adjudication_created_at,
                       d.updated_at AS adjudication_updated_at
                FROM annotations AS a
                JOIN participants AS p
                  ON p.study_id = a.study_id AND p.participant_id = a.participant_id
                JOIN studies AS s ON s.study_id = a.study_id
                JOIN items AS i
                  ON i.study_id = a.study_id AND i.item_id = a.item_id
                LEFT JOIN adjudications AS d
                  ON d.study_id = a.study_id AND d.item_id = a.item_id
                  AND d.phase = a.phase
                WHERE a.study_id = ? AND p.status = 'active' AND p.consented = 1
                  AND p.consent_version = s.consent_version
                ORDER BY a.item_id, a.phase, p.pseudonym
                """,
                (study_id,),
            ).fetchall()
            panel_keys: dict[tuple[str, str], list[tuple[Any, ...]]] = {}
            for row in rows:
                phase = row["phase"]
                payload = _json_loads(row["payload_json"], {})
                panel_keys.setdefault((row["item_id"], phase), []).append(self._agreement_key(phase, payload))
            expected_labels = int(study["labels_per_item"] or 0)
            adjudication_eligible = {
                key
                for key, keys in panel_keys.items()
                if expected_labels and len(keys) == expected_labels and len(set(keys)) > 1
            }
            exported: list[dict[str, Any]] = []
            for row in rows:
                payload = _json_loads(row["payload_json"], {})
                if not include_notes:
                    payload.pop("note", None)
                panel_key = (row["item_id"], row["phase"])
                adjudication_payload = (
                    _json_loads(row["adjudication_json"]) if panel_key in adjudication_eligible else None
                )
                adjudication = None
                if adjudication_payload is not None:
                    if not include_notes:
                        adjudication_payload.pop("note", None)
                    adjudication = {
                        **adjudication_payload,
                        "adjudicator_pseudonym": row["adjudicator_pseudonym"],
                        "revision": row["adjudication_revision"],
                        "created_date": row["adjudication_created_at"][:10],
                        "updated_date": row["adjudication_updated_at"][:10],
                    }
                    if include_exact_timestamps:
                        adjudication["created_at"] = row["adjudication_created_at"]
                        adjudication["updated_at"] = row["adjudication_updated_at"]
                export = {
                    "study_id": study_id,
                    "annotation_id": row["annotation_id"],
                    "assignment_id": row["assignment_id"],
                    "item_id": row["item_id"],
                    "item_group": row["item_group"],
                    "participant_pseudonym": row["participant_pseudonym"],
                    "phase": row["phase"],
                    "category": row["category"],
                    "surface": row["surface"],
                    **payload,
                    "qwen_answer": _json_loads(row["qwen_answer_json"]),
                    "gemma_answer": _json_loads(row["gemma_answer_json"]),
                    "stratum": row["stratum"],
                    "population": _json_loads(row["population_json"]),
                    "sample": _json_loads(row["sample_json"]),
                    "weight": row["weight"],
                    "sampling_weight": row["weight"],
                    "population_size": row["population_size"],
                    "sample_probability": row["sample_probability"],
                    "repeat_of": row["repeat_of"],
                    "adjudication": adjudication,
                    "revision": row["revision"],
                    "created_date": row["created_at"][:10],
                    "updated_date": row["updated_at"][:10],
                }
                if include_exact_timestamps:
                    export["created_at"] = row["created_at"]
                    export["updated_at"] = row["updated_at"]
                exported.append(export)
            return exported

    def backup(self, destination: str | Path, *, overwrite: bool = False) -> Path:
        """Create a consistent SQLite backup, refusing overwrite by default."""

        destination = Path(destination)
        if destination.resolve() == self.path.resolve():
            raise ValidationError("backup destination must differ from the source database")
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            source = self._connect()
            target = sqlite3.connect(temporary)
            try:
                source.backup(target)
                target.commit()
            finally:
                target.close()
                source.close()
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination
