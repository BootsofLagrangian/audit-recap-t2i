#!/usr/bin/env python3
"""Build, materialize, operate, validate, and export the blinded human CBU study."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
import math
import os
import re
import secrets
import sqlite3
import ssl
import stat
import sys
import tempfile
import webbrowser
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from aiohttp import web

from audit_recap_t2i.human_cbu.assets import (
    materialize_selected_images,
    verify_canonical_manifest,
)
from audit_recap_t2i.human_cbu.builder import (
    build_sample,
    join_candidate_pool,
    load_claim_records,
    load_judge_records,
    validate_study_design,
)
from audit_recap_t2i.human_cbu.export import write_export_bundle
from audit_recap_t2i.human_cbu.store import (
    ADJUDICATION_PACKET_VERSION,
    ADJUDICATION_RESPONSE_VERSION,
    ADJUDICATION_REVIEW_VERSION,
    HumanCBUStore,
    ethics_review_basis_error,
)
from audit_recap_t2i.human_cbu.web import (
    create_author_practice_app,
    create_preview_app,
    create_self_enrollment_test_app,
    create_study_app,
    load_preview_item,
    validate_participant_information,
)

PRIVATE_MODE = 0o600
PRIVATE_DIRECTORY_MODE = 0o700
DEFAULT_ITEMS_NAME = "items.private.jsonl"
DEFAULT_SAMPLE_MANIFEST_NAME = "study_manifest.json"
DEFAULT_CANONICAL_REPORT_NAME = "canonical_manifest_report.private.json"
DEFAULT_MATERIALIZATION_REPORT_NAME = "materialization_report.private.json"
DEFAULT_INIT_REPORT_NAME = "init_db_report.private.json"
HUMAN_CBU_V1_PROTOCOL = "human_cbu_v1"
ADJUDICATION_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
MAX_ADJUDICATION_PACKET_BYTES = 32 * 1024 * 1024
MAX_ADJUDICATION_REVIEW_BYTES = 4 * 1024 * 1024
MAX_ADJUDICATION_RESPONSE_BYTES = 1024 * 1024
MAX_ADJUDICATION_IMAGE_BYTES = 256 * 1024 * 1024


def _add_database_study_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", required=True, help="Initialized human-CBU SQLite database")
    parser.add_argument("--study-id", required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_preview = subparsers.add_parser(
        "build-preview",
        help="Materialize one real artifact-backed preview item",
    )
    build_preview.add_argument("--config", required=True)
    build_preview.add_argument("--question-id", required=True)
    build_preview.add_argument("--output-dir", required=True)

    preview = subparsers.add_parser(
        "preview",
        help="Serve the visual UI against one local preview item",
    )
    preview.add_argument("--item-json", required=True)
    preview.add_argument("--image", required=True)
    preview.add_argument("--phase", choices=["caption", "image"], default="caption")
    preview.add_argument("--host", default="127.0.0.1")
    preview.add_argument("--port", type=int, default=8765)

    participant_test = subparsers.add_parser(
        "participant-test",
        help=(
            "Exercise self-enrollment and caption/image tasks in volatile "
            "response-discarding memory"
        ),
    )
    participant_test.add_argument("--item-json", required=True)
    participant_test.add_argument("--image", required=True)
    participant_test.add_argument(
        "--additional-item-json",
        action="append",
        default=[],
        help="Additional preview item to rotate into the test; repeat with its image",
    )
    participant_test.add_argument(
        "--additional-image",
        action="append",
        default=[],
        help="Image paired by position with --additional-item-json",
    )
    participant_test.add_argument("--pairs", type=int, default=3)
    participant_test.add_argument("--host", default="127.0.0.1")
    participant_test.add_argument("--port", type=int, default=8768)

    practice_codes = subparsers.add_parser(
        "author-practice-codes",
        help="Create exactly three anonymous author-rehearsal codes in a private file",
    )
    practice_codes.add_argument("--output", required=True)

    practice_serve = subparsers.add_parser(
        "author-practice-serve",
        help="Serve a volatile three-author rehearsal whose response values are discarded",
    )
    _add_database_study_arguments(practice_serve)
    practice_serve.add_argument("--codes", required=True)
    practice_serve.add_argument("--asset-root")
    practice_serve.add_argument("--host", default="127.0.0.1")
    practice_serve.add_argument("--port", type=int, default=8767)

    sample = subparsers.add_parser(
        "sample",
        help="Build the frozen stratified item sample as JSONL",
    )
    sample.add_argument("--config", required=True)
    sample.add_argument("--output-dir", required=True)
    sample.add_argument("--claims-per-stratum", type=int)
    sample.add_argument("--no-fingerprint", action="store_true")

    materialize = subparsers.add_parser(
        "materialize",
        help="Fail-closed manifest verification and selected-image extraction",
    )
    materialize.add_argument("--config", required=True)
    materialize.add_argument("--sample-dir", required=True)
    materialize.add_argument("--items")
    materialize.add_argument("--asset-root")
    materialize.add_argument("--workers", type=int)
    materialize.add_argument("--dataset-prefix")

    init_db = subparsers.add_parser(
        "init-db",
        help="Normalize the frozen sample and reports into a new draft database",
    )
    init_db.add_argument("--config", required=True)
    init_db.add_argument("--sample-dir", required=True)
    init_db.add_argument("--db", required=True)
    init_db.add_argument("--items")
    init_db.add_argument("--sample-manifest")
    init_db.add_argument("--canonical-report")
    init_db.add_argument("--materialization-report")
    init_db.add_argument("--init-report")

    invites = subparsers.add_parser(
        "invites",
        help="Create pseudonymous invites and write raw codes once to a mode-0600 JSON file",
    )
    _add_database_study_arguments(invites)
    invites.add_argument("--count", type=int, required=True)
    invites.add_argument("--output", required=True)
    invites.add_argument("--expires-in-seconds", type=int)

    provision_adjudicator = subparsers.add_parser(
        "provision-adjudicator",
        help="Provision the predeclared adjudicator credential once while draft",
    )
    _add_database_study_arguments(provision_adjudicator)
    provision_adjudicator.add_argument("--output", required=True)

    assign = subparsers.add_parser(
        "assign",
        help="Create the deterministic two-phase assignment plan while the study is draft",
    )
    _add_database_study_arguments(assign)
    assign.add_argument("--config")
    assign.add_argument("--labels-per-item", type=int)
    assign.add_argument("--seed")
    assign.add_argument(
        "--remaining-first",
        action="store_true",
        help=(
            "Freeze the roster without preallocating items; consenting primaries "
            "atomically claim the least-covered remaining work"
        ),
    )

    replace_participant = subparsers.add_parser(
        "replace-participant",
        help=("While paused, replace a declined/withdrawn primary with an active predeclared reserve"),
    )
    _add_database_study_arguments(replace_participant)
    replace_participant.add_argument("--source-participant-id", required=True)
    replace_participant.add_argument("--replacement-participant-id", required=True)
    replace_participant.add_argument(
        "--confirm-source-participant-id",
        required=True,
        help="Must exactly repeat --source-participant-id",
    )

    validate = subparsers.add_parser(
        "validate",
        help="Report pre-launch validation failures without changing study state",
    )
    _add_database_study_arguments(validate)
    validate.add_argument(
        "--allow-missing-assets",
        action="store_true",
        help="Diagnostic only; sealing always requires every selected asset",
    )

    seal = subparsers.add_parser(
        "seal",
        help="Persist the ethics-review basis and participant notice, validate, and freeze the draft",
    )
    _add_database_study_arguments(seal)
    seal.add_argument(
        "--ethics-determination-id",
        required=True,
        help=(
            "Use 'review-not-required:<authority-or-basis>' or a nonempty "
            "institutional determination identifier"
        ),
    )
    seal.add_argument(
        "--participant-information-json",
        required=True,
        help="Participant-notice fields as JSON",
    )

    open_study = subparsers.add_parser(
        "open",
        help="Open a ready study for participant sessions (explicit confirmation required)",
    )
    _add_database_study_arguments(open_study)
    open_study.add_argument(
        "--confirm-open",
        required=True,
        metavar="STUDY_ID",
        help="Must exactly equal --study-id; opening begins participant collection",
    )

    pause = subparsers.add_parser("pause", help="Pause an open study")
    _add_database_study_arguments(pause)

    close = subparsers.add_parser("close", help="Close an open or paused study")
    _add_database_study_arguments(close)
    close.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Emergency close with pending assignments; exports remain gated as incomplete",
    )

    serve = subparsers.add_parser("serve", help="Serve the production blinded annotation UI")
    _add_database_study_arguments(serve)
    serve.add_argument(
        "--asset-root",
        help=(
            "Root containing the relative asset references stored by init-db; "
            "defaults to the private root recorded during init-db"
        ),
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--tls-cert")
    serve.add_argument("--tls-key")
    serve.add_argument(
        "--self-enrollment-invites",
        help=(
            "Mode-0600 preprovisioned invite artifact used to issue one "
            "participant code on the first page"
        ),
    )

    status = subparsers.add_parser("status", help="Print operational study counts")
    _add_database_study_arguments(status)

    backup = subparsers.add_parser("backup", help="Create a consistent SQLite backup")
    backup.add_argument("--db", required=True)
    backup.add_argument("--output", required=True)
    backup.add_argument("--overwrite", action="store_true")

    export = subparsers.add_parser(
        "export",
        help="Generate deterministic private/public analysis and rebuttal artifacts",
    )
    _add_database_study_arguments(export)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--public-id-namespace")
    export.add_argument("--bootstrap-resamples", type=int, default=10_000)
    export.add_argument("--bootstrap-seed", type=int, default=1477)
    export.add_argument("--include-notes", action="store_true")
    export.add_argument("--include-exact-timestamps", action="store_true")
    export.add_argument(
        "--include-public-rows",
        action="store_true",
        help="Opt in to a privacy-reviewed row-level public file; aggregate-only is the default",
    )

    adjudication_packets = subparsers.add_parser(
        "adjudication-packets",
        help="Write one private self-contained directory per closed-study disagreement",
    )
    _add_database_study_arguments(adjudication_packets)
    adjudication_packets.add_argument(
        "--phase",
        choices=["caption", "image"],
        required=True,
        help="Generate exactly one blinded adjudication phase",
    )
    adjudication_packets.add_argument(
        "--output-dir",
        required=True,
        help="New private directory to create; existing paths are never reused",
    )
    adjudication_packets.add_argument(
        "--asset-root",
        help=("Root containing frozen relative image assets; defaults to the asset_serve_root recorded by init-db"),
    )

    adjudicate = subparsers.add_parser(
        "adjudicate",
        help="Record an already-published packet-scoped response after collection closes",
    )
    _add_database_study_arguments(adjudicate)
    adjudicate.add_argument(
        "--phase",
        choices=["caption", "image"],
        required=True,
    )
    adjudicate.add_argument("--packet-dir")
    adjudicate.add_argument(
        "--credential-file",
        help="Private mode-0600 credential file produced by provision-adjudicator",
    )
    adjudicate.add_argument(
        "--item-id",
        help="Legacy protocols only",
    )
    adjudicate.add_argument("--payload-json", help="Legacy protocols only")
    adjudicate.add_argument("--adjudicator-pseudonym", help="Legacy protocols only")

    adjudication_review = subparsers.add_parser(
        "adjudication-review",
        help="Review and submit exactly one private adjudication packet over loopback",
    )
    _add_database_study_arguments(adjudication_review)
    adjudication_review.add_argument("--packet-dir", required=True)
    adjudication_review.add_argument(
        "--credential-file",
        required=True,
        help="Private mode-0600 credential file produced by provision-adjudicator",
    )
    adjudication_review.add_argument("--host", default="127.0.0.1")
    adjudication_review.add_argument("--port", type=int, default=8766)
    adjudication_review.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the one-session URL without opening the default browser",
    )

    return parser.parse_args(argv)


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def study_settings(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    study = config.get("study")
    inputs = config.get("inputs")
    images = config.get("images")
    if not isinstance(study, dict) or not isinstance(inputs, dict) or not isinstance(images, dict):
        raise ValueError("configuration requires study, inputs, and images mappings")
    return study, inputs, images


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _json_text(payload: Any) -> str:
    return (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def ensure_private_directory(path: str | Path) -> Path:
    """Create/chmod a study-owned directory so private artifacts cannot leak."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(PRIVATE_DIRECTORY_MODE)
    return directory


def _ensure_parent(path: Path) -> None:
    """Create a missing parent privately without changing a pre-existing parent."""

    if path.parent.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(PRIVATE_DIRECTORY_MODE)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: Path, payload: Any, *, private: bool = False) -> None:
    """Atomically replace a generated JSON artifact."""

    _ensure_parent(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_json_text(payload))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(PRIVATE_MODE if private else 0o644)
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], *, private: bool = True) -> None:
    """Atomically replace a generated JSONL artifact."""

    _ensure_parent(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(PRIVATE_MODE if private else 0o644)
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def secure_write_once_json(path: Path, payload: Any) -> None:
    """Atomically create one secret JSON file, refusing replacement."""

    secure_write_once_bytes(path, _json_text(payload).encode("utf-8"))


def secure_write_once_bytes(path: Path, raw: bytes) -> None:
    """Atomically create one private file, refusing every replacement."""

    _ensure_parent(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(PRIVATE_MODE)
        # Hard-link publication is atomic and, unlike replace(), fails if a
        # one-time secret target already exists.
        os.link(temporary, path)
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_private_regular_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> bytes:
    """Read one private, non-symlink, non-hardlinked file from a stable FD."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} must be an existing non-symlink file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be a private regular non-hardlinked file")
        if before.st_mode & 0o077:
            raise ValueError(f"{label} must not grant group or world permissions")
        if before.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds its maximum allowed size")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(
            descriptor,
            min(1024 * 1024, maximum_bytes + 1 - total),
        ):
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError(f"{label} exceeds its maximum allowed size")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        before_signature = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_signature = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_signature != after_signature:
            raise ValueError(f"{label} changed while it was being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    """Parse a UTF-8 JSON object while rejecting duplicate keys and NaN."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains invalid JSON constant {value!r}")

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON root must be an object")
    return payload


def _load_private_json(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_private_regular_file(
        path,
        label=label,
        maximum_bytes=maximum_bytes,
    )
    return _strict_json_object(raw, label=label), raw


def _private_packet_directory(path: str | Path) -> Path:
    """Resolve the one accepted ``<phase-root>/packets/ap_<hash>`` shape."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.absolute()
    packets_directory = candidate.parent
    phase_root = packets_directory.parent
    if packets_directory.name != "packets":
        raise ValueError("adjudication packet directory must be directly inside <phase-root>/packets")
    for directory, label in (
        (phase_root, "phase root"),
        (packets_directory, "packets directory"),
        (candidate, "packet directory"),
    ):
        try:
            metadata = directory.lstat()
            resolved = directory.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"adjudication {label} does not exist") from error
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or directory.is_symlink()
            or metadata.st_mode & 0o077
            or resolved != directory
        ):
            raise ValueError(f"adjudication {label} must be a private canonical non-symlink directory")
    if re.fullmatch(r"ap_[0-9a-f]{64}", candidate.name) is None:
        raise ValueError("adjudication packet directory name is not a valid packet_id")
    return candidate


def _load_adjudicator_credential(
    path: str | Path,
    *,
    study_id: str,
    frozen_pseudonym: Any,
) -> str:
    payload, _ = _load_private_json(
        Path(path),
        label="adjudicator credential file",
        maximum_bytes=64 * 1024,
    )
    if set(payload) != {
        "study_id",
        "adjudicator_pseudonym",
        "credential",
    }:
        raise ValueError("--credential-file violates the exact credential allowlist")
    if payload["study_id"] != study_id:
        raise ValueError("--credential-file study_id does not match --study-id")
    if payload["adjudicator_pseudonym"] != frozen_pseudonym:
        raise ValueError("--credential-file adjudicator_pseudonym does not match the frozen study pseudonym")
    credential = payload["credential"]
    if not isinstance(credential, str) or not credential:
        raise ValueError("--credential-file does not contain a valid credential")
    return credential


def _frozen_adjudication_categories(study: Mapping[str, Any]) -> list[str]:
    metadata = study.get("metadata")
    categories = metadata.get("semantic_visual_claim_types") if isinstance(metadata, Mapping) else None
    if (
        not isinstance(categories, list)
        or not categories
        or not all(isinstance(value, str) and value for value in categories)
        or len(categories) != len(set(categories))
    ):
        raise ValueError("human_cbu_v1 study metadata must freeze unique semantic_visual_claim_types")
    return list(categories)


def secure_write_once_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Atomically create one private JSONL file, refusing replacement."""

    _ensure_parent(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(PRIVATE_MODE)
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def is_loopback_host(host: str) -> bool:
    """Accept only the explicit localhost name or an IP loopback address."""

    normalized = host.strip()
    if normalized.casefold() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv4Address):
        return address in ipaddress.ip_network("127.0.0.0/8")
    return address == ipaddress.ip_address("::1")


def validate_bind_security(
    host: str,
    *,
    preview: bool,
    tls_enabled: bool,
) -> None:
    """Reject unsafe network exposure before opening data or a database."""

    if preview:
        if not is_loopback_host(host):
            raise ValueError("preview may bind only to localhost, 127.0.0.0/8, or ::1")
        return
    if not tls_enabled:
        raise ValueError(
            "production serving requires both --tls-cert and --tls-key on every bind; "
            "use preview for loopback-only visual inspection"
        )


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return payload


def load_jsonl_with_sha256(
    path: str | Path,
) -> tuple[list[dict[str, Any]], str]:
    """Parse JSONL and hash the exact bytes that were parsed."""

    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: JSONL must be UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
        rows.append(payload)
    if not rows:
        raise ValueError(f"{path}: JSONL contains no rows")
    return rows, hashlib.sha256(raw).hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows, _ = load_jsonl_with_sha256(path)
    return rows


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sample_items_sha256(
    sample_manifest: Mapping[str, Any],
    actual_sha256: str,
    *,
    required: bool,
) -> dict[str, Any]:
    """Verify the manifest binding for the exact frozen JSONL bytes."""

    expected = sample_manifest.get("items_sha256")
    if expected is None:
        if required:
            raise ValueError("human_cbu_v1 sample manifest must bind items.private.jsonl with items_sha256")
        return {
            "required": False,
            "verified": False,
            "items_sha256": actual_sha256,
        }
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("sample manifest items_sha256 must be a lowercase SHA-256 hex digest")
    if not secrets.compare_digest(expected, actual_sha256):
        raise ValueError("sample manifest items_sha256 does not match the exact items.private.jsonl bytes")
    return {
        "required": required,
        "verified": True,
        "items_sha256": actual_sha256,
    }


def build_from_config(
    config: Mapping[str, Any],
    *,
    claims_per_stratum: int | None = None,
    fingerprint_inputs: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    study, inputs, _ = study_settings(config)
    judges = inputs.get("judges")
    if not isinstance(judges, dict):
        raise ValueError("inputs.judges must be a mapping")
    claim_types, _ = validate_study_design(
        study["surface_groups"],
        study.get("semantic_visual_claim_types"),
    )
    configured_claims_per_stratum = _positive_integer(
        study.get("claims_per_stratum"),
        "study.claims_per_stratum",
    )
    if claims_per_stratum is not None:
        requested_claims_per_stratum = _positive_integer(
            claims_per_stratum,
            "--claims-per-stratum",
        )
        if requested_claims_per_stratum != configured_claims_per_stratum:
            raise ValueError("--claims-per-stratum does not match the frozen study.claims_per_stratum")
    result = build_sample(
        claim_response_paths=list(inputs["claimed_cbu_responses"]),
        judge_response_paths={name: list(paths) for name, paths in judges.items()},
        surface_groups={name: list(surfaces) for name, surfaces in study["surface_groups"].items()},
        semantic_visual_claim_types=claim_types,
        token_budget=int(study["token_budget"]),
        claims_per_stratum=configured_claims_per_stratum,
        seed=int(study["seed"]),
        study_namespace=str(study["id"]),
        repeat_fraction=float(study.get("repeat_fraction", 0.0)),
        required_judges=tuple(judges),
        fingerprint_inputs=fingerprint_inputs,
    )
    return result.items, result.report


def command_sample(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    items_path = output / DEFAULT_ITEMS_NAME
    manifest_path = output / DEFAULT_SAMPLE_MANIFEST_NAME
    existing = [path for path in (items_path, manifest_path) if path.exists()]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to overwrite a frozen human-CBU sample; use a new version/destination: {paths}"
        )
    config = load_config(args.config)
    items, report = build_from_config(
        config,
        claims_per_stratum=args.claims_per_stratum,
        fingerprint_inputs=not args.no_fingerprint,
    )
    output = ensure_private_directory(output)
    items_published = False
    items_sha256: str | None = None
    try:
        secure_write_once_jsonl(items_path, items)
        items_published = True
        items_sha256 = _sha256_file(items_path)
        secure_write_once_json(
            manifest_path,
            {
                **report,
                "items_sha256": items_sha256,
            },
        )
    except Exception:
        if items_published:
            items_path.unlink(missing_ok=True)
            _fsync_directory(output)
        raise
    print(
        _json_text(
            {
                "items": len(items),
                "items_sha256": items_sha256,
                "output_dir": str(output.resolve()),
                "private_items_mode": oct(items_path.stat().st_mode & 0o777),
            }
        ),
        end="",
    )
    return 0


def command_build_preview(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    study, inputs, images = study_settings(config)
    token_budget = int(study["token_budget"])
    claims, claim_report = load_claim_records(
        inputs["claimed_cbu_responses"],
        token_budget=token_budget,
    )
    claim = claims.get(args.question_id)
    if claim is None:
        raise ValueError(f"question_id not found in extractor responses: {args.question_id}")
    judges: dict[str, dict[str, dict[str, Any]]] = {}
    judge_reports: dict[str, Any] = {}
    for name, paths in inputs["judges"].items():
        records, report = load_judge_records(
            paths,
            token_budget=token_budget,
            judge_name=name,
        )
        judges[name] = records
        judge_reports[name] = report
    candidates, join_report = join_candidate_pool(
        {args.question_id: claim},
        judges,
        surface_groups=study["surface_groups"],
        required_judges=tuple(judges),
    )
    if len(candidates) != 1:
        raise ValueError(f"preview question did not produce exactly one joined candidate: {join_report}")
    candidate = candidates[0]
    candidate["item_id"] = f"preview-{args.question_id.replace(':', '-')}"
    output = ensure_private_directory(args.output_dir)
    ensure_private_directory(output / "assets")
    materialization = materialize_selected_images(
        [candidate],
        wds_root=images["canonical_wds_root"],
        asset_root=output / "assets",
        workers=1,
    )
    if materialization["materialized_unique_images"] != 1:
        raise RuntimeError(f"preview image materialization failed: {materialization['failures']}")
    asset = materialization["assets"][0]
    preview_item = {
        "item_id": candidate["item_id"],
        "category": candidate["category"],
        "unit": candidate["unit"],
        "span": candidate["span"],
        "target": candidate["target"],
        "caption": candidate["caption"],
        "token_budget": candidate["token_budget"],
    }
    write_json(output / "preview_item.json", preview_item, private=True)
    write_json(
        output / "preview_report.json",
        {
            "question_id": args.question_id,
            "claim_load": claim_report,
            "judge_load": judge_reports,
            "join": join_report,
            "image": asset,
        },
        private=True,
    )
    print(
        _json_text(
            {
                "item_json": str((output / "preview_item.json").resolve()),
                "image": asset["path"],
                "image_sha256": asset["sha256"],
            }
        ),
        end="",
    )
    return 0


def command_preview(args: argparse.Namespace) -> int:
    validate_bind_security(args.host, preview=True, tls_enabled=False)
    item = load_preview_item(args.item_json)
    application = create_preview_app(
        item=item,
        image_path=args.image,
        initial_phase=args.phase,
    )
    web.run_app(application, host=args.host, port=args.port, access_log=None)
    return 0


def command_participant_test(args: argparse.Namespace) -> int:
    validate_bind_security(args.host, preview=True, tls_enabled=False)
    item = load_preview_item(args.item_json)
    if len(args.additional_item_json) != len(args.additional_image):
        raise ValueError(
            "--additional-item-json and --additional-image must be supplied in pairs"
        )
    additional_items = tuple(load_preview_item(path) for path in args.additional_item_json)
    fixture_count = 1 + len(additional_items)
    if args.pairs > fixture_count:
        print(
            (
                f"warning: --pairs={args.pairs} exceeds {fixture_count} distinct "
                "test fixtures; fixtures will repeat in supplied order"
            ),
            file=sys.stderr,
            flush=True,
        )
    application = create_self_enrollment_test_app(
        item=item,
        image_path=args.image,
        pair_count=args.pairs,
        additional_items=additional_items,
        additional_image_paths=tuple(args.additional_image),
    )
    print(
        _json_text(
            {
                "test_mode": True,
                "responses_persisted": False,
                "codes_persisted": False,
                "bind": f"http://{args.host}:{args.port}",
                "pairs_per_test_session": args.pairs,
                "distinct_test_fixtures": fixture_count,
                "fixtures_repeat": args.pairs > fixture_count,
            }
        ),
        end="",
        flush=True,
    )
    web.run_app(application, host=args.host, port=args.port, access_log=None)
    return 0


def command_author_practice_codes(args: argparse.Namespace) -> int:
    output = Path(args.output)
    slots = [
        {
            "practice_id": f"author_practice_{index:02d}",
            "code": "hcap_" + secrets.token_urlsafe(24),
        }
        for index in range(1, 4)
    ]
    secure_write_once_json(
        output,
        {
            "schema_version": 1,
            "purpose": "author_ui_rehearsal_non_analysis",
            "responses_persisted": False,
            "slots": slots,
        },
    )
    print(
        _json_text(
            {
                "private_output": str(output.resolve()),
                "practice_slots": len(slots),
                "responses_persisted": False,
                "mode": oct(output.stat().st_mode & 0o777),
            }
        ),
        end="",
    )
    return 0


def _load_author_practice_codes(path: str | Path) -> dict[str, str]:
    code_path = Path(path)
    if code_path.is_symlink() or not code_path.is_file():
        raise ValueError("author practice code file must be a regular non-symlink file")
    if stat.S_IMODE(code_path.stat().st_mode) != PRIVATE_MODE:
        raise ValueError("author practice code file must have mode 0600")
    payload = load_json(code_path)
    if (
        payload.get("purpose") != "author_ui_rehearsal_non_analysis"
        or payload.get("responses_persisted") is not False
        or not isinstance(payload.get("slots"), list)
        or len(payload["slots"]) != 3
    ):
        raise ValueError("author practice code file has an invalid contract")
    codes: dict[str, str] = {}
    for row in payload["slots"]:
        if not isinstance(row, Mapping):
            raise ValueError("author practice slot must be an object")
        practice_id = row.get("practice_id")
        code = row.get("code")
        if not isinstance(practice_id, str) or not isinstance(code, str):
            raise ValueError("author practice slot is missing its identifier or code")
        codes[practice_id] = code
    if len(codes) != 3 or len(set(codes.values())) != 3:
        raise ValueError("author practice identifiers and codes must be unique")
    return codes


def _load_self_enrollment_codes(
    path: str | Path,
    *,
    study_id: str,
    expected_count: int,
) -> tuple[str, ...]:
    code_path = Path(path)
    if code_path.is_symlink() or not code_path.is_file():
        raise ValueError(
            "self-enrollment invite file must be a regular non-symlink file"
        )
    if stat.S_IMODE(code_path.stat().st_mode) != PRIVATE_MODE:
        raise ValueError("self-enrollment invite file must have mode 0600")
    payload = load_json(code_path)
    if payload.get("study_id") != study_id or not isinstance(payload.get("invites"), list):
        raise ValueError("self-enrollment invite file has an invalid contract")
    codes = []
    participant_ids = []
    for row in payload["invites"]:
        if not isinstance(row, Mapping):
            raise ValueError("self-enrollment invite row must be an object")
        code = row.get("invite_code")
        participant_id = row.get("participant_id")
        if not isinstance(code, str) or not code or not isinstance(participant_id, str):
            raise ValueError("self-enrollment invite row is incomplete")
        codes.append(code)
        participant_ids.append(participant_id)
    if (
        len(codes) != expected_count
        or len(codes) != len(set(codes))
        or len(participant_ids) != len(set(participant_ids))
    ):
        raise ValueError(
            "self-enrollment invite file does not match the frozen unique-code count"
        )
    return tuple(codes)


def _load_author_practice_items(
    db_path: str | Path,
    *,
    study_id: str,
    slots: Sequence[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        study = connection.execute(
            "SELECT metadata_json FROM studies WHERE study_id = ?",
            (study_id,),
        ).fetchone()
        if study is None:
            raise ValueError(f"unknown study_id: {study_id}")
        rows = connection.execute(
            """
            SELECT item_id, caption, unit, span_text, target, category,
                   image_asset_json, asset_status
            FROM items
            WHERE study_id = ?
            ORDER BY item_id
            """,
            (study_id,),
        ).fetchall()
        if not rows:
            raise ValueError("author practice source study has no items")
        items: list[dict[str, Any]] = []
        for row in rows:
            if row["asset_status"] != "available" or not row["image_asset_json"]:
                raise ValueError(f"author practice item {row['item_id']} has no verified image")
            asset = json.loads(row["image_asset_json"])
            relative = asset.get("asset_relpath")
            if not isinstance(relative, str) or not relative:
                raise ValueError(f"author practice item {row['item_id']} has no asset_relpath")
            items.append(
                {
                    "item_id": row["item_id"],
                    "caption": row["caption"],
                    "unit": row["unit"],
                    "span": row["span_text"],
                    "target": row["target"],
                    "category": row["category"],
                    "asset_ref": relative,
                }
            )
        ordered_slots = sorted(slots)
        allocated = {slot: [] for slot in ordered_slots}
        for index, item in enumerate(items):
            allocated[ordered_slots[index % len(ordered_slots)]].append(item)
        return allocated, json.loads(study["metadata_json"])
    finally:
        connection.close()


def command_author_practice_serve(args: argparse.Namespace) -> int:
    validate_bind_security(args.host, preview=True, tls_enabled=False)
    codes = _load_author_practice_codes(args.codes)
    author_items, metadata = _load_author_practice_items(
        args.db,
        study_id=args.study_id,
        slots=tuple(codes),
    )
    asset_root_value = args.asset_root or metadata.get("asset_serve_root")
    if not isinstance(asset_root_value, str) or not asset_root_value:
        raise ValueError("--asset-root is required because the source study records none")
    application = create_author_practice_app(
        author_codes=codes,
        author_items=author_items,
        asset_root=asset_root_value,
    )
    print(
        _json_text(
            {
                "practice_mode": True,
                "responses_persisted": False,
                "bind": f"http://{args.host}:{args.port}",
                "tasks_per_author": {
                    slot: len(items) for slot, items in sorted(author_items.items())
                },
            }
        ),
        end="",
        flush=True,
    )
    web.run_app(application, host=args.host, port=args.port, access_log=None)
    return 0


def _complete_manifest_verification(report: Mapping[str, Any]) -> None:
    requested = int(report.get("requested_unique_samples") or 0)
    verified = int(report.get("verified_unique_samples") or 0)
    failed = int(report.get("failed_unique_samples") or 0)
    if requested <= 0 or failed or requested != verified:
        raise RuntimeError(
            "canonical manifest verification failed closed: "
            f"requested={requested}, verified={verified}, failed={failed}"
        )


def _complete_materialization(report: Mapping[str, Any]) -> None:
    requested = int(report.get("requested_unique_images") or 0)
    materialized = int(report.get("materialized_unique_images") or 0)
    failed = int(report.get("failed_unique_images") or 0)
    if requested <= 0 or failed or requested != materialized:
        raise RuntimeError(
            "selected image materialization failed closed: "
            f"requested={requested}, materialized={materialized}, failed={failed}"
        )


def command_materialize(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _, _, images = study_settings(config)
    sample_dir = ensure_private_directory(args.sample_dir)
    items_path = Path(args.items) if args.items else sample_dir / DEFAULT_ITEMS_NAME
    if args.items is None:
        items_path.chmod(PRIVATE_MODE)
        sample_manifest_path = sample_dir / DEFAULT_SAMPLE_MANIFEST_NAME
        if sample_manifest_path.is_file():
            sample_manifest_path.chmod(PRIVATE_MODE)
    asset_root = ensure_private_directory(Path(args.asset_root) if args.asset_root else sample_dir / "assets")
    items = load_jsonl(items_path)
    canonical_report = verify_canonical_manifest(
        items,
        manifest_path=images["canonical_manifest"],
        dataset_prefix=args.dataset_prefix or str(images.get("canonical_dataset_prefix") or "cc12m_202603"),
    )
    canonical_report_path = sample_dir / DEFAULT_CANONICAL_REPORT_NAME
    write_json(canonical_report_path, canonical_report, private=True)
    _complete_manifest_verification(canonical_report)

    verified_by_image: dict[str, dict[str, Any]] = {}
    for row in canonical_report["verified"]:
        image_key = str(row["image_key"])
        prior = verified_by_image.get(image_key)
        if prior is not None and prior != row:
            raise RuntimeError(f"image key resolves to multiple canonical samples: {image_key}")
        verified_by_image[image_key] = row
    extraction_items: list[dict[str, Any]] = []
    for item in items:
        enriched = dict(item)
        verified = verified_by_image.get(str(item.get("image_key") or ""))
        if verified is None:
            raise RuntimeError(f"selected image lacks a canonical manifest record: {item.get('image_key')!r}")
        # The manifest's ``sha256_raw`` can be an upstream sidecar digest
        # (e.g. pre-img2dataset bytes), not necessarily the bytes stored in
        # ``storage_uri``. The exact manifest locator and URL establish
        # identity; materialization computes and records the stored-byte hash.
        extraction_items.append(enriched)

    materialization_report = materialize_selected_images(
        extraction_items,
        wds_root=images["canonical_wds_root"],
        asset_root=asset_root,
        workers=args.workers or int(images.get("materialize_workers") or 16),
    )
    for asset in materialization_report.get("assets", []):
        path = asset.get("path")
        if isinstance(path, str):
            Path(path).chmod(PRIVATE_MODE)
    materialization_report["canonical_report_sha256"] = _sha256_file(canonical_report_path)
    materialization_report_path = sample_dir / DEFAULT_MATERIALIZATION_REPORT_NAME
    write_json(materialization_report_path, materialization_report, private=True)
    _complete_materialization(materialization_report)
    print(
        _json_text(
            {
                "verified_unique_samples": canonical_report["verified_unique_samples"],
                "materialized_unique_images": materialization_report["materialized_unique_images"],
                "asset_root": materialization_report["asset_root"],
                "canonical_report": str(canonical_report_path.resolve()),
                "materialization_report": str(materialization_report_path.resolve()),
            }
        ),
        end="",
    )
    return 0


def resolve_exact_span(
    caption: str,
    raw_span: Any,
) -> tuple[dict[str, int] | None, str]:
    """Resolve only provably exact caption offsets; never guess a fuzzy match."""

    if isinstance(raw_span, Mapping):
        start, end = raw_span.get("start"), raw_span.get("end")
        if type(start) is int and type(end) is int and 0 <= start <= end <= len(caption):
            return {"start": start, "end": end}, "provided_offsets"
        raise ValueError("provided span offsets are invalid")
    if isinstance(raw_span, Sequence) and not isinstance(raw_span, (str, bytes)) and len(raw_span) == 2:
        start, end = raw_span
        if type(start) is int and type(end) is int and 0 <= start <= end <= len(caption):
            return {"start": start, "end": end}, "provided_offsets"
        raise ValueError("provided span offsets are invalid")
    if raw_span is None or raw_span == "":
        return None, "missing"
    if not isinstance(raw_span, str):
        raise ValueError("span must be text, exact offsets, or null")
    starts: list[int] = []
    offset = 0
    while True:
        found = caption.find(raw_span, offset)
        if found < 0:
            break
        starts.append(found)
        offset = found + 1
    if len(starts) == 1:
        start = starts[0]
        return {"start": start, "end": start + len(raw_span)}, "exact_unique_text"
    if not starts:
        return None, "not_found"
    return None, "ambiguous"


def _safe_asset_relpath(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("materialized asset_relpath must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"unsafe materialized asset_relpath: {value!r}")
    return str(path)


def _index_unique(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            raise ValueError(f"{label} row is missing {key}")
        normalized = dict(row)
        prior = indexed.get(value)
        if prior is not None and prior != normalized:
            raise ValueError(f"{label} contains conflicting rows for {key}={value!r}")
        indexed[value] = normalized
    return indexed


def normalize_store_items(
    builder_rows: Sequence[Mapping[str, Any]],
    *,
    canonical_report: Mapping[str, Any],
    materialization_report: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project rich builder rows to the strict store schema."""

    _complete_manifest_verification(canonical_report)
    _complete_materialization(materialization_report)
    canonical_by_image = _index_unique(
        canonical_report.get("verified", []),
        key="image_key",
        label="canonical report",
    )
    assets_by_image = _index_unique(
        materialization_report.get("assets", []),
        key="image_key",
        label="materialization report",
    )
    asset_serve_root = Path(str(materialization_report["asset_root"])).resolve().parent
    normalized: list[dict[str, Any]] = []
    span_counts: Counter[str] = Counter()
    for row in builder_rows:
        image_key = str(row.get("image_key") or "")
        canonical = canonical_by_image.get(image_key)
        asset = assets_by_image.get(image_key)
        if canonical is None or asset is None:
            raise ValueError(f"item {row.get('item_id')!r} has no verified materialized image")
        stable_image_id = str(canonical.get("stable_image_id") or "")
        sample_id = str(canonical.get("sample_id") or "")
        if not stable_image_id or not sample_id:
            raise ValueError(f"canonical image record lacks stable identity for {image_key!r}")
        asset_relpath = _safe_asset_relpath(asset.get("asset_relpath"))
        asset_path = (asset_serve_root / asset_relpath).resolve()
        if asset_serve_root not in asset_path.parents or not asset_path.is_file():
            raise ValueError(f"materialized asset is outside the serve root or missing: {asset_relpath}")
        observed_bytes = asset_path.stat().st_size
        if int(asset.get("bytes") or -1) != observed_bytes:
            raise ValueError(f"materialized asset size changed: {asset_relpath}")
        materialized_sha256 = asset.get("sha256")
        if not isinstance(materialized_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", materialized_sha256):
            raise ValueError(f"materialized asset lacks a valid byte SHA-256: {asset_relpath}")
        materialized_sha256 = materialized_sha256.lower()
        observed_sha256 = _sha256_file(asset_path)
        if observed_sha256 != materialized_sha256:
            raise ValueError(f"materialized asset SHA-256 changed: {asset_relpath}")

        caption = str(row.get("caption") or "")
        span, span_resolution = resolve_exact_span(caption, row.get("span"))
        span_counts[span_resolution] += 1
        population_size = int(row["stratum_population"])
        stratum_sample_size = int(row["stratum_sample_size"])
        if population_size <= 0 or not 0 < stratum_sample_size <= population_size:
            raise ValueError(f"invalid stratum counts for item {row.get('item_id')!r}")
        base_probability = stratum_sample_size / population_size
        repeat_of = row.get("repeat_of")
        is_repeat = bool(row.get("is_repeat")) or repeat_of is not None
        if is_repeat != (repeat_of is not None):
            raise ValueError(f"repeat metadata is inconsistent for item {row.get('item_id')!r}")
        judges = row.get("judge_answers")
        if not isinstance(judges, Mapping) or "qwen" not in judges or "gemma" not in judges:
            raise ValueError(f"item {row.get('item_id')!r} lacks both judge answers")
        population = {
            "family": row.get("family"),
            "surface_group": row.get("surface_group"),
            "stratum": row.get("stratum"),
            "population_claims": population_size,
        }
        sample = {
            "question_id": row.get("question_id"),
            "caption_id": row.get("caption_id"),
            "extractor_request_id": row.get("extractor_request_id"),
            "source_row": row.get("source_row"),
            "is_repeat": is_repeat,
            "span_text": row.get("span"),
            "span_resolution": span_resolution,
            "stratum_sample_size": stratum_sample_size,
            "base_sample_probability": base_probability,
        }
        normalized.append(
            {
                "item_id": str(row["item_id"]),
                "caption": caption,
                "unit": str(row["unit"]),
                "span": span,
                "target": row.get("target"),
                "category": str(row["category"]),
                "surface": str(row["surface"]),
                "item_group": stable_image_id,
                "image_locator": {
                    "dataset": sample_id.split(":", 1)[0],
                    "sample_id": sample_id,
                    "stable_image_id": stable_image_id,
                    "source_url_sha1": canonical.get("source_url_sha1"),
                    "storage_uri": canonical.get("storage_uri"),
                },
                # The participant-facing image endpoint receives only this
                # opaque relative reference; no surface, caption, URL, judge,
                # or source path is embedded in the served asset metadata.
                "image_asset": {
                    "asset_relpath": asset_relpath,
                    "sha256": materialized_sha256,
                },
                "asset_status": "available",
                "qwen_answer": judges["qwen"],
                "gemma_answer": judges["gemma"],
                "stratum": str(row["stratum"]),
                "population": population,
                "sample": sample,
                "weight": 0.0 if is_repeat else float(row["sampling_weight"]),
                "population_size": population_size,
                "sample_probability": None if is_repeat else base_probability,
                "repeat_of": repeat_of,
            }
        )
    item_ids = {item["item_id"] for item in normalized}
    for item in normalized:
        repeat_of = item["repeat_of"]
        if repeat_of is not None and repeat_of not in item_ids:
            raise ValueError(f"repeat item {item['item_id']!r} references absent base item {repeat_of!r}")
    report = {
        "items": len(normalized),
        "base_items": sum(item["repeat_of"] is None for item in normalized),
        "repeat_items": sum(item["repeat_of"] is not None for item in normalized),
        "unique_image_groups": len({item["item_group"] for item in normalized}),
        "span_resolution": dict(sorted(span_counts.items())),
        "asset_serve_root": str(asset_serve_root),
        "hidden_repeat_primary_weight": 0.0,
        "hidden_repeat_sample_probability": None,
    }
    return normalized, report


_REPEAT_ALLOWED_TOP_LEVEL_DIFFERENCES = frozenset(
    {
        "item_id",
        "repeat_of",
        "is_repeat",
        "repeat_index",
        "weight",
        "sampling_weight",
        "sample_probability",
        "position",
        "order",
        "order_key",
    }
)
_REPEAT_ALLOWED_SAMPLE_DIFFERENCES = frozenset(
    {
        "is_repeat",
        "repeat_index",
        "position",
        "order",
        "order_key",
    }
)
_MISSING_REPEAT_FIELD = object()


def _repeat_payload_difference(
    repeat_value: Any,
    original_value: Any,
    *,
    path: str,
) -> str | None:
    if isinstance(repeat_value, Mapping) and isinstance(original_value, Mapping):
        allowed = _REPEAT_ALLOWED_SAMPLE_DIFFERENCES if path == "sample" else frozenset()
        for key in sorted((set(repeat_value) | set(original_value)) - allowed):
            nested_path = f"{path}.{key}"
            difference = _repeat_payload_difference(
                repeat_value.get(key, _MISSING_REPEAT_FIELD),
                original_value.get(key, _MISSING_REPEAT_FIELD),
                path=nested_path,
            )
            if difference is not None:
                return difference
        return None
    return path if repeat_value != original_value else None


def _validate_repeat_payload_identity(
    repeat: Mapping[str, Any],
    original: Mapping[str, Any],
) -> None:
    """Require every UI/analysis payload field to match its hidden original."""

    repeat_id = repeat.get("item_id")
    original_id = original.get("item_id")
    fields = (set(repeat) | set(original)) - _REPEAT_ALLOWED_TOP_LEVEL_DIFFERENCES
    for field in sorted(fields):
        repeat_value = repeat.get(field, _MISSING_REPEAT_FIELD)
        original_value = original.get(field, _MISSING_REPEAT_FIELD)
        difference = _repeat_payload_difference(
            repeat_value,
            original_value,
            path=field,
        )
        if difference is not None:
            raise ValueError(
                f"repeat item {repeat_id!r} semantic payload does not match base item {original_id!r} on {difference}"
            )


def validate_frozen_sample_design(
    items: Sequence[Mapping[str, Any]],
    *,
    surface_groups: Mapping[str, Sequence[str]],
    semantic_visual_claim_types: Sequence[str],
    expected_strata: Sequence[str],
    claims_per_stratum: int,
    expected_repeat_items: int | None = None,
    validate_sampling_metadata: bool = False,
) -> dict[str, Any]:
    """Reject any frozen sample that differs from its configured factorial design."""

    surface_to_group = {str(surface): str(group) for group, surfaces in surface_groups.items() for surface in surfaces}
    allowed_types = set(semantic_visual_claim_types)
    expected = set(expected_strata)
    by_id = {str(item["item_id"]): item for item in items}
    base_counts: Counter[str] = Counter()
    population_by_stratum: dict[str, int] = {}
    sampling_weight_by_stratum: dict[str, float] = {}
    for item in items:
        item_id = str(item["item_id"])
        surface = str(item["surface"])
        category = str(item["category"])
        group = surface_to_group.get(surface)
        if group is None:
            raise ValueError(f"frozen sample item {item_id!r} has surface {surface!r} outside study.surface_groups")
        if category not in allowed_types:
            raise ValueError(
                f"frozen sample item {item_id!r} has category {category!r} outside study.semantic_visual_claim_types"
            )
        expected_stratum = f"{group}/{category}"
        if str(item["stratum"]) != expected_stratum:
            raise ValueError(
                f"frozen sample item {item_id!r} has stratum {item['stratum']!r}; "
                f"expected {expected_stratum!r} from its surface and category"
            )
        if item.get("repeat_of") is None:
            base_counts[expected_stratum] += 1
            if not validate_sampling_metadata:
                continue
            population_size = item.get("population_size")
            sample_probability = item.get("sample_probability")
            sampling_weight = item.get("weight")
            sample_metadata = item.get("sample")
            stratum_sample_size = (
                sample_metadata.get("stratum_sample_size") if isinstance(sample_metadata, Mapping) else None
            )
            if (
                isinstance(population_size, bool)
                or not isinstance(population_size, int)
                or population_size < claims_per_stratum
                or stratum_sample_size != claims_per_stratum
            ):
                raise ValueError(f"frozen sample item {item_id!r} has invalid population/sample counts")
            expected_probability = claims_per_stratum / population_size
            expected_weight = population_size / claims_per_stratum
            if (
                isinstance(sample_probability, bool)
                or not isinstance(sample_probability, (int, float))
                or not math.isclose(
                    float(sample_probability),
                    expected_probability,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                or isinstance(sampling_weight, bool)
                or not isinstance(sampling_weight, (int, float))
                or not math.isclose(
                    float(sampling_weight),
                    expected_weight,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(f"frozen sample item {item_id!r} has inconsistent inverse-probability weight")
            previous_population = population_by_stratum.setdefault(
                expected_stratum,
                population_size,
            )
            previous_weight = sampling_weight_by_stratum.setdefault(
                expected_stratum,
                expected_weight,
            )
            if previous_population != population_size or not math.isclose(
                previous_weight,
                expected_weight,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"frozen sample stratum {expected_stratum!r} has inconsistent sampling metadata")

    observed = set(base_counts)
    if observed != expected:
        raise ValueError("frozen sample surface×type cells do not match the configured study design")
    wrong_counts = {
        stratum: base_counts[stratum] for stratum in sorted(expected) if base_counts[stratum] != claims_per_stratum
    }
    expected_base_items = len(expected) * claims_per_stratum
    base_items = sum(base_counts.values())
    if wrong_counts or base_items != expected_base_items:
        details = ", ".join(f"{stratum}={count}/{claims_per_stratum}" for stratum, count in wrong_counts.items())
        raise ValueError(
            "frozen sample does not contain exactly study.claims_per_stratum "
            f"base items in every surface×type cell: {details}"
        )

    repeat_count = 0
    for item in items:
        repeat_of = item.get("repeat_of")
        if repeat_of is None:
            continue
        repeat_count += 1
        original = by_id[str(repeat_of)]
        if original.get("repeat_of") is not None:
            raise ValueError(f"repeat item {item['item_id']!r} must reference a base item")
        _validate_repeat_payload_identity(item, original)
    if expected_repeat_items is not None and repeat_count != expected_repeat_items:
        raise ValueError(
            "frozen sample hidden-repeat count does not match the configured protocol: "
            f"{repeat_count}/{expected_repeat_items}"
        )
    return {
        "base_items": base_items,
        "repeat_items": repeat_count,
        "claims_per_stratum": claims_per_stratum,
        "base_items_by_stratum": dict(sorted(base_counts.items())),
        "population_by_stratum": dict(sorted(population_by_stratum.items())),
        "sampling_weight_by_stratum": dict(sorted(sampling_weight_by_stratum.items())),
    }


def command_init_db(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    study, _, _ = study_settings(config)
    protocol_version = str(study.get("protocol_version") or "")
    claims_per_stratum = _positive_integer(
        study.get("claims_per_stratum"),
        "study.claims_per_stratum",
    )
    required_labels_per_item = study.get("required_labels_per_item")
    if (
        isinstance(required_labels_per_item, bool)
        or not isinstance(required_labels_per_item, int)
        or required_labels_per_item < 2
    ):
        raise ValueError("study.required_labels_per_item must be an integer >= 2")
    if protocol_version == HUMAN_CBU_V1_PROTOCOL and required_labels_per_item != 2:
        raise ValueError("human_cbu_v1 requires exactly 2 initial labels per item")
    adjudicator_pseudonym = study.get("adjudicator_pseudonym")
    if protocol_version == HUMAN_CBU_V1_PROTOCOL and (
        not isinstance(adjudicator_pseudonym, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", adjudicator_pseudonym)
    ):
        raise ValueError("human_cbu_v1 requires study.adjudicator_pseudonym as a short pseudonymous token")
    repeat_fraction_raw = study.get("repeat_fraction", 0.0)
    if isinstance(repeat_fraction_raw, bool) or not isinstance(repeat_fraction_raw, (int, float)):
        raise ValueError("study.repeat_fraction must be a number in [0, 1]")
    repeat_fraction = float(repeat_fraction_raw)
    if not 0.0 <= repeat_fraction <= 1.0:
        raise ValueError("study.repeat_fraction must be a number in [0, 1]")
    primary_participant_count = study.get("primary_participant_count")
    if primary_participant_count is not None and (
        isinstance(primary_participant_count, bool)
        or not isinstance(primary_participant_count, int)
        or primary_participant_count < required_labels_per_item
    ):
        raise ValueError("study.primary_participant_count must be an integer >= study.required_labels_per_item")
    reserve_participant_count = study.get("reserve_participant_count", 0)
    if (
        isinstance(reserve_participant_count, bool)
        or not isinstance(reserve_participant_count, int)
        or reserve_participant_count < 0
    ):
        raise ValueError("study.reserve_participant_count must be a non-negative integer")
    target_items_per_participant = study.get("target_items_per_participant")
    if target_items_per_participant is not None and (
        isinstance(target_items_per_participant, bool)
        or not isinstance(target_items_per_participant, int)
        or target_items_per_participant <= 0
    ):
        raise ValueError(
            "study.target_items_per_participant must be a positive integer"
        )
    optional_extension_batch_size = study.get("optional_extension_batch_size")
    optional_extension_max_items = study.get("optional_extension_max_items")
    overlap_target_items_per_stratum = study.get("overlap_target_items_per_stratum", 0)
    if (
        isinstance(overlap_target_items_per_stratum, bool)
        or not isinstance(overlap_target_items_per_stratum, int)
        or overlap_target_items_per_stratum < 0
    ):
        raise ValueError(
            "study.overlap_target_items_per_stratum must be a non-negative integer"
        )
    extension_values = (optional_extension_batch_size, optional_extension_max_items)
    if any(value is not None for value in extension_values):
        if target_items_per_participant is None:
            raise ValueError("optional work requires study.target_items_per_participant")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in extension_values
        ):
            raise ValueError(
                "study optional extension batch size and maximum must both be positive integers"
            )
        if optional_extension_max_items < target_items_per_participant:
            raise ValueError(
                "study.optional_extension_max_items must be at least the participant target"
            )
    self_enrollment_limit = study.get("self_enrollment_limit")
    if self_enrollment_limit is not None and (
        isinstance(self_enrollment_limit, bool)
        or not isinstance(self_enrollment_limit, int)
        or self_enrollment_limit <= 0
    ):
        raise ValueError("study.self_enrollment_limit must be a positive integer")
    if (
        self_enrollment_limit is not None
        and primary_participant_count is not None
        and self_enrollment_limit != primary_participant_count
    ):
        raise ValueError(
            "study.self_enrollment_limit must match study.primary_participant_count"
        )
    sample_dir = ensure_private_directory(args.sample_dir)
    items_path = Path(args.items) if args.items else sample_dir / DEFAULT_ITEMS_NAME
    sample_manifest_path = (
        Path(args.sample_manifest) if args.sample_manifest else sample_dir / DEFAULT_SAMPLE_MANIFEST_NAME
    )
    canonical_report_path = (
        Path(args.canonical_report) if args.canonical_report else sample_dir / DEFAULT_CANONICAL_REPORT_NAME
    )
    materialization_report_path = (
        Path(args.materialization_report)
        if args.materialization_report
        else sample_dir / DEFAULT_MATERIALIZATION_REPORT_NAME
    )
    init_report_path = Path(args.init_report) if args.init_report else sample_dir / DEFAULT_INIT_REPORT_NAME
    for private_input in (
        items_path,
        sample_manifest_path,
        canonical_report_path,
        materialization_report_path,
    ):
        private_input.chmod(PRIVATE_MODE)
    destination = Path(args.db)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite an existing study database: {destination}")
    builder_rows, items_sha256 = load_jsonl_with_sha256(items_path)
    sample_manifest = load_json(sample_manifest_path)
    requires_items_binding = (
        protocol_version == HUMAN_CBU_V1_PROTOCOL or sample_manifest.get("protocol") == HUMAN_CBU_V1_PROTOCOL
    )
    items_binding = validate_sample_items_sha256(
        sample_manifest,
        items_sha256,
        required=requires_items_binding,
    )
    canonical_report = load_json(canonical_report_path)
    materialization_report = load_json(materialization_report_path)
    normalized, normalization_report = normalize_store_items(
        builder_rows,
        canonical_report=canonical_report,
        materialization_report=materialization_report,
    )
    configured_types, configured_strata = validate_study_design(
        study["surface_groups"],
        study.get("semantic_visual_claim_types"),
    )
    expected_base_items = len(configured_strata) * claims_per_stratum
    expected_repeat_items = round(expected_base_items * repeat_fraction)
    design_report = validate_frozen_sample_design(
        normalized,
        surface_groups=study["surface_groups"],
        semantic_visual_claim_types=configured_types,
        expected_strata=configured_strata,
        claims_per_stratum=claims_per_stratum,
        expected_repeat_items=(expected_repeat_items if protocol_version == HUMAN_CBU_V1_PROTOCOL else None),
        validate_sampling_metadata=(protocol_version == HUMAN_CBU_V1_PROTOCOL),
    )
    manifest_types = sample_manifest.get("semantic_visual_claim_types")
    manifest_strata = sample_manifest.get("expected_strata")
    if manifest_types != configured_types or manifest_strata != configured_strata:
        raise ValueError("sample manifest design does not match the configured study design")
    if sample_manifest.get("surface_groups") != study["surface_groups"]:
        raise ValueError("sample manifest surface_groups do not match the configured study design")
    if sample_manifest.get("claims_per_stratum") != claims_per_stratum:
        raise ValueError("sample manifest claims_per_stratum does not match study.claims_per_stratum")
    if protocol_version == HUMAN_CBU_V1_PROTOCOL:
        if sample_manifest.get("seed") != int(study["seed"]):
            raise ValueError("sample manifest seed does not match study.seed")
        manifest_sample = sample_manifest.get("sample")
        manifest_strata_rows = manifest_sample.get("strata") if isinstance(manifest_sample, Mapping) else None
        if not isinstance(manifest_strata_rows, Sequence) or isinstance(
            manifest_strata_rows,
            (str, bytes),
        ):
            raise ValueError("sample manifest must contain the frozen sampling strata")
        manifest_strata = {str(row.get("stratum")): row for row in manifest_strata_rows if isinstance(row, Mapping)}
        if set(manifest_strata) != set(configured_strata) or len(manifest_strata_rows) != len(configured_strata):
            raise ValueError("sample manifest sampling strata do not match the configured design")
        for stratum in configured_strata:
            row = manifest_strata[stratum]
            expected_population = design_report["population_by_stratum"][stratum]
            expected_weight = design_report["sampling_weight_by_stratum"][stratum]
            if (
                row.get("population_claims") != expected_population
                or row.get("sampled_claims") != claims_per_stratum
                or isinstance(row.get("sampling_weight"), bool)
                or not isinstance(row.get("sampling_weight"), (int, float))
                or not math.isclose(
                    float(row["sampling_weight"]),
                    expected_weight,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(f"sample manifest sampling metadata does not match frozen stratum {stratum!r}")
        if sample_manifest.get("repeat_fraction") != repeat_fraction:
            raise ValueError("sample manifest repeat_fraction does not match study.repeat_fraction")
        manifest_repeat_items = manifest_sample.get("repeat_items") if isinstance(manifest_sample, Mapping) else None
        if manifest_repeat_items != expected_repeat_items:
            raise ValueError("sample manifest repeat_items does not match the configured hidden-repeat count")
    readiness = dict(study.get("readiness", {}))
    for field, configured_design in (
        ("expected_proposed_types", sorted(configured_types)),
        ("expected_strata", configured_strata),
    ):
        configured = readiness.get(field)
        if configured is not None:
            if not isinstance(configured, Sequence) or isinstance(configured, (str, bytes)):
                raise ValueError(f"study.readiness.{field} must be a sequence")
            if sorted(str(value) for value in configured) != configured_design:
                raise ValueError(f"study.readiness.{field} does not match the configured design")
        readiness[field] = configured_design
    if str(sample_manifest.get("study_namespace")) != str(study["id"]):
        raise ValueError("sample manifest study_namespace does not match config study.id")
    _ensure_parent(destination)
    with tempfile.TemporaryDirectory(
        prefix=".human-cbu-init-",
        dir=destination.parent,
    ) as temporary_dir:
        draft_path = Path(temporary_dir) / "draft.sqlite3"
        store = HumanCBUStore.initialize(draft_path)
        metadata = {
            "purpose": "blinded human-anchored CBU validation",
            "token_budget": int(study["token_budget"]),
            "surface_groups": study["surface_groups"],
            "collect": study.get("collect", {}),
            "readiness": readiness,
            "asset_serve_root": normalization_report["asset_serve_root"],
            "config_sha256": _sha256_file(args.config),
            "sample_manifest_sha256": _sha256_file(sample_manifest_path),
            "items_sha256": items_sha256,
            "items_manifest_binding": items_binding,
            "canonical_report_sha256": _sha256_file(canonical_report_path),
            "materialization_report_sha256": _sha256_file(materialization_report_path),
            "sample_seed": int(study["seed"]),
            "assignment_seed": str(study["seed"]),
            "required_labels_per_item": required_labels_per_item,
            "adjudicator_pseudonym": adjudicator_pseudonym,
            "repeat_fraction": repeat_fraction,
            "expected_repeat_items": expected_repeat_items,
            "claims_per_stratum": claims_per_stratum,
            "primary_participant_count": primary_participant_count,
            "reserve_participant_count": reserve_participant_count,
            "target_items_per_participant": target_items_per_participant,
            "optional_extension_batch_size": optional_extension_batch_size,
            "optional_extension_max_items": optional_extension_max_items,
            "overlap_target_items_per_stratum": overlap_target_items_per_stratum,
            "self_enrollment_limit": self_enrollment_limit,
            "semantic_visual_claim_types": configured_types,
            "expected_strata": configured_strata,
            "ethics_or_irb_equivalent_determination_id": None,
        }
        created = store.create_study(
            str(study["id"]),
            title=str(study["title"]),
            protocol_version=str(study["protocol_version"]),
            consent_version=str(study["consent_version"]),
            metadata=metadata,
        )
        store.add_items(str(study["id"]), normalized)
        store.backup(destination)
    destination.chmod(PRIVATE_MODE)
    report = {
        **normalization_report,
        "design": design_report,
        "items_manifest_binding": items_binding,
        "database": str(destination.resolve()),
        "database_mode": oct(destination.stat().st_mode & 0o777),
        "study": created,
        "status": "draft",
        "ethics_gate": "not_recorded",
    }
    write_json(init_report_path, report, private=True)
    print(_json_text(report), end="")
    return 0


def _open_store(args: argparse.Namespace) -> HumanCBUStore:
    return HumanCBUStore.open(args.db)


def command_invites(args: argparse.Namespace) -> int:
    output = Path(args.output)
    _ensure_parent(output)
    if output.exists():
        raise FileExistsError(output)
    reservation = output.with_name(f".{output.name}.reservation")
    descriptor = os.open(
        reservation,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        PRIVATE_MODE,
    )
    try:
        os.close(descriptor)
        descriptor = -1
        store = _open_store(args)
        invites = store.create_participant_invites(
            args.study_id,
            count=args.count,
            expires_in_seconds=args.expires_in_seconds,
        )
        secure_write_once_json(
            output,
            {"study_id": args.study_id, "invites": invites},
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        reservation.unlink(missing_ok=True)
    print(
        _json_text(
            {
                "study_id": args.study_id,
                "invites_created": len(invites),
                "private_output": str(output.resolve()),
                "mode": oct(output.stat().st_mode & 0o777),
            }
        ),
        end="",
    )
    return 0


def command_provision_adjudicator(args: argparse.Namespace) -> int:
    output = Path(args.output)
    _ensure_parent(output)
    if output.exists():
        raise FileExistsError(output)
    reservation = output.with_name(f".{output.name}.reservation")
    descriptor = os.open(
        reservation,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        PRIVATE_MODE,
    )
    try:
        os.close(descriptor)
        descriptor = -1
        provisioned = _open_store(args).provision_adjudicator(args.study_id)
        secure_write_once_json(output, provisioned)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        reservation.unlink(missing_ok=True)
    print(
        _json_text(
            {
                "study_id": args.study_id,
                "adjudicator_pseudonym": provisioned["adjudicator_pseudonym"],
                "private_output": str(output.resolve()),
                "mode": oct(output.stat().st_mode & 0o777),
            }
        ),
        end="",
    )
    return 0


def command_assign(args: argparse.Namespace) -> int:
    config_study: Mapping[str, Any] = {}
    if args.config:
        config_study, _, _ = study_settings(load_config(args.config))
        if str(config_study["id"]) != args.study_id:
            raise ValueError("config study.id does not match --study-id")
    store = _open_store(args)
    frozen_study = store.get_study(args.study_id)
    frozen_labels = frozen_study["metadata"].get("required_labels_per_item")
    if isinstance(frozen_labels, bool) or not isinstance(frozen_labels, int) or frozen_labels < 2:
        raise ValueError("study metadata must freeze required_labels_per_item >= 2")
    configured_labels = config_study.get("required_labels_per_item")
    if configured_labels is not None and int(configured_labels) != frozen_labels:
        raise ValueError("config required_labels_per_item does not match the frozen study metadata")
    if args.labels_per_item is not None and args.labels_per_item != frozen_labels:
        raise ValueError("--labels-per-item does not match the frozen required_labels_per_item")
    labels = frozen_labels
    frozen_primary = frozen_study["metadata"].get("primary_participant_count")
    frozen_reserves = frozen_study["metadata"].get("reserve_participant_count", 0)
    configured_primary = config_study.get("primary_participant_count")
    configured_reserves = config_study.get("reserve_participant_count")
    if configured_primary is not None and configured_primary != frozen_primary:
        raise ValueError("config primary_participant_count does not match the frozen study metadata")
    if configured_reserves is not None and configured_reserves != frozen_reserves:
        raise ValueError("config reserve_participant_count does not match the frozen study metadata")
    frozen_target = frozen_study["metadata"].get("target_items_per_participant")
    configured_target = config_study.get("target_items_per_participant")
    if configured_target is not None and configured_target != frozen_target:
        raise ValueError(
            "config target_items_per_participant does not match the frozen study metadata"
        )
    frozen_overlap_target = frozen_study["metadata"].get(
        "overlap_target_items_per_stratum", 0
    )
    configured_overlap_target = config_study.get("overlap_target_items_per_stratum")
    if (
        configured_overlap_target is not None
        and configured_overlap_target != frozen_overlap_target
    ):
        raise ValueError(
            "config overlap_target_items_per_stratum does not match frozen study metadata"
        )
    frozen_seed = frozen_study["metadata"].get("assignment_seed")
    if not isinstance(frozen_seed, str) or not frozen_seed:
        raise ValueError("study metadata must freeze a nonempty assignment_seed")
    configured_seed = config_study.get("seed")
    if configured_seed is not None and str(configured_seed) != frozen_seed:
        raise ValueError("config seed does not match the frozen assignment_seed")
    if args.seed is not None and args.seed != frozen_seed:
        raise ValueError("--seed does not match the frozen assignment_seed")
    seed = frozen_seed
    assignment_method = (
        store.plan_remaining_first_assignments
        if getattr(args, "remaining_first", False)
        else store.assign_items
    )
    report = assignment_method(
        args.study_id,
        labels_per_item=labels,
        primary_participant_count=frozen_primary,
        reserve_participant_count=frozen_reserves,
        seed=seed,
    )
    print(_json_text(report), end="")
    return 0


def command_replace_participant(args: argparse.Namespace) -> int:
    if args.confirm_source_participant_id != args.source_participant_id:
        raise ValueError("--confirm-source-participant-id must exactly match --source-participant-id")
    report = _open_store(args).replace_participant(
        args.study_id,
        source_participant_id=args.source_participant_id,
        replacement_participant_id=args.replacement_participant_id,
    )
    print(_json_text(report), end="")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    report = _open_store(args).validation_report(
        args.study_id,
        require_assets=not args.allow_missing_assets,
    )
    print(_json_text(report), end="")
    return 0 if report["valid"] else 2


def command_seal(args: argparse.Namespace) -> int:
    determination = args.ethics_determination_id.strip()
    determination_error = ethics_review_basis_error(determination)
    if determination_error is not None:
        raise ValueError(determination_error)
    participant_information = validate_participant_information(load_json(args.participant_information_json))
    store = _open_store(args)
    current = store.get_study(args.study_id)
    if participant_information["approved_version"] != current["consent_version"]:
        raise ValueError("participant-information approved_version does not match the study consent_version")
    if current["status"] == "draft":
        metadata = dict(current["metadata"])
        existing = metadata.get("ethics_or_irb_equivalent_determination_id")
        if existing not in {None, determination}:
            raise ValueError("draft already records a different ethics-review basis")
        existing_information = metadata.get("participant_information")
        if existing_information is not None and existing_information != participant_information:
            raise ValueError("draft already records a different participant notice")
        metadata["ethics_or_irb_equivalent_determination_id"] = determination
        metadata["participant_information"] = participant_information
        store.update_study_metadata(args.study_id, metadata=metadata)
    elif current["status"] == "ready":
        existing = current["metadata"].get("ethics_or_irb_equivalent_determination_id")
        if existing != determination:
            raise ValueError("sealed study ethics-review basis does not match the supplied value")
        if current["metadata"].get("participant_information") != participant_information:
            raise ValueError("sealed study participant notice does not match the supplied form")
    else:
        raise ValueError(f"cannot seal a study with status {current['status']!r}")
    report = store.seal_validation(args.study_id, require_assets=True)
    report["ethics_or_irb_equivalent_determination_id"] = determination
    print(_json_text(report), end="")
    return 0


def command_open(args: argparse.Namespace) -> int:
    if args.confirm_open != args.study_id:
        raise ValueError("--confirm-open must exactly equal --study-id; no collection was opened")
    result = _open_store(args).set_study_status(args.study_id, "open")
    print(_json_text(result), end="")
    return 0


def command_pause(args: argparse.Namespace) -> int:
    result = _open_store(args).set_study_status(args.study_id, "paused")
    print(_json_text(result), end="")
    return 0


def command_close(args: argparse.Namespace) -> int:
    result = _open_store(args).set_study_status(
        args.study_id,
        "closed",
        allow_incomplete=args.allow_incomplete,
    )
    print(_json_text(result), end="")
    return 0


def command_serve(args: argparse.Namespace) -> int:
    if bool(args.tls_cert) != bool(args.tls_key):
        raise ValueError("--tls-cert and --tls-key must be provided together")
    tls_enabled = bool(args.tls_cert and args.tls_key)
    validate_bind_security(args.host, preview=False, tls_enabled=tls_enabled)
    ssl_context = None
    if tls_enabled:
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(args.tls_cert, args.tls_key)
    store = _open_store(args)
    study = store.get_study(args.study_id)
    if study["status"] not in {"open", "paused", "closed"}:
        raise ValueError(
            "production participant serving requires an open, paused, or closed "
            f"study, got {study['status']!r}; use the preview command for visual inspection"
        )
    asset_root_value = args.asset_root or study["metadata"].get("asset_serve_root")
    if not isinstance(asset_root_value, str) or not asset_root_value:
        raise ValueError("--asset-root is required because the study metadata has no asset_serve_root")
    asset_root = Path(asset_root_value).resolve()
    if not asset_root.is_dir():
        raise FileNotFoundError(asset_root)
    enrollment_limit = study["metadata"].get("self_enrollment_limit")
    if enrollment_limit is not None and (
        isinstance(enrollment_limit, bool)
        or not isinstance(enrollment_limit, int)
        or enrollment_limit <= 0
    ):
        raise ValueError("study metadata contains an invalid self_enrollment_limit")
    self_enrollment_invites = getattr(args, "self_enrollment_invites", None)
    if enrollment_limit is not None and not self_enrollment_invites:
        raise ValueError(
            "--self-enrollment-invites is required by the frozen study plan"
        )
    if enrollment_limit is None and self_enrollment_invites:
        raise ValueError(
            "--self-enrollment-invites is not allowed when the study plan disables it"
        )
    enrollment_codes = (
        ()
        if enrollment_limit is None
        else _load_self_enrollment_codes(
            self_enrollment_invites,
            study_id=args.study_id,
            expected_count=enrollment_limit,
        )
    )
    application = create_study_app(
        store=store,
        study_id=args.study_id,
        asset_root=asset_root,
        participant_information=study["metadata"].get("participant_information"),
        self_enrollment_codes=enrollment_codes,
    )
    web.run_app(
        application,
        host=args.host,
        port=args.port,
        access_log=None,
        ssl_context=ssl_context,
    )
    return 0


def command_status(args: argparse.Namespace) -> int:
    store = _open_store(args)
    study = store.get_study(args.study_id)
    include_analysis = study["status"] in {"closed", "archived"}
    print(
        _json_text(
            store.status_summary(
                args.study_id,
                include_analysis=include_analysis,
            )
        ),
        end="",
    )
    return 0


def command_backup(args: argparse.Namespace) -> int:
    output = _open_store(args).backup(args.output, overwrite=args.overwrite)
    output.chmod(PRIVATE_MODE)
    print(
        _json_text(
            {
                "backup": str(output.resolve()),
                "mode": oct(output.stat().st_mode & 0o777),
            }
        ),
        end="",
    )
    return 0


def command_export(args: argparse.Namespace) -> int:
    store = _open_store(args)
    study = store.get_study(args.study_id)
    study_status = study["status"]
    if study_status not in {"closed", "archived"}:
        raise ValueError(f"export requires a closed or archived study; current status is {study_status!r}")
    status = store.status_summary(args.study_id, include_analysis=True)
    rows = store.export_rows(
        args.study_id,
        include_exact_timestamps=args.include_exact_timestamps,
        include_notes=args.include_notes,
    )
    output_dir = ensure_private_directory(args.output_dir)
    result = write_export_bundle(
        rows,
        output_dir,
        status=status,
        participant_summary=store.participant_summary(args.study_id),
        study_provenance={
            "ethics_review_basis": study["metadata"].get(
                "ethics_or_irb_equivalent_determination_id"
            ),
            "participant_notice_version": study["metadata"]
            .get("participant_information", {})
            .get("approved_version"),
        },
        readiness_thresholds=study["metadata"].get("readiness"),
        include_public_rows=args.include_public_rows,
        public_id_namespace=(
            (args.public_id_namespace or f"{args.study_id}:public-v1") if args.include_public_rows else None
        ),
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(_json_text(result), end="")
    return 0


def _load_adjudication_packet_for_review(
    packet_dir: Path,
    *,
    semantic_categories: Sequence[str],
) -> tuple[dict[str, Any], bytes, bytes, Path | None]:
    """Load and bind the exact sidecar, deterministic page, and optional image."""

    packet, packet_raw = _load_private_json(
        packet_dir / "packet.json",
        label="adjudication packet sidecar",
        maximum_bytes=MAX_ADJUDICATION_PACKET_BYTES,
    )
    phase = packet.get("phase")
    image_sources = {str(packet.get("packet_id") or ""): "packet-local"} if phase == "image" else {}
    _validate_adjudication_packet_inputs(
        [packet],
        image_sources,
        expected_phase=str(phase),
    )
    if packet["packet_id"] != packet_dir.name:
        raise ValueError("adjudication packet directory name does not match packet_id")

    from audit_recap_t2i.human_cbu.adjudication import (
        render_adjudication_html_bytes,
    )

    expected_review = render_adjudication_html_bytes(
        packet,
        semantic_categories=semantic_categories,
    )
    review_raw = _read_private_regular_file(
        packet_dir / "review.html",
        label="adjudication review page",
        maximum_bytes=MAX_ADJUDICATION_REVIEW_BYTES,
    )
    if not secrets.compare_digest(review_raw, expected_review):
        raise ValueError("adjudication review page does not deterministically match its packet sidecar")

    image_path: Path | None = None
    allowed_entries = {"packet.json", "review.html"}
    if phase == "image":
        image_name = packet["evidence"]["image_file"]
        image_path = packet_dir / image_name
        image_raw = _read_private_regular_file(
            image_path,
            label="adjudication packet image",
            maximum_bytes=MAX_ADJUDICATION_IMAGE_BYTES,
        )
        if not secrets.compare_digest(
            hashlib.sha256(image_raw).hexdigest(),
            packet["evidence"]["image_sha256"],
        ):
            raise ValueError("adjudication packet image SHA-256 does not match reviewed evidence")
        allowed_entries.add(image_name)
    if (packet_dir / "response.json").exists():
        allowed_entries.add("response.json")
    if {entry.name for entry in packet_dir.iterdir()} != allowed_entries:
        raise ValueError("adjudication packet directory contains missing or unexpected review artifacts")
    return packet, packet_raw, review_raw, image_path


def _normalize_adjudication_decision(
    *,
    phase: str,
    decision: Mapping[str, Any],
    packet: Mapping[str, Any],
    semantic_categories: Sequence[str],
) -> dict[str, Any]:
    """Prevalidate fully before publishing the immutable response artifact."""

    if phase == "image" and "control_usefulness" in decision:
        raise ValueError("control_usefulness is not allowed in image adjudication")
    normalized = HumanCBUStore.validate_annotation_payload(phase, decision)
    if phase == "image":
        # ``validate_annotation_payload`` shares the participant schema and
        # materializes this optional field as null.  The frozen adjudication
        # protocol deliberately excludes the exploratory usefulness question.
        normalized.pop("control_usefulness", None)
        return normalized

    if normalized["category_check"] != "incorrect":
        # The shared validator materializes this optional key as null.  Remove
        # it so a second validation in Store remains schema-valid.
        normalized.pop("corrected_category", None)
        return normalized
    corrected = normalized.get("corrected_category")
    proposed = packet["evidence"]["category"]
    if corrected not in semantic_categories:
        raise ValueError("corrected_category must be one of the frozen semantic visual-claim types")
    if corrected == proposed:
        raise ValueError("corrected_category must differ from the packet's proposed category")
    return normalized


def _publish_adjudication_response(
    packet_dir: Path,
    response: Mapping[str, Any],
) -> tuple[Path, bool]:
    """Publish fixed ``response.json`` once; allow only byte-exact replay."""

    response_path = packet_dir / "response.json"
    raw = _json_text(response).encode("utf-8")
    if len(raw) > MAX_ADJUDICATION_RESPONSE_BYTES:
        raise ValueError("adjudication response exceeds its maximum allowed size")
    try:
        secure_write_once_bytes(response_path, raw)
        return response_path, False
    except FileExistsError:
        existing = _read_private_regular_file(
            response_path,
            label="adjudication response",
            maximum_bytes=MAX_ADJUDICATION_RESPONSE_BYTES,
        )
        if not secrets.compare_digest(existing, raw):
            raise FileExistsError(
                "response.json already exists with a different decision; the accepted artifact was not overwritten"
            ) from None
        return response_path, True


def _adjudication_submitter(
    *,
    store: HumanCBUStore,
    study_id: str,
    packet_dir: Path,
    packet: Mapping[str, Any],
    packet_raw: bytes,
    semantic_categories: Sequence[str],
    adjudicator_credential: str,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Build a decision-only callback whose context has no browser selector."""

    phase = str(packet["phase"])

    def submit(decision: Mapping[str, Any]) -> dict[str, Any]:
        normalized = _normalize_adjudication_decision(
            phase=phase,
            decision=decision,
            packet=packet,
            semantic_categories=semantic_categories,
        )
        response = {
            "response_version": ADJUDICATION_RESPONSE_VERSION,
            "review_version": ADJUDICATION_REVIEW_VERSION,
            "packet_file_sha256": hashlib.sha256(packet_raw).hexdigest(),
            "packet": dict(packet),
            "decision": normalized,
        }
        _, response_replay = _publish_adjudication_response(
            packet_dir,
            response,
        )
        result = store.adjudicate_packet_response(
            study_id,
            phase,
            packet_dir=packet_dir,
            adjudicator_credential=adjudicator_credential,
        )
        return {
            **result,
            "response_artifact_replay": response_replay,
        }

    return submit


def command_adjudicate(args: argparse.Namespace) -> int:
    store = _open_store(args)
    study = store.get_study(args.study_id)
    if study["protocol_version"] == HUMAN_CBU_V1_PROTOCOL:
        if args.item_id is not None or args.payload_json is not None:
            raise ValueError("human_cbu_v1 forbids legacy --item-id/--payload-json adjudication")
        if args.adjudicator_pseudonym is not None:
            raise ValueError("human_cbu_v1 uses the frozen adjudicator pseudonym")
        if not args.packet_dir or not args.credential_file:
            raise ValueError("human_cbu_v1 requires --packet-dir and --credential-file")
        frozen_pseudonym = study["metadata"].get("adjudicator_pseudonym")
        raw_credential = _load_adjudicator_credential(
            args.credential_file,
            study_id=args.study_id,
            frozen_pseudonym=frozen_pseudonym,
        )
        result = store.adjudicate_packet_response(
            args.study_id,
            args.phase,
            packet_dir=args.packet_dir,
            adjudicator_credential=raw_credential,
        )
    else:
        if args.packet_dir is not None:
            raise ValueError("legacy protocols do not accept packet-bound adjudication")
        if args.credential_file is not None:
            raise ValueError("legacy protocols do not accept adjudicator credentials")
        if not args.item_id or not args.payload_json:
            raise ValueError("legacy adjudication requires --item-id and --payload-json")
        payload = load_json(args.payload_json)
        result = store.adjudicate(
            args.study_id,
            args.item_id,
            args.phase,
            payload,
            adjudicator_pseudonym=args.adjudicator_pseudonym or "adjudicator",
        )
    print(_json_text(result), end="")
    return 0


def command_adjudication_review(args: argparse.Namespace) -> int:
    """Serve one packet over loopback until one valid decision is committed."""

    if not is_loopback_host(args.host):
        raise ValueError("adjudication-review may bind only to localhost, 127.0.0.0/8, or ::1")
    if not 0 <= args.port <= 65_535:
        raise ValueError("--port must be between 0 and 65535")

    packet_dir = _private_packet_directory(args.packet_dir)
    store = _open_store(args)
    study = store.get_study(args.study_id)
    if study["protocol_version"] != HUMAN_CBU_V1_PROTOCOL:
        raise ValueError("adjudication-review is defined only for human_cbu_v1 studies")
    if study["status"] != "closed":
        raise ValueError(f"adjudication-review requires a closed study; current status is {study['status']!r}")
    semantic_categories = _frozen_adjudication_categories(study)
    packet, packet_raw, review_raw, image_path = _load_adjudication_packet_for_review(
        packet_dir,
        semantic_categories=semantic_categories,
    )
    frozen_pseudonym = study["metadata"].get("adjudicator_pseudonym")
    adjudicator_credential = _load_adjudicator_credential(
        args.credential_file,
        study_id=args.study_id,
        frozen_pseudonym=frozen_pseudonym,
    )
    submit = _adjudication_submitter(
        store=store,
        study_id=args.study_id,
        packet_dir=packet_dir,
        packet=packet,
        packet_raw=packet_raw,
        semantic_categories=semantic_categories,
        adjudicator_credential=adjudicator_credential,
    )

    async def review_one_packet() -> None:
        from audit_recap_t2i.human_cbu.adjudication import (
            start_adjudication_server,
        )

        submitted = asyncio.Event()

        async def submit_and_signal(
            decision: dict[str, Any],
        ) -> dict[str, Any]:
            result = submit(decision)
            submitted.set()
            return result

        server = await start_adjudication_server(
            packet,
            on_submit=submit_and_signal,
            image_path=image_path,
            semantic_categories=semantic_categories,
            review_html=review_raw,
            host=args.host,
            port=args.port,
        )
        try:
            print(
                _json_text(
                    {
                        "study_id": args.study_id,
                        "packet_id": packet["packet_id"],
                        "phase": packet["phase"],
                        "review_url": server.url,
                        "origin": server.origin,
                        "packet_dir": str(packet_dir),
                    }
                ),
                end="",
                flush=True,
            )
            if not args.no_browser:
                webbrowser.open(server.url, new=2)
            await submitted.wait()
        finally:
            await server.close()

    asyncio.run(review_one_packet())
    return 0


def _validate_adjudication_packet_inputs(
    packets: Any,
    image_sources: Any,
    *,
    expected_phase: str,
) -> None:
    """Fail closed unless store output matches the phase-safe packet schema."""

    if not isinstance(packets, list):
        raise ValueError("adjudication packets must be a list")
    if not isinstance(image_sources, Mapping):
        raise ValueError("adjudication image sources must be a mapping")
    packet_ids: set[str] = set()
    image_packet_ids: set[str] = set()
    packet_keys = {
        "packet_version",
        "packet_id",
        "packet_sha256",
        "phase",
        "evidence",
    }
    caption_evidence_keys = {"unit", "target", "category", "caption", "span"}
    image_evidence_keys = {
        "unit",
        "target",
        "category",
        "image_file",
        "image_sha256",
    }
    for packet in packets:
        if not isinstance(packet, Mapping) or set(packet) != packet_keys:
            raise ValueError("adjudication packet violates the exact top-level allowlist")
        packet_id = packet["packet_id"]
        if not isinstance(packet_id, str) or not re.fullmatch(r"ap_[0-9a-f]{64}", packet_id):
            raise ValueError("adjudication packet_id is invalid")
        if packet_id in packet_ids:
            raise ValueError(f"duplicate adjudication packet_id: {packet_id}")
        packet_ids.add(packet_id)
        if packet["packet_version"] != ADJUDICATION_PACKET_VERSION:
            raise ValueError("adjudication packet version mismatch")
        phase = packet["phase"]
        if phase != expected_phase:
            raise ValueError("adjudication packet set contains a packet from the wrong phase")
        evidence = packet["evidence"]
        if not isinstance(evidence, Mapping):
            raise ValueError("adjudication packet evidence must be a mapping")
        if phase == "caption":
            if set(evidence) != caption_evidence_keys:
                raise ValueError("caption packet violates the evidence allowlist")
        elif phase == "image":
            if set(evidence) != image_evidence_keys:
                raise ValueError("image packet violates the evidence allowlist")
            image_packet_ids.add(packet_id)
            image_sha256 = evidence["image_sha256"]
            if not isinstance(image_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", image_sha256):
                raise ValueError("image packet has an invalid frozen SHA-256")
            image_file = evidence["image_file"]
            if not isinstance(image_file, str) or "\\" in image_file:
                raise ValueError("image packet has an unsafe packet-local filename")
            image_path = PurePosixPath(image_file)
            if (
                image_path.is_absolute()
                or len(image_path.parts) != 1
                or image_path.name != f"image{image_path.suffix}"
                or image_path.suffix.lower() not in ADJUDICATION_IMAGE_SUFFIXES
            ):
                raise ValueError("image packet has an unsafe packet-local filename")
        else:
            raise ValueError(f"unknown adjudication packet phase: {phase!r}")

        projection = {key: packet[key] for key in ("packet_version", "packet_id", "phase", "evidence")}
        expected_packet_sha256 = hashlib.sha256(
            json.dumps(
                projection,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if packet["packet_sha256"] != expected_packet_sha256:
            raise ValueError("adjudication packet SHA-256 does not match its projection")

    if set(image_sources) != image_packet_ids:
        raise ValueError("private image-source mapping must exactly match image-phase packets")
    for packet_id, source_ref in image_sources.items():
        if not isinstance(source_ref, str) or not source_ref:
            raise ValueError(f"adjudication image source is invalid for packet {packet_id}")


def _copy_adjudication_packet_image(
    *,
    asset_root: Path,
    source_ref: str,
    packet_dir: Path,
    image_file: str,
    expected_sha256: str,
) -> None:
    """Copy one verified regular file to an opaque packet-local filename."""

    if "\\" in source_ref or "\x00" in source_ref:
        raise ValueError("adjudication image source is not a safe relative path")
    source_relpath = PurePosixPath(source_ref)
    if (
        source_relpath.is_absolute()
        or not source_relpath.parts
        or any(part in {"", ".", ".."} for part in source_relpath.parts)
    ):
        raise ValueError("adjudication image source is not a safe relative path")
    source_path = (asset_root / Path(*source_relpath.parts)).resolve(strict=True)
    if asset_root != source_path and asset_root not in source_path.parents:
        raise ValueError("adjudication image source escapes the frozen asset root")

    image_relpath = PurePosixPath(image_file)
    destination = packet_dir / Path(*image_relpath.parts)
    if (
        len(image_relpath.parts) != 1
        or image_relpath.name != f"image{image_relpath.suffix}"
        or image_relpath.suffix.lower() not in ADJUDICATION_IMAGE_SUFFIXES
        or destination.parent != packet_dir
        or os.path.lexists(destination)
    ):
        raise ValueError("adjudication packet destination is unsafe or already exists")

    open_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    source_descriptor = os.open(source_path, open_flags)
    temporary: Path | None = None
    try:
        source_stat = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError("adjudication image source must be a regular file")
        destination_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        with (
            os.fdopen(source_descriptor, "rb", closefd=False) as source,
            os.fdopen(
                destination_descriptor,
                "wb",
            ) as target,
        ):
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        observed_sha256 = digest.hexdigest()
        if not secrets.compare_digest(observed_sha256, expected_sha256):
            raise ValueError("adjudication image bytes do not match the frozen materialization SHA-256")
        temporary.chmod(PRIVATE_MODE)
        os.link(temporary, destination)
        temporary.unlink()
        temporary = None
        destination.chmod(PRIVATE_MODE)
        _fsync_directory(destination.parent)
    finally:
        os.close(source_descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def command_adjudication_packets(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    if os.path.lexists(output_dir):
        raise FileExistsError(f"refusing to reuse an existing adjudication phase directory: {output_dir}")
    _ensure_parent(output_dir)

    store = _open_store(args)
    packet_inputs = store.adjudication_packet_bundle_inputs(
        args.study_id,
        args.phase,
    )
    packets = packet_inputs["packets"]
    image_sources = packet_inputs["image_sources"]
    _validate_adjudication_packet_inputs(
        packets,
        image_sources,
        expected_phase=args.phase,
    )
    study = store.get_study(args.study_id)
    semantic_categories = _frozen_adjudication_categories(study)

    image_packets = [packet for packet in packets if packet["phase"] == "image"]
    asset_root: Path | None = None
    if image_packets:
        if args.asset_root:
            raw_asset_root = args.asset_root
        else:
            metadata = study.get("metadata", {})
            raw_asset_root = metadata.get("asset_serve_root") if isinstance(metadata, Mapping) else None
            if not isinstance(raw_asset_root, str) or not raw_asset_root:
                raise ValueError("no frozen asset_serve_root is available; provide --asset-root")
        asset_root = Path(raw_asset_root).resolve(strict=True)
        if not asset_root.is_dir():
            raise ValueError(f"adjudication asset root is not a directory: {asset_root}")

    os.mkdir(output_dir, PRIVATE_DIRECTORY_MODE)
    output_dir.chmod(PRIVATE_DIRECTORY_MODE)
    packets_dir = output_dir / "packets"
    os.mkdir(packets_dir, PRIVATE_DIRECTORY_MODE)
    packets_dir.chmod(PRIVATE_DIRECTORY_MODE)

    from audit_recap_t2i.human_cbu.adjudication import (
        render_adjudication_html_bytes,
    )

    packet_directories: list[str] = []
    review_files: list[str] = []
    for packet in packets:
        packet_id = packet["packet_id"]
        review_bytes = render_adjudication_html_bytes(
            packet,
            semantic_categories=semantic_categories,
        )
        packet_dir = packets_dir / packet_id
        os.mkdir(packet_dir, PRIVATE_DIRECTORY_MODE)
        packet_dir.chmod(PRIVATE_DIRECTORY_MODE)
        if packet["phase"] == "image":
            assert asset_root is not None
            source_ref = image_sources[packet_id]
            _copy_adjudication_packet_image(
                asset_root=asset_root,
                source_ref=source_ref,
                packet_dir=packet_dir,
                image_file=packet["evidence"]["image_file"],
                expected_sha256=packet["evidence"]["image_sha256"],
            )
        secure_write_once_json(packet_dir / "packet.json", packet)
        review_file = packet_dir / "review.html"
        secure_write_once_bytes(review_file, review_bytes)
        _fsync_directory(packet_dir)
        packet_directories.append(str(packet_dir.resolve()))
        review_files.append(str(review_file.resolve()))
    _fsync_directory(packets_dir)
    _fsync_directory(output_dir)
    print(
        _json_text(
            {
                "study_id": args.study_id,
                "phase": args.phase,
                "packet_count": len(packets),
                "image_count": len(image_packets),
                "private_output_dir": str(output_dir.resolve()),
                "packets_dir": str(packets_dir.resolve()),
                "packet_directories": packet_directories,
                "review_files": review_files,
                "response_workflow": (
                    "run adjudication-review with exactly one packet directory; "
                    "the loopback server constructs and publishes response.json"
                ),
                "directory_mode": oct(output_dir.stat().st_mode & 0o777),
                "packet_directory_mode": oct(packets_dir.stat().st_mode & 0o777),
                "file_mode": oct(PRIVATE_MODE),
            }
        ),
        end="",
    )
    return 0


COMMANDS = {
    "sample": command_sample,
    "build-preview": command_build_preview,
    "preview": command_preview,
    "participant-test": command_participant_test,
    "author-practice-codes": command_author_practice_codes,
    "author-practice-serve": command_author_practice_serve,
    "materialize": command_materialize,
    "init-db": command_init_db,
    "invites": command_invites,
    "provision-adjudicator": command_provision_adjudicator,
    "assign": command_assign,
    "replace-participant": command_replace_participant,
    "validate": command_validate,
    "seal": command_seal,
    "open": command_open,
    "pause": command_pause,
    "close": command_close,
    "serve": command_serve,
    "status": command_status,
    "backup": command_backup,
    "export": command_export,
    "adjudication-packets": command_adjudication_packets,
    "adjudication-review": command_adjudication_review,
    "adjudicate": command_adjudicate,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
