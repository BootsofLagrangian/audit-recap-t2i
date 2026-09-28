"""Materialize only the selected images needed by a human CBU study."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import pyarrow.dataset as ds
from PIL import Image

from .builder import canonicalize_url

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _scan_tar_members(tar_path: Path, wanted_names: set[str]) -> dict[str, str]:
    matches: dict[str, str] = {}
    proc = subprocess.Popen(
        ["tar", "-tf", str(tar_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        member = line.strip()
        suffix = PurePosixPath(member).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            continue
        name = PurePosixPath(member).name
        if name in wanted_names and name not in matches:
            matches[name] = member
            if len(matches) == len(wanted_names):
                proc.terminate()
                break
    _, stderr = proc.communicate()
    if proc.returncode not in {0, -15}:
        raise RuntimeError(f"failed to scan {tar_path}: {stderr.strip()}")
    return matches


def verify_canonical_manifest(
    items: Iterable[Mapping[str, Any]],
    *,
    manifest_path: str | Path,
    dataset_prefix: str = "cc12m_202603",
) -> dict[str, Any]:
    """Fail closed unless every exact WDS locator matches the canonical manifest.

    URL identity alone is intentionally insufficient because one CC12M URL can
    have multiple stored materializations. The historical shard/member locator
    selects an exact canonical sample, whose URL is then cross-checked.
    """

    unique: dict[str, Mapping[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for item in items:
        locator = item.get("wds_locator")
        if not isinstance(locator, Mapping):
            failures.append({"image_key": item.get("image_key"), "reason": "missing_wds_locator"})
            continue
        shard = str(locator.get("shard") or "")
        key = str(locator.get("key") or "")
        suffix = str(locator.get("suffix") or "").lower()
        if not shard or not key or suffix not in IMAGE_SUFFIXES:
            failures.append({"image_key": item.get("image_key"), "reason": "invalid_wds_locator"})
            continue
        sample_id = f"{dataset_prefix}:{shard}:{key}"
        prior = unique.get(sample_id)
        if prior is not None:
            if canonicalize_url(prior.get("image_url")) != canonicalize_url(item.get("image_url")):
                failures.append(
                    {
                        "image_key": item.get("image_key"),
                        "sample_id": sample_id,
                        "reason": "conflicting_sample_urls",
                    }
                )
            continue
        unique[sample_id] = item

    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    sample_ids = sorted(unique)
    rows: list[dict[str, Any]] = []
    if sample_ids:
        table = ds.dataset(manifest, format="parquet").to_table(
            columns=[
                "sample_id",
                "canonical_url",
                "source_url_sha1",
                "stable_image_id",
                "storage_uri",
                "media_ext",
                "width",
                "height",
                "bytes",
                "sha256_raw",
            ],
            filter=ds.field("sample_id").isin(sample_ids),
        )
        rows = table.to_pylist()

    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[str(row["sample_id"])].append(row)
    verified: list[dict[str, Any]] = []
    for sample_id, item in sorted(unique.items()):
        matches = by_sample.get(sample_id, [])
        if len(matches) != 1:
            failures.append(
                {
                    "image_key": item.get("image_key"),
                    "sample_id": sample_id,
                    "reason": "manifest_missing" if not matches else "manifest_duplicate",
                    "matches": len(matches),
                }
            )
            continue
        row = matches[0]
        request_url = canonicalize_url(item.get("image_url"))
        manifest_url = canonicalize_url(row.get("canonical_url"))
        if request_url != manifest_url:
            failures.append(
                {
                    "image_key": item.get("image_key"),
                    "sample_id": sample_id,
                    "reason": "canonical_url_mismatch",
                    "request_url": request_url,
                    "manifest_url": manifest_url,
                }
            )
            continue
        locator = item["wds_locator"]
        expected_fragment = f"#{locator['key']}{locator['suffix']}"
        storage_uri = str(row.get("storage_uri") or "")
        if not storage_uri.endswith(expected_fragment):
            failures.append(
                {
                    "image_key": item.get("image_key"),
                    "sample_id": sample_id,
                    "reason": "storage_uri_mismatch",
                    "storage_uri": storage_uri,
                    "expected_fragment": expected_fragment,
                }
            )
            continue
        verified.append(
            {
                "image_key": item.get("image_key"),
                "sample_id": sample_id,
                "canonical_url": row.get("canonical_url"),
                "source_url_sha1": row.get("source_url_sha1"),
                "stable_image_id": row.get("stable_image_id"),
                "storage_uri": storage_uri,
                "media_ext": row.get("media_ext"),
                "source_width": row.get("width"),
                "source_height": row.get("height"),
                "source_bytes": row.get("bytes"),
                "source_sha256_raw": row.get("sha256_raw"),
            }
        )
    verified.sort(key=lambda row: str(row["sample_id"]))
    failures.sort(key=lambda row: (str(row.get("sample_id") or ""), str(row.get("image_key") or "")))
    return {
        "requested_unique_samples": len(unique),
        "verified_unique_samples": len(verified),
        "failed_unique_samples": len(failures),
        "manifest_path": str(manifest.resolve()),
        "verified": verified,
        "failures": failures,
    }


def _extract_member(tar_path: Path, member: str, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=".human-cbu-", dir=output_path.parent)
    try:
        with os.fdopen(file_descriptor, "wb") as output:
            proc = subprocess.Popen(["tar", "-xOf", str(tar_path), member], stdout=subprocess.PIPE)
            assert proc.stdout is not None
            while chunk := proc.stdout.read(8 * 1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
            return_code = proc.wait()
            if return_code != 0:
                raise RuntimeError(f"failed to extract {member} from {tar_path}")
            output.flush()
            os.fsync(output.fileno())
        temporary_path = Path(temporary_name)
        with Image.open(temporary_path) as image:
            image.verify()
        with Image.open(temporary_path) as image:
            width, height = image.size
            image_format = image.format
        os.replace(temporary_path, output_path)
        return {
            "path": str(output_path),
            "sha256": digest.hexdigest(),
            "bytes": output_path.stat().st_size,
            "width": width,
            "height": height,
            "format": image_format,
        }
    finally:
        temporary_path = Path(temporary_name)
        if temporary_path.exists():
            temporary_path.unlink()


def _materialize_shard(
    *,
    shard: str,
    items: list[Mapping[str, Any]],
    wds_root: Path,
    asset_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tar_path = wds_root / f"{shard}.tar"
    if not tar_path.exists():
        return [], [
            {"image_key": item["image_key"], "reason": "missing_tar", "tar_path": str(tar_path)} for item in items
        ]
    names = {f"{item['wds_locator']['key']}{str(item['wds_locator']['suffix']).lower()}" for item in items}
    matches = _scan_tar_members(tar_path, names)
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    by_name = {f"{item['wds_locator']['key']}{str(item['wds_locator']['suffix']).lower()}": item for item in items}
    for name, item in by_name.items():
        member = matches.get(name)
        if member is None:
            failures.append(
                {
                    "image_key": item["image_key"],
                    "reason": "missing_member",
                    "tar_path": str(tar_path),
                    "member_name": name,
                }
            )
            continue
        suffix = PurePosixPath(member).suffix.lower()
        digest_name = hashlib.blake2b(str(item["image_key"]).encode("utf-8"), digest_size=16).hexdigest()
        output_path = asset_root / f"{digest_name}{suffix}"
        try:
            result = _extract_member(tar_path, member, output_path)
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(
                {
                    "image_key": item["image_key"],
                    "reason": "extract_or_verify_failed",
                    "detail": str(exc),
                    "tar_path": str(tar_path),
                    "member": member,
                }
            )
            continue
        expected = item.get("image_sha256_expected")
        if isinstance(expected, str) and expected and result["sha256"] != expected.lower():
            output_path.unlink(missing_ok=True)
            failures.append(
                {
                    "image_key": item["image_key"],
                    "reason": "sha256_mismatch",
                    "expected": expected.lower(),
                    "observed": result["sha256"],
                }
            )
            continue
        successes.append(
            {
                "image_key": item["image_key"],
                "asset_relpath": str(output_path.relative_to(asset_root.parent)),
                "materialization_source": "canonical_wds",
                "source_tar": str(tar_path),
                "source_member": member,
                **result,
            }
        )
    return successes, failures


def materialize_selected_images(
    items: Iterable[Mapping[str, Any]],
    *,
    wds_root: str | Path,
    asset_root: str | Path,
    workers: int = 16,
) -> dict[str, Any]:
    """Extract one verified local asset for every unique selected image."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    unique: dict[str, Mapping[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for item in items:
        image_key = str(item["image_key"])
        locator = item.get("wds_locator")
        if image_key in unique:
            prior = unique[image_key].get("wds_locator")
            if locator != prior:
                raise ValueError(f"conflicting WDS locators for {image_key}")
            continue
        if not isinstance(locator, Mapping) or not locator.get("shard") or not locator.get("key"):
            failures.append({"image_key": image_key, "reason": "missing_wds_locator"})
            continue
        unique[image_key] = item

    by_shard: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in unique.values():
        by_shard[str(item["wds_locator"]["shard"])].append(item)
    root = Path(wds_root)
    assets = Path(asset_root)
    assets.mkdir(parents=True, exist_ok=True, mode=0o700)
    assets.chmod(0o700)
    successes: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _materialize_shard,
                shard=shard,
                items=shard_items,
                wds_root=root,
                asset_root=assets,
            ): shard
            for shard, shard_items in sorted(by_shard.items())
        }
        for future in as_completed(futures):
            shard_successes, shard_failures = future.result()
            successes.extend(shard_successes)
            failures.extend(shard_failures)
    successes.sort(key=lambda row: str(row["image_key"]))
    failures.sort(key=lambda row: str(row["image_key"]))
    return {
        "requested_unique_images": len(unique)
        + sum(1 for failure in failures if failure["reason"] == "missing_wds_locator"),
        "materialized_unique_images": len(successes),
        "failed_unique_images": len(failures),
        "wds_root": str(root.resolve()),
        "asset_root": str(assets.resolve()),
        "assets": successes,
        "failures": failures,
    }
