#!/usr/bin/env python3
"""
Production recap runner — long captions via 8× independent vLLM instances.

Write path: append directly to DDN JSONL shards with per-image `.done` checkpoints.
Resume: skip processed image IDs using shard-local `.done` files plus a strict
manifest check so shard geometry cannot drift silently between restarts.

Usage:
    # Sanity run: 2 shards from CC12M
    uv run python scripts/vllm/run_recap.py \
      --dataset cc12m --input-dir data/cc12m-wds/ \
      --limit 20000

    # Full run
    uv run python scripts/vllm/run_recap.py \
      --dataset cc12m --input-dir data/cc12m-wds/

    # Dry run (discovery + resume check, no inference)
    uv run python scripts/vllm/run_recap.py \
      --dataset cc12m --input-dir data/cc12m-wds/ --dry-run

    # Resume after crash (same command — skips completed)
    uv run python scripts/vllm/run_recap.py \
      --dataset cc12m --input-dir data/cc12m-wds/
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import contextlib
import hashlib
import json
import logging
import os
import random
import re
import resource
import shlex
import signal
import shutil
import sqlite3
import tarfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import aiohttp
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECAP_CONFIG_DIR = PROJECT_ROOT / "configs" / "recap"
DEFAULT_OUTPUT_DDN = PROJECT_ROOT / "outputs" / "recap"
DEFAULT_VLLM_HOST = "http://localhost"
VLLM_PORTS = tuple(range(8000, 8008))
DEFAULT_SERVE_MULTI_SCRIPT = PROJECT_ROOT / "scripts" / "vllm" / "serve_multi.sh"
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
MAX_IMAGE_PX = 1024
MAX_SHORT_EDGE = 768  # match canon: 768px shortest edge resize
MAX_FILE_URL_BYTES = 64 * 1024 * 1024
JPEG_QUALITY = 90
CHECKPOINT_INTERVAL_SEC = 60
DEFAULT_NUM_SHARDS = 64
MAX_RETRIES = 3
RETRY_DELAY = 2.0
DEFAULT_FLUSH_BATCH_SIZE = 128
DEFAULT_ENCODE_QUEUE_SIZE = 256
MANIFEST_VERSION = 1
MANIFEST_CACHE_VERSION = 1
DEFAULT_REQUEST_TIMEOUT = 180  # seconds per vLLM request
CIRCUIT_BREAKER_THRESHOLD = 64  # consecutive failures to mark unhealthy
CIRCUIT_BREAKER_COOLDOWN = 30  # seconds before re-probing unhealthy endpoint
ENDPOINT_WAIT_MIN_SEC = 0.01
ENDPOINT_WAIT_MAX_SEC = 5.0
ENDPOINT_WAIT_JITTER_SEC = 0.02
RETRY_JITTER_SEC = 1.0
PRODUCER_PUT_TIMEOUT = 60  # seconds between backpressure warnings
PRODUCER_PUT_POLL_SEC = 1.0
SHUTDOWN_DRAIN_TIMEOUT = 30  # seconds to wait for inflight drain on first SIGTERM
SHUTDOWN_ABORT_TIMEOUT = 10  # seconds after drain before force cancel
ENDPOINT_OVERLOAD_WAITING = 128
ENDPOINT_OVERLOAD_RUNNING = 1088
ENDPOINT_OVERLOAD_KV_CACHE = 0.995
ENDPOINT_OVERLOAD_COOLDOWN = 5
ENDPOINT_PROBE_INTERVAL_SEC = 10.0
ENDPOINT_STALL_TIMEOUT_SEC = 60
ENDPOINT_STALL_MIN_RUNNING = 64
DEFAULT_ENDPOINT_RESTART_TIMEOUT = 900
DEFAULT_HTTP_KEEPALIVE_TIMEOUT = 20.0
DEFAULT_HTTP_CONNECTION_HEADROOM = 128
DEFAULT_CONNECT_TIMEOUT = 3.0
DEFAULT_SOCK_CONNECT_TIMEOUT = 3.0
ENDPOINT_SOFT_FAILURE_THRESHOLD = 16
ENDPOINT_CAPACITY_BACKOFF_RATIO = 0.85
ENDPOINT_CAPACITY_MIN = 512
ENDPOINT_CAPACITY_RECOVERY_SEC = 10.0
ENDPOINT_CAPACITY_RECOVERY_JITTER_SEC = 10.0
ENDPOINT_CAPACITY_RECOVERY_STEP = 64
ENDPOINT_FAILURE_HOLDOFF_SEC = 3.0
ENDPOINT_FAILURE_HOLDOFF_JITTER_SEC = 6.0
ENDPOINT_ADMISSION_INTERVAL_SEC = 0.006
ENDPOINT_ADMISSION_JITTER_SEC = 0.006
ENDPOINT_INITIAL_CAPACITY = 0
ENDPOINT_TRANSPORT_FAILURE_THRESHOLD = 8
ENDPOINT_TRANSPORT_BACKOFF_REQUIRES_REMOTE_PRESSURE = True
ENDPOINT_TRANSPORT_FAILURE_HOLDOFF_SEC = 8.0
ENDPOINT_TRANSPORT_FAILURE_HOLDOFF_JITTER_SEC = 12.0
ENDPOINT_REMOTE_IDLE_LOCAL_INFLIGHT = 0
ENDPOINT_REMOTE_IDLE_HOLDOFF_SEC = 3.0
ENDPOINT_REMOTE_IDLE_GRACE_SEC = 5.0
QUEUE_GET_TIMEOUT = 1.0
SLOW_REQUEST_SEC = 180.0
EXIT_STOP_SIGNAL = 2
EXIT_HARDCAP = 5
NULL_TEXT_BLOCK_RE = re.compile(
    r"(?:\n\s*){0,2}TEXT:\s*[\[\(<{]?\s*(?:"
    r"<\s*no\s+text\s*>|"
    r"none(?:\s*(?:[.,;:]|\(|visible\b|legible\b|clearly\b|in\b|as\b|beyond\b).*)?|"
    r"no\s+(?:clearly\s+)?(?:(?:readable|legible)\s+)?text(?:\s*(?:[.,;:]|visible\b|appears\b|is\b|beyond\b|as\b|in\b|present\b).*)?|"
    r"n/?a|null"
    r")(?:\s*\n\s*(?:none|n/?a|null)\s*)?\.?\s*[\]\)>}]?\s*$",
    re.IGNORECASE,
)
CGROUP_CURRENT_PATHS = (
    Path("/sys/fs/cgroup/memory.current"),
    Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
)
CGROUP_LIMIT_PATHS = (
    Path("/sys/fs/cgroup/memory.max"),
    Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
)

logger = logging.getLogger("recap")

_MANIFEST_IMAGE_ID_RE = re.compile(rb'"image_id":\s*"((?:\\.|[^"\\])*)"')
_MANIFEST_REL_PATH_RE = re.compile(rb'"rel_path":\s*"((?:\\.|[^"\\])*)"')
_MANIFEST_URL_RE = re.compile(rb'"url":\s*"((?:\\.|[^"\\])*)"')


def _decode_json_string_value(raw: bytes) -> str:
    if b"\\" not in raw:
        return raw.decode("utf-8")
    return str(json.loads(b'"' + raw + b'"'))


def _parse_manifest_cache_fields(raw_line: bytes) -> tuple[str, str, str | None]:
    """Extract cache fields without parsing huge canonicalize manifest rows.

    DataComp canonical rows carry long provenance/caption fields that are not
    needed for recap discovery. Regex-extracting the three small string fields
    keeps resume catch-up proportional to tail bytes without spending most CPU
    in full JSON decoding. Fall back to `json.loads` if the expected fields are
    absent, so older/nonstandard manifests still work.
    """
    image_match = _MANIFEST_IMAGE_ID_RE.search(raw_line)
    rel_match = _MANIFEST_REL_PATH_RE.search(raw_line)
    if image_match is None or rel_match is None:
        row = json.loads(raw_line)
        return str(row["image_id"]), str(row["rel_path"]), row.get("url")
    url_match = _MANIFEST_URL_RE.search(raw_line)
    url = _decode_json_string_value(url_match.group(1)) if url_match is not None else None
    return (
        _decode_json_string_value(image_match.group(1)),
        _decode_json_string_value(rel_match.group(1)),
        url,
    )


# ── Run lock ────────────────────────────────────────────────

class RecapRunLock:
    """Advisory file lock to prevent duplicate recap runs for the same dataset/domain.

    Uses fcntl.flock(LOCK_EX|LOCK_NB) — kernel releases automatically on SIGKILL.
    """

    def __init__(self, ddn_root: Path, *, name: str = "run.lock", shared: bool = False):
        if "/" in name or name in {"", ".", ".."}:
            raise ValueError(f"invalid run lock name: {name!r}")
        self._lock_path = ddn_root / name
        self._shared = shared
        self._fd: int | None = None

    def acquire(self) -> None:
        import fcntl
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self._lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            mode = fcntl.LOCK_SH if self._shared else fcntl.LOCK_EX
            fcntl.flock(fd, mode | fcntl.LOCK_NB)
        except OSError:
            # Lock held by another process — read owner info and fail
            try:
                with open(fd, closefd=False) as f:
                    owner = f.read(4096)
                os.close(fd)
            except Exception:
                os.close(fd)
                owner = "(unreadable)"
            raise SystemExit(
                f"ERROR: Another recap run is active for this dataset.\n"
                f"Lock file: {self._lock_path}\n"
                f"Owner: {owner}\n"
                f"If the previous run crashed, remove the lock file manually."
            )
        self._fd = fd
        if self._shared:
            return
        # Write owner metadata
        meta = json.dumps({
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "cmdline": " ".join(os.sys.argv),
        })
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, meta.encode())

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
            if not self._shared:
                try:
                    self._lock_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


def _read_proc_status_mib(field: str) -> float | None:
    try:
        with Path("/proc/self/status").open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith(f"{field}:"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) / 1024.0
    except (FileNotFoundError, OSError, ValueError):
        return None
    return None


def _read_cgroup_mib(paths: tuple[Path, ...]) -> float | None:
    for path in paths:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not raw or raw == "max":
            return None
        try:
            return int(raw) / (1024.0 * 1024.0)
        except ValueError:
            continue
    return None


def _memory_status() -> str:
    rss = _read_proc_status_mib("VmRSS")
    hwm = _read_proc_status_mib("VmHWM")
    current = _read_cgroup_mib(CGROUP_CURRENT_PATHS)
    limit = _read_cgroup_mib(CGROUP_LIMIT_PATHS)

    parts: list[str] = []
    if rss is None:
        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if ru_maxrss > 0:
            rss = ru_maxrss / 1024.0
    if rss is not None:
        parts.append(f"rss={rss:.0f}MiB")
    if hwm is not None:
        parts.append(f"hwm={hwm:.0f}MiB")
    if current is not None and limit is not None:
        parts.append(f"cgroup={current:.0f}/{limit:.0f}MiB")
    elif current is not None:
        parts.append(f"cgroup={current:.0f}MiB")
    return " ".join(parts) if parts else "mem=n/a"


def disk_usage_for(path: Path) -> shutil._ntuple_diskusage:
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe)


def disk_used_pct(path: Path) -> float:
    usage = disk_usage_for(path)
    return usage.used / usage.total * 100.0


def check_hardcap(path: Path, hardcap_use_pct: float) -> bool:
    if hardcap_use_pct <= 0:
        return False
    return disk_used_pct(path) >= hardcap_use_pct


def image_id_to_shard(image_id: str, num_shards: int) -> int:
    """Deterministic shard from image_id bits. No hash needed for numeric IDs."""
    try:
        return int(image_id) % num_shards
    except ValueError:
        import zlib
        return zlib.crc32(image_id.encode()) % num_shards


def normalize_caption_text(caption: str) -> str:
    """Drop model-added null OCR markers such as trailing `TEXT: None`."""
    return NULL_TEXT_BLOCK_RE.sub("", caption).rstrip()


# ── Endpoint router with circuit breaker ─────────────────

@dataclass
class EndpointProgress:
    prompt_tokens: int = 0
    generation_tokens: int = 0
    responses_total: int = 0
    last_progress_at: float = 0.0
    last_probe_at: float = 0.0

class EndpointRouter:
    """Health-aware endpoint router. Marks endpoints unhealthy on consecutive
    failures, probes them in the background, and routes only to healthy ones."""

    def __init__(
        self,
        endpoints: list[str],
        max_inflight_per_ep: int = 500,
        *,
        overload_waiting: int = ENDPOINT_OVERLOAD_WAITING,
        overload_running: int = ENDPOINT_OVERLOAD_RUNNING,
        overload_kv_cache: float = ENDPOINT_OVERLOAD_KV_CACHE,
        overload_cooldown: int = ENDPOINT_OVERLOAD_COOLDOWN,
        stall_timeout_sec: int = ENDPOINT_STALL_TIMEOUT_SEC,
        stall_min_running: int = ENDPOINT_STALL_MIN_RUNNING,
        soft_failure_threshold: int = ENDPOINT_SOFT_FAILURE_THRESHOLD,
        capacity_backoff_ratio: float = ENDPOINT_CAPACITY_BACKOFF_RATIO,
        capacity_min: int = ENDPOINT_CAPACITY_MIN,
        capacity_recovery_sec: float = ENDPOINT_CAPACITY_RECOVERY_SEC,
        capacity_recovery_jitter_sec: float = ENDPOINT_CAPACITY_RECOVERY_JITTER_SEC,
        capacity_recovery_step: int = ENDPOINT_CAPACITY_RECOVERY_STEP,
        failure_holdoff_sec: float = ENDPOINT_FAILURE_HOLDOFF_SEC,
        failure_holdoff_jitter_sec: float = ENDPOINT_FAILURE_HOLDOFF_JITTER_SEC,
        admission_interval_sec: float = ENDPOINT_ADMISSION_INTERVAL_SEC,
        admission_jitter_sec: float = ENDPOINT_ADMISSION_JITTER_SEC,
        initial_capacity: int = ENDPOINT_INITIAL_CAPACITY,
        transport_failure_threshold: int = ENDPOINT_TRANSPORT_FAILURE_THRESHOLD,
        transport_backoff_requires_remote_pressure: bool = ENDPOINT_TRANSPORT_BACKOFF_REQUIRES_REMOTE_PRESSURE,
        transport_failure_holdoff_sec: float = ENDPOINT_TRANSPORT_FAILURE_HOLDOFF_SEC,
        transport_failure_holdoff_jitter_sec: float = ENDPOINT_TRANSPORT_FAILURE_HOLDOFF_JITTER_SEC,
        remote_idle_local_inflight: int = ENDPOINT_REMOTE_IDLE_LOCAL_INFLIGHT,
        remote_idle_holdoff_sec: float = ENDPOINT_REMOTE_IDLE_HOLDOFF_SEC,
        remote_idle_grace_sec: float = ENDPOINT_REMOTE_IDLE_GRACE_SEC,
    ):
        self._endpoints = endpoints
        self._healthy: dict[str, bool] = {ep: True for ep in endpoints}
        self._consecutive_fails: dict[str, int] = {ep: 0 for ep in endpoints}
        self._transport_fails: dict[str, int] = {ep: 0 for ep in endpoints}
        self._cooldown_until: dict[str, float] = {ep: 0.0 for ep in endpoints}
        self._inflight: dict[str, int] = {ep: 0 for ep in endpoints}
        self._remote_running: dict[str, int] = {ep: 0 for ep in endpoints}
        self._remote_waiting: dict[str, int] = {ep: 0 for ep in endpoints}
        self._remote_kv_cache: dict[str, float] = {ep: 0.0 for ep in endpoints}
        self._overloaded_until: dict[str, float] = {ep: 0.0 for ep in endpoints}
        self._progress: dict[str, EndpointProgress] = {ep: EndpointProgress() for ep in endpoints}
        self._stalled: dict[str, bool] = {ep: False for ep in endpoints}
        self._restart_pending: dict[str, bool] = {ep: False for ep in endpoints}
        self._restart_requested: dict[str, bool] = {ep: False for ep in endpoints}
        self._overload_waiting = overload_waiting
        self._overload_running = overload_running
        self._overload_kv_cache = overload_kv_cache
        self._overload_cooldown = overload_cooldown
        self._stall_timeout_sec = stall_timeout_sec
        self._stall_min_running = stall_min_running
        self._max_inflight = max_inflight_per_ep
        initial = max_inflight_per_ep if initial_capacity <= 0 else min(max_inflight_per_ep, max(1, initial_capacity))
        self._initial_capacity = initial
        self._capacity_limit: dict[str, int] = {ep: initial for ep in endpoints}
        self._capacity_min = min(max_inflight_per_ep, max(1, capacity_min))
        self._capacity_backoff_ratio = min(0.99, max(0.1, capacity_backoff_ratio))
        self._soft_failure_threshold = max(0, soft_failure_threshold)
        self._capacity_recovery_sec = max(0.0, capacity_recovery_sec)
        self._capacity_recovery_jitter_sec = max(0.0, capacity_recovery_jitter_sec)
        self._capacity_recovery_step = max(1, capacity_recovery_step)
        self._failure_holdoff_sec = max(0.0, failure_holdoff_sec)
        self._failure_holdoff_jitter_sec = max(0.0, failure_holdoff_jitter_sec)
        self._admission_interval_sec = max(0.0, admission_interval_sec)
        self._admission_jitter_sec = max(0.0, admission_jitter_sec)
        self._transport_failure_threshold = max(0, transport_failure_threshold)
        self._transport_backoff_requires_remote_pressure = transport_backoff_requires_remote_pressure
        self._transport_failure_holdoff_sec = max(0.0, transport_failure_holdoff_sec)
        self._transport_failure_holdoff_jitter_sec = max(0.0, transport_failure_holdoff_jitter_sec)
        self._remote_idle_local_inflight = max(0, remote_idle_local_inflight)
        self._remote_idle_holdoff_sec = max(0.0, remote_idle_holdoff_sec)
        self._remote_idle_grace_sec = max(0.0, remote_idle_grace_sec)
        self._last_capacity_recovery: dict[str, float] = {ep: 0.0 for ep in endpoints}
        now = time.monotonic()
        next_recovery = (
            now + self._capacity_recovery_sec + random.uniform(0.0, self._capacity_recovery_jitter_sec)
            if initial < max_inflight_per_ep
            else 0.0
        )
        self._next_capacity_recovery_at: dict[str, float] = {ep: next_recovery for ep in endpoints}
        self._next_admission_at: dict[str, float] = {ep: 0.0 for ep in endpoints}
        self._lock = asyncio.Lock()

    def _score(self, ep: str) -> tuple[int, int, int]:
        return (
            self._remote_waiting.get(ep, 0),
            self._inflight.get(ep, 0),
            self._remote_running.get(ep, 0),
        )

    def get_endpoint(self) -> str | None:
        """Pick the least-loaded healthy endpoint with available capacity."""
        now = time.monotonic()
        eligible = [
            ep for ep in self._endpoints
            if self._healthy.get(ep, False)
            and not self._restart_pending.get(ep, False)
            and not self._stalled.get(ep, False)
            and self._inflight.get(ep, 0) < self._capacity_limit.get(ep, self._max_inflight)
            and now >= self._next_admission_at.get(ep, 0.0)
            and now >= self._cooldown_until.get(ep, 0)
            and now >= self._overloaded_until.get(ep, 0)
        ]
        if eligible:
            return min(eligible, key=self._score)
        return None

    def route_available_delay(self) -> float:
        """Bounded wait before checking routing again.

        Consumers call this instead of sleeping for the full circuit-breaker
        cooldown. That keeps endpoint refill continuous and avoids synchronized
        30-second retry waves after transient vLLM disconnects.
        """
        now = time.monotonic()
        delays: list[float] = []
        for ep in self._endpoints:
            if self._restart_pending.get(ep, False) or self._stalled.get(ep, False):
                continue
            if self._inflight.get(ep, 0) >= self._capacity_limit.get(ep, self._max_inflight):
                continue
            admission_delay = self._next_admission_at.get(ep, 0.0) - now
            if not self._healthy.get(ep, False):
                delays.append(max(admission_delay, self._cooldown_until.get(ep, 0) - now))
                continue
            delays.append(max(
                admission_delay,
                self._cooldown_until.get(ep, 0) - now,
                self._overloaded_until.get(ep, 0) - now,
            ))
        if not delays:
            return ENDPOINT_WAIT_MAX_SEC
        return min(
            ENDPOINT_WAIT_MAX_SEC,
            max(ENDPOINT_WAIT_MIN_SEC, min(max(0.0, delay) for delay in delays)),
        )

    def acquire(self, ep: str) -> None:
        self._inflight[ep] = self._inflight.get(ep, 0) + 1
        if self._admission_interval_sec > 0 or self._admission_jitter_sec > 0:
            self._next_admission_at[ep] = time.monotonic() + self._admission_interval_sec + random.uniform(
                0.0,
                self._admission_jitter_sec,
            )

    def release(self, ep: str) -> None:
        self._inflight[ep] = max(0, self._inflight.get(ep, 0) - 1)

    def report_success(self, ep: str) -> None:
        was_healthy = self._healthy.get(ep, True)
        self._consecutive_fails[ep] = 0
        self._transport_fails[ep] = 0
        if was_healthy:
            self._cooldown_until[ep] = 0.0
        self._recover_capacity(ep)
        if not was_healthy:
            self._healthy[ep] = True
            logger.info("Endpoint %s recovered — marking healthy", ep)

    def report_failure(self, ep: str, is_timeout: bool = False) -> None:
        self._consecutive_fails[ep] = self._consecutive_fails.get(ep, 0) + 1
        fail_count = self._consecutive_fails[ep]
        backed_off = False
        if is_timeout and self._transport_failure_threshold > 0:
            self._transport_fails[ep] = self._transport_fails.get(ep, 0) + 1
            transport_count = self._transport_fails[ep]
            if transport_count >= self._transport_failure_threshold and transport_count % self._transport_failure_threshold == 0:
                if (
                    not self._transport_backoff_requires_remote_pressure
                    or self._has_remote_pressure(ep)
                ):
                    self._backoff_capacity(ep, fail_count=transport_count, reason="transport failures")
                    self._overloaded_until[ep] = max(
                        self._overloaded_until.get(ep, 0.0),
                        time.monotonic()
                        + self._transport_failure_holdoff_sec
                        + random.uniform(0.0, self._transport_failure_holdoff_jitter_sec),
                    )
                    backed_off = True
                else:
                    logger.warning(
                        "Endpoint %s saw %d transport failures but remote metrics show no pressure "
                        "(running=%d waiting=%d kv=%.3f); preserving capacity=%d",
                        ep,
                        transport_count,
                        self._remote_running.get(ep, 0),
                        self._remote_waiting.get(ep, 0),
                        self._remote_kv_cache.get(ep, 0.0),
                        self._capacity_limit.get(ep, self._max_inflight),
                    )
        if (
            not backed_off
            and
            self._soft_failure_threshold > 0
            and fail_count >= self._soft_failure_threshold
            and fail_count % self._soft_failure_threshold == 0
        ):
            self._backoff_capacity(ep, fail_count=fail_count, reason="consecutive failures")
        if self._consecutive_fails[ep] >= CIRCUIT_BREAKER_THRESHOLD:
            if self._healthy.get(ep, True):
                logger.warning("Endpoint %s breaker OPEN after %d consecutive failures",
                               ep, self._consecutive_fails[ep])
            self._healthy[ep] = False
            self._cooldown_until[ep] = time.monotonic() + CIRCUIT_BREAKER_COOLDOWN

    def _has_remote_pressure(self, ep: str) -> bool:
        running = self._remote_running.get(ep, 0)
        waiting = self._remote_waiting.get(ep, 0)
        kv_cache_usage = self._remote_kv_cache.get(ep, 0.0)
        return (
            waiting >= self._overload_waiting
            or running >= self._overload_running
            or (kv_cache_usage >= self._overload_kv_cache and waiting > 0)
        )

    def _backoff_capacity(self, ep: str, *, fail_count: int, reason: str = "failures") -> None:
        old = self._capacity_limit.get(ep, self._max_inflight)
        new = max(self._capacity_min, int(old * self._capacity_backoff_ratio))
        if new >= old:
            return
        self._capacity_limit[ep] = new
        self._overloaded_until[ep] = max(
            self._overloaded_until.get(ep, 0.0),
            time.monotonic()
            + self._overload_cooldown
            + self._failure_holdoff_sec
            + random.uniform(0.0, self._failure_holdoff_jitter_sec),
        )
        self._next_capacity_recovery_at[ep] = max(
            self._next_capacity_recovery_at.get(ep, 0.0),
            time.monotonic()
            + self._capacity_recovery_sec
            + random.uniform(0.0, self._capacity_recovery_jitter_sec),
        )
        logger.warning(
            "Endpoint %s soft-throttled after %d %s: capacity %d -> %d",
            ep,
            fail_count,
            reason,
            old,
            new,
        )

    def _recover_capacity(self, ep: str) -> None:
        old = self._capacity_limit.get(ep, self._max_inflight)
        if old >= self._max_inflight:
            return
        now = time.monotonic()
        if now < self._next_capacity_recovery_at.get(ep, 0.0):
            return
        if now - self._last_capacity_recovery.get(ep, 0.0) < self._capacity_recovery_sec:
            return
        new = min(self._max_inflight, old + self._capacity_recovery_step)
        self._capacity_limit[ep] = new
        self._last_capacity_recovery[ep] = now
        self._next_capacity_recovery_at[ep] = now + self._capacity_recovery_sec + random.uniform(
            0.0,
            self._capacity_recovery_jitter_sec,
        )
        logger.info("Endpoint %s capacity recovered: %d -> %d", ep, old, new)

    def healthy_count(self) -> int:
        return sum(1 for v in self._healthy.values() if v)

    def update_metrics(
        self,
        ep: str,
        *,
        running: int,
        waiting: int,
        kv_cache_usage: float = 0.0,
        prompt_tokens: int = 0,
        generation_tokens: int = 0,
        responses_total: int = 0,
        now: float | None = None,
    ) -> None:
        now = time.monotonic() if now is None else now
        self._remote_running[ep] = running
        self._remote_waiting[ep] = waiting
        self._remote_kv_cache[ep] = kv_cache_usage
        overloaded = (
            waiting >= self._overload_waiting
            or running >= self._overload_running
            or (kv_cache_usage >= self._overload_kv_cache and waiting > 0)
        )
        if overloaded:
            if now >= self._overloaded_until.get(ep, 0):
                logger.warning(
                    "Endpoint %s overloaded — running=%d waiting=%d kv=%.3f, pausing new traffic for %ds",
                    ep, running, waiting, kv_cache_usage, self._overload_cooldown,
                )
            self._overloaded_until[ep] = now + self._overload_cooldown

        progress = self._progress[ep]
        metrics_moved = (
            prompt_tokens != progress.prompt_tokens
            or generation_tokens != progress.generation_tokens
            or responses_total != progress.responses_total
        )
        progress.prompt_tokens = prompt_tokens
        progress.generation_tokens = generation_tokens
        progress.responses_total = responses_total
        progress.last_probe_at = now
        if progress.last_progress_at == 0.0 or metrics_moved:
            progress.last_progress_at = now
            self._stalled[ep] = False

        idle_sec = now - progress.last_progress_at if progress.last_progress_at else 0.0
        local_inflight = self._inflight.get(ep, 0)
        local_pressure = (
            self._remote_idle_local_inflight > 0
            and self._remote_idle_holdoff_sec > 0
            and local_inflight >= self._remote_idle_local_inflight
            and running == 0
            and waiting == 0
            and idle_sec >= self._remote_idle_grace_sec
        )
        if local_pressure:
            if now >= self._overloaded_until.get(ep, 0.0):
                logger.warning(
                    "Endpoint %s has local connection pressure with remote idle — local=%d running=%d waiting=%d idle=%.0fs; pausing new traffic for %.1fs",
                    ep,
                    local_inflight,
                    running,
                    waiting,
                    idle_sec,
                    self._remote_idle_holdoff_sec,
                )
            self._overloaded_until[ep] = max(
                self._overloaded_until.get(ep, 0.0),
                now + self._remote_idle_holdoff_sec,
            )
        remote_stuck = running >= self._stall_min_running and waiting == 0
        local_stuck = local_inflight >= self._stall_min_running and running == 0 and waiting == 0
        if (
            (remote_stuck or local_stuck)
            and idle_sec >= self._stall_timeout_sec
            and not self._restart_pending.get(ep, False)
        ):
            self._mark_stalled(
                ep,
                idle_sec=idle_sec,
                running=running,
                waiting=waiting,
                local_inflight=local_inflight,
                now=now,
            )

    def _mark_stalled(
        self,
        ep: str,
        *,
        idle_sec: float,
        running: int,
        waiting: int,
        local_inflight: int,
        now: float,
    ) -> None:
        if not self._stalled.get(ep, False):
            logger.warning(
                "Endpoint %s stalled — no metric progress for %.0fs with local=%d running=%d waiting=%d; quarantining and requesting restart",
                ep,
                idle_sec,
                local_inflight,
                running,
                waiting,
            )
        self._stalled[ep] = True
        self._healthy[ep] = False
        self._restart_requested[ep] = True
        self._cooldown_until[ep] = now + max(self._stall_timeout_sec, CIRCUIT_BREAKER_COOLDOWN)

    def consume_restart_requests(self) -> list[str]:
        requested = [ep for ep, needed in self._restart_requested.items() if needed and not self._restart_pending.get(ep, False)]
        for ep in requested:
            self._restart_requested[ep] = False
        return requested

    def begin_restart(self, ep: str) -> None:
        self._restart_pending[ep] = True
        self._healthy[ep] = False
        self._cooldown_until[ep] = time.monotonic() + CIRCUIT_BREAKER_COOLDOWN

    def finish_restart(self, ep: str, *, recovered: bool) -> None:
        self._restart_pending[ep] = False
        self._consecutive_fails[ep] = 0
        self._transport_fails[ep] = 0
        progress = self._progress[ep]
        progress.prompt_tokens = 0
        progress.generation_tokens = 0
        progress.responses_total = 0
        progress.last_progress_at = time.monotonic()
        progress.last_probe_at = progress.last_progress_at
        self._remote_running[ep] = 0
        self._remote_waiting[ep] = 0
        self._stalled[ep] = False
        if recovered:
            self._healthy[ep] = True
            self._cooldown_until[ep] = 0.0
            if self._initial_capacity < self._max_inflight:
                self._capacity_limit[ep] = self._initial_capacity
                self._next_capacity_recovery_at[ep] = (
                    time.monotonic()
                    + self._capacity_recovery_sec
                    + random.uniform(0.0, self._capacity_recovery_jitter_sec)
                )
            logger.info("Endpoint %s restart completed — routing restored", ep)
        else:
            self._healthy[ep] = False
            self._cooldown_until[ep] = time.monotonic() + CIRCUIT_BREAKER_COOLDOWN
            logger.warning("Endpoint %s restart failed — keeping endpoint quarantined", ep)

    async def probe_endpoints(self, session: aiohttp.ClientSession) -> None:
        """Refresh health and load metrics for all endpoints.

        If /metrics responds successfully, the endpoint is reachable and serving —
        use metric movement as a health signal before routing traffic back.
        """
        now = time.monotonic()
        for ep in self._endpoints:
            base = ep.rsplit("/v1/", 1)[0] if "/v1/" in ep else ep
            try:
                async with session.get(f"{base}/metrics", timeout=aiohttp.ClientTimeout(total=5)) as r:
                    metrics_text = await r.text()
                running = _parse_prometheus_metric(metrics_text, "vllm:num_requests_running")
                waiting = _parse_prometheus_metric(metrics_text, "vllm:num_requests_waiting")
                kv_cache_usage = _parse_prometheus_metric_float(metrics_text, "vllm:kv_cache_usage_perc")
                prompt_tokens = _parse_prometheus_metric(metrics_text, "vllm:prompt_tokens_total")
                generation_tokens = _parse_prometheus_metric(metrics_text, "vllm:generation_tokens_total")
                responses_total = _parse_prometheus_metric_prefix(
                    metrics_text,
                    'http_requests_total{handler="/v1/chat/completions",method="POST",status="2xx"}',
                )
                self.update_metrics(
                    ep,
                    running=running,
                    waiting=waiting,
                    kv_cache_usage=kv_cache_usage,
                    prompt_tokens=prompt_tokens,
                    generation_tokens=generation_tokens,
                    responses_total=responses_total,
                    now=now,
                )
                if not self._stalled.get(ep, False) and not self._restart_pending.get(ep, False):
                    self.report_success(ep)
            except Exception:
                if ep in self._remote_running:
                    self._remote_running[ep] = 0
                if ep in self._remote_waiting:
                    self._remote_waiting[ep] = 0
                if ep in self._remote_kv_cache:
                    self._remote_kv_cache[ep] = 0.0

            # If endpoint was unhealthy, check /health to confirm recovery
            if (
                not self._healthy.get(ep, True)
                and not self._stalled.get(ep, False)
                and not self._restart_pending.get(ep, False)
            ):
                try:
                    async with session.get(f"{base}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                        if r.status == 200:
                            self.report_success(ep)
                            logger.info("Endpoint %s recovered via /health probe", ep)
                except Exception:
                    self._cooldown_until[ep] = now + CIRCUIT_BREAKER_COOLDOWN

    def snapshot(self) -> list[dict[str, object]]:
        now = time.monotonic()
        rows: list[dict[str, object]] = []
        for ep in self._endpoints:
            rows.append(
                {
                    "endpoint": ep,
                    "healthy": self._healthy.get(ep, False),
                    "stalled": self._stalled.get(ep, False),
                    "restart_pending": self._restart_pending.get(ep, False),
                    "capacity_limit": self._capacity_limit.get(ep, self._max_inflight),
                    "local_inflight": self._inflight.get(ep, 0),
                    "remote_running": self._remote_running.get(ep, 0),
                    "remote_waiting": self._remote_waiting.get(ep, 0),
                    "remote_kv_cache": self._remote_kv_cache.get(ep, 0.0),
                    "idle_sec": self._idle_seconds(ep, now),
                    "cooldown_sec": max(0, int(self._cooldown_until.get(ep, 0) - now)),
                    "overloaded_sec": max(0, int(self._overloaded_until.get(ep, 0) - now)),
                }
            )
        return rows

    def endpoint_state(self, ep: str) -> dict[str, object]:
        now = time.monotonic()
        return {
            "endpoint": ep,
            "healthy": self._healthy.get(ep, False),
            "stalled": self._stalled.get(ep, False),
            "restart_pending": self._restart_pending.get(ep, False),
            "capacity_limit": self._capacity_limit.get(ep, self._max_inflight),
            "local_inflight": self._inflight.get(ep, 0),
            "remote_running": self._remote_running.get(ep, 0),
            "remote_waiting": self._remote_waiting.get(ep, 0),
            "remote_kv_cache": self._remote_kv_cache.get(ep, 0.0),
            "idle_sec": self._idle_seconds(ep, now),
            "cooldown_sec": max(0, int(self._cooldown_until.get(ep, 0) - now)),
            "overloaded_sec": max(0, int(self._overloaded_until.get(ep, 0) - now)),
        }

    def _idle_seconds(self, ep: str, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        last_progress = self._progress[ep].last_progress_at
        if last_progress == 0.0:
            return 0
        return max(0, int(now - last_progress))


# ── Data types ───────────────────────────────────────────

@dataclass
class RecapImage:
    global_index: int
    image_id: str
    image_path: str  # relative path or tar member
    source_kind: str  # "file" | "tar" | "parquet"
    image_file: Path | None = None
    tar_path: Path | None = None
    tar_member: str | None = None
    parquet_path: Path | None = None
    parquet_row_group: int = 0
    parquet_row_offset: int = 0  # row offset within the row group
    url: str | None = None


@dataclass
class Stats:
    discovered: int = 0
    skipped: int = 0
    success: int = 0
    failed: int = 0


@dataclass
class ResumeState:
    completed_ids: set[str] | "LazyDoneIDs" | "DBBackedDoneIDs"
    skipped: int
    bootstrapped_from_jsonl: bool = False


@dataclass
class ManifestCacheInfo:
    image_count: int
    manifest_offset: int
    manifest_size: int
    discovery_fingerprint: str | None = None
    last_line_sha256: str = ""


class LazyDoneIDs:
    """Lazily load a shard done file when pending iteration needs membership."""

    def __init__(self, done_path: Path) -> None:
        self.done_path = done_path
        self._ids: set[str] | None = None

    def _load(self) -> set[str]:
        if self._ids is None:
            self._ids = read_done_file(self.done_path)
        return self._ids

    def __contains__(self, image_id: object) -> bool:
        return isinstance(image_id, str) and image_id in self._load()

    def __len__(self) -> int:
        return len(self._load())


class DBBackedDoneIDs:
    """Membership view backed by the manifest-cache done table."""

    def __init__(self, cache_db: Path, shard_id: int) -> None:
        self.cache_db = cache_db
        self.shard_id = shard_id

    def __contains__(self, image_id: object) -> bool:
        if not isinstance(image_id, str):
            return False
        conn = _connect_manifest_cache(self.cache_db)
        try:
            row = conn.execute(
                "SELECT 1 FROM done WHERE shard = ? AND image_id = ? LIMIT 1",
                (self.shard_id, image_id),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def __len__(self) -> int:
        return _manifest_cache_done_count(self.cache_db, self.shard_id)


EncodePayload = tuple[RecapImage, str | None, str | None]


# ── Config ───────────────────────────────────────────────

def load_domain_config(domain: str) -> dict[str, Any]:
    base = RECAP_CONFIG_DIR / "base.yaml"
    overlay_path = RECAP_CONFIG_DIR / "domains" / f"{domain}.yaml"
    if not overlay_path.exists():
        raise FileNotFoundError(f"Domain config not found: {overlay_path}")
    with base.open() as f:
        cfg = yaml.safe_load(f) or {}
    with overlay_path.open() as f:
        overlay = yaml.safe_load(f) or {}
    _deep_merge(cfg, overlay)
    return cfg


def _parse_prometheus_metric(metrics_text: str, metric_name: str) -> int:
    prefix = f"{metric_name}{{"
    for line in metrics_text.splitlines():
        if line.startswith(prefix):
            try:
                return int(float(line.rsplit(" ", 1)[-1]))
            except ValueError:
                return 0
    return 0


def _parse_prometheus_metric_float(metrics_text: str, metric_name: str) -> float:
    prefix = f"{metric_name}{{"
    for line in metrics_text.splitlines():
        if line.startswith(prefix):
            try:
                return float(line.rsplit(" ", 1)[-1])
            except ValueError:
                return 0.0
    return 0.0


def _parse_prometheus_metric_prefix(metrics_text: str, metric_prefix: str) -> int:
    for line in metrics_text.splitlines():
        if line.startswith(metric_prefix):
            try:
                return int(float(line.rsplit(" ", 1)[-1]))
            except ValueError:
                return 0
    return 0


def _deep_merge(base: dict, overlay: dict) -> None:
    for k, v in overlay.items():
        if k == "extends":
            continue
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def get_prompts(cfg: dict) -> tuple[str, str, int, list[str]]:
    cc = cfg.get("caption", {})
    sys_prompt = str(cc.get("system_prompt", "")).strip()
    long = cc.get("stages", {}).get("long", {})
    user_prompt = str(long.get("prompt", "")).strip()
    max_tokens = int(long.get("max_tokens", 512))
    context_fields = list(long.get("context_fields", []))
    if not sys_prompt or not user_prompt:
        raise ValueError("Config must define caption.system_prompt and caption.stages.long.prompt")
    return sys_prompt, user_prompt, max_tokens, context_fields


# ── Booru metadata grounding ─────────────────────────────

def _read_manifest_image_ids(manifest_path: Path, limit: int | None = None) -> set[str]:
    image_ids: set[str] = set()
    with manifest_path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            image_id, _, _ = _parse_manifest_cache_fields(line)
            image_ids.add(str(image_id))
            if limit and len(image_ids) >= limit:
                break
    return image_ids


def load_booru_metadata(
    parquet_dir: Path,
    image_ids: set[str] | None = None,
) -> dict[str, dict]:
    """Load Danbooru metadata parquets → {post_id: {tags...}} dict."""
    import pandas as pd

    parquets = sorted(parquet_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files in {parquet_dir}")

    columns = [
        "id", "tag_string_general", "tag_string_character",
        "tag_string_copyright", "tag_string_artist", "rating",
    ]
    needed_ids: set[int] | None = None
    if image_ids is not None:
        needed_ids = set()
        invalid_image_ids = 0
        for image_id in image_ids:
            try:
                needed_ids.add(int(image_id))
            except ValueError:
                invalid_image_ids += 1
                continue
        if image_ids and not needed_ids:
            logger.warning(
                "Booru metadata filter received %d image IDs but none were numeric; grounding will be empty",
                len(image_ids),
            )
        elif invalid_image_ids:
            logger.warning(
                "Booru metadata filter ignored %d/%d non-numeric image IDs",
                invalid_image_ids,
                len(image_ids),
            )
        logger.info(
            "Loading %d metadata parquets from %s for %d manifest image IDs...",
            len(parquets),
            parquet_dir,
            len(image_ids),
        )
    else:
        logger.info("Loading %d metadata parquets from %s...", len(parquets), parquet_dir)

    dfs = []
    scanned_rows = 0
    for parquet_path in parquets:
        df_part = pd.read_parquet(parquet_path, columns=columns)
        scanned_rows += len(df_part)
        if needed_ids is not None:
            if needed_ids:
                df_part = df_part[df_part["id"].isin(needed_ids)]
            else:
                df_part = df_part.iloc[0:0]
        if not df_part.empty:
            dfs.append(df_part)
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=columns)
    logger.info("Loaded %d/%d metadata rows", len(df), scanned_rows)

    def split_tags(value: Any) -> list[str]:
        if value is None:
            return []
        if pd.isna(value):
            return []
        return str(value).split()

    def clean_text(value: Any) -> str:
        if value is None:
            return ""
        if pd.isna(value):
            return ""
        return str(value)

    meta: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        pid = str(int(row.id))
        meta[pid] = {
            "general_tags": split_tags(row.tag_string_general),
            "character_tags": split_tags(row.tag_string_character),
            "copyright_tags": split_tags(row.tag_string_copyright),
            "artist_tags": split_tags(row.tag_string_artist),
            "rating": clean_text(row.rating),
        }
    return meta


def build_grounding_context(
    image_id: str,
    metadata: dict[str, dict] | None,
    context_fields: list[str],
    cfg: dict,
) -> str:
    """Build grounding context string from booru metadata for prompt injection."""
    if not metadata or not context_fields:
        return ""

    meta = metadata.get(image_id)
    if not meta:
        return ""

    # Compute confirmed_characters if needed
    from audit_recap_t2i.recap.tag_grounding import derive_confirmed_characters
    confirmed_cfg = cfg.get("caption", {}).get("context_policy", {}).get("confirmed_characters", {})
    confirmed = []
    if confirmed_cfg.get("enabled") and "confirmed_characters" in context_fields:
        confirmed = derive_confirmed_characters(
            meta.get("general_tags"),
            meta.get("character_tags"),
            require_solo=bool(confirmed_cfg.get("require_solo", True)),
        )

    context = {**meta, "confirmed_characters": confirmed}

    labels = {
        "general_tags": "general tags",
        "character_tags": "character tags",
        "copyright_tags": "copyright",
        "artist_tags": "artist",
        "confirmed_characters": "confirmed characters",
        "rating": "rating",
    }

    lines = []
    for field in context_fields:
        val = context.get(field)
        if val is None:
            continue
        if isinstance(val, list):
            if not val:
                continue
            lines.append(f"{labels.get(field, field)}: {', '.join(str(v).replace('_', ' ') for v in val)}")
        else:
            text = str(val).strip()
            if text:
                lines.append(f"{labels.get(field, field)}: {text}")

    if not lines:
        return ""
    context_str = "\n\nGrounding context:\n" + "\n".join(f"- {line}" for line in lines)
    # Rough token estimate: ~4 chars/token. Keep context under ~1500 tokens to stay
    # within max-model-len=4096 after image tokens (~1030) + system/user prompt (~500).
    max_context_chars = 6000
    if len(context_str) > max_context_chars:
        # Truncate general_tags (longest field) to fit; keep character/copyright/rating intact
        priority_lines = [l for l in lines if not l.startswith("general tags:")]
        general_lines = [l for l in lines if l.startswith("general tags:")]
        base = "\n\nGrounding context:\n" + "\n".join(f"- {line}" for line in priority_lines)
        remaining = max_context_chars - len(base) - 20  # margin
        if general_lines and remaining > 40:
            gl = general_lines[0]
            if len(gl) > remaining:
                gl = gl[:remaining].rsplit(",", 1)[0]  # cut at last comma
            base += f"\n- {gl}"
        context_str = base
    return context_str


# ── Input discovery ──────────────────────────────────────

def _discover_from_manifest(
    input_dir: Path, manifest_path: Path, limit: int | None = None,
) -> list[RecapImage]:
    """Read image list from canonicalize manifest — O(N) read, no filesystem scan."""
    entries: list[RecapImage] = []
    seen_ids: set[str] = set()
    duplicate_rows = 0
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_id = row["image_id"]
            rel_path = row["rel_path"]
            if image_id in seen_ids:
                duplicate_rows += 1
                continue
            seen_ids.add(image_id)
            entries.append(RecapImage(
                global_index=len(entries),
                image_id=image_id,
                image_path=rel_path,
                source_kind="file",
                image_file=input_dir / rel_path,
                url=row.get("url"),
            ))
            if limit and len(entries) >= limit:
                break
    if duplicate_rows:
        logger.warning(
            "Manifest %s contains %d duplicate image_id rows; deduplicated during discovery",
            manifest_path,
            duplicate_rows,
        )
    return entries


def _connect_manifest_cache(cache_db: Path) -> sqlite3.Connection:
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cache_db, timeout=60.0)
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            image_id TEXT PRIMARY KEY,
            global_index INTEGER NOT NULL,
            rel_path TEXT NOT NULL,
            url TEXT,
            shard INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS done (
            image_id TEXT PRIMARY KEY,
            shard INTEGER NOT NULL,
            global_index INTEGER NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS images_shard_idx ON images(shard, global_index)")
    conn.execute("CREATE INDEX IF NOT EXISTS done_shard_global_idx ON done(shard, global_index)")
    return conn


def _read_manifest_cache_info(cache_db: Path) -> ManifestCacheInfo | None:
    if not cache_db.exists():
        return None
    conn = _connect_manifest_cache(cache_db)
    try:
        meta = _manifest_cache_meta(conn)
        if not meta:
            return None
        return ManifestCacheInfo(
            image_count=int(meta.get("image_count", "0") or 0),
            manifest_offset=int(meta.get("manifest_offset", "0") or 0),
            manifest_size=int(meta.get("manifest_size", "0") or 0),
            discovery_fingerprint=meta.get("discovery_fingerprint"),
            last_line_sha256=meta.get("last_line_sha256", ""),
        )
    finally:
        conn.close()


def _manifest_cache_meta(conn: sqlite3.Connection) -> dict[str, str]:
    return {key: value for key, value in conn.execute("SELECT key, value FROM meta")}


def _set_manifest_cache_meta(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        [(key, str(value)) for key, value in payload.items()],
    )


def _reset_manifest_cache(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM images")
    conn.execute("DELETE FROM done")
    conn.execute("DELETE FROM meta")
    conn.commit()


def _manifest_first_line_sha256(manifest_path: Path) -> str:
    with manifest_path.open("rb") as handle:
        line = handle.readline().rstrip(b"\n")
    return hashlib.sha256(line).hexdigest()


def _manifest_line_before_offset_sha256(manifest_path: Path, offset: int) -> str:
    if offset <= 0:
        return ""
    read_size = min(offset, 1024 * 1024)
    with manifest_path.open("rb") as handle:
        handle.seek(offset - read_size)
        chunk = handle.read(read_size)
    lines = chunk.rstrip(b"\n").splitlines()
    if not lines:
        return ""
    return hashlib.sha256(lines[-1]).hexdigest()


def _sync_manifest_cache(
    input_dir: Path,
    manifest_path: Path,
    cache_db: Path,
    *,
    num_shards: int,
    hardcap_path: Path | None = None,
    hardcap_use_pct: float = 0.0,
    max_new_rows: int | None = None,
) -> ManifestCacheInfo:
    """Update the byte-offset SQLite cache for a canonicalize manifest.

    This keeps restart cost proportional to newly appended manifest bytes instead
    of reparsing the full JSONL manifest every time recap is resumed.
    """
    manifest_stat = manifest_path.stat()
    manifest_first_line_hash = _manifest_first_line_sha256(manifest_path)
    conn = _connect_manifest_cache(cache_db)
    try:
        if hardcap_path is not None and check_hardcap(hardcap_path, hardcap_use_pct):
            raise RuntimeError(
                f"Hardcap reached while syncing manifest cache: {hardcap_path} "
                f"used={disk_used_pct(hardcap_path):.2f}% hardcap={hardcap_use_pct:.2f}%"
            )
        meta = _manifest_cache_meta(conn)
        expected = {
            "version": str(MANIFEST_CACHE_VERSION),
            "input_dir": str(input_dir.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "num_shards": str(num_shards),
            "manifest_device": str(manifest_stat.st_dev),
            "manifest_inode": str(manifest_stat.st_ino),
            "manifest_first_line_sha256": manifest_first_line_hash,
        }
        reset_reasons: list[str] = []
        if meta:
            reset_reasons.extend(
                key for key, value in expected.items()
                if meta.get(key) != value
            )
        offset = int(meta.get("manifest_offset", "0") or 0)
        image_count = int(meta.get("image_count", "0") or 0)
        next_global_index = int(meta.get("next_global_index", "0") or 0)
        if image_count and next_global_index < image_count:
            row = conn.execute("SELECT COALESCE(MAX(global_index) + 1, 0) FROM images").fetchone()
            next_global_index = int(row[0] or image_count)
        if offset > manifest_stat.st_size:
            reset_reasons.append("manifest_shrank")
        if meta and offset > 0:
            expected_tail_hash = meta.get("last_line_sha256")
            current_tail_hash = _manifest_line_before_offset_sha256(manifest_path, offset)
            if not expected_tail_hash or current_tail_hash != expected_tail_hash:
                reset_reasons.append("manifest_tail_mismatch")
        if reset_reasons:
            logger.info("Manifest cache reset (%s): %s", ", ".join(reset_reasons), cache_db)
            _reset_manifest_cache(conn)
            meta = {}
            offset = 0
            image_count = 0
            next_global_index = 0

        inserted = 0
        duplicate_rows = 0
        parsed_rows = 0
        last_offset = offset
        last_line_hash = meta.get("last_line_sha256", "")
        pending_rows: list[tuple[str, int, str, str | None, int]] = []

        def flush_manifest_rows() -> None:
            nonlocal inserted, duplicate_rows, next_global_index
            if not pending_rows:
                return
            before_changes = conn.total_changes
            conn.executemany(
                """
                INSERT OR IGNORE INTO images(image_id, global_index, rel_path, url, shard)
                VALUES (?, ?, ?, ?, ?)
                """,
                pending_rows,
            )
            inserted_batch = conn.total_changes - before_changes
            inserted += inserted_batch
            duplicate_rows += len(pending_rows) - inserted_batch
            next_global_index = max(next_global_index, max(row[1] for row in pending_rows) + 1)
            pending_rows.clear()

        with manifest_path.open("rb") as handle:
            handle.seek(offset)
            while True:
                if max_new_rows is not None and parsed_rows >= max_new_rows:
                    break
                line_offset = handle.tell()
                if line_offset >= manifest_stat.st_size:
                    break
                raw_line = handle.readline()
                if not raw_line:
                    break
                if not raw_line.endswith(b"\n"):
                    # Canonicalize may be appending concurrently; leave partial tail for next run.
                    last_offset = line_offset
                    break
                last_offset = handle.tell()
                last_line_hash = hashlib.sha256(raw_line.rstrip(b"\n")).hexdigest()
                line = raw_line.strip()
                if not line:
                    continue
                image_id, rel_path, url = _parse_manifest_cache_fields(line)
                shard = image_id_to_shard(image_id, num_shards)
                pending_rows.append(
                    (image_id, next_global_index + len(pending_rows), rel_path, url, shard),
                )
                parsed_rows += 1
                if parsed_rows % 100_000 == 0:
                    flush_manifest_rows()
                    if hardcap_path is not None and check_hardcap(hardcap_path, hardcap_use_pct):
                        raise RuntimeError(
                            f"Hardcap reached while syncing manifest cache: {hardcap_path} "
                            f"used={disk_used_pct(hardcap_path):.2f}% hardcap={hardcap_use_pct:.2f}%"
                        )
                    if inserted:
                        conn.execute("DELETE FROM meta WHERE key = 'discovery_fingerprint'")
                    _set_manifest_cache_meta(
                        conn,
                        {
                            **expected,
                            "manifest_offset": last_offset,
                            "manifest_size": manifest_stat.st_size,
                            "image_count": image_count + inserted,
                            "next_global_index": next_global_index,
                            "last_line_sha256": last_line_hash,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    conn.commit()
                    logger.info(
                        "Manifest cache ingest progress: parsed=%d inserted=%d duplicate=%d offset=%d db=%s",
                        parsed_rows,
                        inserted,
                        duplicate_rows,
                        last_offset,
                        cache_db,
                    )

        flush_manifest_rows()
        if hardcap_path is not None and check_hardcap(hardcap_path, hardcap_use_pct):
            raise RuntimeError(
                f"Hardcap reached while syncing manifest cache: {hardcap_path} "
                f"used={disk_used_pct(hardcap_path):.2f}% hardcap={hardcap_use_pct:.2f}%"
            )
        if inserted:
            conn.execute("DELETE FROM meta WHERE key = 'discovery_fingerprint'")
        _set_manifest_cache_meta(
            conn,
            {
                **expected,
                "manifest_offset": last_offset,
                "manifest_size": manifest_stat.st_size,
                "image_count": image_count + inserted,
                "next_global_index": next_global_index,
                "last_line_sha256": last_line_hash,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        conn.commit()
        if parsed_rows or duplicate_rows:
            logger.info(
                "Manifest cache ingest: parsed=%d inserted=%d duplicate=%d offset=%d db=%s",
                parsed_rows,
                inserted,
                duplicate_rows,
                last_offset,
                cache_db,
            )

        meta = _manifest_cache_meta(conn)
        return ManifestCacheInfo(
            image_count=int(meta.get("image_count", "0") or 0),
            manifest_offset=int(meta.get("manifest_offset", "0") or 0),
            manifest_size=int(meta.get("manifest_size", "0") or 0),
            discovery_fingerprint=meta.get("discovery_fingerprint"),
            last_line_sha256=meta.get("last_line_sha256", ""),
        )
    finally:
        conn.close()


def _manifest_cache_fingerprint(cache_db: Path) -> str:
    conn = _connect_manifest_cache(cache_db)
    try:
        meta = _manifest_cache_meta(conn)
        cached = meta.get("discovery_fingerprint")
        if cached:
            return cached
        digest = hashlib.sha256()
        for (image_id,) in conn.execute("SELECT image_id FROM images ORDER BY image_id"):
            digest.update(str(image_id).encode("utf-8"))
            digest.update(b"\0")
            digest.update(b"file")
            digest.update(b"\n")
        fingerprint = digest.hexdigest()
        _set_manifest_cache_meta(conn, {"discovery_fingerprint": fingerprint})
        conn.commit()
        return fingerprint
    finally:
        conn.close()


def _manifest_cache_append_fingerprint(input_manifest_path: Path, cache_info: ManifestCacheInfo) -> str:
    digest = hashlib.sha256()
    digest.update(str(input_manifest_path.resolve()).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(cache_info.image_count).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(cache_info.manifest_offset).encode("utf-8"))
    digest.update(b"\0")
    digest.update(cache_info.last_line_sha256.encode("utf-8"))
    return digest.hexdigest()


def _load_manifest_cache_shard(input_dir: Path, cache_db: Path, shard: int) -> list[RecapImage]:
    conn = _connect_manifest_cache(cache_db)
    try:
        return [
            RecapImage(
                global_index=global_index,
                image_id=image_id,
                image_path=rel_path,
                source_kind="file",
                image_file=input_dir / rel_path,
                url=url,
            )
            for global_index, image_id, rel_path, url in conn.execute(
                """
                SELECT global_index, image_id, rel_path, url
                FROM images
                WHERE shard = ?
                ORDER BY global_index
                """,
                (shard,),
            )
        ]
    finally:
        conn.close()


def _manifest_cache_shard_count(cache_db: Path, shard: int) -> int:
    conn = _connect_manifest_cache(cache_db)
    try:
        row = conn.execute("SELECT count(*) FROM images WHERE shard = ?", (shard,)).fetchone()
        return int(row[0] or 0)
    finally:
        conn.close()


def _manifest_cache_done_overlap_count(cache_db: Path, done_ids: set[str]) -> int:
    if not done_ids:
        return 0
    conn = _connect_manifest_cache(cache_db)
    try:
        total = 0
        ids = list(done_ids)
        chunk_size = 900
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start:start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            row = conn.execute(
                f"SELECT count(*) FROM images WHERE image_id IN ({placeholders})",
                chunk,
            ).fetchone()
            total += int(row[0] or 0)
        return total
    finally:
        conn.close()


def _manifest_cache_done_count(cache_db: Path, shard_id: int) -> int:
    conn = _connect_manifest_cache(cache_db)
    try:
        row = conn.execute("SELECT count(*) FROM done WHERE shard = ?", (shard_id,)).fetchone()
        return int(row[0] or 0)
    finally:
        conn.close()


def _manifest_cache_current_done_count(cache_db: Path, shard_id: int) -> int:
    conn = _connect_manifest_cache(cache_db)
    try:
        row = conn.execute(
            """
            SELECT count(*)
            FROM done AS d
            JOIN images AS i ON i.image_id = d.image_id
            WHERE d.shard = ? AND i.shard = ?
            """,
            (shard_id, shard_id),
        ).fetchone()
        return int(row[0] or 0)
    finally:
        conn.close()


def _done_mirror_meta_payload(shard_id: int, done_path: Path, done_count: int) -> dict[str, Any]:
    stat = _done_file_stat(done_path)
    prefix = f"done_mirror.shard_{shard_id:04d}"
    return {
        f"{prefix}.done_count": done_count,
        f"{prefix}.done_file_size": stat["done_file_size"],
        f"{prefix}.done_file_mtime_ns": stat["done_file_mtime_ns"],
    }


def _done_mirror_meta_matches(
    cache_db: Path,
    *,
    shard_id: int,
    done_path: Path,
    done_count: int,
) -> bool:
    if not done_path.exists():
        return done_count == 0 and _manifest_cache_done_count(cache_db, shard_id) == 0
    conn = _connect_manifest_cache(cache_db)
    try:
        meta = _manifest_cache_meta(conn)
        expected = _done_mirror_meta_payload(shard_id, done_path, done_count)
        db_count = conn.execute("SELECT count(*) FROM done WHERE shard = ?", (shard_id,)).fetchone()
        if int(db_count[0] or 0) != done_count:
            return False
        return all(meta.get(key) == str(value) for key, value in expected.items())
    finally:
        conn.close()


def _insert_manifest_cache_done_rows(
    conn: sqlite3.Connection,
    rows: list[tuple[str, int, int]],
) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT OR IGNORE INTO done(image_id, shard, global_index)
        VALUES (?, ?, ?)
        """,
        rows,
    )


def _replace_manifest_cache_done_ids(
    cache_db: Path,
    done_ids: set[str],
    *,
    shard_id: int,
    num_shards: int,
    done_path: Path,
) -> int:
    """Mirror DDN `.done` IDs into SQLite for cache-aware pending queries.

    The `.done` files remain authoritative. This table is only an acceleration
    index and is populated by joining completed IDs against the manifest cache,
    so stale/foreign IDs cannot hide current images.
    """
    foreign_ids = {image_id for image_id in done_ids if image_id_to_shard(image_id, num_shards) != shard_id}
    if foreign_ids:
        sample = ", ".join(sorted(foreign_ids)[:5])
        raise RuntimeError(
            f"Done IDs outside shard-{shard_id:04d} ({sample}). "
            "Resume geometry changed or the done file is corrupted."
        )

    conn = _connect_manifest_cache(cache_db)
    try:
        conn.execute("DELETE FROM done WHERE shard = ?", (shard_id,))
        if not done_ids:
            _set_manifest_cache_meta(conn, _done_mirror_meta_payload(shard_id, done_path, 0))
            conn.commit()
            return 0
        rows: list[tuple[str, int, int]] = []
        for image_id in done_ids:
            rows.append((image_id, shard_id, -1))
            if len(rows) >= 100_000:
                _insert_manifest_cache_done_rows(conn, rows)
                rows.clear()
                conn.commit()
        _insert_manifest_cache_done_rows(conn, rows)
        _set_manifest_cache_meta(conn, _done_mirror_meta_payload(shard_id, done_path, len(done_ids)))
        conn.commit()
        row = conn.execute(
            """
            SELECT count(*)
            FROM done AS d
            JOIN images AS i ON i.image_id = d.image_id
            WHERE d.shard = ? AND i.shard = ?
            """,
            (shard_id, shard_id),
        ).fetchone()
        return int(row[0] or 0)
    finally:
        conn.close()


def bootstrap_done_from_jsonl_by_shard(
    *,
    jsonl_path: Path,
    shard_id: int,
    num_shards: int,
    done_path: Path,
) -> set[str]:
    completed: set[str] = set()
    ordered_ids: list[str] = []
    duplicate_ids = 0

    if not jsonl_path.exists():
        sync_done_file(done_path, [])
        return completed

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL in {jsonl_path}:{line_no}: {exc}") from exc
            image_id = str(record.get("image_id", "")).strip()
            if not image_id:
                raise RuntimeError(f"Missing image_id in {jsonl_path}:{line_no}")
            if image_id_to_shard(image_id, num_shards) != shard_id:
                raise RuntimeError(
                    f"{jsonl_path}:{line_no} contains image_id={image_id!r} outside shard-{shard_id:04d}. "
                    "Resume geometry changed; refusing unsafe replay."
                )
            if image_id in completed:
                duplicate_ids += 1
                continue
            completed.add(image_id)
            ordered_ids.append(image_id)

    if duplicate_ids:
        logger.warning("%s contains %d image_id duplicates; done state was deduplicated", jsonl_path, duplicate_ids)
    sync_done_file(done_path, ordered_ids)
    return completed


def load_manifest_cache_resume_state(
    *,
    cache_db: Path,
    shard_id: int,
    num_shards: int,
    ddn_jsonl: Path,
    ddn_done: Path,
    ddn_offset: Path,
    ddn_state: Path,
    trust_done_mirror: bool = False,
) -> tuple[ResumeState, int, int]:
    """Resume a manifest-cache shard without materializing all shard rows."""
    total_rows = _manifest_cache_shard_count(cache_db, shard_id)
    offset = read_offset_file(ddn_offset)
    shard_state = read_shard_state(ddn_state)
    jsonl_size = ddn_jsonl.stat().st_size if ddn_jsonl.exists() else 0
    state_jsonl_size = int(shard_state.get("jsonl_size", -1))
    state_done_count = int(shard_state.get("done_count", -1))
    state_done_file_size = int(shard_state.get("done_file_size", -1))
    state_done_file_mtime_ns = int(shard_state.get("done_file_mtime_ns", -1))
    done_stat = _done_file_stat(ddn_done)
    fast_state_ok = (
        bool(shard_state)
        and ddn_done.exists()
        and state_jsonl_size == jsonl_size
        and state_done_count >= 0
        and state_done_count <= total_rows
        and offset <= state_done_count
        and state_done_file_size == done_stat["done_file_size"]
        and state_done_file_mtime_ns == done_stat["done_file_mtime_ns"]
    )
    if fast_state_ok:
        if _done_mirror_meta_matches(
            cache_db,
            shard_id=shard_id,
            done_path=ddn_done,
            done_count=state_done_count,
        ):
            db_done_count = (
                state_done_count
                if trust_done_mirror
                else _manifest_cache_current_done_count(cache_db, shard_id)
            )
        else:
            done_ids = read_done_file(ddn_done)
            db_done_count = _replace_manifest_cache_done_ids(
                cache_db,
                done_ids,
                shard_id=shard_id,
                num_shards=num_shards,
                done_path=ddn_done,
            )
        skipped = min(db_done_count, total_rows)
        write_offset_file(ddn_offset, skipped)
        return (
            ResumeState(completed_ids=DBBackedDoneIDs(cache_db, shard_id), skipped=skipped),
            total_rows,
            max(0, total_rows - skipped),
        )

    done_ids = read_done_file(ddn_done)
    foreign_ids = {image_id for image_id in done_ids if image_id_to_shard(image_id, num_shards) != shard_id}
    if foreign_ids:
        sample = ", ".join(sorted(foreign_ids)[:5])
        raise RuntimeError(
            f"{ddn_done} contains IDs outside shard-{shard_id:04d} ({sample}). "
            "Resume geometry changed or the done file is corrupted."
        )

    state_mismatch = (
        ddn_jsonl.exists()
        and jsonl_size > 0
        and (
            not shard_state
            or state_jsonl_size != jsonl_size
            or state_done_count != len(done_ids)
        )
    )
    needs_bootstrap = (
        ddn_jsonl.exists()
        and (
            not ddn_done.exists()
            or (not done_ids and ddn_jsonl.stat().st_size > 0)
            or offset > len(done_ids)
            or state_mismatch
        )
    )
    bootstrapped = False
    if needs_bootstrap:
        if state_mismatch:
            logger.info(
                "  %s: shard state mismatch, reconciling done from JSONL "
                "(state_jsonl=%s current_jsonl=%d state_done=%s current_done=%d)",
                ddn_jsonl.name,
                state_jsonl_size,
                jsonl_size,
                state_done_count,
                len(done_ids),
            )
        done_ids = bootstrap_done_from_jsonl_by_shard(
            jsonl_path=ddn_jsonl,
            shard_id=shard_id,
            num_shards=num_shards,
            done_path=ddn_done,
        )
        bootstrapped = ddn_jsonl.exists()

    overlap = _replace_manifest_cache_done_ids(
        cache_db,
        done_ids,
        shard_id=shard_id,
        num_shards=num_shards,
        done_path=ddn_done,
    )
    if overlap == 0 and done_ids:
        overlap = _manifest_cache_done_overlap_count(cache_db, done_ids)
    write_offset_file(ddn_offset, len(done_ids))
    write_shard_state(ddn_state, done_count=len(done_ids), jsonl_size=jsonl_size, done_path=ddn_done)
    return (
        ResumeState(completed_ids=DBBackedDoneIDs(cache_db, shard_id), skipped=overlap, bootstrapped_from_jsonl=bootstrapped),
        total_rows,
        max(0, total_rows - overlap),
    )


class ManifestCachePendingShard:
    def __init__(
        self,
        *,
        input_dir: Path,
        cache_db: Path,
        shard_id: int,
        completed_ids: set[str] | LazyDoneIDs | DBBackedDoneIDs,
        pending_count: int,
    ) -> None:
        self.input_dir = input_dir
        self.cache_db = cache_db
        self.shard_id = shard_id
        self.completed_ids = completed_ids
        self.pending_count = pending_count

    def __len__(self) -> int:
        return self.pending_count

    def __iter__(self):
        conn = _connect_manifest_cache(self.cache_db)
        yielded = 0
        try:
            if isinstance(self.completed_ids, DBBackedDoneIDs):
                for global_index, image_id, rel_path, url in conn.execute(
                    """
                    SELECT i.global_index, i.image_id, i.rel_path, i.url
                    FROM images AS i
                    LEFT JOIN done AS d ON d.image_id = i.image_id
                    WHERE i.shard = ? AND d.image_id IS NULL
                    ORDER BY i.global_index
                    """,
                    (self.shard_id,),
                ):
                    yielded += 1
                    yield RecapImage(
                        global_index=global_index,
                        image_id=image_id,
                        image_path=rel_path,
                        source_kind="file",
                        image_file=self.input_dir / rel_path,
                        url=url,
                    )
                return

            for global_index, image_id, rel_path, url in conn.execute(
                """
                SELECT global_index, image_id, rel_path, url
                FROM images
                WHERE shard = ?
                ORDER BY global_index
                """,
                (self.shard_id,),
            ):
                if image_id in self.completed_ids:
                    continue
                yielded += 1
                yield RecapImage(
                    global_index=global_index,
                    image_id=image_id,
                    image_path=rel_path,
                    source_kind="file",
                    image_file=self.input_dir / rel_path,
                    url=url,
                )
        finally:
            conn.close()


def _discover_from_manifest_cache(
    input_dir: Path,
    manifest_path: Path,
    cache_db: Path,
    *,
    num_shards: int,
    limit: int | None = None,
) -> list[RecapImage]:
    """Read canonicalize manifest through a byte-offset SQLite cache."""
    if limit is not None:
        return _discover_from_manifest(input_dir, manifest_path, limit)

    _sync_manifest_cache(input_dir, manifest_path, cache_db, num_shards=num_shards)
    conn = _connect_manifest_cache(cache_db)
    try:
        entries: list[RecapImage] = []
        for global_index, image_id, rel_path, url in conn.execute(
            # Rows are inserted in canonical manifest order. Reading by rowid
            # avoids a full temp sort over tens of millions of rows on resume.
            "SELECT global_index, image_id, rel_path, url FROM images ORDER BY rowid"
        ):
            entries.append(
                RecapImage(
                    global_index=global_index,
                    image_id=image_id,
                    image_path=rel_path,
                    source_kind="file",
                    image_file=input_dir / rel_path,
                    url=url,
                )
            )
        return entries
    finally:
        conn.close()


def should_use_sharded_manifest_cache(
    *,
    input_manifest_path: Path,
    manifest_cache_db: Path | None,
    limit: int | None,
    parquet_id_col: str | None,
) -> bool:
    return (
        input_manifest_path.exists()
        and manifest_cache_db is not None
        and limit is None
        and parquet_id_col is None
    )


def explicit_manifest_cache_requires_manifest(
    *,
    manifest_cache_arg: str | None,
    input_manifest_path: Path,
    limit: int | None,
    parquet_id_col: str | None,
) -> None:
    if (
        manifest_cache_arg
        and limit is None
        and parquet_id_col is None
        and not input_manifest_path.exists()
    ):
        raise RuntimeError(
            "--manifest-cache-db was supplied, but input _manifest.jsonl is missing: "
            f"{input_manifest_path}. Refusing filesystem discovery because it would "
            "change the resume/canonical ID surface."
        )


def _discover_from_parquets(
    input_dir: Path,
    id_col: str = "photoid",
    image_col: str = "jpg",
    limit: int | None = None,
) -> list[RecapImage]:
    """Discover images from HF-style parquet dataset (images embedded in columns).

    Scans parquet metadata only (no image decode), enumerating rows with their
    parquet file, row group, and row offset for later streaming reads.
    """
    import pyarrow.parquet as pq

    parquets = sorted(input_dir.rglob("*.parquet"))
    if not parquets:
        return []

    logger.info("Scanning %d parquet files for image rows (id_col=%s)...", len(parquets), id_col)
    entries: list[RecapImage] = []
    seen_ids: set[str] = set()
    for pq_path in parquets:
        try:
            pf = pq.ParquetFile(pq_path)
        except Exception as exc:
            logger.warning("Skipping bad parquet %s: %s", pq_path, exc)
            continue
        for rg_idx in range(pf.metadata.num_row_groups):
            # Read only the ID column from this row group (no image decode)
            id_table = pf.read_row_group(rg_idx, columns=[id_col])
            id_array = id_table.column(id_col)
            for row_off in range(len(id_array)):
                raw_id = id_array[row_off].as_py()
                image_id = str(int(raw_id)) if isinstance(raw_id, (int, float)) else str(raw_id)
                if image_id in seen_ids:
                    continue
                seen_ids.add(image_id)
                try:
                    rel = pq_path.relative_to(input_dir)
                except ValueError:
                    rel = pq_path.name
                entries.append(RecapImage(
                    global_index=len(entries),
                    image_id=image_id,
                    image_path=f"{rel}:rg{rg_idx}:{row_off}",
                    source_kind="parquet",
                    parquet_path=pq_path,
                    parquet_row_group=rg_idx,
                    parquet_row_offset=row_off,
                ))
                if limit and len(entries) >= limit:
                    break
            if limit and len(entries) >= limit:
                break
        if limit and len(entries) >= limit:
            break

    logger.info("Discovered %d images from parquets", len(entries))
    return entries


def discover_images(
    input_dir: Path,
    limit: int | None = None,
    parquet_id_col: str | None = None,
    parquet_image_col: str | None = None,
    manifest_cache_db: Path | None = None,
    manifest_cache_shards: int = DEFAULT_NUM_SHARDS,
) -> list[RecapImage]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input not found: {input_dir}")

    # Parquet mode: if parquet columns are specified or directory contains parquets
    if parquet_id_col is not None:
        entries = _discover_from_parquets(
            input_dir,
            id_col=parquet_id_col,
            image_col=parquet_image_col or "jpg",
            limit=limit,
        )
        if entries:
            return entries
        raise FileNotFoundError(f"No parquet images found in {input_dir}")

    # Manifest-first: use canonicalize manifest if available (instant vs minutes of scanning)
    manifest_path = input_dir / "_manifest.jsonl"
    if manifest_path.exists():
        logger.info("Using manifest: %s", manifest_path)
        if manifest_cache_db is not None:
            entries = _discover_from_manifest_cache(
                input_dir,
                manifest_path,
                manifest_cache_db,
                num_shards=manifest_cache_shards,
                limit=limit,
            )
            if entries:
                return entries
            raise RuntimeError(f"Manifest cache returned no entries for non-empty manifest: {manifest_path}")
        try:
            entries = _discover_from_manifest(input_dir, manifest_path, limit)
            if entries:
                return entries
            logger.warning("Manifest empty, falling back to filesystem scan")
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("Manifest corrupt (%s), falling back to filesystem scan", exc)

    entries: list[RecapImage] = []

    # Try WebDataset tars first
    tars = sorted(
        p for p in input_dir.iterdir()
        if p.suffix in {".tar", ".gz", ".tgz"} or p.name.endswith(".tar.gz")
    ) if input_dir.is_dir() else []

    if tars:
        for tar_path in tars:
            try:
                with tarfile.open(tar_path, "r:*") as tf:
                    for member in tf.getmembers():
                        if not member.isfile():
                            continue
                        if Path(member.name).suffix.lower() not in IMAGE_EXTENSIONS:
                            continue
                        entries.append(RecapImage(
                            global_index=len(entries),
                            image_id=Path(member.name).stem,
                            image_path=member.name,
                            source_kind="tar",
                            tar_path=tar_path,
                            tar_member=member.name,
                        ))
                        if limit and len(entries) >= limit:
                            break
            except tarfile.TarError as e:
                logger.warning("Skipping bad tar %s: %s", tar_path, e)
            if limit and len(entries) >= limit:
                break
    else:
        # Flat directory — use subprocess find for speed (300x faster than rglob on millions of files)
        import subprocess
        ext_args = []
        for ext in IMAGE_EXTENSIONS:
            if ext_args:
                ext_args.append("-o")
            ext_args.extend(["-name", f"*{ext}"])
        result = subprocess.run(
            ["find", str(input_dir), "-type", "f", "("] + ext_args + [")"],
            capture_output=True, text=True,
        )
        paths = sorted(result.stdout.strip().split("\n")) if result.stdout.strip() else []
        for p_str in paths:
            path = Path(p_str)
            entries.append(RecapImage(
                global_index=len(entries),
                image_id=path.stem,
                image_path=str(path.relative_to(input_dir)),
                source_kind="file",
                image_file=path,
            ))
            if limit and len(entries) >= limit:
                break

    if not entries:
        raise FileNotFoundError(f"No images found in {input_dir}")
    return entries


# ── Image encoding ───────────────────────────────────────

class TarCache:
    """Pre-load tar member index for random access without re-scanning."""

    def __init__(self):
        self._handles: dict[Path, tarfile.TarFile] = {}
        self._members: dict[Path, dict[str, tarfile.TarInfo]] = {}

    def read_image(self, tar_path: Path, member_name: str) -> bytes:
        if tar_path not in self._handles:
            tf = tarfile.open(tar_path, "r:")
            self._handles[tar_path] = tf
            self._members[tar_path] = {m.name: m for m in tf.getmembers()}
        info = self._members[tar_path].get(member_name)
        if info is None:
            raise FileNotFoundError(f"'{member_name}' not in {tar_path.name}")
        f = self._handles[tar_path].extractfile(info)
        if f is None:
            raise FileNotFoundError(member_name)
        data = f.read()
        f.close()
        return data

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles.clear()
        self._members.clear()


class ParquetCache:
    """Streaming reader for HF parquet datasets with embedded images.

    Reads row groups lazily via pyarrow, caches the current row group table
    to amortize I/O. Images are read as raw bytes (no PIL decode) and handed
    off to _to_data_url for resize + base64 encoding.
    """

    def __init__(self, image_col: str = "jpg", id_col: str = "photoid"):
        import pyarrow.parquet as pq  # noqa: F811
        self._pq = pq
        self._image_col = image_col
        self._id_col = id_col
        self._open_files: dict[Path, Any] = {}  # Path → pq.ParquetFile
        self._cached_rg: dict[tuple[Path, int], Any] = {}  # (path, rg_idx) → Table

    _MAX_OPEN_FILES = 8  # bound file descriptors per producer thread

    def _get_file(self, parquet_path: Path):
        if parquet_path not in self._open_files:
            # Evict oldest file handle if at capacity
            if len(self._open_files) >= self._MAX_OPEN_FILES:
                oldest_key = next(iter(self._open_files))
                del self._open_files[oldest_key]
                # Also evict any row groups from the closed file
                self._cached_rg = {
                    k: v for k, v in self._cached_rg.items() if k[0] != oldest_key
                }
            self._open_files[parquet_path] = self._pq.ParquetFile(parquet_path)
        return self._open_files[parquet_path]

    def _get_row_group(self, parquet_path: Path, rg_idx: int):
        key = (parquet_path, rg_idx)
        if key not in self._cached_rg:
            # Evict oldest to bound memory — keep at most 2 row groups
            if len(self._cached_rg) >= 2:
                oldest = next(iter(self._cached_rg))
                del self._cached_rg[oldest]
            pf = self._get_file(parquet_path)
            self._cached_rg[key] = pf.read_row_group(rg_idx, columns=[self._image_col])
        return self._cached_rg[key]

    def read_image_bytes(self, parquet_path: Path, rg_idx: int, row_offset: int) -> bytes:
        """Read raw image bytes from a parquet row group at the given offset."""
        table = self._get_row_group(parquet_path, rg_idx)
        col = table.column(self._image_col)
        # HF datasets store images as struct {bytes, path} or raw bytes
        cell = col[row_offset]
        if hasattr(cell, "as_py"):
            cell = cell.as_py()
        if isinstance(cell, dict):
            return cell["bytes"]
        if isinstance(cell, bytes):
            return cell
        raise ValueError(f"Unexpected image cell type: {type(cell)}")

    def close(self):
        self._cached_rg.clear()
        self._open_files.clear()


def encode_image(
    record: RecapImage,
    tar_cache: TarCache | None = None,
    parquet_cache: ParquetCache | None = None,
) -> str:
    if record.source_kind == "file":
        with Image.open(record.image_file) as img:
            return _to_data_url(img)
    elif record.source_kind == "tar" and tar_cache:
        data = tar_cache.read_image(record.tar_path, record.tar_member)
        with Image.open(BytesIO(data)) as img:
            return _to_data_url(img)
    elif record.source_kind == "parquet" and parquet_cache:
        raw = parquet_cache.read_image_bytes(
            record.parquet_path, record.parquet_row_group, record.parquet_row_offset,
        )
        with Image.open(BytesIO(raw)) as img:
            return _to_data_url(img)
    raise ValueError(f"Cannot load {record.image_id}")


def _to_data_url(img: Image.Image) -> str:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    short = min(w, h)
    if short > MAX_SHORT_EDGE:
        # Resize so shortest edge = MAX_SHORT_EDGE, preserve aspect ratio (matches canon)
        s = MAX_SHORT_EDGE / short
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ── Resume: DDN checkpoint ───────────────────────────────

def read_offset_file(offset_path: Path) -> int:
    """Read the legacy offset file if present."""
    if offset_path.exists():
        try:
            return int(offset_path.read_text(encoding="utf-8").strip() or 0)
        except ValueError:
            logger.warning("Invalid offset in %s, resetting to 0", offset_path)
    return 0


def write_offset_file(offset_path: Path, offset: int) -> None:
    """Persist a tiny compatibility checkpoint of how many IDs are marked done."""
    tmp = offset_path.with_suffix(".tmp")
    tmp.write_text(f"{offset}\n", encoding="utf-8")
    tmp.replace(offset_path)


def read_shard_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Invalid shard state %s: %s", state_path, exc)
        return {}


def _done_file_stat(done_path: Path) -> dict[str, int]:
    if not done_path.exists():
        return {"done_file_size": 0, "done_file_mtime_ns": 0}
    stat = done_path.stat()
    return {"done_file_size": stat.st_size, "done_file_mtime_ns": stat.st_mtime_ns}


def write_shard_state(state_path: Path, *, done_count: int, jsonl_size: int, done_path: Path | None = None) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "done_count": done_count,
        "jsonl_size": jsonl_size,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if done_path is not None:
        payload.update(_done_file_stat(done_path))
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(state_path)


def compute_discovery_fingerprint(images: list[RecapImage]) -> str:
    """Order-independent fingerprint: sort by image_id before hashing."""
    digest = hashlib.sha256()
    for image in sorted(images, key=lambda i: i.image_id):
        digest.update(image.image_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(image.source_kind.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_run_manifest(args: argparse.Namespace, images: list[RecapImage]) -> dict[str, Any]:
    return build_run_manifest_from_stats(
        args,
        total_images=len(images),
        discovery_fingerprint=compute_discovery_fingerprint(images),
    )


def build_run_manifest_from_stats(
    args: argparse.Namespace,
    *,
    total_images: int,
    discovery_fingerprint: str,
    input_manifest_path: Path | None = None,
    input_manifest_offset: int | None = None,
    input_manifest_last_line_sha256: str | None = None,
    input_manifest_size: int | None = None,
    append_validation_base: ManifestCacheInfo | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "dataset": args.dataset,
        "domain": args.domain,
        "num_shards": int(getattr(args, "num_shards", 64)),
        "num_workers": args.num_workers,
        "use_file_urls": bool(args.use_file_urls),
        "total_images": total_images,
        "discovery_fingerprint": discovery_fingerprint,
        "input_dir": str(Path(args.input_dir).resolve()),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if input_manifest_path is not None:
        payload["input_manifest_path"] = str(input_manifest_path.resolve())
    if input_manifest_offset is not None:
        payload["input_manifest_offset"] = int(input_manifest_offset)
    if input_manifest_last_line_sha256 is not None:
        payload["input_manifest_last_line_sha256"] = input_manifest_last_line_sha256
    if input_manifest_size is not None:
        payload["input_manifest_size"] = int(input_manifest_size)
    if append_validation_base is not None:
        payload["_append_validation_base"] = {
            "image_count": append_validation_base.image_count,
            "manifest_offset": append_validation_base.manifest_offset,
            "last_line_sha256": append_validation_base.last_line_sha256,
            "discovery_fingerprint": append_validation_base.discovery_fingerprint,
        }
    return payload


def public_manifest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(public_manifest_payload(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def reshard_legacy_outputs(ddn_root: Path, num_shards: int) -> int:
    """Reshard existing JSONL+done files from any geometry to the new bit-mapped layout.

    Reads all shard-*.jsonl from one fixed recap surface, reassigns each
    record by image_id_to_shard(), writes new shard files, and returns the
    total migrated records.

    This is an intra-surface repair step only. It must not be treated as a
    cross-surface merge rule for alternative materializations that happen to
    share local-looking image_id formats.
    """
    old_jsonls = sorted(ddn_root.glob("shard-*.jsonl"))
    if not old_jsonls:
        return 0

    # Read all records
    records_by_shard: dict[int, list[tuple[str, str]]] = {i: [] for i in range(num_shards)}
    total = 0
    for jsonl_path in old_jsonls:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                image_id = str(record.get("image_id", "")).strip()
                if not image_id:
                    continue
                sid = image_id_to_shard(image_id, num_shards)
                records_by_shard[sid].append((image_id, raw_line))
                total += 1

    if total == 0:
        return 0

    # Archive old files
    archive_dir = ddn_root / "_legacy_shards"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for old in old_jsonls:
        old.rename(archive_dir / old.name)
    for old in ddn_root.glob("shard-*.offset"):
        old.rename(archive_dir / old.name)
    old_done_dir = ddn_root / "done"
    if old_done_dir.exists():
        for old in old_done_dir.glob("shard-*.done"):
            old.rename(archive_dir / old.name)

    # Write new shard files
    done_dir = ddn_root / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    migrated = 0
    for sid, entries in records_by_shard.items():
        if not entries:
            continue
        # Deduplicate by image_id (keep first)
        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for image_id, raw_line in entries:
            if image_id not in seen:
                seen.add(image_id)
                unique.append((image_id, raw_line))

        # Write JSONL
        jsonl_path = ddn_root / f"shard-{sid:04d}.jsonl"
        tmp = jsonl_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for _, raw_line in unique:
                f.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(jsonl_path)

        # Write done file
        done_path = done_dir / f"shard-{sid:04d}.done"
        tmp = done_path.with_suffix(".done.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for image_id, _ in unique:
                f.write(f"{image_id}\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(done_path)

        migrated += len(unique)

    logger.info("Resharded %d records from %d old files → %d new shards (legacy archived to %s)",
                migrated, len(old_jsonls), num_shards, archive_dir)
    return migrated


def has_existing_recap_outputs(ddn_root: Path) -> bool:
    if any(ddn_root.glob("shard-*.jsonl")):
        return True
    done_dir = ddn_root / "done"
    return done_dir.exists() and any(done_dir.glob("shard-*.done"))


def validate_or_write_manifest(
    manifest_path: Path,
    current: dict[str, Any],
    *,
    has_outputs: bool,
    incremental: bool = False,
    rolling_corpus: bool = False,
) -> None:
    if not manifest_path.exists():
        write_json_file(manifest_path, current)
        return

    existing = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Keys exempt from strict checking in incremental mode (corpus may grow)
    _INCREMENTAL_EXEMPT = frozenset(
        {
            "total_images",
            "discovery_fingerprint",
            "input_manifest_offset",
            "input_manifest_last_line_sha256",
            "input_manifest_size",
        }
    )

    manifest_keys = (
        "version",
        "dataset",
        "domain",
        "num_shards",
        "num_workers",
        "use_file_urls",
        "total_images",
        "discovery_fingerprint",
        "input_dir",
        "input_manifest_path",
        "input_manifest_offset",
        "input_manifest_last_line_sha256",
        "input_manifest_size",
    )
    mismatches: list[str] = []
    incremental_mismatches: list[str] = []
    rolling_exempt = _INCREMENTAL_EXEMPT | {"input_manifest_path"}
    for key in manifest_keys:
        if existing.get(key) != current.get(key):
            if rolling_corpus and key in rolling_exempt:
                incremental_mismatches.append(
                    f"{key}: existing={existing.get(key)!r} current={current.get(key)!r}"
                )
            elif incremental and (key in _INCREMENTAL_EXEMPT or (key == "input_manifest_path" and key not in existing)):
                incremental_mismatches.append(
                    f"{key}: existing={existing.get(key)!r} current={current.get(key)!r}"
                )
            else:
                mismatches.append(
                    f"{key}: existing={existing.get(key)!r} current={current.get(key)!r}"
                )

    if mismatches and has_outputs:
        details = "\n".join(f"  - {item}" for item in mismatches)
        raise RuntimeError(
            "Resume manifest mismatch detected. Refusing unsafe replay.\n"
            f"Manifest: {manifest_path}\n"
            f"{details}\n"
            "Use the same corpus/shard geometry or start with a new output directory."
        )

    if incremental_mismatches and has_outputs and rolling_corpus:
        for item in incremental_mismatches:
            logger.info("Rolling corpus: manifest field changed — %s", item)
    elif incremental_mismatches and has_outputs:
        old_total = existing.get("total_images", 0)
        new_total = current.get("total_images", 0)
        if new_total < old_total:
            raise RuntimeError(
                f"Corpus shrank: total_images went from {old_total} to {new_total}. "
                "Incremental mode only supports growing corpus. "
                "If images were intentionally removed, start with a new output directory."
            )
        fingerprint_changed = existing.get("discovery_fingerprint") != current.get("discovery_fingerprint")
        if fingerprint_changed and not incremental_append_is_verified(existing, current):
            raise RuntimeError(
                "Incremental manifest fingerprint changed without append-only proof. "
                "Refusing to mix a potentially different corpus into existing recap outputs. "
                "Use a new output directory, restore the previous manifest cache, or run an explicit repair."
            )
        for item in incremental_mismatches:
            logger.info("Incremental: manifest field changed — %s", item)

    all_changed = mismatches + incremental_mismatches
    if all_changed:
        if has_outputs:
            logger.info("Updating manifest at %s for incremental run", manifest_path)
        else:
            logger.warning("Manifest updated at %s (no existing recap outputs to protect)", manifest_path)
        write_json_file(manifest_path, current)


def incremental_append_is_verified(existing: dict[str, Any], current: dict[str, Any]) -> bool:
    old_total = int(existing.get("total_images") or 0)
    new_total = int(current.get("total_images") or 0)
    if new_total <= old_total:
        return False
    input_manifest_path = current.get("input_manifest_path")
    if not isinstance(input_manifest_path, str):
        return False
    manifest_path = Path(input_manifest_path)

    old_offset = existing.get("input_manifest_offset")
    old_tail_hash = existing.get("input_manifest_last_line_sha256")
    if old_offset is not None and old_tail_hash:
        if existing.get("input_manifest_path") != current.get("input_manifest_path"):
            return False
        try:
            old_offset_int = int(old_offset)
            current_offset = int(current.get("input_manifest_offset") or 0)
        except (TypeError, ValueError):
            return False
        if current_offset <= old_offset_int:
            return False
        return _manifest_line_before_offset_sha256(manifest_path, old_offset_int) == str(old_tail_hash)

    legacy_base = current.get("_append_validation_base")
    if not isinstance(legacy_base, dict):
        return False
    try:
        base_count = int(legacy_base.get("image_count") or 0)
        base_offset = int(legacy_base.get("manifest_offset") or 0)
    except (TypeError, ValueError):
        return False
    if base_count != old_total:
        return False
    if legacy_base.get("discovery_fingerprint") != existing.get("discovery_fingerprint"):
        return False
    if not legacy_base.get("last_line_sha256"):
        return False
    try:
        current_offset = int(current.get("input_manifest_offset") or 0)
    except (TypeError, ValueError):
        return False
    if current_offset <= base_offset:
        return False
    return _manifest_line_before_offset_sha256(manifest_path, base_offset) == str(legacy_base["last_line_sha256"])


def read_done_file(done_path: Path) -> set[str]:
    if not done_path.exists():
        return set()

    ids: set[str] = set()
    duplicates = 0
    for raw_line in done_path.read_text(encoding="utf-8").splitlines():
        image_id = raw_line.strip()
        if not image_id:
            continue
        if image_id in ids:
            duplicates += 1
            continue
        ids.add(image_id)
    if duplicates:
        logger.warning("%s contains %d duplicate IDs; deduplicating in memory", done_path, duplicates)
    return ids


def sync_done_file(done_path: Path, ordered_ids: list[str]) -> None:
    done_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = done_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for image_id in ordered_ids:
            handle.write(f"{image_id}\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(done_path)


def bootstrap_done_from_jsonl(
    *,
    jsonl_path: Path,
    shard_image_ids: set[str],
    done_path: Path,
    incremental: bool = False,
) -> set[str]:
    completed: set[str] = set()
    ordered_ids: list[str] = []
    duplicate_ids = 0
    foreign_ids = 0

    if not jsonl_path.exists():
        sync_done_file(done_path, [])
        return completed

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL in {jsonl_path}:{line_no}: {exc}") from exc
            image_id = str(record.get("image_id", "")).strip()
            if not image_id:
                raise RuntimeError(f"Missing image_id in {jsonl_path}:{line_no}")
            if image_id not in shard_image_ids:
                if incremental:
                    foreign_ids += 1
                    if image_id in completed:
                        duplicate_ids += 1
                        continue
                    completed.add(image_id)
                    ordered_ids.append(image_id)
                    continue
                raise RuntimeError(
                    f"{jsonl_path}:{line_no} contains image_id={image_id!r} outside the current shard. "
                    "Resume geometry changed; refusing unsafe replay."
                )
            if image_id in completed:
                duplicate_ids += 1
                continue
            completed.add(image_id)
            ordered_ids.append(image_id)

    if foreign_ids:
        logger.info("  %s: bootstrap skipped %d foreign IDs (incremental — corpus still growing)", jsonl_path.name, foreign_ids)
    if duplicate_ids:
        logger.warning("%s contains %d duplicate image_id rows; done state was deduplicated", jsonl_path, duplicate_ids)
    sync_done_file(done_path, ordered_ids)
    return completed


def load_resume_state(
    *,
    shard: list[RecapImage],
    ddn_jsonl: Path,
    ddn_done: Path,
    ddn_offset: Path,
    ddn_state: Path | None = None,
    incremental: bool = False,
) -> ResumeState:
    if ddn_state is None:
        ddn_state = ddn_jsonl.with_suffix(ddn_jsonl.suffix + ".state.json")
    shard_ids = {image.image_id for image in shard}
    if len(shard_ids) != len(shard):
        raise RuntimeError(f"Duplicate image IDs detected in current shard for {ddn_jsonl.name}")

    done_ids = read_done_file(ddn_done)
    offset = read_offset_file(ddn_offset)
    shard_state = read_shard_state(ddn_state)
    jsonl_size = ddn_jsonl.stat().st_size if ddn_jsonl.exists() else 0
    state_jsonl_size = int(shard_state.get("jsonl_size", -1))
    state_done_count = int(shard_state.get("done_count", -1))
    state_mismatch = (
        ddn_jsonl.exists()
        and jsonl_size > 0
        and (
            not shard_state
            or state_jsonl_size != jsonl_size
            or state_done_count != len(done_ids)
        )
    )
    needs_bootstrap = (
        ddn_jsonl.exists()
        and (
            not ddn_done.exists()
            or (not done_ids and ddn_jsonl.stat().st_size > 0)
            or offset > len(done_ids)
            or state_mismatch
        )
    )

    if needs_bootstrap:
        if state_mismatch:
            logger.info(
                "  %s: shard state mismatch, reconciling done from JSONL "
                "(state_jsonl=%s current_jsonl=%d state_done=%s current_done=%d)",
                ddn_jsonl.name,
                state_jsonl_size,
                jsonl_size,
                state_done_count,
                len(done_ids),
            )
        completed_ids = bootstrap_done_from_jsonl(
            jsonl_path=ddn_jsonl,
            shard_image_ids=shard_ids,
            done_path=ddn_done,
            incremental=incremental,
        )
        write_offset_file(ddn_offset, len(completed_ids))
        write_shard_state(
            ddn_state,
            done_count=len(completed_ids),
            jsonl_size=ddn_jsonl.stat().st_size if ddn_jsonl.exists() else 0,
            done_path=ddn_done,
        )
        return ResumeState(
            completed_ids=completed_ids,
            skipped=len(completed_ids),
            bootstrapped_from_jsonl=ddn_jsonl.exists(),
        )

    foreign_ids = done_ids - shard_ids
    if foreign_ids:
        if incremental:
            logger.info(
                "  %s: %d done IDs not in current corpus (incremental — corpus still growing), preserving in done file",
                ddn_done.name, len(foreign_ids),
            )
        else:
            sample = ", ".join(sorted(foreign_ids)[:5])
            raise RuntimeError(
                f"{ddn_done} contains IDs outside the current shard ({sample}). "
                "Resume geometry changed or the done file is corrupted."
            )

    ordered_ids = [image.image_id for image in shard if image.image_id in done_ids]
    if incremental and foreign_ids:
        # In incremental mode, write the union (current shard ordering + foreign tail)
        # so future runs see the same done set without re-bootstrap.
        all_ids = ordered_ids + sorted(foreign_ids)
        if ddn_done.exists():
            sync_done_file(ddn_done, all_ids)
        write_offset_file(ddn_offset, len(ordered_ids))
        write_shard_state(ddn_state, done_count=len(all_ids), jsonl_size=jsonl_size, done_path=ddn_done)
        return ResumeState(completed_ids=done_ids, skipped=len(ordered_ids))

    if ddn_done.exists():
        sync_done_file(ddn_done, ordered_ids)
    write_offset_file(ddn_offset, len(ordered_ids))
    write_shard_state(ddn_state, done_count=len(ordered_ids), jsonl_size=jsonl_size, done_path=ddn_done)
    return ResumeState(completed_ids=done_ids, skipped=len(ordered_ids))


class DDNAppender:
    """Append-only DDN writer that flushes JSONL and done IDs together."""

    def __init__(
        self,
        jsonl_path: Path,
        done_path: Path,
        offset_path: Path,
        state_path: Path,
        *,
        start_done: int,
        flush_batch_size: int,
        flush_interval_sec: float,
        manifest_cache_db: Path | None = None,
        shard_id: int | None = None,
    ):
        self._jsonl_path = jsonl_path
        self._done_path = done_path
        self._offset_path = offset_path
        self._state_path = state_path
        self._flush_batch_size = flush_batch_size
        self._flush_interval_sec = flush_interval_sec
        self._manifest_cache_db = manifest_cache_db
        self._shard_id = shard_id
        self._done_count = start_done
        self._jsonl_buffer: list[str] = []
        self._done_buffer: list[str] = []
        self._done_cache_buffer: list[tuple[str, int, int]] = []
        self._last_flush = time.perf_counter()
        self._lock = asyncio.Lock()

    @property
    def done_count(self) -> int:
        return self._done_count

    async def append(self, record: dict[str, Any], *, global_index: int | None = None) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        image_id = str(record["image_id"])
        async with self._lock:
            self._jsonl_buffer.append(line)
            self._done_buffer.append(image_id)
            if self._manifest_cache_db is not None and self._shard_id is not None and global_index is not None:
                self._done_cache_buffer.append((image_id, self._shard_id, global_index))
            now = time.perf_counter()
            if (
                len(self._jsonl_buffer) >= self._flush_batch_size
                or now - self._last_flush >= self._flush_interval_sec
            ):
                await self._flush_locked()

    async def flush(self) -> None:
        async with self._lock:
            if self._jsonl_buffer:
                await self._flush_locked()

    async def _flush_locked(self) -> None:
        if not self._jsonl_buffer:
            return
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._done_path.parent.mkdir(parents=True, exist_ok=True)
        with self._jsonl_path.open("a", encoding="utf-8") as f:
            for line in self._jsonl_buffer:
                f.write(line)
            f.flush()
            os.fsync(f.fileno())
        with self._done_path.open("a", encoding="utf-8") as f:
            for image_id in self._done_buffer:
                f.write(f"{image_id}\n")
            f.flush()
            os.fsync(f.fileno())
        if self._manifest_cache_db is not None and self._done_cache_buffer:
            try:
                conn = _connect_manifest_cache(self._manifest_cache_db)
                try:
                    _insert_manifest_cache_done_rows(conn, self._done_cache_buffer)
                    _set_manifest_cache_meta(
                        conn,
                        _done_mirror_meta_payload(
                            self._shard_id or 0,
                            self._done_path,
                            self._done_count + len(self._done_buffer),
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                logger.warning(
                    "Manifest-cache done mirror update failed for %s: %s; DDN .done remains authoritative",
                    self._done_path.name,
                    exc,
                )
        self._done_count += len(self._done_buffer)
        self._jsonl_buffer.clear()
        self._done_buffer.clear()
        self._done_cache_buffer.clear()
        write_offset_file(self._offset_path, self._done_count)
        write_shard_state(
            self._state_path,
            done_count=self._done_count,
            jsonl_size=self._jsonl_path.stat().st_size,
            done_path=self._done_path,
        )
        self._last_flush = time.perf_counter()


class FailureAppender:
    """Append-only per-shard failure log for retry selection.

    Failure rows are intentionally not part of durable done state. A later
    selector must cross-check `.done` before retrying because the same image may
    have succeeded after an earlier transient failure.
    """

    def __init__(
        self,
        path: Path,
        *,
        flush_batch_size: int,
        flush_interval_sec: float,
    ) -> None:
        self._path = path
        self._flush_batch_size = flush_batch_size
        self._flush_interval_sec = flush_interval_sec
        self._buffer: list[str] = []
        self._last_flush = time.perf_counter()
        self._lock = asyncio.Lock()

    async def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        async with self._lock:
            self._buffer.append(line)
            now = time.perf_counter()
            if (
                len(self._buffer) >= self._flush_batch_size
                or now - self._last_flush >= self._flush_interval_sec
            ):
                await self._flush_locked()

    async def flush(self) -> None:
        async with self._lock:
            if self._buffer:
                await self._flush_locked()

    async def _flush_locked(self) -> None:
        if not self._buffer:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            for line in self._buffer:
                handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        self._buffer.clear()
        self._last_flush = time.perf_counter()


def build_failure_record(
    image: RecapImage,
    *,
    dataset: str,
    shard_id: int,
    consumer_id: int,
    error: BaseException,
) -> dict[str, Any]:
    message = str(error)
    if len(message) > 4000:
        message = message[:4000] + "...[truncated]"
    record: dict[str, Any] = {
        "image_id": image.image_id,
        "source": dataset,
        "shard": shard_id,
        "consumer": consumer_id,
        "global_index": image.global_index,
        "image_path": image.image_path,
        "source_kind": image.source_kind,
        "error_type": type(error).__name__,
        "error_message": message,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "retryable": True,
    }
    if image.url is not None:
        record["url"] = image.url
    if image.tar_path is not None:
        record["tar_path"] = str(image.tar_path)
    if image.tar_member is not None:
        record["tar_member"] = image.tar_member
    if image.parquet_path is not None:
        record["parquet_path"] = str(image.parquet_path)
        record["parquet_row_group"] = image.parquet_row_group
        record["parquet_row_offset"] = image.parquet_row_offset
    return record


def _producer_thread(
    loop: asyncio.AbstractEventLoop,
    wid: int,
    images: Any,
    q: asyncio.Queue[EncodePayload | None],
    stop: asyncio.Event,
    num_consumers: int,
    use_file_urls: bool = False,
    parquet_image_col: str | None = None,
    parquet_id_col: str | None = None,
) -> None:
    """Encode images (or resolve file:// URLs) and feed the async query queue."""
    tar_cache = TarCache() if not use_file_urls else None
    has_parquets = False if use_file_urls else any(img.source_kind == "parquet" for img in images[:10])
    parquet_cache = (
        ParquetCache(image_col=parquet_image_col or "jpg", id_col=parquet_id_col or "photoid")
        if has_parquets
        else None
    )
    try:
        for image in images:
            if stop.is_set():
                break
            try:
                if use_file_urls and image.source_kind != "parquet":
                    # Canonicalized dir: just send file:// URL, no encode
                    if image.image_file is None:
                        raise ValueError(f"file-url mode requires flat dir input, not tar: {image.image_id}")
                    # Bound file:// payloads with stat() only; resize-capable
                    # paths handle high-quality images before request encoding.
                    fsize = image.image_file.stat().st_size
                    if fsize > MAX_FILE_URL_BYTES:
                        raise ValueError(f"skip large file: {fsize}B")
                    url = f"file://{image.image_file.resolve()}"
                    payload: EncodePayload = (image, url, None)
                else:
                    payload = (image, encode_image(image, tar_cache, parquet_cache), None)
            except Exception as e:
                logger.warning("[w%d] encode failed for %s: %s", wid, image.image_id, e)
                payload = (image, None, str(e))
            fut = asyncio.run_coroutine_threadsafe(q.put(payload), loop)
            deadline = time.monotonic() + PRODUCER_PUT_TIMEOUT
            while True:
                try:
                    fut.result(timeout=PRODUCER_PUT_POLL_SEC)
                    break
                except TimeoutError:
                    if stop.is_set():
                        fut.cancel()
                        return
                    if time.monotonic() >= deadline:
                        logger.warning(
                            "[w%d] producer put timeout (queue full %ds) — backpressure, waiting to enqueue %s",
                            wid, PRODUCER_PUT_TIMEOUT, image.image_id,
                        )
                        deadline = time.monotonic() + PRODUCER_PUT_TIMEOUT
    finally:
        if tar_cache:
            tar_cache.close()
        if parquet_cache:
            parquet_cache.close()
        if stop.is_set():
            return
        for _ in range(num_consumers):
            try:
                asyncio.run_coroutine_threadsafe(q.put(None), loop).result(timeout=10)
            except (TimeoutError, Exception):
                pass


# ── Health check ─────────────────────────────────────────

async def check_endpoints(host: str, ports: tuple[int, ...]) -> list[str]:
    urls = [f"{host}:{p}" for p in ports]
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        results = await asyncio.gather(
            *[_check_one(s, u) for u in urls], return_exceptions=True
        )
    live = []
    for url, ok in zip(urls, results):
        if isinstance(ok, str):
            logger.info("  %s → %s", url, ok)
            live.append(f"{url}/v1/chat/completions")
        else:
            logger.warning("  %s → DEAD (%s)", url, ok)
    if not live:
        raise RuntimeError("No healthy vLLM endpoints")
    return live


async def _check_one(s: aiohttp.ClientSession, url: str) -> str:
    async with s.get(f"{url}/v1/models") as r:
        data = await r.json()
        return data["data"][0]["id"]


def endpoint_port(ep: str) -> int:
    base = ep.rsplit("/v1/", 1)[0] if "/v1/" in ep else ep
    return int(base.rsplit(":", 1)[-1])


def endpoint_gpu(ep: str) -> int:
    return endpoint_port(ep) - VLLM_PORTS[0]


def build_endpoint_restart_command(ep: str, template: str | None = None) -> list[str]:
    gpu = endpoint_gpu(ep)
    port = endpoint_port(ep)
    if template:
        return shlex.split(template.format(endpoint=ep, gpu=gpu, port=port, script=DEFAULT_SERVE_MULTI_SCRIPT))
    return ["/bin/bash", str(DEFAULT_SERVE_MULTI_SCRIPT), "restart-one", str(gpu)]


async def restart_endpoint(
    router: EndpointRouter,
    ep: str,
    *,
    command_template: str | None,
    timeout_sec: int,
) -> None:
    cmd = build_endpoint_restart_command(ep, command_template)
    logger.warning("Restarting stalled endpoint %s via %s", ep, shlex.join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        router.finish_restart(ep, recovered=False)
        logger.error("Endpoint %s restart timed out after %ds", ep, timeout_sec)
        return

    output = stdout.decode("utf-8", errors="replace").strip()
    if proc.returncode == 0:
        router.finish_restart(ep, recovered=True)
        return

    router.finish_restart(ep, recovered=False)
    tail = "\n".join(output.splitlines()[-10:]) if output else "(no output)"
    logger.error("Endpoint %s restart failed with rc=%d\n%s", ep, proc.returncode, tail)


# ── vLLM query with retry + circuit breaker ──────────────

class ShutdownRequested(Exception):
    """Raised when stop signal is detected during query."""


async def _sleep_with_stop(delay: float, stop: asyncio.Event | None) -> None:
    if delay <= 0:
        return
    try:
        if stop is not None:
            await asyncio.wait_for(stop.wait(), timeout=delay)
        else:
            await asyncio.sleep(delay)
    except asyncio.TimeoutError:
        return
    if stop is not None and stop.is_set():
        raise ShutdownRequested("shutdown during backoff wait")


def _jittered_delay(base_delay: float, *, jitter_sec: float) -> float:
    return base_delay + random.uniform(0.0, jitter_sec)


def _http_connection_limit_per_endpoint(args: argparse.Namespace, max_inflight_per_ep: int) -> int:
    configured = int(getattr(args, "http_connection_limit_per_endpoint", 0) or 0)
    if configured > 0:
        return configured
    headroom = int(getattr(args, "http_connection_headroom", DEFAULT_HTTP_CONNECTION_HEADROOM))
    return max(1, max_inflight_per_ep + max(0, headroom))


def _parse_vllm_ports(value: str | None, *, num_workers: int) -> tuple[int, ...]:
    if not value:
        return VLLM_PORTS[:num_workers]
    ports: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        port = int(raw)
        if port <= 0 or port > 65535:
            raise ValueError(f"invalid port: {port}")
        ports.append(port)
    if not ports:
        raise ValueError("--vllm-ports did not contain any ports")
    return tuple(ports)


async def _wait_for_routable_endpoint(
    router: EndpointRouter,
    *,
    stop: asyncio.Event | None,
) -> str:
    while True:
        if stop is not None and stop.is_set():
            raise ShutdownRequested("shutdown before endpoint acquire")
        ep = router.get_endpoint()
        if ep is not None:
            return ep
        await _sleep_with_stop(
            _jittered_delay(router.route_available_delay(), jitter_sec=ENDPOINT_WAIT_JITTER_SEC),
            stop,
        )


async def _sleep_before_retry(attempt: int, stop: asyncio.Event | None) -> None:
    await _sleep_with_stop(
        _jittered_delay(RETRY_DELAY * (attempt + 1), jitter_sec=RETRY_JITTER_SEC),
        stop,
    )


def _exception_detail(exc: BaseException) -> str:
    parts = [type(exc).__name__, repr(exc)]
    os_error = getattr(exc, "os_error", None)
    if os_error is not None:
        parts.append(f"os_error={os_error!r}")
        errno = getattr(os_error, "errno", None)
        strerror = getattr(os_error, "strerror", None)
        if errno is not None:
            parts.append(f"errno={errno}")
        if strerror:
            parts.append(f"strerror={strerror!r}")
    return " ".join(parts)


async def query_caption(
    session: aiohttp.ClientSession,
    router: EndpointRouter,
    image_id: str,
    *,
    model: str,
    sys_prompt: str,
    user_prompt: str,
    max_tokens: int,
    data_url: str,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    sock_connect_timeout: float = DEFAULT_SOCK_CONNECT_TIMEOUT,
    stop: asyncio.Event | None = None,
) -> tuple[str, dict, dict[str, object]]:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req_timeout = aiohttp.ClientTimeout(
        total=request_timeout + 10,
        sock_read=request_timeout,
        connect=connect_timeout,
        sock_connect=sock_connect_timeout,
    )
    last_err = None
    last_trace: dict[str, object] | None = None
    for attempt in range(MAX_RETRIES):
        if stop is not None and stop.is_set():
            raise ShutdownRequested(f"shutdown before attempt {attempt + 1}")
        ep = await _wait_for_routable_endpoint(router, stop=stop)
        router.acquire(ep)
        started = time.perf_counter()
        try:
            async with asyncio.timeout(request_timeout):
                async with session.post(
                    ep, json=payload,
                    headers={"Authorization": "Bearer sk-fake"},
                    timeout=req_timeout,
                ) as resp:
                    body = await resp.json()
                    if resp.status >= 400:
                        raise RuntimeError(f"HTTP {resp.status}: {str(body)[:200]}")
            elapsed = time.perf_counter() - started
            content = body["choices"][0]["message"].get("content")
            if not isinstance(content, str):
                raise RuntimeError(f"non-string response content: {str(body)[:200]}")
            text = content.strip()
            router.report_success(ep)
            trace = {
                "endpoint": ep,
                "attempt": attempt + 1,
                "latency_sec": round(elapsed, 3),
                **router.endpoint_state(ep),
            }
            if attempt > 0 or elapsed >= SLOW_REQUEST_SEC:
                logger.info(
                    "[%s] request ok via %s attempt=%d latency=%.1fs waiting=%s running=%s local=%s",
                    image_id,
                    ep,
                    attempt + 1,
                    elapsed,
                    trace["remote_waiting"],
                    trace["remote_running"],
                    trace["local_inflight"],
                )
            return text, body.get("usage", {}), trace
        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError,
                aiohttp.ClientConnectionError, aiohttp.ServerDisconnectedError) as e:
            router.report_failure(ep, is_timeout=True)
            last_err = e
            elapsed = time.perf_counter() - started
            last_trace = {
                "endpoint": ep,
                "attempt": attempt + 1,
                "latency_sec": round(elapsed, 3),
                **router.endpoint_state(ep),
            }
            logger.warning(
                "[%s] timeout/disconnect on %s attempt=%d latency=%.1fs waiting=%s running=%s local=%s err=%s",
                image_id,
                ep,
                attempt + 1,
                elapsed,
                last_trace["remote_waiting"],
                last_trace["remote_running"],
                last_trace["local_inflight"],
                _exception_detail(e),
            )
            if attempt < MAX_RETRIES - 1:
                await _sleep_before_retry(attempt, stop)
        except Exception as e:
            router.report_failure(ep)
            last_err = e
            elapsed = time.perf_counter() - started
            last_trace = {
                "endpoint": ep,
                "attempt": attempt + 1,
                "latency_sec": round(elapsed, 3),
                **router.endpoint_state(ep),
            }
            logger.warning(
                "[%s] request error on %s attempt=%d latency=%.1fs waiting=%s running=%s local=%s err=%s",
                image_id,
                ep,
                attempt + 1,
                elapsed,
                last_trace["remote_waiting"],
                last_trace["remote_running"],
                last_trace["local_inflight"],
                _exception_detail(e),
            )
            if attempt < MAX_RETRIES - 1:
                await _sleep_before_retry(attempt, stop)
        finally:
            router.release(ep)
    if last_trace is not None:
        logger.error(
            "[%s] request failed after %d attempts last_ep=%s last_latency=%.1fs waiting=%s running=%s",
            image_id,
            MAX_RETRIES,
            last_trace["endpoint"],
            last_trace["latency_sec"],
            last_trace["remote_waiting"],
            last_trace["remote_running"],
        )
    raise last_err  # type: ignore


# ── Worker ───────────────────────────────────────────────

async def worker(
    wid: int,
    images: Any,
    router: EndpointRouter,
    session: aiohttp.ClientSession,
    *,
    model: str,
    sys_prompt: str,
    user_prompt: str,
    max_tokens: int,
    dataset: str,
    ddn_jsonl: Path,
    ddn_done: Path,
    ddn_offset: Path,
    ddn_state: Path,
    failure_path: Path,
    start_done: int,
    stop: asyncio.Event,
    global_stats: Stats,
    shard_stats: Stats,
    inflight: int,
    queue_size: int,
    flush_batch_size: int,
    flush_interval_sec: float,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    sock_connect_timeout: float = DEFAULT_SOCK_CONNECT_TIMEOUT,
    use_file_urls: bool = False,
    metadata: dict[str, dict] | None = None,
    context_fields: list[str] | None = None,
    domain_cfg: dict | None = None,
    parquet_image_col: str | None = None,
    parquet_id_col: str | None = None,
    manifest_cache_db: Path | None = None,
) -> None:
    if not images:
        return

    q: asyncio.Queue[EncodePayload | None] = asyncio.Queue(maxsize=queue_size)
    loop = asyncio.get_running_loop()
    appender = DDNAppender(
        jsonl_path=ddn_jsonl,
        done_path=ddn_done,
        offset_path=ddn_offset,
        state_path=ddn_state,
        start_done=start_done,
        flush_batch_size=flush_batch_size,
        flush_interval_sec=flush_interval_sec,
        manifest_cache_db=manifest_cache_db,
        shard_id=wid,
    )
    failure_appender = FailureAppender(
        failure_path,
        flush_batch_size=flush_batch_size,
        flush_interval_sec=flush_interval_sec,
    )

    has_parquets = False if use_file_urls else any(img.source_kind == "parquet" for img in images[:10])
    mode = "parquet→base64" if has_parquets else ("file://" if use_file_urls else "base64")
    logger.info("[w%d] starting: %d images, %d consumers, mode=%s", wid, len(images), inflight, mode)
    producer = threading.Thread(
        target=_producer_thread,
        args=(loop, wid, images, q, stop, inflight, use_file_urls),
        kwargs={"parquet_image_col": parquet_image_col, "parquet_id_col": parquet_id_col},
        daemon=True,
    )
    producer.start()

    async def do_one(cid: int) -> None:
        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=QUEUE_GET_TIMEOUT)
            except asyncio.TimeoutError:
                if stop.is_set():
                    return
                continue
            if payload is None:
                q.task_done()
                return
            if stop.is_set():
                q.task_done()
                return
            img, data_url, encode_err = payload
            try:
                if encode_err:
                    raise RuntimeError(encode_err)
                # Inject booru grounding context if metadata provided
                final_prompt = user_prompt
                if metadata and context_fields and domain_cfg:
                    grounding = build_grounding_context(
                        img.image_id, metadata, context_fields, domain_cfg,
                    )
                    if grounding:
                        final_prompt = user_prompt + grounding
                caption, usage, trace = await query_caption(
                    session, router, img.image_id, model=model,
                    sys_prompt=sys_prompt, user_prompt=final_prompt,
                    max_tokens=max_tokens, data_url=data_url,
                    request_timeout=request_timeout,
                    connect_timeout=connect_timeout,
                    sock_connect_timeout=sock_connect_timeout,
                    stop=stop,
                )
                record: dict[str, object] = {
                    "image_id": img.image_id,
                    "source": dataset,
                    "image_path": img.image_path,
                    "caption": normalize_caption_text(caption),
                    "completion_tokens": usage.get("completion_tokens"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "model": model,
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                if img.url is not None:
                    record["url"] = img.url
                await appender.append(record, global_index=img.global_index)
                if trace["attempt"] != 1 or trace["latency_sec"] >= SLOW_REQUEST_SEC:
                    logger.info(
                        "[w%d c%d] traced image=%s endpoint=%s attempt=%s latency=%.1fs waiting=%s running=%s",
                        wid,
                        cid,
                        img.image_id,
                        trace["endpoint"],
                        trace["attempt"],
                        trace["latency_sec"],
                        trace["remote_waiting"],
                        trace["remote_running"],
                    )
                global_stats.success += 1
                shard_stats.success += 1
            except ShutdownRequested:
                break  # Don't count as failed — will resume on next run
            except Exception as e:
                logger.warning("[w%d c%d] %s failed: %s: %r", wid, cid, img.image_id, type(e).__name__, e)
                await failure_appender.append(
                    build_failure_record(
                        img,
                        dataset=dataset,
                        shard_id=wid,
                        consumer_id=cid,
                        error=e,
                    ),
                )
                global_stats.failed += 1
                shard_stats.failed += 1
            finally:
                q.task_done()

    consumers = [asyncio.create_task(do_one(cid)) for cid in range(inflight)]
    await asyncio.gather(*consumers, return_exceptions=True)
    await appender.flush()
    await failure_appender.flush()
    producer.join()
    logger.info(
        "[w%d] done — %d new records, done=%d",
        wid,
        appender.done_count - start_done,
        appender.done_count,
    )


# ── Main orchestrator ────────────────────────────────────

async def run(args: argparse.Namespace) -> int:
    cfg = load_domain_config(args.domain)
    sys_prompt, user_prompt, max_tokens, context_fields = get_prompts(cfg)
    hardcap_path = Path(args.hardcap_path)
    if check_hardcap(hardcap_path, args.hardcap_use_pct):
        raise SystemExit(
            f"ERROR: {hardcap_path} is at {disk_used_pct(hardcap_path):.2f}% "
            f"(hardcap={args.hardcap_use_pct:.2f}%). Refusing to start recap."
        )

    endpoints: list[str] = []
    model = "unknown"
    vllm_ports = _parse_vllm_ports(args.vllm_ports, num_workers=args.num_workers)
    if args.vllm_ports and len(vllm_ports) != args.num_workers:
        logger.info(
            "--vllm-ports selected %d endpoints; overriding --num-workers=%d for routing/inflight partitioning",
            len(vllm_ports),
            args.num_workers,
        )
    if not args.dry_run:
        logger.info("Checking vLLM endpoints...")
        endpoints = await check_endpoints(DEFAULT_VLLM_HOST, vllm_ports)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            async with s.get(f"{DEFAULT_VLLM_HOST}:{vllm_ports[0]}/v1/models") as r:
                model = (await r.json())["data"][0]["id"]
        logger.info("Model: %s", model)

    effective_incremental = getattr(args, "incremental", False) or getattr(args, "rolling_corpus", False)
    if effective_incremental and args.limit:
        raise SystemExit(
            "ERROR: --incremental/--rolling-corpus and --limit cannot be combined. "
            "A limited subset may exclude previously-completed images, breaking resume safety."
        )

    input_dir = Path(args.input_dir)
    input_manifest_path = Path(args.input_manifest) if args.input_manifest else input_dir / "_manifest.jsonl"
    pq_id = getattr(args, "parquet_id_col", None)
    pq_img = getattr(args, "parquet_image_col", None)
    explicit_manifest_cache_requires_manifest(
        manifest_cache_arg=args.manifest_cache_db,
        input_manifest_path=input_manifest_path,
        limit=args.limit,
        parquet_id_col=pq_id,
    )

    # Load only metadata rows that can ground this run's image IDs. Full
    # Danbooru exports are much larger than the monthly crawl manifest, and
    # materializing all rows as Python dicts can stall startup for minutes.
    booru_metadata: dict[str, dict] | None = None
    if args.metadata:
        metadata_image_ids: set[str] | None = None
        if input_manifest_path.exists() and pq_id is None:
            metadata_image_ids = _read_manifest_image_ids(input_manifest_path, limit=args.limit)
            logger.info("Metadata grounding manifest IDs: %d from %s", len(metadata_image_ids), input_manifest_path)
        booru_metadata = load_booru_metadata(Path(args.metadata), image_ids=metadata_image_ids)
        logger.info("Booru grounding: %d metadata entries, context_fields=%s", len(booru_metadata), context_fields)
    elif context_fields:
        logger.info("Context fields configured but no --metadata provided, skipping grounding")

    # Output dirs. The manifest cache defaults beside recap outputs for
    # durability, but high-churn DataComp runs can place it on NVMe because it
    # is reconstructable from the canonicalize manifest.
    ddn_root = Path(args.output_dir) / args.dataset / args.domain
    ddn_root.mkdir(parents=True, exist_ok=True)
    manifest_cache_db = (
        Path(args.manifest_cache_db)
        if args.manifest_cache_db
        else ddn_root / "_resume" / "manifest_cache.sqlite3"
    )
    if args.shard_lane_count < 1:
        raise SystemExit("--shard-lane-count must be >= 1")
    if args.shard_lane_index < 0 or args.shard_lane_index >= args.shard_lane_count:
        raise SystemExit("--shard-lane-index must satisfy 0 <= index < count")
    lane_shards = {
        sid for sid in range(args.num_shards)
        if sid % args.shard_lane_count == args.shard_lane_index
    }
    if not lane_shards:
        raise SystemExit(
            f"Shard lane {args.shard_lane_index}/{args.shard_lane_count} owns no shards "
            f"for num_shards={args.num_shards}"
        )

    # Acquire before discovery/cache sync so only one process can mutate resume
    # surfaces for a dataset/domain at a time.
    run_locks: list[RecapRunLock] = []
    group_lock = RecapRunLock(ddn_root, shared=args.run_lock_shared)
    group_lock.acquire()
    run_locks.append(group_lock)
    logger.info("Run lock acquired: %s shared=%s", ddn_root / "run.lock", args.run_lock_shared)
    if args.run_lock_name != "run.lock":
        lane_lock = RecapRunLock(ddn_root, name=args.run_lock_name)
        lane_lock.acquire()
        run_locks.append(lane_lock)
        logger.info("Run lock acquired: %s", ddn_root / args.run_lock_name)

    def release_run_locks() -> None:
        for lock in reversed(run_locks):
            lock.release()

    use_sharded_manifest_cache = should_use_sharded_manifest_cache(
        input_manifest_path=input_manifest_path,
        manifest_cache_db=manifest_cache_db,
        limit=args.limit,
        parquet_id_col=pq_id,
    )
    append_validation_base = (
        _read_manifest_cache_info(manifest_cache_db)
        if use_sharded_manifest_cache
        else None
    )

    logger.info("Discovering images from %s ...", args.input_dir)
    if use_sharded_manifest_cache:
        if args.manifest_cache_no_sync:
            cache_info = _read_manifest_cache_info(manifest_cache_db)
            if cache_info is None or cache_info.image_count <= 0:
                raise RuntimeError(
                    f"--manifest-cache-no-sync requires an existing manifest cache: {manifest_cache_db}"
                )
            logger.info(
                "Using existing manifest cache without sync | offset=%d size=%d count=%d db=%s",
                cache_info.manifest_offset,
                cache_info.manifest_size,
                cache_info.image_count,
                manifest_cache_db,
            )
        else:
            cache_info = _sync_manifest_cache(
                input_dir,
                input_manifest_path,
                manifest_cache_db,
                num_shards=args.num_shards,
                hardcap_path=hardcap_path,
                hardcap_use_pct=args.hardcap_use_pct,
                max_new_rows=args.manifest_cache_max_sync_rows,
            )
        discovery_fingerprint = cache_info.discovery_fingerprint or _manifest_cache_append_fingerprint(
            input_manifest_path,
            cache_info,
        )
        manifest_payload = build_run_manifest_from_stats(
            args,
            total_images=cache_info.image_count,
            discovery_fingerprint=discovery_fingerprint,
            input_manifest_path=input_manifest_path,
            input_manifest_offset=cache_info.manifest_offset,
            input_manifest_last_line_sha256=cache_info.last_line_sha256,
            input_manifest_size=cache_info.manifest_size,
            append_validation_base=append_validation_base,
        )
        images: list[RecapImage] | None = None
        discovered_total = cache_info.image_count
        logger.info(
            "Discovered %d images from manifest cache | offset=%d size=%d | %s",
            discovered_total,
            cache_info.manifest_offset,
            cache_info.manifest_size,
            _memory_status(),
        )
    else:
        images = discover_images(
            input_dir, limit=args.limit,
            parquet_id_col=pq_id, parquet_image_col=pq_img,
            manifest_cache_db=manifest_cache_db,
            manifest_cache_shards=args.num_shards,
        )
        discovered_total = len(images)
        manifest_payload = build_run_manifest(args, images)
        logger.info("Discovered %d images | %s", discovered_total, _memory_status())

    # One-time legacy reshard
    if args.reshard and has_existing_recap_outputs(ddn_root):
        reshard_legacy_outputs(ddn_root, args.num_shards)
        # Remove stale manifest so it gets rewritten
        (ddn_root / "manifest.json").unlink(missing_ok=True)

    manifest_path = ddn_root / "manifest.json"
    has_outputs = has_existing_recap_outputs(ddn_root)
    if manifest_path.exists() or not has_outputs:
        validate_or_write_manifest(
            manifest_path,
            manifest_payload,
            has_outputs=has_outputs,
            incremental=effective_incremental,
            rolling_corpus=getattr(args, "rolling_corpus", False),
        )

    # Shard partition — bit-mapped from image_id, independent of discovery order
    num_shards = args.num_shards
    num_w = len(vllm_ports)
    shards: list[Any] = [[] for _ in range(num_shards)]
    global_stats = Stats(discovered=0)
    shard_stats = [Stats() for _ in range(num_shards)]
    logger.info(
        "Shard lane: index=%d count=%d owns=%d/%d shards",
        args.shard_lane_index,
        args.shard_lane_count,
        len(lane_shards),
        num_shards,
    )
    if images is None:
        logger.info("Preparing %d SQLite-backed manifest-cache shards ...", num_shards)
        for sid in range(num_shards):
            if sid not in lane_shards:
                continue
            ddn_jsonl = ddn_root / f"shard-{sid:04d}.jsonl"
            ddn_done = ddn_root / "done" / f"shard-{sid:04d}.done"
            ddn_offset = ddn_root / f"shard-{sid:04d}.offset"
            ddn_state = ddn_root / "_resume" / f"shard-{sid:04d}.state.json"
            state, shard_total, shard_pending = load_manifest_cache_resume_state(
                cache_db=manifest_cache_db,
                shard_id=sid,
                num_shards=num_shards,
                ddn_jsonl=ddn_jsonl,
                ddn_done=ddn_done,
                ddn_offset=ddn_offset,
                ddn_state=ddn_state,
                trust_done_mirror=args.manifest_cache_trust_done_mirror,
            )
            shard_stats[sid].discovered = shard_total
            shard_stats[sid].skipped = state.skipped
            global_stats.discovered += shard_total
            global_stats.skipped += state.skipped
            if state.bootstrapped_from_jsonl:
                logger.info("  shard-%04d: bootstrapped done file from legacy JSONL (%d current IDs)", sid, state.skipped)
            elif state.skipped > 0:
                logger.info("  shard-%04d: resume from %d completed current image IDs", sid, state.skipped)
            shards[sid] = ManifestCachePendingShard(
                input_dir=input_dir,
                cache_db=manifest_cache_db,
                shard_id=sid,
                completed_ids=state.completed_ids,
                pending_count=shard_pending,
            )
        logger.info("Manifest-cache shard resume ready (%d shards) | %s", num_shards, _memory_status())
    else:
        for img in images:
            sid = image_id_to_shard(img.image_id, num_shards)
            if sid in lane_shards:
                shards[sid].append(img)
        del images
        logger.info("Shard partition ready (%d shards) | %s", num_shards, _memory_status())
        # Resume: check DDN checkpoints
        for sid, shard in enumerate(shards):
            if sid not in lane_shards:
                continue
            ddn_jsonl = ddn_root / f"shard-{sid:04d}.jsonl"
            ddn_done = ddn_root / "done" / f"shard-{sid:04d}.done"
            ddn_offset = ddn_root / f"shard-{sid:04d}.offset"
            ddn_state = ddn_root / "_resume" / f"shard-{sid:04d}.state.json"
            state = load_resume_state(
                shard=shard,
                ddn_jsonl=ddn_jsonl,
                ddn_done=ddn_done,
                ddn_offset=ddn_offset,
                ddn_state=ddn_state,
                incremental=effective_incremental,
            )
            shard_stats[sid].discovered = len(shard)
            shard_stats[sid].skipped = state.skipped
            global_stats.discovered += len(shard)
            global_stats.skipped += state.skipped
            if state.bootstrapped_from_jsonl:
                logger.info("  shard-%04d: bootstrapped done file from legacy JSONL (%d IDs)", sid, state.skipped)
            elif state.skipped > 0:
                logger.info("  shard-%04d: resume from %d completed image IDs", sid, state.skipped)
            shards[sid] = [image for image in shard if image.image_id not in state.completed_ids]

    # Scale per-shard inflight so total ≈ inflight_per_worker × num_workers
    inflight_per_shard = max(8, args.inflight_per_worker * num_w // max(1, len(lane_shards)))

    pending = global_stats.discovered - global_stats.skipped
    logger.info("Resume: skipped=%d pending=%d", global_stats.skipped, pending)

    if not manifest_path.exists():
        write_json_file(manifest_path, manifest_payload)

    if args.dry_run:
        for sid, st in enumerate(shard_stats):
            logger.info(
                "  shard-%04d: total=%d skipped=%d pending=%d",
                sid, st.discovered, st.skipped, st.discovered - st.skipped,
            )
        logger.info("Dry run complete.")
        release_run_locks()
        logger.info("Run lock released")
        return 0

    if pending == 0:
        logger.info("Nothing to do — all images already processed.")
        release_run_locks()
        logger.info("Run lock released")
        return 0

    # Signal handling — two-phase shutdown
    stop = asyncio.Event()
    _signal_count = 0
    stop_reason: str | None = None

    def _stop():
        nonlocal _signal_count, stop_reason
        _signal_count += 1
        if _signal_count == 1:
            stop_reason = stop_reason or "signal"
            logger.warning("Received stop signal — draining inflight (max %ds), send again to force abort",
                           SHUTDOWN_DRAIN_TIMEOUT)
            stop.set()
        else:
            logger.warning("Second signal — force aborting all tasks")
            for t in asyncio.all_tasks():
                if t is not asyncio.current_task():
                    t.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass

    # Progress reporter
    t0 = time.perf_counter()

    async def reporter():
        nonlocal stop_reason
        while not stop.is_set():
            await asyncio.sleep(30)
            used_pct = disk_used_pct(hardcap_path)
            if check_hardcap(hardcap_path, args.hardcap_use_pct):
                stop_reason = "hardcap"
                logger.error(
                    "Hardcap reached: %s use=%.2f%% hardcap=%.2f%%; stopping recap",
                    hardcap_path,
                    used_pct,
                    args.hardcap_use_pct,
                )
                stop.set()
                return
            done = sum(st.success + st.failed for st in shard_stats) + global_stats.skipped
            total = sum(st.discovered for st in shard_stats)
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta_h = (total - done) / rate / 3600 if rate > 0 else float("inf")
            logger.info(
                "Progress: %d/%d (%.1f%%) | ok=%d fail=%d | %.1f img/s | ETA %.1fh | %s %.2f%% | %s",
                done, total, done / total * 100 if total else 0,
                global_stats.success, global_stats.failed, rate, eta_h,
                hardcap_path, used_pct, _memory_status(),
            )
            for sid, st in enumerate(shard_stats):
                done_shard = st.success + st.failed + st.skipped
                logger.info(
                    "  shard-%04d: done=%d/%d ok=%d fail=%d",
                    sid, done_shard, st.discovered,
                    st.success, st.failed,
                )

    timeout = aiohttp.ClientTimeout(total=args.request_timeout + 30)
    request_timeout = args.request_timeout

    # Health-aware endpoint router with circuit breaker
    max_inflight_per_ep = args.inflight_per_worker
    router = EndpointRouter(
        endpoints,
        max_inflight_per_ep=max_inflight_per_ep,
        overload_waiting=args.endpoint_overload_waiting,
        overload_running=args.endpoint_overload_running,
        overload_kv_cache=args.endpoint_overload_kv_cache,
        overload_cooldown=args.endpoint_overload_cooldown,
        stall_timeout_sec=args.endpoint_stall_sec,
        stall_min_running=args.endpoint_stall_min_running,
        soft_failure_threshold=args.endpoint_soft_failures,
        capacity_backoff_ratio=args.endpoint_capacity_backoff,
        capacity_min=args.endpoint_capacity_min,
        capacity_recovery_sec=args.endpoint_capacity_recovery_sec,
        capacity_recovery_jitter_sec=args.endpoint_capacity_recovery_jitter_sec,
        capacity_recovery_step=args.endpoint_capacity_recovery_step,
        failure_holdoff_sec=args.endpoint_failure_holdoff_sec,
        failure_holdoff_jitter_sec=args.endpoint_failure_holdoff_jitter_sec,
        admission_interval_sec=args.endpoint_admission_interval_sec,
        admission_jitter_sec=args.endpoint_admission_jitter_sec,
        initial_capacity=args.endpoint_initial_capacity,
        transport_failure_threshold=args.endpoint_transport_failure_threshold,
        transport_backoff_requires_remote_pressure=args.endpoint_transport_backoff_requires_remote_pressure,
        transport_failure_holdoff_sec=args.endpoint_transport_failure_holdoff_sec,
        transport_failure_holdoff_jitter_sec=args.endpoint_transport_failure_holdoff_jitter_sec,
        remote_idle_local_inflight=args.endpoint_remote_idle_local_inflight,
        remote_idle_holdoff_sec=args.endpoint_remote_idle_holdoff_sec,
        remote_idle_grace_sec=args.endpoint_remote_idle_grace_sec,
    )
    logger.info(
        "EndpointRouter: %d endpoints, max_inflight=%d/ep, request_timeout=%ds, "
        "overload_waiting=%d overload_running=%d overload_kv=%.3f overload_cooldown=%ds "
        "soft_failures=%d capacity_backoff=%.2f capacity_min=%d capacity_recovery=%d/%ss+jitter%s "
        "failure_holdoff=%s+jitter%s admission_interval=%s+jitter%s initial_capacity=%d "
        "transport_failures=%d remote_pressure_required=%s holdoff=%s+jitter%s "
        "remote_idle_local=%d holdoff=%ss",
        len(endpoints),
        max_inflight_per_ep,
        request_timeout,
        args.endpoint_overload_waiting,
        args.endpoint_overload_running,
        args.endpoint_overload_kv_cache,
        args.endpoint_overload_cooldown,
        args.endpoint_soft_failures,
        args.endpoint_capacity_backoff,
        args.endpoint_capacity_min,
        args.endpoint_capacity_recovery_step,
        args.endpoint_capacity_recovery_sec,
        args.endpoint_capacity_recovery_jitter_sec,
        args.endpoint_failure_holdoff_sec,
        args.endpoint_failure_holdoff_jitter_sec,
        args.endpoint_admission_interval_sec,
        args.endpoint_admission_jitter_sec,
        args.endpoint_initial_capacity,
        args.endpoint_transport_failure_threshold,
        args.endpoint_transport_backoff_requires_remote_pressure,
        args.endpoint_transport_failure_holdoff_sec,
        args.endpoint_transport_failure_holdoff_jitter_sec,
        args.endpoint_remote_idle_local_inflight,
        args.endpoint_remote_idle_holdoff_sec,
    )
    http_limit_per_ep = _http_connection_limit_per_endpoint(args, max_inflight_per_ep)
    connector_kwargs: dict[str, Any] = {
        "limit": http_limit_per_ep * max(1, len(endpoints)),
        "limit_per_host": http_limit_per_ep,
        "enable_cleanup_closed": True,
    }
    if args.http_force_close:
        connector_kwargs["force_close"] = True
    else:
        connector_kwargs["keepalive_timeout"] = args.http_keepalive_timeout
    connector = aiohttp.TCPConnector(**connector_kwargs)
    logger.info(
        "HTTP connector: limit=%d limit_per_host=%d keepalive=%s force_close=%s",
        connector_kwargs["limit"],
        connector_kwargs["limit_per_host"],
        "disabled" if args.http_force_close else f"{args.http_keepalive_timeout:.1f}s",
        args.http_force_close,
    )

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        await router.probe_endpoints(session)
        for row in router.snapshot():
            logger.info(
                "Endpoint init: %s healthy=%s stalled=%s restarting=%s cap=%d local=%d running=%d waiting=%d kv=%.3f idle=%ss cooldown=%ss overload=%ss",
                row["endpoint"],
                row["healthy"],
                row["stalled"],
                row["restart_pending"],
                row["capacity_limit"],
                row["local_inflight"],
                row["remote_running"],
                row["remote_waiting"],
                row["remote_kv_cache"],
                row["idle_sec"],
                row["cooldown_sec"],
                row["overloaded_sec"],
            )
        report_task = asyncio.create_task(reporter())
        worker_tasks = []
        restart_tasks: dict[str, asyncio.Task[None]] = {}

        # Background health probe for circuit breaker recovery
        async def _health_probe():
            while not stop.is_set():
                await asyncio.sleep(ENDPOINT_PROBE_INTERVAL_SEC)
                await router.probe_endpoints(session)
                for ep, task in list(restart_tasks.items()):
                    if task.done():
                        with contextlib.suppress(Exception):
                            task.result()
                        restart_tasks.pop(ep, None)
                if not args.no_endpoint_auto_restart:
                    for ep in router.consume_restart_requests():
                        if ep in restart_tasks:
                            continue
                        router.begin_restart(ep)
                        restart_tasks[ep] = asyncio.create_task(
                            restart_endpoint(
                                router,
                                ep,
                                command_template=args.endpoint_restart_command,
                                timeout_sec=args.endpoint_restart_timeout,
                            )
                        )
                healthy = router.healthy_count()
                if healthy < len(endpoints):
                    logger.warning("Healthy endpoints: %d/%d", healthy, len(endpoints))
                for row in router.snapshot():
                    if (
                        not row["healthy"]
                        or row["stalled"]
                        or row["restart_pending"]
                        or row["remote_waiting"]
                        or row["overloaded_sec"]
                        or row["cooldown_sec"]
                        or row["capacity_limit"] != max_inflight_per_ep
                    ):
                        logger.warning(
                            "Endpoint probe: %s healthy=%s stalled=%s restarting=%s cap=%d local=%d running=%d waiting=%d kv=%.3f idle=%ss cooldown=%ss overload=%ss",
                            row["endpoint"],
                            row["healthy"],
                            row["stalled"],
                            row["restart_pending"],
                            row["capacity_limit"],
                            row["local_inflight"],
                            row["remote_running"],
                            row["remote_waiting"],
                            row["remote_kv_cache"],
                            row["idle_sec"],
                            row["cooldown_sec"],
                            row["overloaded_sec"],
                        )
        probe_task = asyncio.create_task(_health_probe())

        for sid, shard in enumerate(shards):
            if not shard:
                continue
            worker_tasks.append(asyncio.create_task(worker(
                sid, shard, router, session,
                model=model, sys_prompt=sys_prompt, user_prompt=user_prompt,
                max_tokens=max_tokens, dataset=args.dataset,
                ddn_jsonl=ddn_root / f"shard-{sid:04d}.jsonl",
                ddn_done=ddn_root / "done" / f"shard-{sid:04d}.done",
                ddn_offset=ddn_root / f"shard-{sid:04d}.offset",
                ddn_state=ddn_root / "_resume" / f"shard-{sid:04d}.state.json",
                failure_path=ddn_root / "failures" / f"shard-{sid:04d}.failures.jsonl",
                start_done=shard_stats[sid].skipped,
                stop=stop,
                global_stats=global_stats,
                shard_stats=shard_stats[sid],
                inflight=inflight_per_shard,
                queue_size=args.encode_queue_size,
                flush_batch_size=args.flush_batch_size,
                flush_interval_sec=args.checkpoint_interval_sec,
                request_timeout=request_timeout,
                connect_timeout=args.connect_timeout,
                sock_connect_timeout=args.sock_connect_timeout,
                use_file_urls=args.use_file_urls,
                metadata=booru_metadata,
                context_fields=context_fields if booru_metadata else None,
                domain_cfg=cfg if booru_metadata else None,
                parquet_image_col=pq_img,
                parquet_id_col=pq_id,
                manifest_cache_db=manifest_cache_db if images is None else None,
            )))

        # Wait for workers with bounded drain on shutdown
        try:
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            logger.warning("Tasks cancelled by force abort")
            results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception) and not isinstance(r, (asyncio.CancelledError, ShutdownRequested)):
                logger.error("Worker %d raised: %s", i, r, exc_info=r)
        stop.set()
        report_task.cancel()
        probe_task.cancel()
        for task in restart_tasks.values():
            task.cancel()
        for task in restart_tasks.values():
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Orphan audit — check for stuck requests after shutdown
        try:
            audit_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5))
            for ep in endpoints:
                base = ep.rsplit("/v1/", 1)[0] if "/v1/" in ep else ep
                try:
                    async with audit_session.get(f"{base}/metrics") as r:
                        text = await r.text()
                    running = _parse_prometheus_metric(text, "vllm:num_requests_running")
                    if running > 0:
                        logger.warning("ORPHAN AUDIT: %s has %d stuck requests — restart this vLLM instance", base, running)
                except Exception:
                    pass
            await audit_session.close()
        except Exception:
            pass

    wall = time.perf_counter() - t0
    done = sum(st.success + st.failed + st.skipped for st in shard_stats)
    rate = done / wall if wall > 0 else 0
    logger.info(
        "DONE. discovered=%d skipped=%d success=%d failed=%d | %.1f img/s | %.1fs",
        global_stats.discovered, global_stats.skipped,
        global_stats.success, global_stats.failed, rate, wall,
    )
    for sid, st in enumerate(shard_stats):
        logger.info(
            "  shard-%04d: done=%d ok=%d fail=%d skipped=%d",
            sid, st.success + st.failed + st.skipped, st.success, st.failed, st.skipped,
        )

    release_run_locks()
    logger.info("Run lock released")
    durable_done = global_stats.skipped + global_stats.success
    if stop_reason is not None:
        logger.error(
            "INCOMPLETE: stop_reason=%s discovered=%d durable_done=%d failed=%d",
            stop_reason,
            global_stats.discovered,
            durable_done,
            global_stats.failed,
        )
        return EXIT_HARDCAP if stop_reason == "hardcap" else EXIT_STOP_SIGNAL
    if global_stats.failed:
        logger.error(
            "INCOMPLETE: failed requests are not durable done state; discovered=%d durable_done=%d failed=%d",
            global_stats.discovered,
            durable_done,
            global_stats.failed,
        )
        return 3
    if durable_done < global_stats.discovered:
        logger.error(
            "INCOMPLETE: discovered=%d durable_done=%d",
            global_stats.discovered,
            durable_done,
        )
        return 4
    return 0


# ── CLI ──────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Production recap runner — long captions via 8× vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Sanity run (20K images)
  uv run python scripts/vllm/run_recap.py --dataset cc12m \\
    --input-dir data/cc12m-wds/ --limit 20000

  # Full CC12M
  uv run python scripts/vllm/run_recap.py --dataset cc12m \\
    --input-dir data/cc12m-wds/

  # Dry run (check discovery + resume)
  uv run python scripts/vllm/run_recap.py --dataset cc12m \\
    --input-dir data/cc12m-wds/ --dry-run

  # Resume after crash (same command)
  uv run python scripts/vllm/run_recap.py --dataset cc12m \\
    --input-dir data/cc12m-wds/
""",
    )
    p.add_argument("--dataset", required=True, help="Dataset name (cc12m, danbooru, etc.)")
    p.add_argument("--domain", default="photorealistic", help="Recap domain config")
    p.add_argument("--input-dir", required=True, help="Image dir or WebDataset tar dir")
    p.add_argument(
        "--input-manifest",
        type=str,
        default=None,
        help="Optional canonicalize manifest path. Defaults to <input-dir>/_manifest.jsonl.",
    )
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DDN), help="DDN output root")
    p.add_argument(
        "--manifest-cache-db",
        type=str,
        default=None,
        help=(
            "Optional SQLite manifest cache path. Defaults under output _resume; "
            "NVMe is faster for large growing canonicalize manifests."
        ),
    )
    p.add_argument(
        "--manifest-cache-max-sync-rows",
        type=int,
        default=None,
        help=(
            "Maximum newly appended manifest rows to ingest before starting recap. "
            "Use on live-growing canonicalize surfaces so restart cost is bounded; "
            "later queue repeats ingest more tail rows."
        ),
    )
    p.add_argument(
        "--manifest-cache-trust-done-mirror",
        action="store_true",
        help=(
            "When shard state and manifest-cache done mirror metadata match, "
            "trust the mirrored done count instead of re-counting done/images overlap. "
            "Use only for append-only fixed manifest surfaces."
        ),
    )
    p.add_argument(
        "--manifest-cache-no-sync",
        action="store_true",
        help=(
            "Use the existing manifest cache without ingesting new manifest rows. "
            "Intended for parallel shard lanes after a parent pre-sync."
        ),
    )
    p.add_argument("--num-workers", type=int, default=8, help="Number of vLLM endpoints to use")
    p.add_argument(
        "--vllm-ports",
        type=str,
        default=None,
        help="Comma-separated vLLM ports. Defaults to 8000.. based on --num-workers; use e.g. 8100 for a DP=8 canary server.",
    )
    p.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SHARDS,
                    help="Number of output shards (bit-mapped from image_id, independent of worker count)")
    p.add_argument(
        "--shard-lane-count",
        type=int,
        default=1,
        help="Run only one modulo partition of output shards; use with --shard-lane-index for parallel clients.",
    )
    p.add_argument(
        "--shard-lane-index",
        type=int,
        default=0,
        help="Modulo shard partition index processed by this client when --shard-lane-count > 1.",
    )
    p.add_argument("--inflight-per-worker", type=int, default=500, help="Concurrent reqs per worker")
    p.add_argument(
        "--encode-queue-size",
        type=int,
        default=DEFAULT_ENCODE_QUEUE_SIZE,
        help="Max encoded items queued per worker",
    )
    p.add_argument(
        "--flush-batch-size",
        type=int,
        default=DEFAULT_FLUSH_BATCH_SIZE,
        help="DDN append batch size",
    )
    p.add_argument(
        "--checkpoint-interval-sec",
        type=float,
        default=CHECKPOINT_INTERVAL_SEC,
        help="DDN checkpoint flush interval (seconds)",
    )
    p.add_argument("--metadata", type=str, default=None,
                    help="Path to booru metadata parquet dir (enables tag grounding for the anime_booru domain)")
    p.add_argument("--limit", type=int, default=None, help="Limit total images (for sanity runs)")
    p.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT,
                    help="Per-request timeout in seconds (circuit breaker triggers on consecutive timeouts)")
    p.add_argument(
        "--connect-timeout",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT,
        help="aiohttp pool/connect timeout in seconds",
    )
    p.add_argument(
        "--sock-connect-timeout",
        type=float,
        default=DEFAULT_SOCK_CONNECT_TIMEOUT,
        help="aiohttp socket connect timeout in seconds",
    )
    p.add_argument(
        "--http-connection-limit-per-endpoint",
        type=int,
        default=0,
        help=(
            "aiohttp connector limit_per_host for each vLLM endpoint. "
            "Default: inflight_per_worker + http_connection_headroom."
        ),
    )
    p.add_argument(
        "--http-connection-headroom",
        type=int,
        default=DEFAULT_HTTP_CONNECTION_HEADROOM,
        help="Extra per-endpoint HTTP connections above inflight_per_worker when limit is not explicit",
    )
    p.add_argument(
        "--http-keepalive-timeout",
        type=float,
        default=DEFAULT_HTTP_KEEPALIVE_TIMEOUT,
        help="Idle aiohttp keepalive timeout in seconds",
    )
    p.add_argument(
        "--http-force-close",
        action="store_true",
        help="Disable HTTP keepalive for recap requests",
    )
    p.add_argument(
        "--endpoint-stall-sec",
        type=int,
        default=ENDPOINT_STALL_TIMEOUT_SEC,
        help="Quarantine/restart endpoint when metrics make no progress for this many seconds while requests are running",
    )
    p.add_argument(
        "--endpoint-stall-min-running",
        type=int,
        default=ENDPOINT_STALL_MIN_RUNNING,
        help="Minimum remote running requests before stall detection can quarantine an endpoint",
    )
    p.add_argument(
        "--endpoint-overload-waiting",
        type=int,
        default=ENDPOINT_OVERLOAD_WAITING,
        help=(
            "Pause new traffic to an endpoint only when vLLM waiting requests reach this threshold. "
            "Small nonzero waiting is normal and keeps batching fed."
        ),
    )
    p.add_argument(
        "--endpoint-overload-running",
        type=int,
        default=ENDPOINT_OVERLOAD_RUNNING,
        help="Pause new traffic when vLLM running requests reach this endpoint-local threshold",
    )
    p.add_argument(
        "--endpoint-overload-kv-cache",
        type=float,
        default=ENDPOINT_OVERLOAD_KV_CACHE,
        help="Pause new traffic when KV cache usage reaches this fraction and waiting is nonzero",
    )
    p.add_argument(
        "--endpoint-overload-cooldown",
        type=int,
        default=ENDPOINT_OVERLOAD_COOLDOWN,
        help="Seconds to pause new traffic after endpoint overload is detected",
    )
    p.add_argument(
        "--endpoint-soft-failures",
        type=int,
        default=ENDPOINT_SOFT_FAILURE_THRESHOLD,
        help="Consecutive per-endpoint connection failures before adaptive capacity backoff",
    )
    p.add_argument(
        "--endpoint-capacity-backoff",
        type=float,
        default=ENDPOINT_CAPACITY_BACKOFF_RATIO,
        help="Multiplier applied to endpoint capacity after each soft-failure threshold",
    )
    p.add_argument(
        "--endpoint-capacity-min",
        type=int,
        default=ENDPOINT_CAPACITY_MIN,
        help="Minimum endpoint capacity after adaptive backoff",
    )
    p.add_argument(
        "--endpoint-capacity-recovery-sec",
        type=float,
        default=ENDPOINT_CAPACITY_RECOVERY_SEC,
        help="Minimum seconds between endpoint capacity recovery steps",
    )
    p.add_argument(
        "--endpoint-capacity-recovery-jitter-sec",
        type=float,
        default=ENDPOINT_CAPACITY_RECOVERY_JITTER_SEC,
        help="Additional random seconds before each endpoint capacity recovery step",
    )
    p.add_argument(
        "--endpoint-capacity-recovery-step",
        type=int,
        default=ENDPOINT_CAPACITY_RECOVERY_STEP,
        help="Endpoint capacity increment after each recovery interval with successful requests",
    )
    p.add_argument(
        "--endpoint-failure-holdoff-sec",
        type=float,
        default=ENDPOINT_FAILURE_HOLDOFF_SEC,
        help="Extra per-endpoint routing holdoff after adaptive soft-throttle",
    )
    p.add_argument(
        "--endpoint-failure-holdoff-jitter-sec",
        type=float,
        default=ENDPOINT_FAILURE_HOLDOFF_JITTER_SEC,
        help="Additional random per-endpoint routing holdoff after adaptive soft-throttle",
    )
    p.add_argument(
        "--endpoint-admission-interval-sec",
        type=float,
        default=ENDPOINT_ADMISSION_INTERVAL_SEC,
        help="Minimum per-endpoint spacing between new request admissions",
    )
    p.add_argument(
        "--endpoint-admission-jitter-sec",
        type=float,
        default=ENDPOINT_ADMISSION_JITTER_SEC,
        help="Additional random per-endpoint spacing between new request admissions",
    )
    p.add_argument(
        "--endpoint-initial-capacity",
        type=int,
        default=ENDPOINT_INITIAL_CAPACITY,
        help="Initial per-endpoint routing capacity before ramp-up. 0 means start at inflight_per_worker.",
    )
    p.add_argument(
        "--endpoint-transport-failure-threshold",
        type=int,
        default=ENDPOINT_TRANSPORT_FAILURE_THRESHOLD,
        help="Per-endpoint timeout/write failures before transport-specific soft-throttle",
    )
    p.add_argument(
        "--endpoint-transport-backoff-requires-remote-pressure",
        action=argparse.BooleanOptionalAction,
        default=ENDPOINT_TRANSPORT_BACKOFF_REQUIRES_REMOTE_PRESSURE,
        help=(
            "Only let transport failures reduce endpoint capacity when vLLM metrics "
            "also show remote pressure. Keeps client-side write/connect bursts from "
            "draining otherwise idle servers."
        ),
    )
    p.add_argument(
        "--endpoint-transport-failure-holdoff-sec",
        type=float,
        default=ENDPOINT_TRANSPORT_FAILURE_HOLDOFF_SEC,
        help="Extra routing holdoff after transport-specific soft-throttle",
    )
    p.add_argument(
        "--endpoint-transport-failure-holdoff-jitter-sec",
        type=float,
        default=ENDPOINT_TRANSPORT_FAILURE_HOLDOFF_JITTER_SEC,
        help="Additional random holdoff after transport-specific soft-throttle",
    )
    p.add_argument(
        "--endpoint-remote-idle-local-inflight",
        type=int,
        default=ENDPOINT_REMOTE_IDLE_LOCAL_INFLIGHT,
        help="Pause new traffic when local inflight is at least this high but vLLM reports running=waiting=0. 0 disables.",
    )
    p.add_argument(
        "--endpoint-remote-idle-holdoff-sec",
        type=float,
        default=ENDPOINT_REMOTE_IDLE_HOLDOFF_SEC,
        help="Routing holdoff when local connection pressure exists but remote metrics are idle",
    )
    p.add_argument(
        "--endpoint-remote-idle-grace-sec",
        type=float,
        default=ENDPOINT_REMOTE_IDLE_GRACE_SEC,
        help="Seconds without metric progress before remote-idle local-pressure holdoff can trigger",
    )
    p.add_argument(
        "--endpoint-restart-command",
        type=str,
        default=None,
        help="Optional restart command template. Supports {endpoint}, {gpu}, {port}, {script}",
    )
    p.add_argument(
        "--endpoint-restart-timeout",
        type=int,
        default=DEFAULT_ENDPOINT_RESTART_TIMEOUT,
        help="Timeout in seconds for a single endpoint restart command",
    )
    p.add_argument(
        "--no-endpoint-auto-restart",
        action="store_true",
        help="Disable automatic per-endpoint restart when a stalled endpoint is detected",
    )
    p.add_argument(
        "--run-lock-name",
        default="run.lock",
        help="Per-process exclusive run lock filename under the recap output root.",
    )
    p.add_argument(
        "--run-lock-shared",
        action="store_true",
        help="Acquire dataset run.lock in shared mode before the per-process exclusive lock. Used by disjoint shard lanes.",
    )
    p.add_argument("--dry-run", action="store_true", help="Discovery + resume check only, no inference")
    p.add_argument("--reshard", action="store_true",
                    help="Reshard legacy outputs to match current num_shards (one-time migration)")
    p.add_argument("--incremental", action="store_true",
                    help="Allow growing corpus: relax total_images/fingerprint manifest checks for partial canon runs")
    p.add_argument("--rolling-corpus", action="store_true",
                    help="Allow fixed-key rolling batches whose manifest is replaced between runs; preserves prior done IDs")
    p.add_argument(
        "--use-file-urls",
        action="store_true",
        help=(
            "Send file:// URLs instead of base64 "
            "(requires canonicalized images + --allowed-local-media-path on server)"
        ),
    )
    p.add_argument(
        "--parquet-id-col",
        type=str,
        default=None,
        help="Parquet streaming mode: column name for image ID (e.g. 'photoid'). "
             "Enables reading images directly from HF parquet files without canonicalization.",
    )
    p.add_argument(
        "--parquet-image-col",
        type=str,
        default="jpg",
        help="Parquet streaming mode: column name for image bytes (default: 'jpg')",
    )
    p.add_argument(
        "--hardcap-path",
        type=str,
        default="outputs",
        help="Filesystem path to monitor for local scratch hardcap",
    )
    p.add_argument(
        "--hardcap-use-pct",
        type=float,
        default=85.0,
        help="Stop recap if hardcap-path reaches this used percentage (<=0 disables)",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
