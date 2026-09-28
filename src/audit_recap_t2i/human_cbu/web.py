"""Aiohttp application for the blinded human CBU annotation interface."""

from __future__ import annotations

import asyncio
import hmac
import html
import io
import ipaddress
import json
import math
import mimetypes
import secrets
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from aiohttp import web
from PIL import Image, ImageOps

from .store import (
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    HumanCBUStoreError,
    NotFoundError,
    StudyStateError,
    ValidationError,
)

STATIC_ROOT = Path(__file__).with_name("static")
_CSP = (
    "default-src 'self'; "
    "base-uri 'none'; "
    "connect-src 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'; "
    "img-src 'self' blob:; "
    "object-src 'none'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'"
)
_ERROR_MESSAGES = {
    400: "Invalid request.",
    401: "Authentication failed.",
    403: "Request is not permitted.",
    404: "Resource not found.",
    405: "Method not allowed.",
    409: "The study is not available for this operation.",
    413: "Request is too large.",
    415: "Request content type is not supported.",
    422: "Request validation failed.",
    500: "The study service could not complete the request.",
}
_PARTICIPANT_INFORMATION_FIELDS = (
    "approved_version",
    "duration",
    "compensation",
    "data_retention",
    "research_contact",
    "withdrawal_policy",
    "information_sheet_url",
)
_PARTICIPANT_TASK_DESCRIPTION = (
    "You will judge one short image–text claim at a time. First, use only a "
    "fixed caption window to answer three text questions. Then reveal the "
    "paired image, judge visible support, and rate how well the displayed "
    "caption window covers important visible content."
)
_PARTICIPANT_CONTENT_WARNING = (
    "Public web images may contain unexpected or sensitive material, including "
    "partial nudity in non-sexual or cultural contexts, violence, or medical "
    "imagery. You may skip any image."
)
_PARTICIPANT_VOLUNTARY_NOTICE = (
    "Participation is voluntary. You may pause, close the page, or stop at any "
    "time using the study controls."
)
_HSTS_ENABLED = web.AppKey("human_cbu_hsts_enabled", bool)
PRESENTATION_IMAGE_PIXEL_BUDGET = 768 * 768
PRESENTATION_IMAGE_WEBP_QUALITY = 90


def _json_error(message: str, *, status: int) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _authorization_token(request: web.Request) -> str | None:
    """Read one bearer credential exclusively from the Authorization header."""

    value = request.headers.get("Authorization", "")
    scheme, separator, token = value.partition(" ")
    if (
        separator
        and scheme.casefold() == "bearer"
        and token
        and token == token.strip()
        and not any(character.isspace() for character in token)
    ):
        return token
    return None


def _secure_headers(response: web.StreamResponse, *, hsts_enabled: bool) -> None:
    response.headers["Cache-Control"] = "private, no-store, max-age=0"
    response.headers["Content-Security-Policy"] = _CSP
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = "camera=(), geolocation=(), microphone=()"
    response.headers["Pragma"] = "no-cache"
    response.headers["Referrer-Policy"] = "no-referrer"
    if hsts_enabled:
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"


def _generic_http_error(error: web.HTTPException) -> web.Response:
    status = error.status if 400 <= error.status < 600 else 500
    response = _json_error(_ERROR_MESSAGES.get(status, "Request failed."), status=status)
    if "Allow" in error.headers:
        response.headers["Allow"] = error.headers["Allow"]
    return response


@web.middleware
async def _privacy_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    """Redact operational failures and attach privacy headers to every response."""

    try:
        response = await handler(request)
    except AuthenticationError:
        response = _json_error(_ERROR_MESSAGES[401], status=401)
    except AuthorizationError as error:
        request.app.logger.warning(
            "Human CBU request forbidden method=%s path=%s reason=%s",
            request.method,
            request.path,
            str(error),
        )
        response = _json_error(_ERROR_MESSAGES[403], status=403)
    except NotFoundError:
        response = _json_error(_ERROR_MESSAGES[404], status=404)
    except ValidationError:
        response = _json_error(_ERROR_MESSAGES[422], status=422)
    except (StudyStateError, ConflictError):
        response = _json_error(_ERROR_MESSAGES[409], status=409)
    except HumanCBUStoreError:
        request.app.logger.exception("Human CBU persistence failure")
        response = _json_error(_ERROR_MESSAGES[500], status=500)
    except (json.JSONDecodeError, UnicodeDecodeError):
        response = _json_error(_ERROR_MESSAGES[400], status=400)
    except web.HTTPException as error:
        response = _generic_http_error(error)
    except Exception:
        request.app.logger.exception("Unhandled human CBU web failure")
        response = _json_error(_ERROR_MESSAGES[500], status=500)
    _secure_headers(
        response,
        hsts_enabled=bool(request.app.get(_HSTS_ENABLED, False)),
    )
    return response


def _is_loopback_host(host_header: str | None) -> bool:
    if (
        not host_header
        or host_header != host_header.strip()
        or any(character.isspace() or ord(character) < 32 for character in host_header)
    ):
        return False
    try:
        parsed = urlsplit(f"//{host_header}")
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        return False
    if (
        hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        return False
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


@web.middleware
async def _preview_host_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    if not _is_loopback_host(request.headers.get("Host")):
        raise web.HTTPMisdirectedRequest()
    return await handler(request)


def _application(
    *,
    preview_host_only: bool = False,
    hsts_enabled: bool = False,
) -> web.Application:
    middlewares = [_privacy_middleware]
    if preview_host_only:
        middlewares.append(_preview_host_middleware)
    application = web.Application(middlewares=middlewares, client_max_size=128 * 1024)
    application[_HSTS_ENABLED] = hsts_enabled
    return application


async def _json_object(request: web.Request) -> dict[str, Any]:
    if request.content_type != "application/json":
        raise web.HTTPUnsupportedMediaType()
    payload = await request.json()
    if not isinstance(payload, dict):
        raise ValidationError("request body must be an object")
    return payload


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field} is required")
    return value


def _safe_count(value: Any) -> int:
    if type(value) is not int or value < 0:
        return 0
    return value


def validate_participant_information(information: Mapping[str, Any]) -> dict[str, str]:
    """Validate and normalize one participant notice."""

    if not isinstance(information, Mapping):
        raise TypeError("participant_information must be a mapping")
    unknown = set(information) - set(_PARTICIPANT_INFORMATION_FIELDS)
    missing = set(_PARTICIPANT_INFORMATION_FIELDS) - set(information)
    if missing:
        raise ValueError("participant_information is missing required fields: " + ", ".join(sorted(missing)))
    if unknown:
        raise ValueError("participant_information has unsupported fields: " + ", ".join(sorted(unknown)))

    normalized: dict[str, str] = {}
    for field in _PARTICIPANT_INFORMATION_FIELDS:
        value = information[field]
        maximum = 2_048 if field == "information_sheet_url" else 4_000
        if not isinstance(value, str) or not value.strip() or len(value) > maximum:
            raise ValueError(f"participant_information.{field} must be a non-empty string")
        if any(ord(character) < 32 and character not in "\n\t" for character in value):
            raise ValueError(f"participant_information.{field} contains unsupported control characters")
        normalized[field] = value.strip()

    information_url = normalized["information_sheet_url"]
    if information_url != "/study-information":
        parsed_url = urlsplit(information_url)
        if (
            parsed_url.scheme != "https"
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
        ):
            raise ValueError(
                "participant_information.information_sheet_url must be "
                "'/study-information' or a credential-free HTTPS URL"
            )
    return normalized


def _normalize_progress(
    progress: Mapping[str, Any] | None,
    *,
    phase: str | None = None,
) -> dict[str, int]:
    """Flatten store phase progress into the participant-facing UI contract."""

    progress = progress if isinstance(progress, Mapping) else {}
    caption = progress.get("caption") if isinstance(progress.get("caption"), Mapping) else {}
    image = progress.get("image") if isinstance(progress.get("image"), Mapping) else {}
    caption_completed = _safe_count(caption.get("completed"))
    caption_total = _safe_count(caption.get("total"))
    image_completed = _safe_count(image.get("completed"))
    image_total = _safe_count(image.get("total"))
    overall_completed = caption_completed + image_completed
    overall_total = caption_total + image_total

    current_phase = phase or progress.get("current_phase")
    if current_phase == "caption":
        completed, total = caption_completed, caption_total
    elif current_phase == "image":
        completed, total = image_completed, image_total
    elif caption or image:
        completed, total = overall_completed, overall_total
    else:
        completed = _safe_count(progress.get("completed"))
        total = _safe_count(progress.get("total"))
        overall_completed = _safe_count(progress.get("overall_completed")) or completed
        overall_total = _safe_count(progress.get("overall_total")) or total
    return {
        "completed": completed,
        "total": total,
        "overall_completed": overall_completed,
        "overall_total": overall_total,
    }


def _task_projection(task: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a second participant-facing allow-list over the store projection."""

    assignment_id = task.get("assignment_id")
    phase = task.get("phase")
    if not isinstance(assignment_id, str) or not assignment_id:
        raise HumanCBUStoreError("store returned a task without an assignment ID")
    if not isinstance(phase, str) or not phase:
        raise HumanCBUStoreError("store returned a task without a phase")
    if phase not in {"caption", "image"}:
        raise HumanCBUStoreError("store returned an unsupported task phase")
    projected: dict[str, Any] = {
        "assignment_id": assignment_id,
        "phase": phase,
        "position": _safe_count(task.get("position")),
        "unit": str(task.get("unit") or ""),
        "target": None if task.get("target") is None else str(task["target"]),
        "category": str(task.get("category") or ""),
    }
    if phase == "caption":
        span = task.get("span")
        if span is not None and not isinstance(span, str):
            raise HumanCBUStoreError("store returned an invalid caption span")
        projected.update(
            {
                "caption": str(task.get("caption") or ""),
                "span": span,
            }
        )
    else:
        projected["image_url"] = f"/api/image/{assignment_id}"
    return projected


def _study_result(
    result: Mapping[str, Any],
    *,
    participant_information: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Convert one store status wrapper to the stable browser contract."""

    status = result.get("status")
    if status not in {
        "consent_required",
        "task",
        "waiting_for_assets",
        "waiting_for_peer_labels",
        "complete",
        "paused",
        "closed",
    }:
        raise HumanCBUStoreError("store returned an unsupported task status")
    task: dict[str, Any] | None = None
    phase: str | None = None
    if status == "task":
        raw_task = result.get("task")
        if not isinstance(raw_task, Mapping):
            raise HumanCBUStoreError("store returned a task without a projection")
        task = _task_projection(raw_task)
        phase = task["phase"]
    projected: dict[str, Any] = {
        "status": status,
        "progress": _normalize_progress(result.get("progress"), phase=phase),
    }
    if participant_information is not None:
        projected["consent_version"] = participant_information["approved_version"]
    if phase is not None and task is not None:
        projected["phase"] = phase
        projected["task"] = task
    if status == "consent_required":
        required_version = str(result.get("required_consent_version") or "")
        projected["required_consent_version"] = required_version
        if participant_information is not None:
            if participant_information["approved_version"] != required_version:
                raise StudyStateError("approved participant information does not match the required consent version")
            projected["participant_information"] = dict(participant_information)
    if status == "complete" and result.get("work_extension") is not None:
        extension = result["work_extension"]
        if not isinstance(extension, Mapping):
            raise HumanCBUStoreError("store returned an invalid optional-work status")
        projected["work_extension"] = {
            "batch_size": int(extension.get("batch_size", 0)),
            "max_items": int(extension.get("max_items", 0)),
            "assigned_items": int(extension.get("assigned_items", 0)),
            "can_extend": bool(extension.get("can_extend", False)),
        }
    return projected


def _safe_static_path(name: str) -> Path:
    candidate = (STATIC_ROOT / name).resolve()
    if STATIC_ROOT.resolve() not in candidate.parents:
        raise web.HTTPNotFound()
    return candidate


async def index(_: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_ROOT / "index.html")


def participant_information_document(information: Mapping[str, str]) -> str:
    """Render the sealed, credential-free participant notice."""

    escaped = {key: html.escape(value) for key, value in information.items()}
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Visual Claim Study · Participant notice</title>
  <link rel="stylesheet" href="/static/app.css" />
</head>
<body>
  <main class="login-layout">
    <article class="login-card consent-card participant-notice-page">
      <div class="brand">
        <span class="brand-mark">i</span>
        <span class="brand-copy">
          <strong>Visual Claim Study</strong>
          <span>Participant notice · {escaped["approved_version"]}</span>
        </span>
      </div>
      <h1>What participation involves</h1>
      <p>This study checks how closely automated image–caption quality judgments
      agree with independent human judgments. It is not a test of you.</p>
      <p>{html.escape(_PARTICIPANT_TASK_DESCRIPTION)}</p>
      <p>{html.escape(_PARTICIPANT_CONTENT_WARNING)}</p>
      <p>{html.escape(_PARTICIPANT_VOLUNTARY_NOTICE)}</p>
      <dl class="participant-information-grid">
        <div><dt>Expected duration</dt><dd>{escaped["duration"]}</dd></div>
        <div><dt>Compensation</dt><dd>{escaped["compensation"]}</dd></div>
        <div><dt>Data retention</dt><dd>{escaped["data_retention"]}</dd></div>
        <div><dt>Research contact</dt><dd>{escaped["research_contact"]}</dd></div>
        <div class="withdrawal-policy">
          <dt>Stopping or withdrawing</dt>
          <dd>{escaped["withdrawal_policy"]}</dd>
        </div>
      </dl>
      <p class="storage-note">The study records pseudonymous answers and response
      timing. It does not collect names, demographics, IP addresses, or browser
      identifiers.</p>
      <p><a class="secondary-button" href="/">Return to the study</a></p>
    </article>
  </main>
</body>
</html>
"""


async def static_file(request: web.Request) -> web.FileResponse:
    path = _safe_static_path(request.match_info["name"])
    if not path.is_file():
        raise web.HTTPNotFound()
    response = web.FileResponse(path)
    content_type, _ = mimetypes.guess_type(path.name)
    if content_type:
        response.content_type = content_type
    return response


@lru_cache(maxsize=1_024)
def _render_presentation_image(
    path_text: str,
    source_size: int,
    source_mtime_ns: int,
) -> bytes:
    """Return a metadata-free, aspect-preserving WebP presentation image."""

    del source_size, source_mtime_ns  # Cache-key components validated by the caller.
    with Image.open(path_text) as source:
        source.seek(0)
        oriented = ImageOps.exif_transpose(source)
        oriented.load()
        image = oriented.copy()
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("presentation image has invalid dimensions")
    area = width * height
    if area > PRESENTATION_IMAGE_PIXEL_BUDGET:
        scale = math.sqrt(PRESENTATION_IMAGE_PIXEL_BUDGET / area)
        width = max(1, int(math.floor(width * scale)))
        height = max(1, int(math.floor(height * scale)))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    has_alpha = image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    )
    image = image.convert("RGBA" if has_alpha else "RGB")
    output = io.BytesIO()
    image.save(
        output,
        format="WEBP",
        quality=PRESENTATION_IMAGE_WEBP_QUALITY,
        method=6,
    )
    return output.getvalue()


async def presentation_image_response(path: Path) -> web.Response:
    """Encode an authorized source image for bounded browser presentation."""

    stat_result = path.stat()
    body = await asyncio.to_thread(
        _render_presentation_image,
        str(path),
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )
    return web.Response(body=body, content_type="image/webp")


def preview_task(item: Mapping[str, Any], *, phase: str) -> dict[str, Any]:
    """Return a browser projection with all private study fields removed."""

    common = {
        "assignment_id": str(item.get("assignment_id") or f"preview-{phase}"),
        "phase": phase,
        "position": int(item.get("position") or 1),
        "category": str(item["category"]),
        "unit": str(item["unit"]),
        "target": str(item.get("target") or "scene"),
    }
    if phase == "caption":
        caption = str(item["caption"])
        span = item.get("span")
        if span is None and type(item.get("span_start")) is int and type(item.get("span_end")) is int:
            span = caption[item["span_start"] : item["span_end"]]
        if span is not None and not isinstance(span, str):
            raise ValueError("preview span must be a string or null")
        return {
            **common,
            "caption": caption,
            "span": span,
        }
    if phase == "image":
        return {
            **common,
            "image_url": f"/api/image/{common['assignment_id']}",
        }
    raise ValueError(f"unsupported phase: {phase}")


def create_preview_app(
    *,
    item: Mapping[str, Any],
    image_path: str | Path,
    initial_phase: str = "caption",
) -> web.Application:
    """Create a self-contained app for browser-level visual review."""

    if initial_phase not in {"caption", "image"}:
        raise ValueError("initial_phase must be caption or image")
    image = Path(image_path)
    if not image.is_file():
        raise FileNotFoundError(image)
    app = _application(preview_host_only=True)
    preview_item = dict(item)
    preview_state: dict[str, str | int] = {"phase": initial_phase, "completed": 16}

    async def bootstrap(_: web.Request) -> web.Response:
        phase = str(preview_state["phase"])
        return web.json_response(
            _study_result(
                {
                    "status": "task",
                    "phase": phase,
                    "task": preview_task(preview_item, phase=phase),
                    "progress": {
                        phase: {
                            "completed": int(preview_state["completed"]),
                            "total": 240,
                        },
                        "current_phase": phase,
                    },
                }
            )
        )

    async def login(_: web.Request) -> web.Response:
        return web.json_response({"token": "preview-session"})

    async def annotation(request: web.Request) -> web.Response:
        await _json_object(request)
        phase = str(preview_state["phase"])
        next_phase = "image" if phase == "caption" else "caption"
        preview_state["phase"] = next_phase
        preview_state["completed"] = int(preview_state["completed"]) + 1
        return await bootstrap(request)

    async def consent(request: web.Request) -> web.Response:
        await _json_object(request)
        return await bootstrap(request)

    async def image_response(_: web.Request) -> web.Response:
        return await presentation_image_response(image)

    async def coverage_caption(_: web.Request) -> web.Response:
        return web.json_response({"caption": str(preview_item["caption"])})

    app.router.add_get("/", index)
    app.router.add_get("/static/{name}", static_file)
    app.router.add_get("/api/bootstrap", bootstrap)
    app.router.add_post("/api/login", login)
    app.router.add_post("/api/consent", consent)
    app.router.add_post("/api/annotation", annotation)
    app.router.add_get("/api/image/{assignment_id}", image_response)
    app.router.add_get("/api/coverage-caption/{assignment_id}", coverage_caption)
    return app


def create_self_enrollment_test_app(
    *,
    item: Mapping[str, Any],
    image_path: str | Path,
    pair_count: int = 3,
    additional_items: tuple[Mapping[str, Any], ...] = (),
    additional_image_paths: tuple[str | Path, ...] = (),
) -> web.Application:
    """Exercise the participant enrollment and task cycle without persistence.

    Participant codes, sessions, consent state, and progress live only in this
    process. Annotation bodies are validated enough to advance the interface
    but are never copied, hashed, logged, or written.
    """

    if type(pair_count) is not int or not 1 <= pair_count <= 20:
        raise ValueError("pair_count must be between 1 and 20")
    if len(additional_items) != len(additional_image_paths):
        raise ValueError("each additional test item requires one additional image")
    preview_items = [dict(item), *(dict(candidate) for candidate in additional_items)]
    images = [Path(image_path), *(Path(candidate) for candidate in additional_image_paths)]
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(image)
    codes: dict[str, str] = {}
    sessions: dict[str, str] = {}
    progress: dict[str, dict[str, Any]] = {}
    app = _application(preview_host_only=True)

    def participant_for(request: web.Request) -> str:
        token = _authorization_token(request)
        code = sessions.get(token or "")
        if code is None:
            raise AuthenticationError("test session is missing")
        return code

    def current_result(code: str) -> dict[str, Any]:
        state = progress[code]
        if not state["consented"]:
            return {
                "status": "consent_required",
                "test_mode": True,
                "responses_persisted": False,
                "required_consent_version": "test-only-v1",
            }
        if state["stopped"] or state["image"] >= pair_count:
            return {
                "status": "complete",
                "test_mode": True,
                "responses_persisted": False,
                "progress": {
                    "completed": state["caption"] + state["image"],
                    "overall_completed": state["caption"] + state["image"],
                },
            }
        phase = "caption" if state["caption"] == state["image"] else "image"
        index = int(state[phase])
        preview_item = preview_items[index % len(preview_items)]
        task = preview_task(preview_item, phase=phase)
        task["assignment_id"] = f"participant-test-{phase}-{index:04d}"
        task["position"] = index + 1
        return {
            **_study_result(
                {
                    "status": "task",
                    "phase": phase,
                    "task": task,
                    "progress": {
                        phase: {"completed": index, "total": pair_count},
                        "current_phase": phase,
                    },
                }
            ),
            "test_mode": True,
            "responses_persisted": False,
        }

    async def bootstrap(request: web.Request) -> web.Response:
        if _authorization_token(request) is None:
            return web.json_response(
                {
                    "requires_login": True,
                    "test_mode": True,
                    "responses_persisted": False,
                }
            )
        return web.json_response(current_result(participant_for(request)))

    async def enroll(request: web.Request) -> web.Response:
        payload = await _json_object(request)
        if payload:
            raise ValidationError("test enrollment accepts no participant fields")
        code = "hctest_" + secrets.token_urlsafe(12)
        token = secrets.token_urlsafe(32)
        codes[code] = code
        sessions[token] = code
        progress[code] = {
            "caption": 0,
            "image": 0,
            "consented": False,
            "stopped": False,
        }
        return web.json_response({"code": code, "token": token, "test_mode": True})

    async def login(request: web.Request) -> web.Response:
        payload = await _json_object(request)
        if set(payload) != {"code"}:
            raise ValidationError("login accepts only a participant code")
        code = _required_string(payload, "code")
        if code not in codes:
            raise AuthenticationError(
                "test code is unknown or the response-discarding server restarted"
            )
        token = secrets.token_urlsafe(32)
        sessions[token] = code
        return web.json_response({"token": token, "test_mode": True})

    async def consent(request: web.Request) -> web.Response:
        code = participant_for(request)
        payload = await _json_object(request)
        consented = payload.get("consented")
        if type(consented) is not bool:
            raise ValidationError("consented must be a boolean")
        progress[code]["consented"] = consented
        progress[code]["stopped"] = not consented
        return web.json_response(current_result(code))

    async def annotation(request: web.Request) -> web.Response:
        code = participant_for(request)
        payload = await _json_object(request)
        if set(payload) - {"assignment_id", "answers", "elapsed_ms"}:
            raise ValidationError("annotation request contains unsupported fields")
        if not isinstance(payload.get("answers"), Mapping):
            raise ValidationError("answers must be an object")
        current = current_result(code)
        if current["status"] != "task":
            raise ConflictError("participant test has no current task")
        if payload.get("assignment_id") != current["task"]["assignment_id"]:
            raise ConflictError("participant test task is stale")
        phase = current["phase"]
        progress[code][phase] += 1
        # Deliberately do not copy, log, hash, or retain answers/elapsed_ms.
        return web.json_response(current_result(code))

    async def image_response(request: web.Request) -> web.Response:
        code = participant_for(request)
        current = current_result(code)
        requested = request.match_info["assignment_id"]
        if (
            current["status"] != "task"
            or current.get("phase") != "image"
            or requested != current["task"]["assignment_id"]
        ):
            raise AuthorizationError("test image is not the current image task")
        image_index = int(progress[code]["image"])
        return await presentation_image_response(images[image_index % len(images)])

    async def coverage_caption(request: web.Request) -> web.Response:
        code = participant_for(request)
        current = current_result(code)
        if (
            current["status"] != "task"
            or current.get("phase") != "image"
            or request.match_info["assignment_id"] != current["task"]["assignment_id"]
        ):
            raise AuthorizationError("test image is not the current image task")
        image_index = int(progress[code]["image"])
        current_item = preview_items[image_index % len(preview_items)]
        return web.json_response({"caption": str(current_item["caption"])})

    async def withdraw(request: web.Request) -> web.Response:
        code = participant_for(request)
        await _json_object(request)
        progress[code]["stopped"] = True
        return web.json_response(current_result(code))

    async def health(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "ok": True,
                "test_mode": True,
                "responses_persisted": False,
                "issued_in_memory": len(codes),
            }
        )

    app.router.add_get("/", index)
    app.router.add_get("/static/{name}", static_file)
    app.router.add_get("/healthz", health)
    app.router.add_get("/api/bootstrap", bootstrap)
    app.router.add_post("/api/enroll", enroll)
    app.router.add_post("/api/login", login)
    app.router.add_post("/api/consent", consent)
    app.router.add_post("/api/annotation", annotation)
    app.router.add_post("/api/withdraw", withdraw)
    app.router.add_get("/api/image/{assignment_id}", image_response)
    app.router.add_get("/api/coverage-caption/{assignment_id}", coverage_caption)
    return app


def create_author_practice_app(
    *,
    author_codes: Mapping[str, str],
    author_items: Mapping[str, list[Mapping[str, Any]]],
    asset_root: str | Path,
) -> web.Application:
    """Serve a volatile, response-discarding rehearsal for exactly three authors.

    Codes identify anonymous practice slots only. Task progress lives in process
    memory and annotation bodies are validated only enough to advance the UI;
    response values and timings are never retained.
    """

    if len(author_codes) != 3 or set(author_codes) != set(author_items):
        raise ValueError("author practice requires exactly three matching code/item slots")
    if any(not code for code in author_codes.values()) or len(set(author_codes.values())) != 3:
        raise ValueError("author practice codes must be nonempty and unique")
    assets = Path(asset_root).resolve()
    if not assets.is_dir():
        raise NotADirectoryError(assets)
    frozen_items = {slot: [dict(item) for item in items] for slot, items in author_items.items()}
    progress = {slot: {"caption": 0, "image": 0, "stopped": False} for slot in author_codes}
    sessions: dict[str, str] = {}
    app = _application(preview_host_only=True)

    def authenticated_slot(request: web.Request) -> str:
        token = _authorization_token(request)
        slot = sessions.get(token or "")
        if slot is None:
            raise AuthenticationError("practice session is missing")
        return slot

    def current_result(slot: str) -> dict[str, Any]:
        state = progress[slot]
        items = frozen_items[slot]
        if state["stopped"] or state["image"] >= len(items):
            return {
                "status": "complete",
                "practice_mode": True,
                "progress": {
                    "caption": {"completed": state["caption"], "total": len(items)},
                    "image": {"completed": state["image"], "total": len(items)},
                    "current_phase": "complete",
                },
            }
        phase = "caption" if state["caption"] == state["image"] else "image"
        index = state[phase]
        task = preview_task(items[index], phase=phase)
        task["assignment_id"] = f"practice-{slot}-{phase}-{index:04d}"
        return {
            **_study_result(
                {
                    "status": "task",
                    "phase": phase,
                    "task": task,
                    "progress": {
                        "caption": {"completed": state["caption"], "total": len(items)},
                        "image": {"completed": state["image"], "total": len(items)},
                        "current_phase": phase,
                    },
                }
            ),
            "practice_mode": True,
        }

    async def bootstrap(request: web.Request) -> web.Response:
        token = _authorization_token(request)
        if token is None:
            return web.json_response({"requires_login": True, "practice_mode": True})
        return web.json_response(current_result(authenticated_slot(request)))

    async def login(request: web.Request) -> web.Response:
        payload = await _json_object(request)
        if set(payload) != {"code"}:
            raise ValidationError("login accepts only a practice code")
        submitted = _required_string(payload, "code")
        slot = next(
            (
                candidate
                for candidate, expected in author_codes.items()
                if hmac.compare_digest(submitted, expected)
            ),
            None,
        )
        if slot is None:
            raise AuthenticationError("practice code is invalid")
        token = secrets.token_urlsafe(32)
        sessions[token] = slot
        return web.json_response({"token": token, "practice_mode": True})

    async def annotation(request: web.Request) -> web.Response:
        slot = authenticated_slot(request)
        payload = await _json_object(request)
        if set(payload) - {"assignment_id", "answers", "elapsed_ms"}:
            raise ValidationError("annotation request contains unsupported fields")
        if not isinstance(payload.get("answers"), Mapping):
            raise ValidationError("answers must be an object")
        current = current_result(slot)
        if current["status"] != "task":
            raise ConflictError("practice slot has no current task")
        assignment_id = current["task"]["assignment_id"]
        if payload.get("assignment_id") != assignment_id:
            raise ConflictError("practice task is stale")
        phase = current["phase"]
        progress[slot][phase] += 1
        # Deliberately do not copy, log, hash, or retain answers/elapsed_ms.
        return web.json_response(current_result(slot))

    async def image_response(request: web.Request) -> web.StreamResponse:
        slot = authenticated_slot(request)
        current = current_result(slot)
        requested = request.match_info["assignment_id"]
        image_index = next(
            (
                index
                for index in range(len(frozen_items[slot]))
                if requested == f"practice-{slot}-image-{index:04d}"
            ),
            None,
        )
        current_image_index = progress[slot]["image"]
        is_current = (
            current["status"] == "task"
            and current.get("phase") == "image"
            and image_index == current_image_index
        )
        is_completed = image_index is not None and image_index < current_image_index
        if not (is_current or is_completed):
            raise AuthorizationError("practice image is not the current image task")
        relative = frozen_items[slot][image_index].get("asset_ref")
        if not isinstance(relative, str) or not relative:
            raise NotFoundError("practice image has no asset reference")
        path = (assets / relative).resolve()
        if assets not in path.parents or not path.is_file():
            raise NotFoundError("practice image is unavailable")
        return await presentation_image_response(path)

    async def coverage_caption(request: web.Request) -> web.Response:
        slot = authenticated_slot(request)
        current = current_result(slot)
        requested = request.match_info["assignment_id"]
        if current["status"] != "task" or current.get("phase") != "image":
            raise AuthorizationError("practice image is not the current image task")
        if current["task"]["assignment_id"] != requested:
            raise AuthorizationError("practice image is not the current image task")
        index = progress[slot]["image"]
        return web.json_response({"caption": str(frozen_items[slot][index]["caption"])})

    async def withdraw(request: web.Request) -> web.Response:
        slot = authenticated_slot(request)
        await _json_object(request)
        progress[slot]["stopped"] = True
        return web.json_response(current_result(slot))

    async def health(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "ok": True,
                "practice_mode": True,
                "responses_persisted": False,
            }
        )

    app.router.add_get("/", index)
    app.router.add_get("/static/{name}", static_file)
    app.router.add_get("/healthz", health)
    app.router.add_get("/api/bootstrap", bootstrap)
    app.router.add_post("/api/login", login)
    app.router.add_post("/api/annotation", annotation)
    app.router.add_post("/api/withdraw", withdraw)
    app.router.add_get("/api/image/{assignment_id}", image_response)
    app.router.add_get("/api/coverage-caption/{assignment_id}", coverage_caption)
    return app


def create_study_app(
    *,
    store: Any,
    study_id: str,
    asset_root: str | Path,
    participant_information: Mapping[str, Any],
    self_enrollment_codes: tuple[str, ...] = (),
) -> web.Application:
    """Create the production annotation app over a ``HumanCBUStore``.

    The storage object owns authorization, phase gating, projections, and
    transactional writes. This HTTP layer intentionally never reads private
    item columns directly.
    """

    assets = Path(asset_root).resolve()
    if not assets.is_dir():
        raise NotADirectoryError(assets)
    approved_information = validate_participant_information(participant_information)
    app = _application(hsts_enabled=True)

    def session_token(request: web.Request) -> str:
        token = _authorization_token(request)
        if token is None:
            raise AuthenticationError("bearer session is missing")
        return token

    async def bootstrap(request: web.Request) -> web.Response:
        token = _authorization_token(request)
        if token is None:
            return web.json_response({"requires_login": True})
        authenticated = store.authenticate_session(token, study_id=study_id)
        study_status = authenticated["study_status"]
        if authenticated["session_scope"] == "withdrawal_only":
            result = {
                "status": study_status if study_status in {"paused", "closed"} else "paused",
                "progress": store.get_progress(token, study_id=study_id),
            }
        elif study_status in {"paused", "closed"}:
            result = {
                "status": study_status,
                "progress": store.get_progress(token, study_id=study_id),
            }
        elif study_status == "open":
            result = store.fetch_next_task(token, study_id=study_id)
        else:
            raise StudyStateError("participant access is unavailable in the current study state")
        return web.json_response(
            _study_result(
                result,
                participant_information=approved_information,
            )
        )

    async def login(request: web.Request) -> web.Response:
        payload = await _json_object(request)
        if set(payload) != {"code"}:
            raise ValidationError("login accepts only a participant code")
        session = store.create_session(_required_string(payload, "code"), study_id=study_id)
        return web.json_response({"token": session["session_token"]})

    async def enroll(request: web.Request) -> web.Response:
        payload = await _json_object(request)
        if payload:
            raise ValidationError("self-enrollment accepts no participant fields")
        if not self_enrollment_codes:
            raise StudyStateError("self-enrollment is unavailable")
        issued = store.claim_preprovisioned_participant_session(
            study_id,
            self_enrollment_codes,
        )
        return web.json_response(
            {
                "code": issued["participant_code"],
                "token": issued["session_token"],
            }
        )

    async def consent(request: web.Request) -> web.Response:
        token = session_token(request)
        payload = await _json_object(request)
        if set(payload) - {"consented", "consent_version", "profile"}:
            raise ValidationError("consent request contains unsupported fields")
        profile = payload.get("profile")
        if profile is not None and not isinstance(profile, Mapping):
            raise ValidationError("profile must be an object")
        authenticated = store.authenticate_session(token, study_id=study_id)
        if (
            payload.get("consented") is False
            and authenticated["participant_status"] == "active"
            and authenticated["consented"]
        ):
            raise ValidationError(
                "active participants must use the participant-code-confirmed withdrawal endpoint"
            )
        result = store.record_consent_profile(
            token,
            study_id=study_id,
            consented=payload.get("consented"),
            consent_version=_required_string(payload, "consent_version"),
            profile=profile,
        )
        if not result["consented"]:
            return web.json_response(
                {
                    "status": "declined",
                    "progress": _normalize_progress(store.get_progress(token, study_id=study_id)),
                }
            )
        return web.json_response(
            _study_result(
                store.fetch_next_task(token, study_id=study_id),
                participant_information=approved_information,
            )
        )

    async def withdraw(request: web.Request) -> web.Response:
        token = session_token(request)
        payload = await _json_object(request)
        if set(payload) != {"participant_code", "consent_version"}:
            raise ValidationError(
                "withdrawal requires participant_code and consent_version"
            )
        authenticated = store.authenticate_session(token, study_id=study_id)
        confirmed = store.authenticate_invite(
            _required_string(payload, "participant_code"),
            study_id=study_id,
        )
        if confirmed["participant_id"] != authenticated["participant_id"]:
            raise AuthorizationError(
                "withdrawal participant code does not match the bearer session"
            )
        result = store.record_consent_profile(
            token,
            study_id=study_id,
            consented=False,
            consent_version=_required_string(payload, "consent_version"),
            profile={},
        )
        return web.json_response(
            {
                "status": "declined",
                "progress": _normalize_progress(
                    store.get_progress(token, study_id=study_id)
                ),
                "withdrawn": not result["consented"],
            }
        )

    async def annotation(request: web.Request) -> web.Response:
        token = session_token(request)
        payload = await _json_object(request)
        if set(payload) - {"assignment_id", "answers", "elapsed_ms"}:
            raise ValidationError("annotation request contains unsupported fields")
        answers = payload.get("answers")
        if not isinstance(answers, Mapping):
            raise ValidationError("answers must be an object")
        store.save_annotation(
            token,
            _required_string(payload, "assignment_id"),
            answers,
            study_id=study_id,
            elapsed_ms=payload.get("elapsed_ms"),
        )
        return web.json_response(
            _study_result(
                store.fetch_next_task(token, study_id=study_id),
                participant_information=approved_information,
            )
        )

    async def image_response(request: web.Request) -> web.StreamResponse:
        token = session_token(request)
        assignment_id = request.match_info["assignment_id"]
        asset = store.authorized_image_asset(token, assignment_id, study_id=study_id)
        relative = asset.get("asset_ref")
        if not isinstance(relative, str) or not relative:
            raise NotFoundError("authorized image asset has no path")
        path = (assets / relative).resolve()
        if assets not in path.parents or not path.is_file():
            raise NotFoundError("authorized image asset is missing")
        return await presentation_image_response(path)

    async def coverage_caption(request: web.Request) -> web.Response:
        result = store.authorized_coverage_caption(
            session_token(request),
            request.match_info["assignment_id"],
            study_id=study_id,
        )
        return web.json_response({"caption": result["caption"]})

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def study_information(_: web.Request) -> web.Response:
        return web.Response(
            text=participant_information_document(approved_information),
            content_type="text/html",
            charset="utf-8",
        )

    async def extend_work(request: web.Request) -> web.Response:
        token = session_token(request)
        if await _json_object(request):
            raise ValidationError("optional-work request accepts no fields")
        store.extend_remaining_first_work(token, study_id=study_id)
        return web.json_response(
            _study_result(
                store.fetch_next_task(token, study_id=study_id),
                participant_information=approved_information,
            )
        )

    app.router.add_get("/", index)
    app.router.add_get("/study-information", study_information)
    app.router.add_get("/static/{name}", static_file)
    app.router.add_get("/healthz", health)
    app.router.add_get("/api/bootstrap", bootstrap)
    app.router.add_post("/api/login", login)
    app.router.add_post("/api/enroll", enroll)
    app.router.add_post("/api/consent", consent)
    app.router.add_post("/api/withdraw", withdraw)
    app.router.add_post("/api/annotation", annotation)
    app.router.add_post("/api/extend-work", extend_work)
    app.router.add_get("/api/image/{assignment_id}", image_response)
    app.router.add_get("/api/coverage-caption/{assignment_id}", coverage_caption)

    async def close_store(_: web.Application) -> None:
        store.close()

    app.on_cleanup.append(close_store)
    return app


def load_preview_item(path: str | Path) -> dict[str, Any]:
    """Load a JSON preview item while rejecting non-object payloads."""

    item = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(item, dict):
        raise ValueError("preview item must be a JSON object")
    return item
