"""Packet-scoped, loopback-only adjudication review interface.

The browser is deliberately not an authority for packet identity.  One app is
constructed around one already-validated packet, and the browser can submit
only the phase-specific decision fields.  The caller binds that decision to
the exact packet sidecar and persists it.
"""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import mimetypes
import re
import secrets
import stat
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlsplit

from aiohttp import web

from .builder import SEMANTIC_VISUAL_CLAIM_TYPES
from .store import (
    ADJUDICATION_PACKET_VERSION,
    AuthenticationError,
    ConflictError,
    HumanCBUStore,
    HumanCBUStoreError,
    ValidationError,
)

DEFAULT_SEMANTIC_CATEGORIES = tuple(SEMANTIC_VISUAL_CLAIM_TYPES)
DEFAULT_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_PACKET_ID_RE = re.compile(r"ap_[0-9a-f]{64}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_SESSION_COOKIE = "human_cbu_adjudication_session"
_CSRF_COOKIE = "human_cbu_adjudication_csrf"
_CSRF_HEADER = "X-Human-CBU-CSRF"
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_IMAGE_BYTES = 256 * 1024 * 1024
_SUBMITTED_EVENT_KEY = web.AppKey("human_cbu_submitted_event", asyncio.Event)

ADJUDICATION_CSP = (
    "default-src 'none'; "
    "base-uri 'none'; "
    "connect-src 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'; "
    "img-src 'self' data:; "
    "script-src 'unsafe-inline'; "
    "style-src 'unsafe-inline'"
)


def security_headers() -> dict[str, str]:
    """Return the fixed headers applied to every loopback response."""

    return {
        "Cache-Control": "no-store, max-age=0",
        "Content-Security-Policy": ADJUDICATION_CSP,
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
        "Permissions-Policy": ("camera=(), microphone=(), geolocation=(), payment=(), usb=()"),
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
    }


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValidationError("adjudication packet must be strict JSON") from error


def _semantic_categories(values: Sequence[str]) -> tuple[str, ...]:
    supported = set(SEMANTIC_VISUAL_CLAIM_TYPES)
    if (
        isinstance(values, (str, bytes))
        or not values
        or any(type(value) is not str or value not in supported for value in values)
        or len(values) != len(set(values))
    ):
        raise ValidationError(
            "semantic_categories must be a non-empty unique sequence of supported semantic visual-claim types"
        )
    return tuple(values)


def _optional_text(
    value: Any,
    field: str,
    *,
    maximum: int,
) -> str | None:
    if value is None:
        return None
    if type(value) is not str or len(value) > maximum:
        raise ValidationError(f"{field} must be text or null")
    return value


def _required_text(value: Any, field: str, *, maximum: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValidationError(f"{field} must be non-empty text")
    return value


@dataclass(frozen=True)
class ValidatedAdjudicationPacket:
    """Validated phase-safe packet data used by both renderer and server."""

    packet: dict[str, Any]
    phase: str
    evidence: dict[str, Any]
    semantic_categories: tuple[str, ...]
    correction_categories: tuple[str, ...]


def validate_adjudication_packet(
    packet: Mapping[str, Any],
    *,
    semantic_categories: Sequence[str] = DEFAULT_SEMANTIC_CATEGORIES,
) -> ValidatedAdjudicationPacket:
    """Validate the exact public packet schema and its logical digest."""

    categories = _semantic_categories(semantic_categories)
    packet_keys = {
        "packet_version",
        "packet_id",
        "packet_sha256",
        "phase",
        "evidence",
    }
    if not isinstance(packet, Mapping) or set(packet) != packet_keys:
        raise ValidationError("adjudication packet violates the exact top-level allowlist")
    packet_id = packet["packet_id"]
    if type(packet_id) is not str or _PACKET_ID_RE.fullmatch(packet_id) is None:
        raise ValidationError("adjudication packet_id is invalid")
    if packet["packet_version"] != ADJUDICATION_PACKET_VERSION:
        raise ValidationError("adjudication packet version mismatch")
    phase = packet["phase"]
    if phase not in {"caption", "image"}:
        raise ValidationError("adjudication packet phase is invalid")
    evidence = packet["evidence"]
    if not isinstance(evidence, Mapping):
        raise ValidationError("adjudication packet evidence must be an object")

    common = {"unit", "target", "category"}
    expected = common | {"caption", "span"} if phase == "caption" else common | {"image_file", "image_sha256"}
    if set(evidence) != expected:
        raise ValidationError(f"{phase} adjudication evidence violates its exact allowlist")
    _required_text(evidence["unit"], "evidence.unit", maximum=100_000)
    _optional_text(evidence["target"], "evidence.target", maximum=10_000)
    category = evidence["category"]
    if type(category) is not str or category not in categories:
        raise ValidationError("packet category is outside the frozen semantic category vocabulary")
    if phase == "caption":
        _required_text(evidence["caption"], "evidence.caption", maximum=1_000_000)
        _optional_text(evidence["span"], "evidence.span", maximum=100_000)
    else:
        image_file = evidence["image_file"]
        if type(image_file) is not str or "\\" in image_file:
            raise ValidationError("image packet has an unsafe packet-local filename")
        image_path = PurePosixPath(image_file)
        if (
            image_path.is_absolute()
            or len(image_path.parts) != 1
            or image_path.name != f"image{image_path.suffix}"
            or image_path.suffix.lower() not in _IMAGE_SUFFIXES
        ):
            raise ValidationError("image packet has an unsafe packet-local filename")
        image_sha256 = evidence["image_sha256"]
        if type(image_sha256) is not str or _SHA256_RE.fullmatch(image_sha256) is None:
            raise ValidationError("image packet SHA-256 is invalid")

    projection = {key: packet[key] for key in ("packet_version", "packet_id", "phase", "evidence")}
    import hashlib

    expected_sha256 = hashlib.sha256(_canonical_json(projection)).hexdigest()
    packet_sha256 = packet["packet_sha256"]
    if type(packet_sha256) is not str or not secrets.compare_digest(packet_sha256, expected_sha256):
        raise ValidationError("adjudication packet SHA-256 does not match its evidence")
    copied_packet = json.loads(_canonical_json(packet).decode("utf-8"))
    copied_evidence = dict(copied_packet["evidence"])
    return ValidatedAdjudicationPacket(
        packet=copied_packet,
        phase=phase,
        evidence=copied_evidence,
        semantic_categories=categories,
        correction_categories=tuple(value for value in categories if value != category),
    )


def _highlight_caption(caption: str, span: str | None) -> str:
    if not span:
        return escape(caption)
    start = caption.find(span)
    if start < 0:
        return escape(caption)
    end = start + len(span)
    return escape(caption[:start]) + "<mark>" + escape(caption[start:end]) + "</mark>" + escape(caption[end:])


def _choice(
    name: str,
    value: str,
    label: str,
    detail: str,
    *,
    shortcut: str,
) -> str:
    return f"""
      <label class="choice">
        <input type="radio" name="{escape(name)}" value="{escape(value)}">
        <span class="choice-key" aria-hidden="true">{escape(shortcut)}</span>
        <span class="choice-copy">
          <strong>{escape(label)}</strong>
          <small>{escape(detail)}</small>
        </span>
        <span class="choice-dot" aria-hidden="true"></span>
      </label>"""


def _caption_decisions(validated: ValidatedAdjudicationPacket) -> str:
    proposed = validated.evidence["category"]
    options = "".join(
        f'<option value="{escape(value)}">{escape(value.replace("_", " ").title())}</option>'
        for value in validated.correction_categories
    )
    licensed_yes = _choice(
        "caption_licensed",
        "yes",
        "Yes",
        "The complete claim is directly stated.",
        shortcut="1",
    )
    licensed_no = _choice(
        "caption_licensed",
        "no",
        "No",
        "Some or all of the claim is not stated.",
        shortcut="2",
    )
    licensed_uncertain = _choice(
        "caption_licensed",
        "uncertain",
        "Unclear",
        "The wording does not support a confident decision.",
        shortcut="3",
    )
    atomic_yes = _choice(
        "atomic_visual_claim",
        "yes",
        "Yes",
        "One visual fact that a prompt could control.",
        shortcut="1",
    )
    atomic_not_visual = _choice(
        "atomic_visual_claim",
        "not_visual",
        "No · not visual",
        "Metadata, captioner wording, or nonvisual content.",
        shortcut="2",
    )
    atomic_not_atomic = _choice(
        "atomic_visual_claim",
        "not_atomic",
        "No · not atomic",
        "Several separable visual facts are bundled.",
        shortcut="3",
    )
    atomic_uncertain = _choice(
        "atomic_visual_claim",
        "uncertain",
        "Unclear",
        "The unit cannot be classified confidently.",
        shortcut="4",
    )
    category_correct = _choice(
        "category_check",
        "correct",
        "Correct",
        "The proposed type is the best fit.",
        shortcut="1",
    )
    category_incorrect = _choice(
        "category_check",
        "incorrect",
        "Incorrect",
        "Choose a different frozen semantic type.",
        shortcut="2",
    )
    category_uncertain = _choice(
        "category_check",
        "uncertain",
        "Unclear",
        "No confident type decision is possible.",
        shortcut="3",
    )
    return f"""
      <section class="question" aria-labelledby="q1">
        <p class="eyebrow coral">Question 1 · explicitness</p>
        <h2 id="q1">Does the caption explicitly state the full claim?</h2>
        <div class="choices">
          {licensed_yes}
          {licensed_no}
          {licensed_uncertain}
        </div>
      </section>
      <section class="question" aria-labelledby="q2">
        <p class="eyebrow coral">Question 2 · CBU validity</p>
        <h2 id="q2">Is this one meaningful visual-control claim?</h2>
        <div class="guidance">
          <strong>Atomic does not mean one word.</strong>
          Keep modifiers together when they specify one property or relation.
          Mark the unit non-atomic only when it bundles independently
          checkable or controllable facts.
        </div>
        <div class="choices">
          {atomic_yes}
          {atomic_not_visual}
          {atomic_not_atomic}
          {atomic_uncertain}
        </div>
      </section>
      <section class="question" id="category-section" aria-labelledby="q3">
        <p class="eyebrow coral">Question 3 · semantic type</p>
        <h2 id="q3">Is the proposed type correct?</h2>
        <div class="proposed">
          <span>Proposed type</span>
          <strong>{escape(proposed.replace("_", " ").title())}</strong>
        </div>
        <div class="choices compact">
          {category_correct}
          {category_incorrect}
          {category_uncertain}
        </div>
        <label class="correction hidden" id="correction">
          <span>Correct semantic type</span>
          <select id="corrected-category" name="corrected_category">
            <option value="">Choose one…</option>
            {options}
          </select>
        </label>
        <p class="auto-note hidden" id="category-na">
          Semantic type is recorded as not applicable for a nonvisual or
          non-atomic unit.
        </p>
      </section>"""


def _image_decisions(validated: ValidatedAdjudicationPacket) -> str:
    category = validated.evidence["category"].replace("_", " ").title()
    support_yes = _choice(
        "image_support",
        "yes",
        "Supported",
        "Visible evidence supports the full claim.",
        shortcut="1",
    )
    support_no = _choice(
        "image_support",
        "no",
        "Unsupported",
        "The claim is contradicted or lacks visible support.",
        shortcut="2",
    )
    support_uncertain = _choice(
        "image_support",
        "uncertain",
        "Uncertain",
        "Occlusion, scale, ambiguity, or unreadability prevents a decision.",
        shortcut="3",
    )
    support_not_visual = _choice(
        "image_support",
        "not_visual",
        "Not a visual claim",
        "The candidate cannot be decided from visible evidence.",
        shortcut="4",
    )
    support_unavailable = _choice(
        "image_support",
        "image_unavailable",
        "Image unusable",
        "The image is missing, corrupt, or cannot be inspected.",
        shortcut="5",
    )
    support_skip = _choice(
        "image_support",
        "prefer_not_to_answer",
        "Prefer not to answer",
        "Do not provide a support label for this packet.",
        shortcut="6",
    )
    return f"""
      <section class="question image-question" aria-labelledby="q1">
        <p class="eyebrow coral">Question 1 · visual support</p>
        <h2 id="q1">Does the image visibly support the complete claim?</h2>
        <div class="choices">
          {support_yes}
          {support_no}
          {support_uncertain}
          {support_not_visual}
          {support_unavailable}
          {support_skip}
        </div>
        <div class="guidance">
          <strong>{escape(category)}:</strong>
          Judge the whole claim and its named target using only visible
          evidence. Do not infer hidden causes or unseen content.
        </div>
      </section>"""


_BASE_CSS = r"""
:root {
  --ink:#17202a; --muted:#69727b; --paper:#fbfaf6; --deep:#f1eee6;
  --line:#d9d5cb; --line2:#bdb7aa; --teal:#0d6964; --teal2:#084b48;
  --tealsoft:#dcece8; --coral:#ad4937; --coralsoft:#f8e3dc;
  --amber:#ffd971; --ambersoft:#fff3c9; --blue:#e5edf7; --white:#fff;
  --shadow:0 24px 70px rgba(34,43,46,.12);
  font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  color:var(--ink); background:var(--deep); font-synthesis:none;
}
*{box-sizing:border-box} html{min-height:100%;background:var(--deep)}
body{min-width:320px;min-height:100vh;margin:0;overflow-x:hidden;background:
linear-gradient(rgba(23,32,42,.026) 1px,transparent 1px),
linear-gradient(90deg,rgba(23,32,42,.026) 1px,transparent 1px),var(--deep);
background-size:32px 32px}
body:before,body:after{content:"";position:fixed;z-index:0;width:390px;height:390px;
border-radius:50%;filter:blur(85px);opacity:.22;pointer-events:none}
body:before{top:-190px;left:-100px;background:#95c9c1}
body:after{right:-150px;bottom:-210px;background:#efa68f}
button,input,select,textarea{font:inherit}.shell{position:relative;z-index:1;
width:min(1480px,100%);min-height:100vh;margin:auto;padding:22px 30px 30px}
.topbar{height:70px;display:grid;grid-template-columns:1fr auto 1fr;align-items:center;
gap:24px}.brand{display:flex;align-items:center;gap:12px}.mark{display:grid;place-items:center;
width:42px;height:42px;border-radius:13px 13px 13px 4px;background:var(--ink);color:white;
font:700 21px Georgia,serif;box-shadow:0 8px 25px rgba(35,44,47,.10)}
.brand strong,.brand div>span{display:block}.brand strong{font-size:15px}.brand div>span{margin-top:3px;
font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.phase{display:flex;align-items:center;gap:10px;font-size:13px;font-weight:750}
.phase-number{display:grid;place-items:center;width:31px;height:31px;border-radius:50%;
background:var(--teal);color:white;box-shadow:0 5px 16px rgba(13,105,100,.22)}
.privacy{justify-self:end;display:flex;align-items:center;gap:8px;padding:9px 12px;
border:1px solid #acd1ca;border-radius:999px;background:var(--tealsoft);color:var(--teal2);
font-size:12px;font-weight:750}.workspace{height:calc(100vh - 122px);min-height:670px;
display:grid;grid-template-columns:minmax(0,1.08fr) minmax(430px,.92fr);gap:22px}
.panel{min-width:0;min-height:0;border:1px solid var(--line);border-radius:27px;
background:rgba(251,250,246,.96);box-shadow:var(--shadow);overflow:hidden}
.evidence-panel{display:flex;flex-direction:column}.panel-head{min-height:68px;padding:16px 24px;
display:flex;align-items:center;justify-content:space-between;gap:14px;border-bottom:1px solid var(--line)}
.eyebrow{margin:0;color:var(--muted);font-size:11px;font-weight:800;letter-spacing:.11em;
text-transform:uppercase}.coral{color:var(--coral)}.badge{display:inline-flex;align-items:center;
min-height:29px;padding:6px 11px;border:1px solid #ead184;border-radius:999px;
background:var(--ambersoft);font-size:11px;font-weight:750;color:#705619}
.evidence-body{min-height:0;flex:1;padding:30px 40px 34px;overflow:auto}
.claim-strip{margin-bottom:22px;padding:17px 18px;border:1px solid #cad8e9;
border-radius:15px;background:var(--blue)}.claim-meta{display:flex;flex-wrap:wrap;gap:7px;
margin-bottom:8px}.pill{padding:5px 9px;border:1px solid #c9d6e7;border-radius:999px;
background:#edf3fa;color:#54606d;font-size:10px;font-weight:800;letter-spacing:.07em;
text-transform:uppercase}.claim{margin:0;font:500 clamp(25px,2.2vw,38px)/1.16 Georgia,serif;
letter-spacing:-.02em}.target{margin:8px 0 0;color:var(--muted);font-size:12px}
.instruction{margin:28px 0 13px}.instruction h1{max-width:650px;margin:8px 0 0;
font:500 clamp(28px,2.7vw,43px)/1.08 Georgia,serif;letter-spacing:-.03em}
.caption-card{padding:34px 40px;border:1px solid var(--line);border-left:5px solid var(--coral);
border-radius:18px;background:white;box-shadow:0 12px 35px rgba(35,44,47,.08);
font:400 clamp(21px,1.75vw,29px)/1.58 Georgia,serif;white-space:pre-wrap}
.caption-card mark{padding:0 .12em;background:var(--amber);color:inherit}.phase-note{display:flex;gap:10px;
margin:20px 0 0;color:var(--muted);font-size:12px;line-height:1.5}
.image-stage{height:calc(100% - 4px);min-height:420px;padding:28px;display:grid;place-items:center;
background:#e6e3dc}.image-frame{width:100%;height:100%;min-height:380px;display:grid;
place-items:center;overflow:hidden;background:#efede7;box-shadow:0 16px 42px rgba(26,32,34,.14)}
.image-frame img{display:block;max-width:100%;max-height:100%;object-fit:contain}
.image-evidence{display:flex;flex-direction:column;overflow:hidden}.image-evidence .claim-strip{
flex:0 0 auto}.image-evidence .image-stage{flex:1 1 auto;height:auto;min-height:0}
.image-evidence .image-frame{height:100%;min-height:0}.image-evidence .phase-note{flex:0 0 auto}
.decision-panel{display:flex;flex-direction:column}.decision-scroll{min-height:0;flex:1;
padding:28px 38px 120px;overflow:auto}.decision-intro{margin-bottom:18px}.decision-intro h1{
margin:7px 0 5px;font:500 clamp(25px,2vw,35px)/1.15 Georgia,serif}
.decision-intro p{margin:0;color:var(--muted);font-size:12px;line-height:1.45}
.question{padding:22px 0 25px;border-top:1px solid var(--line)}
.question:first-of-type{border-top:0}.question h2{margin:7px 0 14px;font-size:16px;line-height:1.32}
.choices{display:grid;gap:8px}.choice{position:relative;display:flex;align-items:center;gap:11px;
min-height:55px;padding:10px 13px;cursor:pointer;border:1px solid var(--line);border-radius:14px;
background:rgba(255,255,255,.78);transition:.15s ease}.choice:hover{border-color:#9dafa9;
transform:translateY(-1px)}.choice:has(input:checked){border-color:var(--teal);
background:var(--tealsoft);box-shadow:0 0 0 2px rgba(13,105,100,.08)}
.choice input{position:absolute;opacity:0;pointer-events:none}.choice-key{display:grid;place-items:center;
flex:0 0 24px;width:24px;height:24px;border:1px solid var(--line);border-radius:7px;
background:white;color:var(--muted);font-size:10px;font-weight:800}.choice-copy{min-width:0;flex:1}
.choice-copy strong,.choice-copy small{display:block}.choice-copy strong{font-size:13px}
.choice-copy small{margin-top:3px;color:var(--muted);font-size:10.5px;line-height:1.32}
.choice-dot{width:17px;height:17px;border:2px solid var(--line2);border-radius:50%}
.choice:has(input:checked) .choice-dot{border:5px solid var(--teal)}
.choice:focus-within{outline:3px solid rgba(13,105,100,.2);outline-offset:2px}
.compact .choice{min-height:52px}.proposed{display:flex;align-items:center;justify-content:space-between;
gap:12px;margin-bottom:10px;padding:11px 13px;border:1px solid #cbd8e8;border-radius:13px;
background:var(--blue);font-size:11px}.proposed span{color:var(--muted);font-weight:700}
.correction{display:grid;gap:7px;margin-top:12px;color:var(--muted);font-size:11px;font-weight:700}
.correction select{width:100%;min-height:44px;padding:0 12px;border:1px solid var(--line2);
border-radius:12px;background:white;color:var(--ink)}.auto-note,.guidance{margin:12px 0 0;
padding:12px 14px;border:1px solid #ebd184;border-radius:12px;background:var(--ambersoft);
color:#6c5620;font-size:11px;line-height:1.45}.hidden{display:none!important}
.actionbar{position:absolute;right:0;bottom:0;left:0;display:flex;align-items:center;
justify-content:space-between;gap:14px;min-height:82px;padding:14px 22px;border-top:1px solid var(--line);
background:rgba(255,255,255,.93);backdrop-filter:blur(12px)}.decision-panel{position:relative}
.status{margin:0;color:var(--muted);font-size:11px;line-height:1.4}.status.error{color:#9b342d}
.status.success{color:var(--teal2);font-weight:700}.submit{min-width:178px;min-height:49px;
padding:11px 20px;border:0;border-radius:14px;background:var(--teal);color:white;
font-size:12px;font-weight:800;cursor:pointer;box-shadow:0 9px 24px rgba(13,105,100,.22)}
.submit:hover:not(:disabled){background:var(--teal2)}.submit:disabled{cursor:not-allowed;
background:#aeb3b5;box-shadow:none}.submitted .choice,.submitted select{pointer-events:none}
@media(max-width:840px){
  .shell{padding:10px 10px 24px}.topbar{height:auto;min-height:76px;grid-template-columns:1fr auto;
  grid-template-areas:"brand privacy" "phase phase";gap:9px 12px;margin-bottom:8px}
  .brand{grid-area:brand}.brand div>span{display:none}.privacy{grid-area:privacy;padding:7px 9px;font-size:10px}
  .phase{grid-area:phase;justify-self:center;font-size:11px}.workspace{height:auto;min-height:0;
  display:block}.panel{border-radius:21px;margin-bottom:12px}.evidence-body{padding:18px}
  .panel-head{min-height:55px;padding:12px 15px}.claim-strip{margin:-18px -18px 20px;
  border-width:0 0 1px;border-radius:0;padding:16px 18px}.instruction{margin:20px 0 13px}
  .instruction h1{font-size:28px}.caption-card{padding:25px 20px;font-size:20px;line-height:1.48}
  .image-evidence{display:block;overflow:visible}.image-stage{height:auto;min-height:390px;padding:14px}
  .image-frame{height:390px;min-height:390px}
  .decision-scroll{padding:23px 18px 105px;overflow:visible}.question{padding:22px 0}
  .question h2{font-size:15px}.choice{min-height:58px}.choice-copy small{font-size:10px}
  .actionbar{position:sticky;bottom:0;z-index:4;margin:0 -18px -105px;padding:12px 12px;
  min-height:76px}.status{max-width:125px}.submit{min-width:164px}
}
@media(max-width:380px){.privacy{display:none}.topbar{grid-template-columns:1fr;
grid-template-areas:"brand" "phase"}.claim{font-size:25px}.submit{min-width:150px}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important}}
"""


def _client_script(phase: str) -> str:
    phase_logic = (
        """
    const explicit = value("caption_licensed");
    const atomic = value("atomic_visual_claim");
    let category = value("category_check");
    const correction = document.getElementById("corrected-category");
    if (atomic === "not_visual" || atomic === "not_atomic") {
      category = "not_applicable";
    }
    if (!explicit || !atomic || !category) return null;
    const decision = {
      caption_licensed: explicit,
      atomic_visual_claim: atomic,
      category_check: category
    };
    if (category === "incorrect") {
      if (!correction.value) return null;
      decision.corrected_category = correction.value;
    }
    return decision;"""
        if phase == "caption"
        else """
    const support = value("image_support");
    return support ? {image_support: support} : null;"""
    )
    phase_updates = (
        """
    const atomic = value("atomic_visual_claim");
    const nonClaim = atomic === "not_visual" || atomic === "not_atomic";
    const categorySection = document.getElementById("category-section");
    const categoryNA = document.getElementById("category-na");
    const correction = document.getElementById("correction");
    if (categorySection) categorySection.classList.toggle("muted-section", nonClaim);
    if (categoryNA) categoryNA.classList.toggle("hidden", !nonClaim);
    document.querySelectorAll('input[name="category_check"]').forEach((input) => {
      input.disabled = nonClaim;
      if (nonClaim) input.checked = false;
    });
    const incorrect = !nonClaim && value("category_check") === "incorrect";
    if (correction) correction.classList.toggle("hidden", !incorrect);
    if (!incorrect) document.getElementById("corrected-category").value = "";"""
        if phase == "caption"
        else """
    const image = document.getElementById("evidence-image");
    if (image && !image.src) {
      image.src = location.protocol === "file:" ? image.dataset.fileSrc : "/image";
    }"""
    )
    return (
        r"""
(() => {
  "use strict";
  const form = document.getElementById("decision-form");
  const submit = document.getElementById("submit");
  const status = document.getElementById("status");
  let sending = false;
  const value = (name) => {
    const selected = form.querySelector(`input[name="${name}"]:checked`);
    return selected ? selected.value : "";
  };
  const decision = () => {"""
        + phase_logic
        + r"""
  };
  const update = () => {"""
        + phase_updates
        + r"""
    const ready = Boolean(decision());
    submit.disabled = sending || !ready;
    if (!sending && !form.classList.contains("submitted")) {
      status.textContent = ready
        ? "Ready to record this packet resolution."
        : "Complete every required decision.";
    }
  };
  const csrf = () => {
    const prefix = "human_cbu_adjudication_csrf=";
    const part = document.cookie.split("; ").find((entry) => entry.startsWith(prefix));
    return part ? decodeURIComponent(part.slice(prefix.length)) : "";
  };
  form.addEventListener("change", update);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = decision();
    if (!payload || sending) return;
    sending = true;
    update();
    status.className = "status";
    status.textContent = "Recording this packet…";
    try {
      const response = await fetch("/decision", {
        method: "POST",
        credentials: "same-origin",
        headers: {
          "Content-Type": "application/json",
          "X-Human-CBU-CSRF": csrf()
        },
        body: JSON.stringify({decision: payload})
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(result.error || "The decision was not recorded.");
      form.classList.add("submitted");
      form.querySelectorAll("input,select,button").forEach((control) => {
        control.disabled = true;
      });
      status.className = "status success";
      status.textContent = "Recorded. This packet is now locked.";
      submit.textContent = "Recorded ✓";
    } catch (error) {
      sending = false;
      status.className = "status error";
      status.textContent = error.message || "The decision was not recorded.";
      update();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey || sending) return;
    const active = document.activeElement;
    if (active && ["SELECT", "TEXTAREA", "INPUT"].includes(active.tagName)) return;
    const section = active && active.closest ? active.closest(".question") : null;
    if (!section || !/^[1-6]$/.test(event.key)) return;
    const choices = [...section.querySelectorAll('input[type="radio"]:not(:disabled)')];
    const selected = choices[Number(event.key) - 1];
    if (selected) { selected.checked = true; selected.dispatchEvent(new Event("change", {bubbles:true})); }
  });
  update();
})();
"""
    )


def render_adjudication_html(
    packet: Mapping[str, Any],
    *,
    semantic_categories: Sequence[str] = DEFAULT_SEMANTIC_CATEGORIES,
) -> str:
    """Render deterministic, self-contained packet review HTML."""

    validated = validate_adjudication_packet(
        packet,
        semantic_categories=semantic_categories,
    )
    evidence = validated.evidence
    phase = validated.phase
    category = evidence["category"].replace("_", " ").title()
    target = evidence["target"] or "unspecified"
    phase_title = "Caption claim" if phase == "caption" else "Image evidence"
    privacy = "Image withheld" if phase == "caption" else "Caption withheld"
    badge = "Matched lexical window" if phase == "caption" else "Visible evidence only"
    instruction = "Resolve the caption-side CBU labels" if phase == "caption" else "Resolve image support for this CBU"
    phase_note = (
        "Judge only what the shown caption explicitly licenses. The paired "
        "image and automated judge outputs are intentionally unavailable."
        if phase == "caption"
        else "Judge only visible image evidence. The caption, surface "
        "identity, and automated judge outputs are intentionally unavailable."
    )
    if phase == "caption":
        evidence_view = f"""
          <div class="instruction">
            <p class="eyebrow">Read only what is stated</p>
            <h1>Is the claim licensed by this caption?</h1>
          </div>
          <div class="caption-card">{_highlight_caption(evidence["caption"], evidence["span"])}</div>"""
        decisions = _caption_decisions(validated)
    else:
        image_name = escape(evidence["image_file"])
        evidence_view = f"""
          <div class="image-stage">
            <div class="image-frame">
              <img id="evidence-image" data-file-src="{image_name}"
                   alt="Packet image evidence">
            </div>
          </div>"""
        decisions = _image_decisions(validated)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="light">
  <meta name="theme-color" content="#f1eee6">
  <title>Human CBU Adjudication · {escape(phase_title)}</title>
  <style>{_BASE_CSS}</style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand">
        <span class="mark" aria-hidden="true">C</span>
        <div><strong>Human CBU Adjudication</strong><span>Private · blinded</span></div>
      </div>
      <div class="phase"><span class="phase-number">1</span>{escape(phase_title)}</div>
      <div class="privacy">⌁ {escape(privacy)}</div>
    </header>
    <div class="workspace">
      <section class="panel evidence-panel" aria-label="{escape(phase_title)}">
        <header class="panel-head">
          <p class="eyebrow">{escape(phase_title)}</p>
          <span class="badge">{escape(badge)}</span>
        </header>
        <div class="evidence-body{" image-evidence" if phase == "image" else ""}">
          <div class="claim-strip">
            <div class="claim-meta">
              <span class="pill">{escape(category)}</span>
              <span class="pill">Target · {escape(str(target))}</span>
            </div>
            <p class="claim">{escape(evidence["unit"])}</p>
          </div>
          {evidence_view}
          <p class="phase-note"><span aria-hidden="true">ⓘ</span><span>{escape(phase_note)}</span></p>
        </div>
      </section>
      <form class="panel decision-panel" id="decision-form" novalidate>
        <div class="decision-scroll">
          <header class="decision-intro">
            <p class="eyebrow">One packet · final resolution</p>
            <h1>{escape(instruction)}</h1>
            <p>Resolve from the evidence shown. Prior human and model labels remain hidden.</p>
          </header>
          {decisions}
        </div>
        <footer class="actionbar">
          <p class="status" id="status" role="status">Complete every required decision.</p>
          <button class="submit" id="submit" type="submit" disabled>Record decision →</button>
        </footer>
      </form>
    </div>
  </main>
  <script>{_client_script(phase)}</script>
</body>
</html>
"""


def render_adjudication_html_bytes(
    packet: Mapping[str, Any],
    *,
    semantic_categories: Sequence[str] = DEFAULT_SEMANTIC_CATEGORIES,
) -> bytes:
    """Return the canonical UTF-8 bytes persisted as ``review.html``."""

    return render_adjudication_html(
        packet,
        semantic_categories=semantic_categories,
    ).encode("utf-8")


def normalize_adjudication_decision(
    validated: ValidatedAdjudicationPacket,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate browser decision fields and return a compact canonical payload."""

    if not isinstance(decision, Mapping):
        raise ValidationError("decision must be an object")
    if validated.phase == "image" and "control_usefulness" in decision:
        raise ValidationError("control_usefulness is not allowed in image adjudication")
    normalized = HumanCBUStore.validate_annotation_payload(
        validated.phase,
        decision,
    )
    if validated.phase == "caption":
        corrected = normalized.get("corrected_category")
        if normalized["category_check"] == "incorrect":
            if corrected not in validated.correction_categories:
                raise ValidationError("corrected_category must be a frozen semantic type other than the proposed type")
        result = {
            "caption_licensed": normalized["caption_licensed"],
            "atomic_visual_claim": normalized["atomic_visual_claim"],
            "category_check": normalized["category_check"],
        }
        if corrected is not None:
            result["corrected_category"] = corrected
    else:
        result = {"image_support": normalized["image_support"]}
    for field in ("reason_tags", "note", "confidence", "elapsed_ms"):
        if field in decision:
            result[field] = normalized[field]
    return result


def _strict_json(raw: bytes) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValidationError(f"request contains duplicate key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValidationError(f"request contains invalid JSON constant {value!r}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError("request must be strict UTF-8 JSON") from error


def _hostname_from_host_header(value: str) -> str | None:
    try:
        return urlsplit(f"//{value}").hostname
    except ValueError:
        return None


def _is_loopback(value: str) -> bool:
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _read_private_image(path: Path) -> tuple[bytes, str]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValidationError("adjudication image must be an existing packet-local file") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.is_symlink()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or metadata.st_size > _MAX_IMAGE_BYTES
    ):
        raise ValidationError("adjudication image must be a private regular non-hardlinked file")
    raw = path.read_bytes()
    if len(raw) != metadata.st_size:
        raise ValidationError("adjudication image changed while it was loaded")
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return raw, content_type


def create_adjudication_app(
    packet: Mapping[str, Any],
    *,
    on_submit: Callable[[dict[str, Any]], Any | Awaitable[Any]],
    image_path: str | Path | None = None,
    semantic_categories: Sequence[str] = DEFAULT_SEMANTIC_CATEGORIES,
    review_html: bytes | None = None,
    allowed_hosts: Sequence[str] = tuple(DEFAULT_LOOPBACK_HOSTS),
    session_token: str | None = None,
) -> tuple[web.Application, str]:
    """Create a one-packet app and return it with its one-launch bearer token."""

    validated = validate_adjudication_packet(
        packet,
        semantic_categories=semantic_categories,
    )
    expected_html = render_adjudication_html_bytes(
        packet,
        semantic_categories=validated.semantic_categories,
    )
    if review_html is not None and not secrets.compare_digest(
        review_html,
        expected_html,
    ):
        raise ValidationError("supplied review_html does not deterministically match the packet")
    html_bytes = expected_html if review_html is None else review_html

    normalized_hosts = frozenset(
        value.strip("[]").lower() for value in allowed_hosts if isinstance(value, str) and value
    )
    if not normalized_hosts or any(not _is_loopback(value) for value in normalized_hosts):
        raise ValidationError("adjudication allowed_hosts must be loopback-only")
    token = session_token or secrets.token_urlsafe(32)
    if type(token) is not str or len(token) < 32:
        raise ValidationError("adjudication session token is invalid")
    csrf_token = secrets.token_urlsafe(32)

    image_bytes: bytes | None = None
    image_content_type: str | None = None
    if validated.phase == "image":
        if image_path is None:
            raise ValidationError("image adjudication requires image_path")
        supplied_image = Path(image_path)
        if supplied_image.name != validated.evidence["image_file"]:
            raise ValidationError("image_path filename does not match packet image_file")
        image_bytes, image_content_type = _read_private_image(supplied_image)
        import hashlib

        if not secrets.compare_digest(
            hashlib.sha256(image_bytes).hexdigest(),
            validated.evidence["image_sha256"],
        ):
            raise ValidationError("adjudication image bytes do not match the packet SHA-256")
    elif image_path is not None:
        raise ValidationError("caption adjudication must not receive image_path")

    submit_lock = asyncio.Lock()
    submitted_event = asyncio.Event()
    accepted_decision: dict[str, Any] | None = None

    @web.middleware
    async def guard(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        try:
            host = _hostname_from_host_header(request.headers.get("Host", ""))
            if host is None or host.lower() not in normalized_hosts:
                raise web.HTTPBadRequest(text="invalid loopback Host header")
            peer = request.transport.get_extra_info("peername") if request.transport else None
            if peer:
                peer_host = str(peer[0])
                if not _is_loopback(peer_host):
                    raise web.HTTPForbidden(text="adjudication is loopback-only")
            response = await handler(request)
        except web.HTTPException as error:
            for key, value in security_headers().items():
                error.headers[key] = value
            raise
        for key, value in security_headers().items():
            response.headers[key] = value
        return response

    app = web.Application(
        middlewares=[guard],
        client_max_size=_MAX_REQUEST_BYTES,
    )

    def authorized(request: web.Request) -> bool:
        supplied = request.cookies.get(_SESSION_COOKIE)
        return isinstance(supplied, str) and secrets.compare_digest(
            supplied,
            token,
        )

    async def page(request: web.Request) -> web.StreamResponse:
        query_token = request.query.get("token")
        if query_token is not None:
            if set(request.query) != {"token"} or not secrets.compare_digest(
                query_token,
                token,
            ):
                raise web.HTTPUnauthorized(text="invalid adjudication bootstrap")
            response = web.Response(status=302, headers={"Location": "/"})
            response.set_cookie(
                _SESSION_COOKIE,
                token,
                httponly=True,
                samesite="Strict",
                path="/",
            )
            response.set_cookie(
                _CSRF_COOKIE,
                csrf_token,
                httponly=False,
                samesite="Strict",
                path="/",
            )
            return response
        if request.query or not authorized(request):
            raise web.HTTPUnauthorized(text="adjudication session required")
        return web.Response(body=html_bytes, content_type="text/html", charset="utf-8")

    async def image(request: web.Request) -> web.StreamResponse:
        if request.query or not authorized(request):
            raise web.HTTPUnauthorized(text="adjudication session required")
        assert image_bytes is not None
        assert image_content_type is not None
        return web.Response(body=image_bytes, content_type=image_content_type)

    async def decision(request: web.Request) -> web.StreamResponse:
        nonlocal accepted_decision
        if request.query or not authorized(request):
            raise web.HTTPUnauthorized(text="adjudication session required")
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType(text="decision requires application/json")
        supplied_csrf = request.headers.get(_CSRF_HEADER, "")
        if not secrets.compare_digest(supplied_csrf, csrf_token):
            raise web.HTTPForbidden(text="invalid adjudication CSRF token")
        host = request.headers.get("Host", "")
        origin = request.headers.get("Origin", "")
        parsed_origin = urlsplit(origin)
        if (
            parsed_origin.scheme != "http"
            or parsed_origin.netloc != host
            or parsed_origin.hostname is None
            or parsed_origin.hostname.lower() not in normalized_hosts
        ):
            raise web.HTTPForbidden(text="invalid adjudication Origin")
        fetch_site = request.headers.get("Sec-Fetch-Site")
        if fetch_site not in {None, "same-origin"}:
            raise web.HTTPForbidden(text="cross-site adjudication is forbidden")
        raw = await request.read()
        try:
            payload = _strict_json(raw)
            if not isinstance(payload, Mapping) or set(payload) != {"decision"}:
                raise ValidationError("browser request must contain exactly {decision: {...}}")
            normalized = normalize_adjudication_decision(
                validated,
                payload["decision"],
            )
            async with submit_lock:
                if accepted_decision is not None:
                    if accepted_decision != normalized:
                        raise ConflictError("this one-packet session already accepted a different decision")
                    return web.json_response({"ok": True, "idempotent": True})
                result = on_submit(dict(normalized))
                if inspect.isawaitable(result):
                    await result
                accepted_decision = dict(normalized)
                submitted_event.set()
            return web.json_response({"ok": True, "idempotent": False})
        except ConflictError as error:
            return web.json_response({"error": str(error)}, status=409)
        except AuthenticationError as error:
            return web.json_response({"error": str(error)}, status=401)
        except (HumanCBUStoreError, ValueError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)

    app.router.add_get("/", page)
    if validated.phase == "image":
        app.router.add_get("/image", image)
    app.router.add_post("/decision", decision)
    app[_SUBMITTED_EVENT_KEY] = submitted_event
    return app, token


@dataclass
class RunningAdjudicationServer:
    """Handle for one running loopback adjudication server."""

    runner: web.AppRunner
    origin: str
    url: str
    session_token: str
    submitted_event: asyncio.Event

    async def wait_submitted(self) -> None:
        """Wait until this one-packet session records a decision."""

        await self.submitted_event.wait()

    async def close(self) -> None:
        await self.runner.cleanup()

    async def __aenter__(self) -> RunningAdjudicationServer:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        await self.close()


async def start_adjudication_server(
    packet: Mapping[str, Any],
    *,
    on_submit: Callable[[dict[str, Any]], Any | Awaitable[Any]],
    image_path: str | Path | None = None,
    semantic_categories: Sequence[str] = DEFAULT_SEMANTIC_CATEGORIES,
    review_html: bytes | None = None,
    allowed_hosts: Sequence[str] = tuple(DEFAULT_LOOPBACK_HOSTS),
    session_token: str | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
) -> RunningAdjudicationServer:
    """Bind one packet app to loopback and return its one-launch URL."""

    normalized_host = host.strip("[]").lower()
    if not _is_loopback(normalized_host):
        raise ValidationError("adjudication server host must be loopback")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValidationError("adjudication server port is invalid")
    host_allowlist = set(allowed_hosts)
    host_allowlist.add(normalized_host)
    app, token = create_adjudication_app(
        packet,
        on_submit=on_submit,
        image_path=image_path,
        semantic_categories=semantic_categories,
        review_html=review_html,
        allowed_hosts=tuple(host_allowlist),
        session_token=session_token,
    )
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    try:
        await site.start()
        server = site._server
        if server is None or not server.sockets:
            raise RuntimeError("adjudication loopback server has no listening socket")
        bound_port = int(server.sockets[0].getsockname()[1])
    except Exception:
        await runner.cleanup()
        raise
    display_host = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    origin = f"http://{display_host}:{bound_port}"
    return RunningAdjudicationServer(
        runner=runner,
        origin=origin,
        url=f"{origin}/?token={quote(token, safe='')}",
        session_token=token,
        submitted_event=app[_SUBMITTED_EVENT_KEY],
    )
