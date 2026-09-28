from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer

from audit_recap_t2i.human_cbu.adjudication import (
    create_adjudication_app,
    render_adjudication_html,
    render_adjudication_html_bytes,
    security_headers,
    validate_adjudication_packet,
)
from audit_recap_t2i.human_cbu.builder import SEMANTIC_VISUAL_CLAIM_TYPES
from audit_recap_t2i.human_cbu.store import (
    ADJUDICATION_PACKET_VERSION,
    ValidationError,
)

_SESSION_COOKIE = "human_cbu_adjudication_session"
_CSRF_COOKIE = "human_cbu_adjudication_csrf"
_CSRF_HEADER = "X-Human-CBU-CSRF"
_HOST = "localhost"
_ORIGIN = "http://localhost"
_TOKEN_A = "a" * 48
_TOKEN_B = "b" * 48


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _packet(
    phase: str,
    *,
    identity: str = "b",
    unit: str = "a red cat",
    target: str | None = "cat",
    category: str = "attribute",
    caption: str = "A red cat sits on a mat.",
    span: str | None = "red cat",
    image_bytes: bytes = b"\x89PNG\r\npacket-image",
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "unit": unit,
        "target": target,
        "category": category,
    }
    if phase == "caption":
        evidence.update({"caption": caption, "span": span})
    elif phase == "image":
        evidence.update(
            {
                "image_file": "image.png",
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            }
        )
    else:
        raise AssertionError(f"unsupported test phase: {phase}")
    projection = {
        "packet_version": ADJUDICATION_PACKET_VERSION,
        "packet_id": f"ap_{identity * 64}",
        "phase": phase,
        "evidence": evidence,
    }
    return {
        **projection,
        "packet_sha256": hashlib.sha256(_canonical_json(projection)).hexdigest(),
    }


async def _client_for(app: Any) -> TestClient:
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def _bootstrap(
    client: TestClient,
    token: str,
    *,
    host: str = _HOST,
) -> str:
    response = await client.get(
        f"/?token={token}",
        headers={"Host": host},
        allow_redirects=False,
    )
    assert response.status == 302
    assert response.headers["Location"] == "/"
    assert response.cookies[_SESSION_COOKIE].value == token
    csrf = response.cookies[_CSRF_COOKIE].value
    assert len(csrf) >= 32
    return csrf


def _decision_headers(
    csrf: str,
    *,
    host: str = _HOST,
    origin: str = _ORIGIN,
) -> dict[str, str]:
    return {
        "Host": host,
        "Origin": origin,
        _CSRF_HEADER: csrf,
    }


def _assert_security_headers(response: Any) -> None:
    for field, expected in security_headers().items():
        assert response.headers[field] == expected


def test_caption_renderer_is_deterministic_escaped_phase_blinded_and_responsive() -> None:
    malicious = '<script>alert("packet")</script>'
    caption = f'Prelude <b>& "quoted"</b>; {malicious}; suffix.'
    packet = _packet(
        "caption",
        unit=f"unit {malicious} & more",
        target='<img src=x onerror="alert(2)">',
        caption=caption,
        span=malicious,
    )

    first = render_adjudication_html(packet)
    second = render_adjudication_html(packet)

    assert first == second
    assert render_adjudication_html_bytes(packet) == first.encode("utf-8")
    assert malicious not in first
    assert "&lt;script&gt;alert(&quot;packet&quot;)&lt;/script&gt;" in first
    assert "<mark>&lt;script&gt;alert(&quot;packet&quot;)&lt;/script&gt;</mark>" in first
    assert "&lt;img src=x onerror=&quot;alert(2)&quot;&gt;" in first

    assert "Image withheld" in first
    assert "Caption withheld" not in first
    assert 'id="evidence-image"' not in first
    assert 'data-file-src="image.' not in first
    assert "image_support" not in first
    assert first.count('<input type="radio" name="caption_licensed"') == 3
    assert first.count('<input type="radio" name="atomic_visual_claim"') == 4
    assert first.count('<input type="radio" name="category_check"') == 3

    assert '<option value="attribute">' not in first
    for alternative in set(SEMANTIC_VISUAL_CLAIM_TYPES) - {"attribute"}:
        assert f'<option value="{alternative}">' in first

    assert '<meta name="viewport" content="width=device-width,initial-scale=1">' in first
    assert "@media(max-width:840px)" in first
    assert 'class="workspace"' in first
    assert 'class="actionbar"' in first
    assert 'aria-label="Caption claim"' in first
    assert 'id="decision-form"' in first
    assert 'fetch("/decision"' in first
    assert "<link " not in first
    assert 'src="http' not in first


def test_image_renderer_is_deterministic_and_withholds_caption_surface() -> None:
    packet = _packet(
        "image",
        unit='a sign reading <FREE> & "NOW"',
        target="<sign>",
        category="text_rendering",
    )

    first = render_adjudication_html(packet)
    second = render_adjudication_html(packet)

    assert first == second
    assert render_adjudication_html_bytes(packet) == first.encode("utf-8")
    assert "Caption withheld" in first
    assert "Image withheld" not in first
    assert "A red cat sits on a mat." not in first
    assert "caption_licensed" not in first
    assert "atomic_visual_claim" not in first
    assert "category_check" not in first
    assert "corrected-category" not in first
    assert 'id="evidence-image"' in first
    assert 'data-file-src="image.png"' in first
    assert first.count('<input type="radio" name="image_support"') == 6
    assert "control_usefulness" not in first
    assert "&lt;FREE&gt; &amp; &quot;NOW&quot;" in first
    assert "@media(max-width:840px)" in first
    assert 'aria-label="Image evidence"' in first


@pytest.mark.parametrize(
    "mutate",
    [
        lambda packet: packet.update({"item_id": "private-item"}),
        lambda packet: packet["evidence"].update({"surface": "ours"}),
        lambda packet: packet["evidence"].update({"image_file": "image.png"}),
        lambda packet: packet.update({"packet_sha256": "0" * 64}),
    ],
)
def test_packet_schema_rejects_identity_fields_cross_phase_fields_and_digest_tampering(
    mutate: Any,
) -> None:
    packet = _packet("caption")
    mutate(packet)

    with pytest.raises(ValidationError):
        validate_adjudication_packet(packet)


def test_app_rejects_cross_phase_assets_non_loopback_hosts_and_changed_review(
    tmp_path: Path,
) -> None:
    caption_packet = _packet("caption")
    image = tmp_path / "image.png"
    image.write_bytes(b"\x89PNG\r\npacket-image")
    image.chmod(0o600)

    with pytest.raises(ValidationError, match="must not receive image_path"):
        create_adjudication_app(
            caption_packet,
            on_submit=lambda decision: None,
            image_path=image,
        )
    with pytest.raises(ValidationError, match="loopback-only"):
        create_adjudication_app(
            caption_packet,
            on_submit=lambda decision: None,
            allowed_hosts=("evil.example",),
        )
    with pytest.raises(ValidationError, match="does not deterministically match"):
        create_adjudication_app(
            caption_packet,
            on_submit=lambda decision: None,
            review_html=b"substituted review",
        )
    with pytest.raises(ValidationError, match="requires image_path"):
        create_adjudication_app(
            _packet("image"),
            on_submit=lambda decision: None,
        )


def test_caption_server_binds_one_packet_and_accepts_decision_only_with_exact_retry() -> None:
    async def scenario() -> None:
        packet_a = _packet("caption", identity="a")
        packet_b = _packet("caption", identity="b")
        callback_calls: list[tuple[str, dict[str, Any]]] = []

        def submit(decision: dict[str, Any]) -> None:
            callback_calls.append((packet_b["packet_id"], decision))

        app, token = create_adjudication_app(
            packet_b,
            on_submit=submit,
            session_token=_TOKEN_B,
        )
        assert token == _TOKEN_B
        client = await _client_for(app)
        try:
            response = await client.get("/", headers={"Host": _HOST})
            assert response.status == 401
            _assert_security_headers(response)

            response = await client.get(
                f"/?token={_TOKEN_A}",
                headers={"Host": _HOST},
                allow_redirects=False,
            )
            assert response.status == 401
            _assert_security_headers(response)
            response = await client.get(
                f"/?token={_TOKEN_B}&packet_id={packet_a['packet_id']}",
                headers={"Host": _HOST},
                allow_redirects=False,
            )
            assert response.status == 401
            _assert_security_headers(response)

            csrf = await _bootstrap(client, token)
            response = await client.get("/", headers={"Host": _HOST})
            assert response.status == 200
            page_bytes = await response.read()
            assert page_bytes == render_adjudication_html_bytes(packet_b)
            assert token.encode() not in page_bytes
            assert packet_b["packet_id"].encode() not in page_bytes
            _assert_security_headers(response)

            response = await client.get("/image", headers={"Host": _HOST})
            assert response.status == 404
            _assert_security_headers(response)

            valid = {
                "caption_licensed": "yes",
                "atomic_visual_claim": "yes",
                "category_check": "correct",
            }
            identity_bodies = [
                {"packet_id": packet_a["packet_id"], "decision": valid},
                {"item_id": "private-item", "decision": valid},
                {"decision": {**valid, "packet_id": packet_a["packet_id"]}},
                {"decision": {**valid, "adjudicator_pseudonym": "human-01"}},
            ]
            for body in identity_bodies:
                response = await client.post(
                    "/decision",
                    headers=_decision_headers(csrf),
                    json=body,
                )
                assert response.status == 400
                _assert_security_headers(response)
                assert callback_calls == []

            response = await client.post(
                "/decision",
                headers=_decision_headers(csrf),
                json={
                    "decision": {
                        **valid,
                        "category_check": "incorrect",
                        "corrected_category": "attribute",
                    }
                },
            )
            assert response.status == 400
            _assert_security_headers(response)
            assert callback_calls == []

            response = await client.post(
                "/decision",
                headers=_decision_headers(csrf),
                json={"decision": valid},
            )
            assert response.status == 200
            assert await response.json() == {"ok": True, "idempotent": False}
            _assert_security_headers(response)
            assert callback_calls == [(packet_b["packet_id"], valid)]
            assert set(callback_calls[0][1]) == {
                "caption_licensed",
                "atomic_visual_claim",
                "category_check",
            }

            response = await client.post(
                "/decision",
                headers=_decision_headers(csrf),
                json={"decision": copy.deepcopy(valid)},
            )
            assert response.status == 200
            assert await response.json() == {"ok": True, "idempotent": True}
            assert callback_calls == [(packet_b["packet_id"], valid)]

            changed = {**valid, "caption_licensed": "no"}
            response = await client.post(
                "/decision",
                headers=_decision_headers(csrf),
                json={"decision": changed},
            )
            assert response.status == 409
            _assert_security_headers(response)
            assert callback_calls == [(packet_b["packet_id"], valid)]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_decision_endpoint_enforces_host_origin_csrf_content_type_and_fetch_site() -> None:
    async def scenario() -> None:
        calls: list[Mapping[str, Any]] = []
        app, token = create_adjudication_app(
            _packet("caption"),
            on_submit=calls.append,
            session_token=_TOKEN_B,
        )
        client = await _client_for(app)
        try:
            response = await client.get(
                f"/?token={token}",
                headers={"Host": "evil.example"},
                allow_redirects=False,
            )
            assert response.status == 400
            _assert_security_headers(response)

            csrf = await _bootstrap(client, token)
            decision = {
                "decision": {
                    "caption_licensed": "yes",
                    "atomic_visual_claim": "yes",
                    "category_check": "correct",
                }
            }

            response = await client.post(
                "/decision",
                headers={"Host": _HOST, "Origin": _ORIGIN},
                json=decision,
            )
            assert response.status == 403
            _assert_security_headers(response)

            response = await client.post(
                "/decision",
                headers=_decision_headers(csrf, origin="http://evil.example"),
                json=decision,
            )
            assert response.status == 403
            _assert_security_headers(response)

            response = await client.post(
                "/decision",
                headers={
                    **_decision_headers(csrf),
                    "Sec-Fetch-Site": "cross-site",
                },
                json=decision,
            )
            assert response.status == 403
            _assert_security_headers(response)

            response = await client.post(
                "/decision",
                headers={
                    **_decision_headers(csrf),
                    "Content-Type": "text/plain",
                },
                data=json.dumps(decision),
            )
            assert response.status == 415
            _assert_security_headers(response)

            response = await client.post(
                "/decision",
                headers={
                    **_decision_headers(csrf),
                    "Content-Type": "application/json",
                },
                data='{"decision":{},"decision":{}}',
            )
            assert response.status == 400
            _assert_security_headers(response)
            assert calls == []
        finally:
            await client.close()

    asyncio.run(scenario())


def test_image_endpoint_serves_exact_private_bytes_and_image_decision_has_no_usefulness(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        image_bytes = b"\x89PNG\r\n\x1a\nprivate-packet-image"
        image_path = tmp_path / "image.png"
        image_path.write_bytes(image_bytes)
        image_path.chmod(0o600)
        packet = _packet("image", image_bytes=image_bytes)
        calls: list[dict[str, Any]] = []
        app, token = create_adjudication_app(
            packet,
            on_submit=calls.append,
            image_path=image_path,
            session_token=_TOKEN_B,
        )
        client = await _client_for(app)
        try:
            response = await client.get("/image", headers={"Host": _HOST})
            assert response.status == 401
            _assert_security_headers(response)

            csrf = await _bootstrap(client, token)
            response = await client.get("/image", headers={"Host": _HOST})
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/png"
            assert await response.read() == image_bytes
            _assert_security_headers(response)

            response = await client.post(
                "/decision",
                headers=_decision_headers(csrf),
                json={
                    "decision": {
                        "image_support": "yes",
                        "control_usefulness": 5,
                    }
                },
            )
            assert response.status == 400
            _assert_security_headers(response)
            assert calls == []

            response = await client.post(
                "/decision",
                headers=_decision_headers(csrf),
                json={"decision": {"image_support": "yes"}},
            )
            assert response.status == 200
            assert await response.json() == {"ok": True, "idempotent": False}
            assert calls == [{"image_support": "yes"}]
        finally:
            await client.close()

    asyncio.run(scenario())
