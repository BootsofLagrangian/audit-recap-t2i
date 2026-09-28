from __future__ import annotations

import asyncio
import io
import shutil
import sqlite3
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from audit_recap_t2i.human_cbu.store import (
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    HumanCBUStore,
    NotFoundError,
    StudyStateError,
    ValidationError,
)
from audit_recap_t2i.human_cbu.web import (
    PRESENTATION_IMAGE_PIXEL_BUDGET,
    create_author_practice_app,
    create_preview_app,
    create_self_enrollment_test_app,
    create_study_app,
    presentation_image_response,
)


def _progress(*, current_phase: str = "caption") -> dict[str, Any]:
    return {
        "caption": {"completed": 2, "total": 5, "remaining": 3},
        "image": {"completed": 1, "total": 5, "remaining": 4},
        "current_phase": current_phase,
        "study_id": "private-study",
        "participant_id": "private-participant",
    }


def _participant_information(*, approved_version: str = "consent-v1") -> dict[str, str]:
    return {
        "approved_version": approved_version,
        "duration": "Approximately 45 minutes.",
        "compensation": "Approved compensation sentinel.",
        "data_retention": "Pseudonymous responses are retained for five years.",
        "research_contact": "Research team: research-contact@example.edu",
        "withdrawal_policy": "Stop at any time; saved responses are withdrawn when you use the study control.",
        "information_sheet_url": "https://example.edu/approved-information-sheet",
    }


def _write_test_image(path: Path, *, size: tuple[int, int] = (64, 48)) -> None:
    Image.new("RGB", size, color=(42, 126, 188)).save(path, format="JPEG", quality=96)


def _assert_webp(body: bytes, *, size: tuple[int, int] = (64, 48)) -> None:
    with Image.open(io.BytesIO(body)) as image:
        assert image.format == "WEBP"
        assert image.size == size


@pytest.mark.parametrize(
    ("source_size", "expected_size"),
    [
        ((1_600, 900), (1_024, 576)),
        ((900, 1_600), (576, 1_024)),
        ((320, 200), (320, 200)),
    ],
)
def test_presentation_image_is_bounded_aspect_preserving_webp(
    tmp_path: Path,
    source_size: tuple[int, int],
    expected_size: tuple[int, int],
) -> None:
    async def scenario() -> None:
        image_path = tmp_path / f"{source_size[0]}x{source_size[1]}.jpg"
        _write_test_image(image_path, size=source_size)
        response = await presentation_image_response(image_path)
        assert response.content_type == "image/webp"
        assert isinstance(response.body, bytes)
        with Image.open(io.BytesIO(response.body)) as image:
            assert image.format == "WEBP"
            assert image.size == expected_size
            assert image.width * image.height <= PRESENTATION_IMAGE_PIXEL_BUDGET
            source_ratio = source_size[0] / source_size[1]
            assert image.width / image.height == pytest.approx(source_ratio, rel=0.002)

    asyncio.run(scenario())


def _caption_result() -> dict[str, Any]:
    return {
        "status": "task",
        "phase": "caption",
        "task": {
            "assignment_id": "a_caption",
            "phase": "caption",
            "position": 3,
            "caption": "A red kite flies above a beach.",
            "span": "red kite",
            "unit": "a red kite",
            "target": "kite",
            "category": "attribute",
            "item_id": "private-item",
            "surface": "private-surface",
            "qwen_answer": "yes",
            "gemma_answer": "no",
            "image_asset": {"path": "private.jpg"},
        },
        "progress": _progress(),
        "private_metadata": {"hypothesis": "ours wins"},
    }


class FakeStore:
    def __init__(self) -> None:
        self.result: dict[str, Any] | Exception = {
            "status": "consent_required",
            "required_consent_version": "consent-v1",
            "progress": _progress(),
        }
        self.calls: list[tuple[Any, ...]] = []
        self.asset_ref = "nested/image.jpg"
        self.study_status = "open"
        self.session_scope = "full"
        self.closed = False

    @staticmethod
    def _check_token(token: str) -> None:
        if token != "session-token":
            raise AuthenticationError("SECRET invalid token detail")

    def create_session(self, invite_code: str, *, study_id: str | None = None) -> dict[str, Any]:
        self.calls.append(("create_session", invite_code, study_id))
        if invite_code != "invite-code":
            raise AuthenticationError("SECRET invalid invite detail")
        return {
            "session_token": "session-token",
            "expires_at": "2030-01-01T00:00:00Z",
            "study_id": study_id,
            "participant_id": "private-participant",
            "pseudonym": "private-pseudonym",
            "session_scope": self.session_scope,
        }

    def authenticate_session(self, token: str, *, study_id: str | None = None) -> dict[str, Any]:
        self._check_token(token)
        self.calls.append(("authenticate_session", token, study_id))
        return {
            "study_id": study_id,
            "participant_id": "private-participant",
            "participant_status": "active",
            "consented": True,
            "consent_version": "consent-v1",
            "study_status": self.study_status,
            "session_scope": self.session_scope,
        }

    def authenticate_invite(
        self,
        invite_code: str,
        *,
        study_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(("authenticate_invite", invite_code, study_id))
        if invite_code != "invite-code":
            raise AuthenticationError("SECRET invalid invite detail")
        return {
            "study_id": study_id,
            "participant_id": "private-participant",
            "pseudonym": "private-pseudonym",
        }

    def fetch_next_task(self, token: str, *, study_id: str | None = None) -> dict[str, Any]:
        self._check_token(token)
        self.calls.append(("fetch_next_task", token, study_id))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def record_consent_profile(
        self,
        token: str,
        *,
        study_id: str | None = None,
        consented: bool,
        consent_version: str,
        profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._check_token(token)
        self.calls.append(("record_consent_profile", token, consented, consent_version, profile, study_id))
        self.result = _caption_result()
        return {
            "participant_id": "private-participant",
            "pseudonym": "private-pseudonym",
            "consented": consented,
            "consent_version": consent_version,
            "profile": profile,
            "status": "active" if consented else "declined",
        }

    def get_progress(self, token: str, *, study_id: str | None = None) -> dict[str, Any]:
        self._check_token(token)
        self.calls.append(("get_progress", token, study_id))
        return _progress()

    def save_annotation(
        self,
        token: str,
        assignment_id: str,
        payload: dict[str, Any],
        *,
        study_id: str | None = None,
        elapsed_ms: int | None = None,
    ) -> dict[str, Any]:
        self._check_token(token)
        self.calls.append(("save_annotation", token, assignment_id, payload, elapsed_ms, study_id))
        self.result = {"status": "complete", "progress": _progress(current_phase="complete")}
        return {
            "annotation_id": "private-annotation",
            "assignment_id": assignment_id,
            "item_id": "private-item",
            "phase": "caption",
            "payload": payload,
        }

    def extend_remaining_first_work(
        self,
        token: str,
        *,
        study_id: str | None = None,
    ) -> dict[str, Any]:
        self._check_token(token)
        self.calls.append(("extend_remaining_first_work", token, study_id))
        self.result = _caption_result()
        return {"granted_items": 5}

    def authorized_image_asset(
        self,
        token: str,
        assignment_id: str,
        *,
        study_id: str | None = None,
    ) -> dict[str, Any]:
        self._check_token(token)
        self.calls.append(("authorized_image_asset", token, assignment_id, study_id))
        return {"assignment_id": assignment_id, "asset_ref": self.asset_ref}

    def authorized_coverage_caption(
        self,
        token: str,
        assignment_id: str,
        *,
        study_id: str | None = None,
    ) -> dict[str, Any]:
        self._check_token(token)
        self.calls.append(("authorized_coverage_caption", token, assignment_id, study_id))
        return {
            "assignment_id": assignment_id,
            "caption": "A protected paired caption.",
        }

    def claim_preprovisioned_participant_session(
        self,
        study_id: str,
        invite_codes: tuple[str, ...],
    ) -> dict[str, Any]:
        self.calls.append(
            ("claim_preprovisioned_participant_session", study_id, invite_codes)
        )
        return {
            "participant_code": "hcu_self_issued",
            "session_token": "self-issued-session",
        }

    def close(self) -> None:
        self.closed = True


async def _client_for(app: Any) -> TestClient:
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def _add_open_real_study(
    store: HumanCBUStore,
    study_id: str,
    *,
    item_suffix: str,
) -> list[dict[str, Any]]:
    store.create_study(
        study_id,
        title=f"Human CBU audit {study_id}",
        protocol_version="protocol-v1",
        consent_version="consent-v1",
        metadata={
            "required_labels_per_item": 2,
            "ethics_or_irb_equivalent_determination_id": "TEST-DETERMINATION-001",
            "participant_information": _participant_information(),
        },
    )
    store.add_item(
        study_id,
        item_id=f"private-item-{item_suffix}",
        caption="A red cat sits on a mat.",
        unit="a red cat",
        span=[0, 9],
        target="cat",
        category="attribute",
        surface="ours",
        item_group=f"image-cluster-{item_suffix}",
        image_locator={"url_hash": f"private-url-hash-{item_suffix}"},
        image_asset={"asset_relpath": "image.jpg"},
        qwen_answer={"support": "yes"},
        gemma_answer={"support": "yes"},
    )
    invites = store.create_participant_invites(study_id, count=2)
    store.assign_items(study_id, labels_per_item=2, seed=f"{study_id}-seed")
    store.seal_validation(study_id)
    store.set_study_status(study_id, "open")
    return invites


def _assert_security_headers(response: Any) -> None:
    assert "no-store" in response.headers["Cache-Control"]
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert "img-src 'self' blob:" in response.headers["Content-Security-Policy"]
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


def _assert_hsts(response: Any) -> None:
    assert response.headers["Strict-Transport-Security"] == "max-age=31536000"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_browser_recovers_authenticated_bootstrap_and_focuses_usefulness() -> None:
    app_js = Path(__file__).parents[1] / "src" / "audit_recap_t2i" / "human_cbu" / "static" / "app.js"
    harness = textwrap.dedent(
        r"""
        const assert = require("node:assert/strict");
        const fs = require("node:fs");
        const vm = require("node:vm");

        const originalSource = fs.readFileSync(process.argv[1], "utf8");
        const source = originalSource.replace(/\nbootstrap\(\);\s*$/, "\n");
        assert.notEqual(source, originalSource, "test harness must suppress automatic bootstrap");

        class MockNode {
          constructor(id = "") {
            this.id = id;
            this.listeners = {};
            this.dataset = {};
            this.disabled = false;
            this.textContent = "";
            this.scrollTop = 99;
            this.focused = false;
            this.scrolled = null;
            this.scrollCount = 0;
            this.attributes = {};
            this.classes = new Set();
            this.classList = {
              toggle: (name, force) => {
                if (force === false) this.classes.delete(name);
                else if (force === true || !this.classes.has(name)) this.classes.add(name);
                else this.classes.delete(name);
              },
            };
          }

          addEventListener(type, callback) {
            this.listeners[type] = callback;
          }

          setAttribute(name, value) {
            this.attributes[name] = String(value);
          }

          focus() {
            this.focused = true;
          }

          scrollIntoView(options) {
            this.scrolled = options;
            this.scrollCount += 1;
          }

          scrollTo() {
            this.scrollTop = 0;
          }

          append() {}
        }

        class MockStorage {
          constructor(values = {}) {
            this.values = new Map(Object.entries(values));
          }

          getItem(key) {
            return this.values.has(key) ? this.values.get(key) : null;
          }

          setItem(key, value) {
            this.values.set(key, String(value));
          }

          removeItem(key) {
            this.values.delete(key);
          }
        }

        function loadApp({
          fetchImpl,
          sessionToken = null,
          consentVersion = null,
          participantCode = null,
        }) {
          let html = "";
          let nodes = new Map();
          let tutorialChoiceNodes = [];
          let answerChoiceNodes = [];
          let confirmationCount = 0;
          let promptCount = 0;
          const app = new MockNode("app");
          const tutorialAnnouncer = new MockNode("tutorial-announcer");
          Object.defineProperty(app, "innerHTML", {
            get() {
              return html;
            },
            set(value) {
              html = value;
              nodes = new Map();
              tutorialChoiceNodes = [
                ...html.matchAll(
                  /<button[^>]*data-tutorial-answer="([^"]+)"[^>]*data-value="([^"]+)"[^>]*>/g,
                ),
              ].map((match) => {
                const node = new MockNode(`tutorial:${match[1]}:${match[2]}`);
                node.dataset.tutorialAnswer = match[1];
                node.dataset.value = match[2];
                return node;
              });
              answerChoiceNodes = [
                ...html.matchAll(
                  /<button[^>]*data-answer="([^"]+)"[^>]*data-value="([^"]+)"[^>]*>/g,
                ),
              ].map((match) => {
                const node = new MockNode(`answer:${match[1]}:${match[2]}`);
                node.dataset.answer = match[1];
                node.dataset.value = match[2];
                return node;
              });
            },
          });
          const selectorIsPresent = (selector) => {
            if (selector.startsWith("#")) return html.includes(`id="${selector.slice(1)}"`);
            if (selector.startsWith(".")) return html.includes(selector.slice(1));
            return false;
          };
          app.querySelector = (selector) => {
            if (selector === "#zoom-image") return null;
            if (!selectorIsPresent(selector)) return null;
            if (!nodes.has(selector)) nodes.set(selector, new MockNode(selector));
            return nodes.get(selector);
          };
          app.querySelectorAll = (selector) => {
            if (selector === "[data-tutorial-answer]") return tutorialChoiceNodes;
            if (selector === "[data-answer]") return answerChoiceNodes;
            const tutorialMatch = selector.match(
              /^\[data-tutorial-answer="([^"]+)"\]$/,
            );
            if (tutorialMatch) {
              return tutorialChoiceNodes.filter(
                (node) => node.dataset.tutorialAnswer === tutorialMatch[1],
              );
            }
            const answerMatch = selector.match(/^\[data-answer="([^"]+)"\]$/);
            if (answerMatch) {
              return answerChoiceNodes.filter(
                (node) => node.dataset.answer === answerMatch[1],
              );
            }
            return [];
          };

          const sessionStorage = new MockStorage(
            sessionToken ? {"human-cbu-session": sessionToken} : {},
          );
          const localValues = {};
          if (consentVersion) localValues["human-cbu-consent-version"] = consentVersion;
          if (participantCode) localValues["human-cbu-participant-code-v1"] = participantCode;
          const localStorage = new MockStorage(localValues);
          const context = {
            CSS: {escape(value) { return value; }},
            Headers,
            URL: {createObjectURL() { return "blob:test"; }, revokeObjectURL() {}},
            app,
            console,
            document: {
              querySelector(selector) {
                if (selector === "#app") return app;
                if (selector === "#tutorial-announcer") return tutorialAnnouncer;
                throw new Error(`unexpected document selector: ${selector}`);
              },
            },
            fetch: fetchImpl,
            localStorage,
            performance: {now() { return 100; }},
            requestAnimationFrame(callback) {
              callback();
              return 1;
            },
            sessionStorage,
              window: {
                alert(message) {
                  throw new Error(`unexpected alert: ${message}`);
                },
                confirm() {
                  confirmationCount += 1;
                  return true;
                },
                prompt() {
                  promptCount += 1;
                  return participantCode || "invite-code";
                },
                scrollTo() {},
              },
          };
          vm.createContext(context);
          vm.runInContext(source, context, {filename: process.argv[1]});
          return {
            app,
            context,
            getNode(selector) {
              return app.querySelector(selector);
            },
            getTutorialNode(name, value) {
              return tutorialChoiceNodes.find(
                (node) =>
                  node.dataset.tutorialAnswer === name &&
                  node.dataset.value === value,
              );
            },
            getAnswerNode(name, value) {
              return answerChoiceNodes.find(
                (node) =>
                  node.dataset.answer === name &&
                  node.dataset.value === value,
              );
            },
            localStorage,
            get confirmationCount() {
              return confirmationCount;
            },
            get promptCount() {
              return promptCount;
            },
            sessionStorage,
          };
        }

        async function testBootstrapRecovery() {
          const calls = [];
          let bootstrapCalls = 0;
          const environment = loadApp({
            sessionToken: "session-token",
            participantCode: "invite-code",
            async fetchImpl(path, options = {}) {
              calls.push({path, options});
              if (path === "/api/bootstrap") {
                bootstrapCalls += 1;
                if (bootstrapCalls <= 2) throw new Error("offline");
                return {
                  ok: true,
                  status: 200,
                  async json() {
                    return {status: "task", consent_version: "consent-v1"};
                  },
                };
              }
              assert.equal(path, "/api/withdraw");
              return {ok: true, status: 200, async json() { return {}; }};
            },
          });

          assert.equal(environment.localStorage.getItem("human-cbu-consent-version"), null);
          await vm.runInContext("bootstrap()", environment.context);
          assert.match(environment.app.innerHTML, /role="alert"/);
          assert.match(environment.app.innerHTML, /id="retry-bootstrap"/);
          assert.match(environment.app.innerHTML, /id="withdraw-study"/);
          assert.equal(environment.getNode("#bootstrap-error").focused, true);

          const retry = environment.getNode("#retry-bootstrap");
          assert.equal(typeof retry.listeners.click, "function");
          retry.listeners.click({currentTarget: retry});
          await new Promise((resolve) => setImmediate(resolve));
          await new Promise((resolve) => setImmediate(resolve));
          assert.equal(calls.filter((call) => call.path === "/api/bootstrap").length, 2);
          assert.match(environment.app.innerHTML, /id="withdraw-study"/);

          const withdraw = environment.getNode("#withdraw-study");
          assert.equal(typeof withdraw.listeners.click, "function");
          await withdraw.listeners.click();
          const withdrawalCall = calls.find((call) => call.path === "/api/withdraw");
          assert.ok(withdrawalCall);
          assert.equal(bootstrapCalls, 3);
          assert.equal(withdrawalCall.options.headers.get("Authorization"), "Bearer session-token");
          assert.equal(JSON.parse(withdrawalCall.options.body).consent_version, "consent-v1");
          assert.equal(JSON.parse(withdrawalCall.options.body).participant_code, "invite-code");
          assert.equal(environment.confirmationCount, 2);
          assert.equal(environment.promptCount, 1);
          assert.match(environment.app.innerHTML, /Withdrawal recorded/);
          assert.equal(environment.sessionStorage.getItem("human-cbu-session"), null);
          assert.equal(environment.localStorage.getItem("human-cbu-consent-version"), null);
        }

        function testPauseKeepsParticipantCodeAndMakesNoRequest() {
          let fetchCount = 0;
          const environment = loadApp({
            sessionToken: "session-token",
            participantCode: "invite-code",
            async fetchImpl() {
              fetchCount += 1;
              throw new Error("pause must not call the server");
            },
          });
          vm.runInContext("renderBootstrapError()", environment.context);
          const pause = environment.getNode("#pause-study");
          assert.ok(pause);
          assert.equal(typeof pause.listeners.click, "function");
          pause.listeners.click();
          assert.equal(fetchCount, 0);
          assert.equal(environment.sessionStorage.getItem("human-cbu-session"), null);
          assert.equal(
            environment.localStorage.getItem("human-cbu-participant-code-v1"),
            "invite-code",
          );
          assert.match(environment.app.innerHTML, /Paused safely/);
          assert.match(environment.app.innerHTML, /Continue on this device/);
        }

        async function testExpiredWithdrawalSession() {
          const environment = loadApp({
            sessionToken: "expired-session",
            consentVersion: "consent-v1",
            participantCode: "invite-code",
            async fetchImpl(path) {
              assert.equal(path, "/api/withdraw");
              return {
                ok: false,
                status: 401,
                async json() {
                  return {error: "Authentication is required."};
                },
              };
            },
          });
          vm.runInContext("renderBootstrapError()", environment.context);
          const withdraw = environment.getNode("#withdraw-study");
          assert.equal(typeof withdraw.listeners.click, "function");
          await withdraw.listeners.click();

          assert.equal(vm.runInContext("state.token", environment.context), null);
          assert.equal(vm.runInContext("state.withdrawing", environment.context), false);
          assert.equal(environment.sessionStorage.getItem("human-cbu-session"), null);
          assert.equal(
            environment.localStorage.getItem("human-cbu-consent-version"),
            "consent-v1",
          );
          assert.match(environment.app.innerHTML, /id="login-form"/);
          assert.match(
            environment.app.innerHTML,
            /session expired before withdrawal was recorded/,
          );
          assert.match(environment.app.innerHTML, /role="alert"/);
          assert.equal(
            typeof environment.getNode("#login-form").listeners.submit,
            "function",
          );
        }

            async function testUsefulnessFocus() {
              const environment = loadApp({
                async fetchImpl(path) {
                  assert.equal(path, "/api/coverage-caption/image-assignment");
                  return {
                    ok: true,
                    status: 200,
                    async json() {
                      return {caption: "A visible phoenix fills the frame."};
                    },
                  };
                },
              });
          await vm.runInContext(
            `(() => {
              state.task = {
                assignment_id: "image-assignment",
                phase: "image",
                unit: "a visible phoenix",
                target: "phoenix",
                category: "object",
                image_url: null,
              };
              state.bootstrap = {progress: {completed: 1, total: 2}};
              state.answers = {image_support: "yes"};
              state.imagePrimaryLocked = false;
              state.imageRevealed = true;
              state.startedAt = 1;
              return saveAndContinue();
            })()`,
            environment.context,
          );
              const question = environment.getNode("#coverage-question");
              assert.ok(question);
              assert.equal(question.focused, true);
              assert.equal(question.scrolled.block, "center");
              assert.match(environment.app.innerHTML, /Paired caption · revealed after support was locked/);
              assert.match(environment.app.innerHTML, /A visible phoenix fills the frame/);
              assert.match(environment.app.innerHTML, /id="coverage-question" tabindex="-1"/);
              assert.match(environment.app.innerHTML, /role="region" aria-labelledby="usefulness-question"/);
              assert.match(environment.app.innerHTML, /id="usefulness-question" tabindex="-1"/);
              assert.equal(vm.runInContext("state.imagePrimaryLocked", environment.context), true);
        }

        function testCorrectionChoicesExcludeProposedType() {
          const environment = loadApp({
            async fetchImpl() {
              throw new Error("fetch must not run while rendering correction choices");
            },
          });
          const markup = vm.runInContext(
            `captionQuestions({category: "attribute"})`,
            environment.context,
          );
          assert.doesNotMatch(markup, /<option value="attribute"/);
          assert.match(markup, /<option value="object"/);
          const proposedIsIncomplete = vm.runInContext(
            `(() => {
              state.answers = {
                caption_licensed: "yes",
                atomic_visual_claim: "yes",
                category_check: "incorrect",
                corrected_category: "attribute",
              };
              return responseComplete({phase: "caption", category: "attribute"});
            })()`,
            environment.context,
          );
          assert.equal(proposedIsIncomplete, false);
          const alternativeIsComplete = vm.runInContext(
            `(() => {
              state.answers.corrected_category = "object";
              return responseComplete({phase: "caption", category: "attribute"});
            })()`,
            environment.context,
          );
          assert.equal(alternativeIsComplete, true);
        }

        function testCaptionCategoryAnswerSurvivesTemporaryQuestionTwoSkip() {
          const environment = loadApp({
            async fetchImpl() {
              throw new Error("fetch must not run during caption answer changes");
            },
          });
          vm.runInContext(
            `renderTask({
              assignment_id: "caption-state-regression",
              phase: "caption",
              caption: "A red sphere floats above a platform.",
              span: "red",
              unit: "red",
              target: "sphere",
              category: "attribute"
            }, {completed: 0, total: 3})`,
            environment.context,
          );
          const captionYes = environment.getAnswerNode("caption_licensed", "yes");
          const captionNo = environment.getAnswerNode("caption_licensed", "no");
          const detailYes = environment.getAnswerNode("atomic_visual_claim", "yes");
          const typeYes = environment.getAnswerNode("category_check", "correct");
          const detailNotVisual = environment.getAnswerNode(
            "atomic_visual_claim",
            "not_visual",
          );
          for (const button of [
            captionYes,
            captionNo,
            detailYes,
            typeYes,
            detailNotVisual,
          ]) {
            assert.ok(button);
            assert.equal(typeof button.listeners.click, "function");
          }

          captionYes.listeners.click();
          assert.equal(environment.getNode("#atomic-claim-question").scrolled.block, "center");
          detailYes.listeners.click();
          assert.equal(environment.getNode("#category-check-question").scrolled.block, "center");
          typeYes.listeners.click();
          assert.equal(environment.getNode("#save-next").disabled, false);
          assert.equal(environment.getNode("#save-next").scrolled.block, "nearest");
          assert.equal(environment.getNode("#save-next").focused, false);
          const q2ScrollCount = environment.getNode("#atomic-claim-question").scrollCount;
          captionNo.listeners.click();
          assert.equal(
            environment.getNode("#atomic-claim-question").scrollCount,
            q2ScrollCount,
            "revising Q1 must not force the participant forward again",
          );
          captionYes.listeners.click();

          const saveScrollCount = environment.getNode("#save-next").scrollCount;
          detailNotVisual.listeners.click();
          assert.equal(
            vm.runInContext("state.answers.category_check", environment.context),
            "not_applicable",
          );
          assert.equal(environment.getNode("#save-next").disabled, false);
          assert.equal(
            environment.getNode("#save-next").scrollCount,
            saveScrollCount,
            "revising Q2 must not auto-scroll to Save",
          );

          detailYes.listeners.click();
          assert.equal(
            vm.runInContext("state.answers.category_check", environment.context),
            "correct",
          );
          assert.equal(environment.getNode("#save-next").disabled, false);
          assert.equal(typeYes.attributes["aria-pressed"], "true");

          const detailSeveral = environment.getAnswerNode(
            "atomic_visual_claim",
            "not_atomic",
          );
          detailSeveral.listeners.click();
          assert.equal(
            vm.runInContext("state.answers.category_check", environment.context),
            "not_applicable",
          );
          detailYes.listeners.click();
          assert.equal(
            vm.runInContext("state.answers.category_check", environment.context),
            "correct",
          );

          const rerenderedSkipped = vm.runInContext(
            `(() => {
              state.answers.category_check = "not_applicable";
              return captionQuestions(state.task);
            })()`,
            environment.context,
          );
          assert.match(rerenderedSkipped, /class="category-active hidden"/);
          assert.match(rerenderedSkipped, /class="category-skipped " id="category-skipped"/);
        }

        function testIncorrectCategoryDraftRestoresWithoutStealingSelectFocus() {
          const environment = loadApp({
            async fetchImpl() {
              throw new Error("fetch must not run during corrected-category changes");
            },
          });
          vm.runInContext(
            `renderTask({
              assignment_id: "caption-correction-regression",
              phase: "caption",
              caption: "A red sphere floats above a platform.",
              span: "red",
              unit: "red",
              target: "sphere",
              category: "attribute"
            }, {completed: 0, total: 3})`,
            environment.context,
          );
          environment.getAnswerNode("caption_licensed", "yes").listeners.click();
          const detailYes = environment.getAnswerNode("atomic_visual_claim", "yes");
          detailYes.listeners.click();
          environment.getAnswerNode("category_check", "incorrect").listeners.click();

          const corrected = environment.getNode("#corrected-category");
          assert.equal(corrected.focused, true);
          corrected.value = "lighting";
          corrected.listeners.change({currentTarget: corrected});
          assert.equal(
            vm.runInContext("state.answers.corrected_category", environment.context),
            "lighting",
          );
          assert.equal(environment.getNode("#save-next").disabled, false);
          assert.equal(environment.getNode("#save-next").focused, false);
          assert.equal(environment.getNode("#save-next").scrolled.block, "nearest");

          environment
            .getAnswerNode("atomic_visual_claim", "not_atomic")
            .listeners.click();
          detailYes.listeners.click();
          assert.equal(
            vm.runInContext("state.answers.category_check", environment.context),
            "incorrect",
          );
          assert.equal(
            vm.runInContext("state.answers.corrected_category", environment.context),
            "lighting",
          );
          assert.equal(corrected.value, "lighting");
        }

        function testDistributedLexicalCueIsExplicitAndEscaped() {
          const environment = loadApp({
            async fetchImpl() {
              throw new Error("fetch must not run while highlighting caption evidence");
            },
          });
          const highlighted = JSON.parse(
            vm.runInContext(
              `JSON.stringify(highlightSpan(
                "A weathered concrete ceiling and floor <remain visible>.",
                null,
                "concrete floor"
              ))`,
              environment.context,
            ),
          );
          assert.equal(highlighted.mode, "lexical");
          assert.match(highlighted.html, /<mark class="lexical-cue">concrete<\/mark>/);
          assert.match(highlighted.html, /<mark class="lexical-cue">floor<\/mark>/);
          assert.match(highlighted.html, /&lt;remain visible&gt;/);

          const extractorSpanWins = JSON.parse(
            vm.runInContext(
              `JSON.stringify(highlightSpan("A white wall.", "white", "white wall"))`,
              environment.context,
            ),
          );
          assert.equal(extractorSpanWins.mode, "extractor");
          assert.match(extractorSpanWins.html, /<mark>white<\/mark>/);
          assert.doesNotMatch(extractorSpanWins.html, /lexical-cue/);

          const tooFarApart = JSON.parse(
            vm.runInContext(
              `JSON.stringify(highlightSpan(
                "concrete " + "very ".repeat(40) + "floor",
                null,
                "concrete floor"
              ))`,
              environment.context,
            ),
          );
          assert.equal(tooFarApart.mode, "none");
          assert.doesNotMatch(tooFarApart.html, /<mark/);
        }

        function testUniversalImageCurtainPreventsEagerImageMarkup() {
          const environment = loadApp({
            async fetchImpl() {
              throw new Error("fetch must not run while rendering the image curtain");
            },
          });
          const task = `({
            assignment_id: "image-assignment",
            phase: "image",
            unit: "four",
            target: "women",
            category: "count",
            image_url: "/api/image/image-assignment"
          })`;
          const covered = vm.runInContext(
            `state.imageRevealed = false; imageEvidence(${task})`,
            environment.context,
          );
          assert.match(covered, /id="reveal-image"/);
          assert.match(covered, /id="skip-image-unseen"/);
          assert.match(covered, /does not describe or classify this particular image/);
          assert.doesNotMatch(covered, /data-protected-image/);
          assert.doesNotMatch(covered, /api\/image/);

          const revealed = vm.runInContext(
            `state.imageRevealed = true; imageEvidence(${task})`,
            environment.context,
          );
          assert.match(revealed, /data-protected-image/);
          assert.match(revealed, /id="zoom-image"/);
          assert.doesNotMatch(revealed, /id="reveal-image"/);
        }

        function testFirstTaskIsGatedByInterfaceTutorial() {
          const environment = loadApp({
            async fetchImpl() {
              throw new Error("fetch must not run while rendering the tutorial");
            },
          });
          const result = `({
            status: "task",
            practice_mode: true,
            progress: {completed: 0, total: 10},
            task: {
              assignment_id: "first-caption",
              phase: "caption",
              caption: "Private first caption.",
              span: "Private",
              unit: "private unit",
              target: "target",
              category: "attribute"
            }
          })`;
          vm.runInContext(`applyStudyResult(${result})`, environment.context);
          assert.match(environment.app.innerHTML, /YOU ARE IN THE TUTORIAL/);
          assert.match(environment.app.innerHTML, /What are we asking you to judge/);
          assert.match(environment.app.innerHTML, /Caption/);
          assert.match(environment.app.innerHTML, /Claim/);
          assert.match(environment.app.innerHTML, /CBU/);
          assert.match(environment.app.innerHTML, /one visual detail/);
          assert.doesNotMatch(environment.app.innerHTML, /Private first caption/);

          environment.sessionStorage.setItem(
            "human-cbu-interface-tutorial-v2",
            "complete",
          );
          vm.runInContext(`applyStudyResult(${result})`, environment.context);
          assert.match(environment.app.innerHTML, /first caption/);
          assert.match(environment.app.innerHTML, /private unit/);
          assert.match(environment.app.innerHTML, /id="open-guide"/);
          assert.doesNotMatch(environment.app.innerHTML, /not atomic|atomicity/i);
          assert.doesNotMatch(environment.app.innerHTML, /CBU validity/i);

          const resumed = loadApp({
            async fetchImpl() {
              throw new Error("fetch must not run while rendering resumed work");
            },
          });
          vm.runInContext(
            `applyStudyResult({
              status: "task",
              progress: {completed: 1, total: 100},
              task: {
                assignment_id: "resumed-caption",
                phase: "caption",
                caption: "A resumed caption.",
                span: "resumed",
                unit: "resumed detail",
                target: "target",
                category: "attribute"
              }
            })`,
            resumed.context,
          );
          assert.match(resumed.app.innerHTML, /resumed detail/);
          assert.doesNotMatch(resumed.app.innerHTML, /YOU ARE IN THE TUTORIAL/);
          assert.equal(
            resumed.sessionStorage.getItem("human-cbu-interface-tutorial-v2"),
            "complete",
          );
        }

        function testTutorialRequiresHandsOnCorrectAnswers() {
          const environment = loadApp({
            async fetchImpl() {
              throw new Error("fetch must not run during synthetic tutorial practice");
            },
          });
          vm.runInContext(
            `(() => {
              state.tutorialTask = {
                assignment_id: "first-caption",
                phase: "caption",
                caption: "Private first caption.",
                unit: "private unit",
                target: "target",
                category: "attribute"
              };
              state.tutorialProgress = {completed: 0, total: 10};
              state.tutorialStep = 0;
              state.tutorialAnswers = {};
              renderTutorial(state.tutorialTask, state.tutorialProgress);
            })()`,
            environment.context,
          );
          assert.match(environment.app.innerHTML, /START HERE/i);
          assert.match(environment.app.innerHTML, /Is “red” one independently checkable visual detail/);
          assert.match(environment.app.innerHTML, /Pulled back in a low ponytail/);
          assert.match(environment.app.innerHTML, /THREE QUICK BOUNDARY CHECKS/);
          assert.match(environment.app.innerHTML, /Legitimate uncertainty/);
          assert.doesNotMatch(environment.app.innerHTML, /Private first caption/);

          vm.runInContext(
            `(() => {
              state.tutorialAnswers = {
                caption: "no", detail: "yes", type: "yes",
                missing: "no", bundle: "not_atomic", ambiguity: "uncertain"
              };
              checkTutorialStep();
            })()`,
            environment.context,
          );
          assert.equal(vm.runInContext("state.tutorialStep", environment.context), 0);
          assert.match(environment.getNode("#tutorial-feedback").textContent, /Try again/);

          vm.runInContext(
            `(() => {
              state.tutorialAnswers = {
                caption: "yes", detail: "yes", type: "yes",
                missing: "no", bundle: "not_atomic", ambiguity: "uncertain"
              };
              checkTutorialStep();
            })()`,
            environment.context,
          );
          assert.equal(vm.runInContext("state.tutorialStep", environment.context), 1);
          assert.match(environment.app.innerHTML, /THE CAPTION IS NOW HIDDEN/);
          assert.match(environment.app.innerHTML, /Does the image visibly support/);
        }

        function testTutorialOneDetailChoicesMatchTheLiveQuestion() {
          const environment = loadApp({
            async fetchImpl() {
              throw new Error("fetch must not run while comparing tutorial labels");
            },
          });
          const practice = vm.runInContext("tutorialCaptionPractice()", environment.context);
          const live = vm.runInContext(
            `captionQuestions({category: "attribute"})`,
            environment.context,
          );
          for (const label of [
            "Yes · one detail",
            "No · not visual",
            "No · several details",
            "Not sure",
          ]) {
            assert.ok(practice.includes(label));
            assert.ok(live.includes(label));
          }
          assert.match(practice, /data-tutorial-answer="bundle" data-value="not_atomic"/);
          assert.match(practice, /data-tutorial-answer="ambiguity" data-value="not_visual"/);
          assert.doesNotMatch(practice, /data-value="several"/);
        }

        function testTutorialPreviewsTheOptionalQuestionAndFixedWindowBoundary() {
          const environment = loadApp({
            async fetchImpl() {
              throw new Error("fetch must not run while checking tutorial parity");
            },
          });
          const coverage = vm.runInContext("tutorialCoveragePractice()", environment.context);
          const evidence = vm.runInContext(
            `captionEvidence({
              caption: "A fixed window that ends",
              span: "fixed",
              unit: "fixed",
              category: "attribute",
              token_budget: 64
            })`,
            environment.context,
          );
          const response = vm.runInContext(
            `responsePanel({
              assignment_id: "fixed-header",
              phase: "caption",
              unit: "phoenix",
              target: "phoenix",
              category: "object"
            })`,
            environment.context,
          );
          assert.match(coverage, /1 required \+ 1 optional/);
          assert.match(coverage, /text-to-image model should draw/);
          assert.match(coverage, /data-tutorial-answer="usefulness" data-value="cannot_judge"/);
          assert.match(evidence, /complete fixed window you are asked to judge/);
          assert.doesNotMatch(evidence, /64-unit window/);
              assert.match(response, /named object such as “phoenix,” “large herd,” or “crowded ensemble”/);
              assert.match(response, /3 main questions/);
              assert.match(response, /one required corrected-type follow-up/);
          assert.ok(
            response.indexOf('class="claim-anchor"') <
              response.indexOf('class="question-map"') &&
              response.indexOf('class="question-map"') <
              response.indexOf('class="response-scroll"'),
          );
        }

        function testCompletedTestHasNoWithdrawalControl() {
          const environment = loadApp({
            async fetchImpl() {
              throw new Error("fetch must not run while rendering completion");
            },
          });
          vm.runInContext(
            `state.testMode = true; state.practiceMode = false; renderComplete({completed: 6})`,
            environment.context,
          );
          assert.match(environment.app.innerHTML, /Interface test complete/);
          assert.doesNotMatch(environment.app.innerHTML, /withdraw-study/);
          assert.ok(!environment.app.innerHTML.includes("Stop / withdraw"));

          vm.runInContext(
            `state.testMode = false; state.practiceMode = true; renderComplete({completed: 6})`,
            environment.context,
          );
          assert.doesNotMatch(environment.app.innerHTML, /withdraw-study/);

          vm.runInContext(
            `state.testMode = false; state.practiceMode = false; renderComplete({completed: 6})`,
            environment.context,
          );
          assert.match(environment.app.innerHTML, /Withdraw my saved responses/);
          assert.ok(!environment.app.innerHTML.includes("Stop / withdraw participation"));

          vm.runInContext(
            `renderComplete({completed: 60}, {can_extend: true, batch_size: 5, max_items: 50})`,
            environment.context,
          );
          assert.match(environment.app.innerHTML, /Want to help a little more/);
          assert.match(environment.app.innerHTML, /id="extend-work"/);
          assert.match(environment.app.innerHTML, /Do 5 more/);
        }

        async function testExpiredTestSessionExplainsVolatileResume() {
          const environment = loadApp({
            sessionToken: "expired-test-session",
            participantCode: "temporary-test-code",
            async fetchImpl(path, options) {
              assert.equal(path, "/api/bootstrap");
              assert.equal(options.headers.get("Authorization"), null);
              return {
                ok: true,
                status: 200,
                async json() {
                  return {
                    requires_login: true,
                    test_mode: true,
                    responses_persisted: false,
                  };
                },
              };
            },
          });
          await vm.runInContext("recoverExpiredBootstrapSession()", environment.context);
          assert.match(environment.app.innerHTML, /Response-discarding test mode/);
          assert.match(environment.app.innerHTML, /Resume temporary test on this device/);
          assert.match(environment.app.innerHTML, /keeps answers and progress in memory/);
          assert.match(environment.app.innerHTML, /none are exported as research data/);
          assert.doesNotMatch(environment.app.innerHTML, /Continue on this device/);

          const production = loadApp({
            sessionToken: "expired-production-session",
            participantCode: "saved-production-code",
            async fetchImpl(path, options) {
              assert.equal(path, "/api/bootstrap");
              assert.equal(options.headers.get("Authorization"), null);
              return {
                ok: true,
                status: 200,
                async json() {
                  return {requires_login: true};
                },
              };
            },
          });
          await vm.runInContext(
            "recoverExpiredBootstrapSession()",
            production.context,
          );
          assert.match(production.app.innerHTML, /Continue on this device/);
          assert.doesNotMatch(production.app.innerHTML, /Response-discarding test mode/);
          assert.doesNotMatch(production.app.innerHTML, /Resume temporary test on this device/);
        }

        function testTutorialClickFlowPreservesPositionAndAcceptsCoverageJudgment() {
          const environment = loadApp({
            async fetchImpl() {
              throw new Error("fetch must not run during synthetic tutorial clicks");
            },
          });
          vm.runInContext(
            `(() => {
              state.tutorialTask = {
                assignment_id: "first-caption",
                phase: "caption",
                caption: "Private first caption.",
                unit: "private unit",
                target: "target",
                category: "attribute"
              };
              state.tutorialProgress = {completed: 0, total: 10};
              state.tutorialStep = 0;
              renderTutorial(state.tutorialTask, state.tutorialProgress);
            })()`,
            environment.context,
          );
          const htmlBeforeClick = environment.app.innerHTML;
          for (const [name, value] of [
            ["caption", "yes"],
            ["detail", "yes"],
            ["type", "yes"],
            ["missing", "no"],
            ["bundle", "not_atomic"],
            ["ambiguity", "uncertain"],
          ]) {
            const button = environment.getTutorialNode(name, value);
            assert.ok(button);
            assert.equal(typeof button.listeners.click, "function");
            button.listeners.click();
            assert.equal(
              environment.app.innerHTML,
              htmlBeforeClick,
              "a selection must not rerender or reset scroll/focus",
            );
          }
          assert.equal(environment.getNode("#tutorial-next").disabled, false);
          const answersBeforeTerms = vm.runInContext(
            "JSON.stringify(state.tutorialAnswers)",
            environment.context,
          );
          environment.getNode("#tutorial-back-to-terms").listeners.click();
          assert.equal(vm.runInContext("state.tutorialStep", environment.context), -1);
          environment.getNode("#tutorial-next").listeners.click();
          assert.equal(vm.runInContext("state.tutorialStep", environment.context), 0);
          assert.equal(
            vm.runInContext("JSON.stringify(state.tutorialAnswers)", environment.context),
            answersBeforeTerms,
          );
          assert.equal(environment.getNode("#tutorial-next").disabled, false);
          environment.getNode("#tutorial-next").listeners.click();
          assert.equal(vm.runInContext("state.tutorialStep", environment.context), 1);

          const support = environment.getTutorialNode("support", "yes");
          support.listeners.click();
          environment.getNode("#tutorial-next").listeners.click();
          assert.equal(vm.runInContext("state.tutorialStep", environment.context), 2);

          const coverage = environment.getTutorialNode("coverage", "3");
          coverage.listeners.click();
          environment.getNode("#tutorial-next").listeners.click();
          assert.equal(vm.runInContext("state.tutorialStep", environment.context), 3);
          assert.match(environment.app.innerHTML, /no single correct number/i);
          environment.getNode("#tutorial-next").listeners.click();
          assert.equal(
            environment.sessionStorage.getItem("human-cbu-interface-tutorial-v2"),
            "complete",
          );
        }

        function testGuideTimeIsExcludedFromElapsedMeasurement() {
          const environment = loadApp({
            async fetchImpl() {
              throw new Error("fetch must not run while measuring local elapsed time");
            },
          });
          const completedGuide = vm.runInContext(
            `(() => {
              state.startedAt = 100;
              state.guideElapsedMs = 30;
              state.guideStartedAt = null;
              return measuredResponseElapsedMs(200);
            })()`,
            environment.context,
          );
          assert.equal(completedGuide, 70);
          const openGuide = vm.runInContext(
            `(() => {
              state.guideElapsedMs = 20;
              state.guideStartedAt = 170;
              return measuredResponseElapsedMs(200);
            })()`,
            environment.context,
          );
          assert.equal(openGuide, 50);
          const clamped = vm.runInContext(
            `(() => {
              state.guideElapsedMs = 500;
              state.guideStartedAt = null;
              return measuredResponseElapsedMs(200);
            })()`,
            environment.context,
          );
          assert.equal(clamped, 0);
        }

        function testConsentCollectsNoProfileQuestionnaire() {
          const environment = loadApp({
            async fetchImpl() {
              throw new Error("fetch must not run while rendering consent");
            },
          });
          vm.runInContext(
            `state.testMode = true; renderConsent("test-only-v1", null)`,
            environment.context,
          );
          assert.match(environment.app.innerHTML, /TEST MODE · NO RESEARCH DATA RETAINED/);
          assert.match(environment.app.innerHTML, /input type="checkbox" name="consented"/);
          assert.doesNotMatch(environment.app.innerHTML, /profile-grid/);
          assert.doesNotMatch(environment.app.innerHTML, /author_status/);
          assert.doesNotMatch(environment.app.innerHTML, /English reading/);
          assert.doesNotMatch(environment.app.innerHTML, /Device/);
        }

        (async () => {
          await testBootstrapRecovery();
          await testExpiredWithdrawalSession();
          testPauseKeepsParticipantCodeAndMakesNoRequest();
          await testUsefulnessFocus();
          testCorrectionChoicesExcludeProposedType();
          testCaptionCategoryAnswerSurvivesTemporaryQuestionTwoSkip();
          testIncorrectCategoryDraftRestoresWithoutStealingSelectFocus();
          testDistributedLexicalCueIsExplicitAndEscaped();
          testUniversalImageCurtainPreventsEagerImageMarkup();
          testFirstTaskIsGatedByInterfaceTutorial();
          testTutorialRequiresHandsOnCorrectAnswers();
          testTutorialOneDetailChoicesMatchTheLiveQuestion();
          testTutorialPreviewsTheOptionalQuestionAndFixedWindowBoundary();
          testCompletedTestHasNoWithdrawalControl();
          await testExpiredTestSessionExplainsVolatileResume();
          testTutorialClickFlowPreservesPositionAndAcceptsCoverageJudgment();
          testGuideTimeIsExcludedFromElapsedMeasurement();
          testConsentCollectsNoProfileQuestionnaire();
        })().catch((error) => {
          console.error(error);
          process.exitCode = 1;
        });
        """
    )
    completed = subprocess.run(
        ["node", "-e", harness, str(app_js)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_preview_uses_production_status_and_task_shape(tmp_path: Path) -> None:
    async def scenario() -> None:
        image = tmp_path / "image.jpg"
        _write_test_image(image)
        app = create_preview_app(
            item={
                "item_id": "must-not-reach-browser",
                "category": "object",
                "unit": "a kite",
                "target": "kite",
                "caption": "A kite flies above the beach.",
                "span": "kite",
                "surface": "private-surface",
            },
            image_path=image,
        )
        client = await _client_for(app)
        try:
            response = await client.get("/api/bootstrap")
            assert response.status == 200
            payload = await response.json()
            assert set(payload) == {"status", "phase", "task", "progress"}
            assert payload["status"] == "task"
            assert payload["task"]["assignment_id"] == "preview-caption"
            assert payload["task"]["span"] == "kite"
            assert "item_id" not in payload["task"]
            assert "surface" not in payload["task"]
            assert set(payload["progress"]) == {
                "completed",
                "total",
                "overall_completed",
                "overall_total",
            }
            _assert_security_headers(response)
            assert "Strict-Transport-Security" not in response.headers

            response = await client.post(
                "/api/annotation",
                json={
                    "assignment_id": "preview-caption",
                    "answers": {"caption_licensed": "yes"},
                    "elapsed_ms": 200,
                },
            )
            payload = await response.json()
            assert payload["task"]["assignment_id"] == "preview-image"
            assert payload["task"]["image_url"] == "/api/image/preview-image"

            response = await client.get("/api/image/preview-image")
            assert response.status == 200
            assert response.content_type == "image/webp"
            _assert_webp(await response.read())
            _assert_security_headers(response)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_participant_test_exercises_self_enrollment_and_discards_responses(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        image = tmp_path / "image.jpg"
        _write_test_image(image)
        app = create_self_enrollment_test_app(
            item={
                "category": "attribute",
                "unit": "a red sphere",
                "target": "sphere",
                "caption": "A red sphere floats above a green platform.",
                "span": "red sphere",
            },
            image_path=image,
            pair_count=1,
        )
        client = await _client_for(app)
        try:
            bootstrap = await client.get("/api/bootstrap")
            assert await bootstrap.json() == {
                "requires_login": True,
                "test_mode": True,
                "responses_persisted": False,
            }

            issued_response = await client.post("/api/enroll", json={})
            issued = await issued_response.json()
            assert issued["code"].startswith("hctest_")
            headers = {"Authorization": f"Bearer {issued['token']}"}

            consent_required = await client.get("/api/bootstrap", headers=headers)
            assert (await consent_required.json())["status"] == "consent_required"
            consent = await client.post(
                "/api/consent",
                headers=headers,
                json={
                    "consented": True,
                    "consent_version": "test-only-v1",
                    "profile": {"ignored": "and not retained"},
                },
            )
            caption = await consent.json()
            assert caption["phase"] == "caption"

            image_task = await client.post(
                "/api/annotation",
                headers=headers,
                json={
                    "assignment_id": caption["task"]["assignment_id"],
                    "answers": {"private": "discard this"},
                    "elapsed_ms": 42,
                },
            )
            image_payload = await image_task.json()
            assert image_payload["phase"] == "image"
            coverage = await client.get(
                f"/api/coverage-caption/{image_payload['task']['assignment_id']}",
                headers=headers,
            )
            assert await coverage.json() == {
                "caption": "A red sphere floats above a green platform."
            }

            complete = await client.post(
                "/api/annotation",
                headers=headers,
                json={
                    "assignment_id": image_payload["task"]["assignment_id"],
                    "answers": {"private": "discard this too"},
                    "elapsed_ms": 84,
                },
            )
            complete_payload = await complete.json()
            assert complete_payload["status"] == "complete"
            assert complete_payload["responses_persisted"] is False

            resumed = await client.post("/api/login", json={"code": issued["code"]})
            assert (await resumed.json())["test_mode"] is True
            health = await client.get("/healthz")
            assert await health.json() == {
                "ok": True,
                "test_mode": True,
                "responses_persisted": False,
                "issued_in_memory": 1,
            }
        finally:
            await client.close()

    asyncio.run(scenario())


def test_participant_test_rotates_distinct_item_and_image_fixtures(tmp_path: Path) -> None:
    async def scenario() -> None:
        first_image = tmp_path / "first.jpg"
        second_image = tmp_path / "second.jpg"
        _write_test_image(first_image, size=(64, 48))
        _write_test_image(second_image, size=(80, 40))
        first_item = {
            "category": "object",
            "unit": "a red sphere",
            "target": "sphere",
            "caption": "A red sphere floats above a platform.",
            "span": "red sphere",
        }
        second_item = {
            "category": "lighting",
            "unit": "lit from the left",
            "target": "scene",
            "caption": "A green cube is lit from the left.",
            "span": "lit from the left",
        }
        app = create_self_enrollment_test_app(
            item=first_item,
            image_path=first_image,
            pair_count=2,
            additional_items=(second_item,),
            additional_image_paths=(second_image,),
        )
        client = await _client_for(app)
        try:
            issued = await (await client.post("/api/enroll", json={})).json()
            headers = {"Authorization": f"Bearer {issued['token']}"}
            first_caption = await (
                await client.post(
                    "/api/consent",
                    headers=headers,
                    json={"consented": True},
                )
            ).json()
            assert first_caption["task"]["unit"] == "a red sphere"

            first_image_task = await (
                await client.post(
                    "/api/annotation",
                    headers=headers,
                    json={
                        "assignment_id": first_caption["task"]["assignment_id"],
                        "answers": {},
                        "elapsed_ms": 1,
                    },
                )
            ).json()
            assert first_image_task["task"]["image_url"] == (
                f"/api/image/{first_image_task['task']['assignment_id']}"
            )
            first_image_response = await client.get(
                first_image_task["task"]["image_url"],
                headers=headers,
            )
            _assert_webp(await first_image_response.read(), size=(64, 48))

            second_caption = await (
                await client.post(
                    "/api/annotation",
                    headers=headers,
                    json={
                        "assignment_id": first_image_task["task"]["assignment_id"],
                        "answers": {},
                        "elapsed_ms": 1,
                    },
                )
            ).json()
            assert second_caption["task"]["unit"] == "lit from the left"

            second_image_task = await (
                await client.post(
                    "/api/annotation",
                    headers=headers,
                    json={
                        "assignment_id": second_caption["task"]["assignment_id"],
                        "answers": {},
                        "elapsed_ms": 1,
                    },
                )
            ).json()
            second_image_response = await client.get(
                second_image_task["task"]["image_url"],
                headers=headers,
            )
            _assert_webp(await second_image_response.read(), size=(80, 40))
        finally:
            await client.close()

    asyncio.run(scenario())


def test_participant_test_rejects_unpaired_or_missing_additional_fixture(
    tmp_path: Path,
) -> None:
    image = tmp_path / "first.jpg"
    _write_test_image(image)
    item = {
        "category": "object",
        "unit": "sphere",
        "target": "sphere",
        "caption": "A sphere.",
        "span": "sphere",
    }
    with pytest.raises(ValueError, match="requires one additional image"):
        create_self_enrollment_test_app(
            item=item,
            image_path=image,
            additional_items=(item,),
        )
    with pytest.raises(FileNotFoundError):
        create_self_enrollment_test_app(
            item=item,
            image_path=image,
            additional_items=(item,),
            additional_image_paths=(tmp_path / "missing.jpg",),
        )


def test_author_practice_is_isolated_resumable_and_response_discarding(tmp_path: Path) -> None:
    async def scenario() -> None:
        image = tmp_path / "image.jpg"
        _write_test_image(image)
        codes = {f"author_practice_{index:02d}": f"secret-{index}" for index in range(1, 4)}
        items = {
            slot: [
                {
                    "item_id": f"private-{slot}-{item_index}",
                    "category": "object",
                    "unit": f"claim {item_index} for {slot}",
                    "target": "scene",
                    "caption": "A kite flies above the beach.",
                    "span": "kite",
                    "asset_ref": "image.jpg",
                }
                for item_index in range(2)
            ]
            for slot in codes
        }
        app = create_author_practice_app(
            author_codes=codes,
            author_items=items,
            asset_root=tmp_path,
        )
        client = await _client_for(app)
        try:
            health = await (await client.get("/healthz")).json()
            assert health == {
                "ok": True,
                "practice_mode": True,
                "responses_persisted": False,
            }
            tokens = {}
            caption_ids = set()
            for slot, code in codes.items():
                login = await client.post("/api/login", json={"code": code})
                tokens[slot] = (await login.json())["token"]
                bootstrap = await client.get(
                    "/api/bootstrap",
                    headers={"Authorization": f"Bearer {tokens[slot]}"},
                )
                payload = await bootstrap.json()
                assert payload["practice_mode"] is True
                assert payload["phase"] == "caption"
                caption_ids.add(payload["task"]["assignment_id"])
            assert len(caption_ids) == 3

            slot = "author_practice_01"
            token = tokens[slot]
            assignment_id = f"practice-{slot}-caption-0000"
            response = await client.post(
                "/api/annotation",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "assignment_id": assignment_id,
                    "answers": {"sentinel_that_must_not_be_retained": "SECRET"},
                    "elapsed_ms": 12345,
                },
            )
            payload = await response.json()
            assert payload["phase"] == "image"
            image_response = await client.get(
                payload["task"]["image_url"],
                headers={"Authorization": f"Bearer {token}"},
            )
            assert image_response.content_type == "image/webp"
            _assert_webp(await image_response.read())
            image_assignment_id = payload["task"]["assignment_id"]
            coverage_response = await client.get(
                f"/api/coverage-caption/{image_assignment_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert await coverage_response.json() == {
                "caption": "A kite flies above the beach."
            }

            reconnect = await client.post("/api/login", json={"code": codes[slot]})
            reconnect_token = (await reconnect.json())["token"]
            resumed = await client.get(
                "/api/bootstrap",
                headers={"Authorization": f"Bearer {reconnect_token}"},
            )
            assert (await resumed.json())["phase"] == "image"
            next_response = await client.post(
                "/api/annotation",
                headers={"Authorization": f"Bearer {reconnect_token}"},
                json={
                    "assignment_id": image_assignment_id,
                    "answers": {"image_support": "yes"},
                    "elapsed_ms": 500,
                },
            )
            assert (await next_response.json())["phase"] == "caption"
            previous_image = await client.get(
                f"/api/image/{image_assignment_id}",
                headers={"Authorization": f"Bearer {reconnect_token}"},
            )
            _assert_webp(await previous_image.read())
        finally:
            await client.close()

    asyncio.run(scenario())


def test_preview_accepts_only_localhost_or_loopback_host_headers(tmp_path: Path) -> None:
    async def scenario() -> None:
        image = tmp_path / "image.jpg"
        _write_test_image(image)
        app = create_preview_app(
            item={
                "category": "object",
                "unit": "a kite",
                "caption": "A kite flies above the beach.",
                "span": "kite",
            },
            image_path=image,
        )
        client = await _client_for(app)
        try:
            for allowed_host in (
                "localhost",
                "localhost:8765",
                "127.0.0.1",
                "127.77.8.9:8765",
                "[::1]",
                "[::1]:8765",
            ):
                response = await client.get(
                    "/api/bootstrap",
                    headers={"Host": allowed_host},
                )
                assert response.status == 200, allowed_host
                assert (await response.json())["status"] == "task"

            for rejected_host in (
                "evil.example",
                "localhost.evil.example",
                "127.0.0.1.evil.example",
                "evil.example:8765",
                "user@localhost",
            ):
                response = await client.get(
                    "/api/bootstrap",
                    headers={"Host": rejected_host},
                )
                assert response.status == 421, rejected_host
                assert await response.json() == {"error": "Request failed."}
                _assert_security_headers(response)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_login_consent_and_caption_projection_are_blinded(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = FakeStore()
        app = create_study_app(
            store=store,
            study_id="study-1",
            asset_root=tmp_path,
            participant_information=_participant_information(),
        )
        client = await _client_for(app)
        try:
            response = await client.get(
                "/api/bootstrap?token=session-token",
                cookies={"session": "session-token"},
            )
            assert await response.json() == {"requires_login": True}
            assert not any(call[0] == "fetch_next_task" for call in store.calls)

            response = await client.post("/api/login", json={"code": "invite-code"})
            assert response.status == 200
            assert await response.json() == {"token": "session-token"}
            assert "Set-Cookie" not in response.headers
            assert ("create_session", "invite-code", "study-1") in store.calls

            headers = {"Authorization": "Bearer session-token"}
            response = await client.get("/api/bootstrap", headers=headers)
            payload = await response.json()
            assert payload == {
                "status": "consent_required",
                "required_consent_version": "consent-v1",
                "consent_version": "consent-v1",
                "participant_information": _participant_information(),
                "progress": {
                    "completed": 2,
                    "total": 5,
                    "overall_completed": 3,
                    "overall_total": 10,
                },
            }

            profile = {
                "english_proficiency": "fluent",
                "image_annotation_experience": "some",
            }
            response = await client.post(
                "/api/consent",
                headers=headers,
                json={"consented": True, "consent_version": "consent-v1", "profile": profile},
            )
            assert response.status == 200
            payload = await response.json()
            assert payload["status"] == "task"
            assert payload["phase"] == "caption"
            assert payload["consent_version"] == "consent-v1"
            assert "participant_information" not in payload
            assert "Approved compensation sentinel." not in str(payload)
            assert payload["progress"] == {
                "completed": 2,
                "total": 5,
                "overall_completed": 3,
                "overall_total": 10,
            }
            assert payload["task"] == {
                "assignment_id": "a_caption",
                "phase": "caption",
                "position": 3,
                "caption": "A red kite flies above a beach.",
                "span": "red kite",
                "unit": "a red kite",
                "target": "kite",
                "category": "attribute",
            }
            serialized = str(payload)
            for private_value in (
                "private-item",
                "private-surface",
                "qwen",
                "gemma",
                "private.jpg",
                "ours wins",
                "private-participant",
            ):
                assert private_value not in serialized
            assert (
                "record_consent_profile",
                "session-token",
                True,
                "consent-v1",
                profile,
                "study-1",
            ) in store.calls
            _assert_security_headers(response)
            _assert_hsts(response)

            response = await client.post(
                "/api/consent",
                headers=headers,
                json={
                    "consented": False,
                    "consent_version": "consent-v1",
                    "profile": {},
                },
            )
            assert response.status == 422

            response = await client.post(
                "/api/withdraw",
                headers=headers,
                json={
                    "participant_code": "wrong-code",
                    "consent_version": "consent-v1",
                },
            )
            assert response.status == 401

            response = await client.post(
                "/api/withdraw",
                headers=headers,
                json={
                    "participant_code": "invite-code",
                    "consent_version": "consent-v1",
                },
            )
            assert response.status == 200
            withdrawal = await response.json()
            assert withdrawal["status"] == "declined"
            assert withdrawal["withdrawn"] is True
            assert "participant_information" not in withdrawal
            assert "Approved compensation sentinel." not in str(withdrawal)
            assert (
                "authenticate_invite",
                "invite-code",
                "study-1",
            ) in store.calls
            assert (
                "record_consent_profile",
                "session-token",
                False,
                "consent-v1",
                {},
                "study-1",
            ) in store.calls
        finally:
            await client.close()
        assert store.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("study_status", ["paused", "closed"])
def test_non_open_bootstrap_is_safe_and_still_permits_withdrawal(
    tmp_path: Path,
    study_status: str,
) -> None:
    async def scenario() -> None:
        store = FakeStore()
        store.study_status = study_status
        app = create_study_app(
            store=store,
            study_id="study-1",
            asset_root=tmp_path,
            participant_information=_participant_information(),
        )
        client = await _client_for(app)
        headers = {"Authorization": "Bearer session-token"}
        try:
            response = await client.get("/api/bootstrap", headers=headers)
            assert response.status == 200
            payload = await response.json()
            assert payload == {
                "status": study_status,
                "consent_version": "consent-v1",
                "progress": {
                    "completed": 2,
                    "total": 5,
                    "overall_completed": 3,
                    "overall_total": 10,
                },
            }
            assert "participant" not in str(payload)
            assert "task" not in payload
            assert not any(call[0] == "fetch_next_task" for call in store.calls)

            response = await client.post(
                "/api/withdraw",
                headers=headers,
                json={
                    "participant_code": "invite-code",
                    "consent_version": "consent-v1",
                },
            )
            assert response.status == 200
            withdrawal = await response.json()
            assert withdrawal["status"] == "declined"
            assert (
                "record_consent_profile",
                "session-token",
                False,
                "consent-v1",
                {},
                "study-1",
            ) in store.calls
        finally:
            await client.close()

    asyncio.run(scenario())


def test_open_bootstrap_never_fetches_with_withdrawal_only_session(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = FakeStore()
        store.study_status = "open"
        store.session_scope = "withdrawal_only"
        store.result = _caption_result()
        app = create_study_app(
            store=store,
            study_id="study-1",
            asset_root=tmp_path,
            participant_information=_participant_information(),
        )
        client = await _client_for(app)
        try:
            response = await client.get(
                "/api/bootstrap",
                headers={"Authorization": "Bearer session-token"},
            )
            assert response.status == 200
            payload = await response.json()
            assert payload["status"] == "paused"
            assert payload["consent_version"] == "consent-v1"
            assert "task" not in payload
            assert not any(call[0] == "fetch_next_task" for call in store.calls)
            assert (
                "get_progress",
                "session-token",
                "study-1",
            ) in store.calls
        finally:
            await client.close()

    asyncio.run(scenario())


def test_waiting_bootstrap_keeps_withdrawal_version_without_participant_information(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = FakeStore()
        store.result = {
            "status": "waiting_for_assets",
            "progress": _progress(current_phase="image"),
        }
        app = create_study_app(
            store=store,
            study_id="study-1",
            asset_root=tmp_path,
            participant_information=_participant_information(),
        )
        client = await _client_for(app)
        try:
            response = await client.get(
                "/api/bootstrap",
                headers={"Authorization": "Bearer session-token"},
            )
            payload = await response.json()
            assert payload["status"] == "waiting_for_assets"
            assert payload["consent_version"] == "consent-v1"
            assert "participant_information" not in payload
            assert "Approved compensation sentinel." not in str(payload)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_annotation_uses_assignment_contract_and_does_not_return_private_save_result(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = FakeStore()
        store.result = _caption_result()
        app = create_study_app(
            store=store,
            study_id="study-1",
            asset_root=tmp_path,
            participant_information=_participant_information(),
        )
        client = await _client_for(app)
        try:
            answers = {
                "caption_licensed": "yes",
                "atomic_visual_claim": "yes",
                "category_check": "correct",
            }
            response = await client.post(
                "/api/annotation",
                headers={"Authorization": "Bearer session-token"},
                json={"assignment_id": "a_caption", "answers": answers, "elapsed_ms": 1_234},
            )
            assert response.status == 200
            payload = await response.json()
            assert payload == {
                "status": "complete",
                "consent_version": "consent-v1",
                "progress": {
                    "completed": 3,
                    "total": 10,
                    "overall_completed": 3,
                    "overall_total": 10,
                },
            }
            assert (
                "save_annotation",
                "session-token",
                "a_caption",
                answers,
                1_234,
                "study-1",
            ) in store.calls
            assert "private-item" not in str(payload)

            response = await client.post(
                "/api/annotation",
                headers={"Authorization": "Bearer session-token"},
                json={
                    "assignment_id": "a_caption",
                    "phase": "caption",
                    "answers": answers,
                    "elapsed_ms": 1,
                },
            )
            assert response.status == 422
            assert await response.json() == {"error": "Request validation failed."}
        finally:
            await client.close()

    asyncio.run(scenario())


def test_optional_work_endpoint_grants_batch_without_accepting_fields(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = FakeStore()
        store.result = {
            "status": "complete",
            "progress": _progress(current_phase="complete"),
            "work_extension": {
                "batch_size": 5,
                "max_items": 50,
                "assigned_items": 30,
                "can_extend": True,
            },
        }
        app = create_study_app(
            store=store,
            study_id="study-1",
            asset_root=tmp_path,
            participant_information=_participant_information(),
        )
        client = await _client_for(app)
        try:
            response = await client.post(
                "/api/extend-work",
                headers={"Authorization": "Bearer session-token"},
                json={},
            )
            assert response.status == 200
            payload = await response.json()
            assert payload["status"] == "task"
            assert payload["task"]["assignment_id"] == "a_caption"
            assert (
                "extend_remaining_first_work",
                "session-token",
                "study-1",
            ) in store.calls

            response = await client.post(
                "/api/extend-work",
                headers={"Authorization": "Bearer session-token"},
                json={"count": 100},
            )
            assert response.status == 422
        finally:
            await client.close()

    asyncio.run(scenario())


def test_image_route_requires_bearer_and_contains_asset_root(tmp_path: Path) -> None:
    async def scenario() -> None:
        nested = tmp_path / "nested"
        nested.mkdir()
        image = nested / "image.jpg"
        _write_test_image(image)
        outside = tmp_path.parent / f"{tmp_path.name}-outside.jpg"
        outside.write_bytes(b"outside")
        store = FakeStore()
        app = create_study_app(
            store=store,
            study_id="study-1",
            asset_root=tmp_path,
            participant_information=_participant_information(),
        )
        client = await _client_for(app)
        try:
            response = await client.get("/api/image/a_image")
            assert response.status == 401
            assert await response.json() == {"error": "Authentication failed."}

            response = await client.get(
                "/api/image/a_image",
                headers={"Authorization": "Bearer session-token"},
            )
            assert response.status == 200
            assert response.content_type == "image/webp"
            _assert_webp(await response.read())
            assert (
                "authorized_image_asset",
                "session-token",
                "a_image",
                "study-1",
            ) in store.calls
            _assert_security_headers(response)

            response = await client.get(
                "/api/coverage-caption/a_image",
                headers={"Authorization": "Bearer session-token"},
            )
            assert response.status == 200
            assert await response.json() == {
                "caption": "A protected paired caption."
            }
            assert (
                "authorized_coverage_caption",
                "session-token",
                "a_image",
                "study-1",
            ) in store.calls

            store.asset_ref = f"../{outside.name}"
            response = await client.get(
                "/api/image/a_image",
                headers={"Authorization": "Bearer session-token"},
            )
            assert response.status == 404
            assert await response.json() == {"error": "Resource not found."}
        finally:
            await client.close()
            outside.unlink()

    asyncio.run(scenario())


def test_self_enrollment_returns_one_code_and_session_without_profile_fields(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = FakeStore()
        app = create_study_app(
            store=store,
            study_id="study-1",
            asset_root=tmp_path,
            participant_information=_participant_information(),
            self_enrollment_codes=("code-a", "code-b"),
        )
        client = await _client_for(app)
        try:
            response = await client.post("/api/enroll", json={})
            assert response.status == 200
            assert await response.json() == {
                "code": "hcu_self_issued",
                "token": "self-issued-session",
            }
            assert (
                "claim_preprovisioned_participant_session",
                "study-1",
                ("code-a", "code-b"),
            ) in store.calls
            response = await client.post(
                "/api/enroll",
                json={"name": "must-not-be-accepted"},
            )
            assert response.status == 422
        finally:
            await client.close()

    asyncio.run(scenario())


def test_real_web_mutations_reject_cross_study_bearer_before_commit(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        _write_test_image(tmp_path / "image.jpg")
        store = HumanCBUStore.initialize(tmp_path / "cross-study.sqlite3")
        _add_open_real_study(store, "study-a", item_suffix="a")
        study_b_invites = _add_open_real_study(store, "study-b", item_suffix="b")
        study_b_session = store.create_session(
            study_b_invites[0]["invite_code"],
            study_id="study-b",
        )
        study_b_token = study_b_session["session_token"]
        store.record_consent_profile(
            study_b_token,
            study_id="study-b",
            consented=True,
            consent_version="consent-v1",
        )
        study_b_task = store.fetch_next_task(
            study_b_token,
            study_id="study-b",
        )["task"]

        with sqlite3.connect(store.path) as connection:
            consent_events_before = connection.execute(
                "SELECT COUNT(*) FROM consent_events WHERE study_id = 'study-b'"
            ).fetchone()[0]
            presentation_before = connection.execute(
                "SELECT presentation_count FROM assignments WHERE assignment_id = ?",
                (study_b_task["assignment_id"],),
            ).fetchone()[0]

        app = create_study_app(
            store=store,
            study_id="study-a",
            asset_root=tmp_path,
            participant_information=_participant_information(),
        )
        client = await _client_for(app)
        headers = {"Authorization": f"Bearer {study_b_token}"}
        try:
            response = await client.post(
                "/api/withdraw",
                headers=headers,
                json={
                    "participant_code": study_b_invites[0]["invite_code"],
                    "consent_version": "consent-v1",
                },
            )
            assert response.status == 401

            response = await client.post(
                "/api/annotation",
                headers=headers,
                json={
                    "assignment_id": study_b_task["assignment_id"],
                    "answers": {
                        "caption_licensed": "yes",
                        "atomic_visual_claim": "yes",
                        "category_check": "correct",
                    },
                    "elapsed_ms": 500,
                },
            )
            assert response.status == 401
        finally:
            await client.close()

        authenticated = store.authenticate_session(
            study_b_token,
            study_id="study-b",
        )
        assert authenticated["participant_status"] == "active"
        assert authenticated["consented"] is True
        with sqlite3.connect(store.path) as connection:
            assert (
                connection.execute("SELECT COUNT(*) FROM consent_events WHERE study_id = 'study-b'").fetchone()[0]
                == consent_events_before
            )
            assert connection.execute("SELECT COUNT(*) FROM annotations WHERE study_id = 'study-b'").fetchone()[0] == 0
            assignment = connection.execute(
                """
                SELECT status, presentation_count
                FROM assignments
                WHERE assignment_id = ?
                """,
                (study_b_task["assignment_id"],),
            ).fetchone()
            assert assignment == ("pending", presentation_before)

    asyncio.run(scenario())


def test_real_web_recovery_session_stays_withdrawal_only_after_reopen(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        _write_test_image(tmp_path / "image.jpg")
        store = HumanCBUStore.initialize(tmp_path / "recovery-scope.sqlite3")
        invites = _add_open_real_study(store, "audit", item_suffix="scope")
        full_session = store.create_session(
            invites[0]["invite_code"],
            study_id="audit",
        )
        store.record_consent_profile(
            full_session["session_token"],
            study_id="audit",
            consented=True,
            consent_version="consent-v1",
        )
        store.set_study_status("audit", "paused")

        app = create_study_app(
            store=store,
            study_id="audit",
            asset_root=tmp_path,
            participant_information=_participant_information(),
        )
        client = await _client_for(app)
        try:
            response = await client.post(
                "/api/login",
                json={"code": invites[0]["invite_code"]},
            )
            assert response.status == 200
            recovery_token = (await response.json())["token"]
            assert (
                store.authenticate_session(
                    recovery_token,
                    study_id="audit",
                )["session_scope"]
                == "withdrawal_only"
            )
            store.set_study_status("audit", "open")
            with sqlite3.connect(store.path) as connection:
                assignment_id, presentation_before = connection.execute(
                    """
                    SELECT assignment_id, presentation_count
                    FROM assignments
                    WHERE study_id = 'audit' AND participant_id = ?
                    ORDER BY position LIMIT 1
                    """,
                    (invites[0]["participant_id"],),
                ).fetchone()
                consent_events_before = connection.execute(
                    "SELECT COUNT(*) FROM consent_events WHERE study_id = 'audit'"
                ).fetchone()[0]

            headers = {"Authorization": f"Bearer {recovery_token}"}
            response = await client.get("/api/bootstrap", headers=headers)
            assert response.status == 200
            bootstrap = await response.json()
            assert bootstrap["status"] == "paused"
            assert bootstrap["consent_version"] == "consent-v1"
            assert "task" not in bootstrap

            response = await client.post(
                "/api/annotation",
                headers=headers,
                json={
                    "assignment_id": assignment_id,
                    "answers": {
                        "caption_licensed": "yes",
                        "atomic_visual_claim": "yes",
                        "category_check": "correct",
                    },
                },
            )
            assert response.status == 403
            response = await client.get(
                f"/api/image/{assignment_id}",
                headers=headers,
            )
            assert response.status == 403
            response = await client.post(
                "/api/consent",
                headers=headers,
                json={
                    "consented": True,
                    "consent_version": "consent-v1",
                    "profile": {},
                },
            )
            assert response.status == 403

            with sqlite3.connect(store.path) as connection:
                assignment = connection.execute(
                    """
                    SELECT status, presentation_count
                    FROM assignments
                    WHERE assignment_id = ?
                    """,
                    (assignment_id,),
                ).fetchone()
                assert assignment == ("pending", presentation_before)
                assert (
                    connection.execute("SELECT COUNT(*) FROM annotations WHERE study_id = 'audit'").fetchone()[0] == 0
                )
                assert (
                    connection.execute("SELECT COUNT(*) FROM consent_events WHERE study_id = 'audit'").fetchone()[0]
                    == consent_events_before
                )

            response = await client.post(
                "/api/withdraw",
                headers=headers,
                json={
                    "participant_code": invites[0]["invite_code"],
                    "consent_version": "consent-v1",
                },
            )
            assert response.status == 200
            assert (await response.json())["status"] == "declined"
            assert (
                store.authenticate_session(
                    recovery_token,
                    study_id="audit",
                )["participant_status"]
                == "withdrawn"
            )
        finally:
            await client.close()

    asyncio.run(scenario())


def test_real_store_browser_contract_end_to_end(tmp_path: Path) -> None:
    async def scenario() -> None:
        image = tmp_path / "image.jpg"
        _write_test_image(image)
        store = HumanCBUStore.initialize(tmp_path / "study.sqlite3")
        store.create_study(
            "audit",
            title="Human CBU audit",
            protocol_version="protocol-v1",
            consent_version="consent-v1",
            metadata={
                "required_labels_per_item": 2,
                "ethics_or_irb_equivalent_determination_id": "TEST-DETERMINATION-001",
                "participant_information": _participant_information(),
            },
        )
        store.add_items(
            "audit",
            [
                {
                    "item_id": "private-item",
                    "caption": "A red cat sits on a mat.",
                    "unit": "a red cat",
                    "span": [0, 9],
                    "target": "cat",
                    "category": "attribute",
                    "surface": "ours",
                    "item_group": "image-cluster",
                    "image_locator": {"url_hash": "private-url-hash"},
                    "image_asset": {"asset_relpath": "image.jpg"},
                    "qwen_answer": {"support": "yes"},
                    "gemma_answer": {"support": "yes"},
                }
            ],
        )
        invite = store.create_participant_invites("audit", count=2)[0]
        store.assign_items("audit", labels_per_item=2, seed="fixed-seed")
        store.seal_validation("audit")
        store.set_study_status("audit", "open")

        app = create_study_app(
            store=store,
            study_id="audit",
            asset_root=tmp_path,
            participant_information=_participant_information(),
        )
        client = await _client_for(app)
        try:
            response = await client.post("/api/login", json={"code": invite["invite_code"]})
            token = (await response.json())["token"]
            headers = {"Authorization": f"Bearer {token}"}

            response = await client.get("/api/bootstrap", headers=headers)
            assert (await response.json())["status"] == "consent_required"

            response = await client.post(
                "/api/consent",
                headers=headers,
                json={
                    "consented": True,
                    "consent_version": "consent-v1",
                    "profile": {
                        "author_status": "non_author",
                        "recruitment_source": "lab_non_author",
                        "english_proficiency": "fluent",
                        "image_annotation_experience": "some",
                        "t2i_experience": "some",
                        "device_class": "laptop",
                    },
                },
            )
            caption_result = await response.json()
            assert caption_result["status"] == "task"
            assert caption_result["task"]["phase"] == "caption"
            assert caption_result["task"]["span"] == "A red cat"
            assert "private-item" not in str(caption_result)
            assert "ours" not in str(caption_result)

            response = await client.post(
                "/api/annotation",
                headers=headers,
                json={
                    "assignment_id": caption_result["task"]["assignment_id"],
                    "answers": {
                        "caption_licensed": "yes",
                        "atomic_visual_claim": "yes",
                        "category_check": "correct",
                    },
                    "elapsed_ms": 500,
                },
            )
            image_result = await response.json()
            assert image_result["status"] == "task"
            assert image_result["task"]["phase"] == "image"
            assert set(image_result["task"]) == {
                "assignment_id",
                "phase",
                "position",
                "unit",
                "target",
                "category",
                "image_url",
            }

            response = await client.get(image_result["task"]["image_url"], headers=headers)
            assert response.status == 200
            assert response.content_type == "image/webp"
            _assert_webp(await response.read())

            response = await client.post(
                "/api/annotation",
                headers=headers,
                json={
                    "assignment_id": image_result["task"]["assignment_id"],
                    "answers": {"image_support": "yes", "control_usefulness": 5},
                    "elapsed_ms": 700,
                },
            )
            complete = await response.json()
            assert complete["status"] == "complete"
            assert complete["consent_version"] == "consent-v1"
            assert complete["progress"] == {
                "completed": 2,
                "total": 2,
                "overall_completed": 2,
                "overall_total": 2,
            }

            # A fresh page has no client-side consent state. The authenticated
            # bootstrap still restores the approved version needed to withdraw.
            response = await client.get("/api/bootstrap", headers=headers)
            fresh_bootstrap = await response.json()
            assert fresh_bootstrap["status"] == "complete"
            assert fresh_bootstrap["consent_version"] == "consent-v1"

            store.set_study_status("audit", "paused")
            response = await client.get("/api/bootstrap", headers=headers)
            paused = await response.json()
            assert paused["status"] == "paused"
            assert paused["consent_version"] == "consent-v1"
            assert "task" not in paused

            response = await client.post(
                "/api/withdraw",
                headers=headers,
                json={
                    "participant_code": invite["invite_code"],
                    "consent_version": "consent-v1",
                },
            )
            assert response.status == 200
            assert (await response.json())["status"] == "declined"
            assert store.authenticate_session(token)["participant_status"] == "withdrawn"
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (AuthenticationError("SECRET auth detail"), 401),
        (AuthorizationError("SECRET authorization detail"), 403),
        (NotFoundError("SECRET path detail"), 404),
        (ValidationError("SECRET field detail"), 422),
        (StudyStateError("SECRET state detail"), 409),
        (ConflictError("SECRET conflict detail"), 409),
    ],
)
def test_store_errors_are_redacted(
    tmp_path: Path,
    error: Exception,
    expected_status: int,
) -> None:
    async def scenario() -> None:
        store = FakeStore()
        store.result = error
        app = create_study_app(
            store=store,
            study_id="study-1",
            asset_root=tmp_path,
            participant_information=_participant_information(),
        )
        client = await _client_for(app)
        try:
            response = await client.get(
                "/api/bootstrap",
                headers={"Authorization": "Bearer session-token"},
            )
            assert response.status == expected_status
            payload = await response.json()
            assert "SECRET" not in str(payload)
            _assert_security_headers(response)
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "participant_information",
    [
        {key: value for key, value in _participant_information().items() if key != "duration"},
        {**_participant_information(), "compensation": "  "},
        {**_participant_information(), "information_sheet_url": "http://example.edu/sheet"},
        {**_participant_information(), "information_sheet_url": "/another-relative-page"},
        {**_participant_information(), "information_sheet_url": "https://user:password@example.edu/sheet"},
        {**_participant_information(), "unapproved_extra": "must fail"},
    ],
)
def test_participant_information_configuration_is_fail_closed(
    tmp_path: Path,
    participant_information: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        create_study_app(
            store=FakeStore(),
            study_id="study-1",
            asset_root=tmp_path,
            participant_information=participant_information,
        )


def test_approved_information_version_must_match_store_consent_version(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = FakeStore()
        app = create_study_app(
            store=store,
            study_id="study-1",
            asset_root=tmp_path,
            participant_information=_participant_information(approved_version="consent-v2"),
        )
        client = await _client_for(app)
        try:
            response = await client.get(
                "/api/bootstrap",
                headers={"Authorization": "Bearer session-token"},
            )
            assert response.status == 409
            payload = await response.json()
            assert payload == {"error": "The study is not available for this operation."}
            assert "Approved compensation sentinel." not in str(payload)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_same_origin_participant_notice_is_public_and_escapes_content(tmp_path: Path) -> None:
    async def scenario() -> None:
        information = {
            **_participant_information(),
            "research_contact": "Study contact via Slack <private>",
            "information_sheet_url": "/study-information",
        }
        app = create_study_app(
            store=FakeStore(),
            study_id="study-1",
            asset_root=tmp_path,
            participant_information=information,
        )
        client = await _client_for(app)
        try:
            response = await client.get("/study-information")
            assert response.status == 200
            assert response.content_type == "text/html"
            page = await response.text()
            assert "Participant notice" in page
            assert "Study contact via Slack &lt;private&gt;" in page
            assert "<private>" not in page
            assert "It is not a test of you." in page
            assert "fixed caption window to answer three text questions" in page
            assert "unexpected or sensitive material" in page
            assert "partial nudity in non-sexual or cultural contexts" in page
            assert "violence, or medical imagery" in page
            assert "Participation is voluntary." in page
            assert "You may pause, close the page, or stop at any time" in page
            assert "does not collect names, demographics, IP addresses" in page
            _assert_security_headers(response)
        finally:
            await client.close()

    asyncio.run(scenario())


def test_invalid_json_and_missing_asset_root_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        create_study_app(
            store=FakeStore(),
            study_id="study-1",
            asset_root=tmp_path / "missing",
            participant_information=_participant_information(),
        )

    async def scenario() -> None:
        app = create_study_app(
            store=FakeStore(),
            study_id="study-1",
            asset_root=tmp_path,
            participant_information=_participant_information(),
        )
        client = await _client_for(app)
        try:
            response = await client.post(
                "/api/login",
                data="{not-json",
                headers={"Content-Type": "application/json"},
            )
            assert response.status == 400
            assert await response.json() == {"error": "Invalid request."}
            _assert_security_headers(response)
        finally:
            await client.close()

    asyncio.run(scenario())
