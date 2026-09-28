from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from audit_recap_t2i.human_cbu import store as store_module
from audit_recap_t2i.human_cbu.adjudication import render_adjudication_html
from audit_recap_t2i.human_cbu.store import (
    ADJUDICATION_PACKET_VERSION,
    ADJUDICATION_RESPONSE_VERSION,
    ADJUDICATION_REVIEW_VERSION,
    DEFAULT_SESSION_TTL_SECONDS,
    SCHEMA_VERSION,
    SESSION_SCOPE_FULL,
    SESSION_SCOPE_WITHDRAWAL_ONLY,
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    HumanCBUStore,
    HumanCBUStoreError,
    StudyStateError,
    ValidationError,
)

CAPTION_YES = {
    "caption_licensed": "yes",
    "atomic_visual_claim": "yes",
    "category_check": "correct",
}
IMAGE_YES = {"image_support": "yes", "control_usefulness": 5}
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


def _launch_metadata() -> dict[str, Any]:
    return {
        "ethics_or_irb_equivalent_determination_id": "TEST-DETERMINATION-001",
        "participant_information": {
            "approved_version": "consent-v1",
            "duration": "Approximately 45 minutes.",
            "compensation": "According to the approved participant notice.",
            "data_retention": "According to the approved participant notice.",
            "research_contact": "Research contact in the approved information sheet.",
            "withdrawal_policy": "Participation may be withdrawn.",
            "information_sheet_url": "https://example.edu/human-cbu-study",
        },
    }


def _v1_metadata(
    *,
    required_labels_per_item: int = 2,
    expected_repeat_items: int = 0,
) -> dict[str, Any]:
    return {
        "purpose": "human anchor",
        "required_labels_per_item": required_labels_per_item,
        "expected_repeat_items": expected_repeat_items,
        "adjudicator_pseudonym": "human-adjudicator-01",
        "sample_manifest_sha256": "a" * 64,
        "semantic_visual_claim_types": list(SEMANTIC_VISUAL_CLAIM_TYPES),
        **_launch_metadata(),
    }


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, *, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def _item(index: int, **overrides: Any) -> dict[str, Any]:
    caption = f"A red cat number {index} sits on a mat."
    item: dict[str, Any] = {
        "item_id": f"item-{index}",
        "caption": caption,
        "unit": "a red cat",
        "span": [0, 9],
        "target": "cat",
        "category": "attribute",
        "surface": "ours" if index % 2 == 0 else "reference",
        "group": f"image-{index}",
        "image_locator": {"kind": "cc12m", "url_hash": f"hash-{index}"},
        "image_asset": {
            "route": f"assets/item-{index}.jpg",
            "sha256": hashlib.sha256(f"image-{index}".encode()).hexdigest(),
            "private_provenance": {
                "surface": "must-not-leak",
                "caption": "must-not-leak",
                "qwen_answer": "must-not-leak",
            },
        },
        "qwen_answer": {"support": "yes"},
        "gemma_answer": {"support": "yes"},
        "stratum": "attribute",
        "population": "cc12m",
        "sample": "calibration-v1",
        "weight": 2.5,
        "population_size": 1000,
        "sample_probability": 0.4,
    }
    item.update(overrides)
    return item


def _draft_store(
    tmp_path: Path,
    *,
    item_count: int = 1,
    participant_count: int = 2,
    clock: MutableClock | None = None,
    target_items_per_participant: int | None = None,
    self_enrollment_limit: int | None = None,
    optional_extension_batch_size: int | None = None,
    optional_extension_max_items: int | None = None,
    overlap_target_items_per_stratum: int = 0,
) -> tuple[HumanCBUStore, list[dict[str, Any]]]:
    store = HumanCBUStore.initialize(tmp_path / "human-cbu.sqlite3", clock=clock or MutableClock())
    store.create_study(
        "audit",
        title="CBU calibration",
        protocol_version="protocol-v1",
        consent_version="consent-v1",
        metadata={
            "purpose": "human anchor",
            "required_labels_per_item": 2,
            **(
                {"target_items_per_participant": target_items_per_participant}
                if target_items_per_participant is not None
                else {}
            ),
            **(
                {"self_enrollment_limit": self_enrollment_limit}
                if self_enrollment_limit is not None
                else {}
            ),
            **(
                {
                    "optional_extension_batch_size": optional_extension_batch_size,
                    "optional_extension_max_items": optional_extension_max_items,
                }
                if optional_extension_batch_size is not None
                or optional_extension_max_items is not None
                else {}
            ),
            "overlap_target_items_per_stratum": overlap_target_items_per_stratum,
            **_launch_metadata(),
        },
    )
    store.add_items("audit", [_item(index) for index in range(item_count)])
    invites = store.create_participant_invites("audit", count=participant_count)
    return store, invites


def _open_store(
    tmp_path: Path,
    *,
    item_count: int = 1,
    participant_count: int = 2,
    labels_per_item: int = 2,
    clock: MutableClock | None = None,
) -> tuple[HumanCBUStore, list[dict[str, Any]], MutableClock]:
    mutable_clock = clock or MutableClock()
    store, invites = _draft_store(
        tmp_path,
        item_count=item_count,
        participant_count=participant_count,
        clock=mutable_clock,
    )
    store.assign_items("audit", labels_per_item=labels_per_item, seed="published-seed")
    report = store.seal_validation("audit")
    assert report["valid"]
    store.set_study_status("audit", "open")
    return store, invites, mutable_clock


def test_remaining_first_claims_least_covered_work_and_gates_images(tmp_path: Path) -> None:
    store, invites = _draft_store(tmp_path, item_count=2, participant_count=2)
    plan = store.plan_remaining_first_assignments(
        "audit",
        labels_per_item=2,
        primary_participant_count=2,
        seed="published-seed",
    )
    assert plan["late_bound"] is True
    assert plan["assignments"] == {"caption": 0, "image": 0}
    validation = store.seal_validation("audit")
    assert validation["valid"] is True
    assert validation["late_binding_incomplete_panels"] == 4
    store.set_study_status("audit", "open")
    first_token = _session_and_consent(store, invites[0])
    second_token = _session_and_consent(store, invites[1])

    first_assignments = []
    for _ in range(2):
        task = store.fetch_next_task(first_token)
        assert task["phase"] == "caption"
        first_assignments.append(task["task"]["assignment_id"])
        store.save_annotation(first_token, task["task"]["assignment_id"], CAPTION_YES)
        image = store.fetch_next_task(first_token)
        assert image["phase"] == "image"
        assert store.authorized_image_asset(first_token, image["task"]["assignment_id"])[
            "asset_ref"
        ].startswith("assets/item-")
        store.save_annotation(first_token, image["task"]["assignment_id"], IMAGE_YES)
    assert len(set(first_assignments)) == 2
    waiting = store.fetch_next_task(first_token)
    assert waiting["status"] == "waiting_for_peer_labels"

    second_assignments = []
    for _ in range(2):
        task = store.fetch_next_task(second_token)
        assert task["phase"] == "caption"
        second_assignments.append(task["task"]["assignment_id"])
        store.save_annotation(second_token, task["task"]["assignment_id"], CAPTION_YES)
        image = store.fetch_next_task(second_token)
        assert image["phase"] == "image"
        store.save_annotation(second_token, image["task"]["assignment_id"], IMAGE_YES)
    assert store.fetch_next_task(first_token)["status"] == "complete"

    with sqlite3.connect(store.path) as connection:
        first_items = {
            row[0]
            for row in connection.execute(
                "SELECT item_id FROM assignments WHERE assignment_id IN (?, ?)",
                first_assignments,
            )
        }
        second_items = {
            row[0]
            for row in connection.execute(
                "SELECT item_id FROM assignments WHERE assignment_id IN (?, ?)",
                second_assignments,
            )
        }
        panels = connection.execute(
            """
            SELECT item_id, phase, COUNT(DISTINCT participant_id)
            FROM assignments GROUP BY item_id, phase ORDER BY item_id, phase
            """
        ).fetchall()
    assert first_items == second_items == {"item-0", "item-1"}
    assert panels == [
        ("item-0", "caption", 2),
        ("item-0", "image", 2),
        ("item-1", "caption", 2),
        ("item-1", "image", 2),
    ]


def test_remaining_first_repeat_does_not_block_matching_base_image(tmp_path: Path) -> None:
    store, invites = _draft_store(
        tmp_path,
        item_count=1,
        participant_count=2,
    )
    repeated = json.loads(json.dumps(_item(0)))
    repeated.update(
        {
            "item_id": "repeat-0",
            "repeat_of": "item-0",
            "weight": 0.0,
            "sample_probability": None,
        }
    )
    store.add_items("audit", [repeated])
    store.plan_remaining_first_assignments(
        "audit",
        labels_per_item=2,
        primary_participant_count=2,
        seed="published-seed",
    )
    assert store.seal_validation("audit")["valid"] is True
    store.set_study_status("audit", "open")
    token = _session_and_consent(store, invites[0])

    caption = store.fetch_next_task(token)
    assert caption["phase"] == "caption"
    store.save_annotation(token, caption["task"]["assignment_id"], CAPTION_YES)

    image = store.fetch_next_task(token)
    assert image["phase"] == "image"
    saved = store.save_annotation(token, image["task"]["assignment_id"], IMAGE_YES)
    assert saved["phase"] == "image"

    repeated_caption = store.fetch_next_task(token)
    assert repeated_caption["phase"] == "caption"


def test_remaining_first_interleaves_repeat_after_memory_gap(tmp_path: Path) -> None:
    store, invites = _draft_store(
        tmp_path,
        item_count=12,
        participant_count=2,
        target_items_per_participant=12,
    )
    participant_id = invites[0]["participant_id"]
    first_item = min(
        (f"item-{index}" for index in range(12)),
        key=lambda item_id: (
            store_module._stable_digest(
                "published-seed",
                "remaining-first",
                "0",
                participant_id,
                item_id,
            ),
            item_id,
        ),
    )
    repeated = json.loads(json.dumps(_item(int(first_item.removeprefix("item-")))))
    repeated.update(
        {
            "item_id": "repeat-first",
            "repeat_of": first_item,
            "weight": 0.0,
            "sample_probability": None,
        }
    )
    store.add_items("audit", [repeated])
    store.plan_remaining_first_assignments(
        "audit",
        labels_per_item=2,
        primary_participant_count=2,
        seed="published-seed",
    )
    store.seal_validation("audit")
    store.set_study_status("audit", "open")
    token = _session_and_consent(store, invites[0])

    observed_items: list[str] = []
    while True:
        result = store.fetch_next_task(token)
        if result["status"] != "task":
            break
        if result["phase"] == "image":
            with sqlite3.connect(store.path) as connection:
                observed_items.append(
                    connection.execute(
                        "SELECT item_id FROM assignments WHERE assignment_id = ?",
                        (result["task"]["assignment_id"],),
                    ).fetchone()[0]
                )
        payload = CAPTION_YES if result["phase"] == "caption" else IMAGE_YES
        store.save_annotation(token, result["task"]["assignment_id"], payload)

    original_index = observed_items.index(first_item)
    repeat_index = observed_items.index("repeat-first")
    assert original_index == 0
    assert repeat_index - original_index >= 10
    assert repeat_index < len(observed_items) - 1
    with sqlite3.connect(store.path) as connection:
        positions = dict(
            connection.execute(
                """
                SELECT i.item_id, a.position
                FROM assignments AS a
                JOIN items AS i ON i.study_id = a.study_id AND i.item_id = a.item_id
                WHERE a.participant_id = ? AND a.phase = 'image'
                  AND i.item_id IN (?, 'repeat-first')
                """,
                (participant_id, first_item),
            )
        )
    assert positions["repeat-first"] - positions[first_item] == 10


def test_remaining_first_omits_repeat_when_work_window_cannot_fit_gap(
    tmp_path: Path,
) -> None:
    store, invites = _draft_store(
        tmp_path,
        item_count=1,
        participant_count=2,
        target_items_per_participant=1,
    )
    repeated = json.loads(json.dumps(_item(0)))
    repeated.update(
        {
            "item_id": "repeat-0",
            "repeat_of": "item-0",
            "weight": 0.0,
            "sample_probability": None,
        }
    )
    store.add_items("audit", [repeated])
    store.plan_remaining_first_assignments(
        "audit", labels_per_item=2, primary_participant_count=2, seed="published-seed"
    )
    store.seal_validation("audit")
    store.set_study_status("audit", "open")
    token = _session_and_consent(store, invites[0])

    caption = store.fetch_next_task(token)
    store.save_annotation(token, caption["task"]["assignment_id"], CAPTION_YES)
    image = store.fetch_next_task(token)
    store.save_annotation(token, image["task"]["assignment_id"], IMAGE_YES)
    assert store.fetch_next_task(token)["status"] == "complete"
    with sqlite3.connect(store.path) as connection:
        repeat_assignments = connection.execute(
            "SELECT COUNT(*) FROM assignments WHERE item_id = 'repeat-0'"
        ).fetchone()[0]
    assert repeat_assignments == 0


def test_remaining_first_concurrent_claim_never_overfills_panel(tmp_path: Path) -> None:
    store, invites = _draft_store(tmp_path, item_count=1, participant_count=3)
    store.plan_remaining_first_assignments(
        "audit",
        labels_per_item=2,
        primary_participant_count=3,
        seed="published-seed",
    )
    store.seal_validation("audit")
    store.set_study_status("audit", "open")
    tokens = [_session_and_consent(store, invite) for invite in invites]

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(store.fetch_next_task, tokens))

    assert sum(result["status"] == "task" for result in results) == 2
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute(
                """
                SELECT COUNT(DISTINCT participant_id) FROM assignments
                WHERE item_id = 'item-0' AND phase = 'caption'
                """
            ).fetchone()[0]
            == 2
        )


def test_remaining_first_caps_work_and_shows_stable_target_progress(
    tmp_path: Path,
) -> None:
    store, invites = _draft_store(
        tmp_path,
        item_count=4,
        participant_count=4,
        target_items_per_participant=2,
    )
    store.plan_remaining_first_assignments(
        "audit",
        labels_per_item=2,
        primary_participant_count=4,
        seed="published-seed",
    )
    store.seal_validation("audit")
    store.set_study_status("audit", "open")
    tokens = [_session_and_consent(store, invite) for invite in invites]

    first = store.fetch_next_task(tokens[0])
    assert first["status"] == "task"
    assert first["progress"]["caption"] == {
        "total": 2,
        "completed": 0,
        "remaining": 2,
    }

    for token in tokens:
        while True:
            result = store.fetch_next_task(token)
            if result["status"] != "task":
                break
            payload = CAPTION_YES if result["phase"] == "caption" else IMAGE_YES
            store.save_annotation(token, result["task"]["assignment_id"], payload)

    with sqlite3.connect(store.path) as connection:
        per_participant = connection.execute(
            """
            SELECT participant_id, COUNT(*)
            FROM assignments
            WHERE phase = 'caption'
            GROUP BY participant_id
            ORDER BY participant_id
            """
        ).fetchall()
        panels = connection.execute(
            """
            SELECT item_id, COUNT(DISTINCT participant_id)
            FROM assignments
            WHERE phase = 'caption'
            GROUP BY item_id
            ORDER BY item_id
            """
        ).fetchall()
    assert all(count <= 2 for _, count in per_participant)
    assert sum(count for _, count in per_participant) == 8
    assert all(count == 2 for _, count in panels)
    assert len(per_participant) == 4
    assert store.fetch_next_task(tokens[0])["status"] == "complete"
    assert store.get_progress(tokens[0])["caption"]["total"] == 2


def test_remaining_first_balances_strata_within_least_covered_items(
    tmp_path: Path,
) -> None:
    store, invites = _draft_store(
        tmp_path,
        item_count=4,
        participant_count=2,
        target_items_per_participant=2,
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE items SET stratum = 'ours/object' WHERE item_id IN ('item-0', 'item-1')")
        connection.execute("UPDATE items SET stratum = 'reference/object' WHERE item_id IN ('item-2', 'item-3')")
        connection.commit()
    store.plan_remaining_first_assignments(
        "audit",
        labels_per_item=2,
        primary_participant_count=2,
        seed="published-seed",
    )
    store.seal_validation("audit")
    store.set_study_status("audit", "open")
    token = _session_and_consent(store, invites[0])

    assigned_strata: list[str] = []
    for _ in range(2):
        caption = store.fetch_next_task(token)
        with sqlite3.connect(store.path) as connection:
            assigned_strata.append(
                connection.execute(
                    """
                    SELECT i.stratum FROM assignments AS a
                    JOIN items AS i ON i.study_id = a.study_id AND i.item_id = a.item_id
                    WHERE a.assignment_id = ?
                    """,
                    (caption["task"]["assignment_id"],),
                ).fetchone()[0]
            )
        store.save_annotation(token, caption["task"]["assignment_id"], CAPTION_YES)
        image = store.fetch_next_task(token)
        store.save_annotation(token, image["task"]["assignment_id"], IMAGE_YES)

    assert set(assigned_strata) == {"ours/object", "reference/object"}


def test_remaining_first_prioritizes_balanced_second_labels(
    tmp_path: Path,
) -> None:
    store, invites = _draft_store(
        tmp_path,
        item_count=4,
        participant_count=2,
        target_items_per_participant=2,
        overlap_target_items_per_stratum=1,
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE items SET stratum = 'ours/object' WHERE item_id IN ('item-0', 'item-1')"
        )
        connection.execute(
            "UPDATE items SET stratum = 'reference/object' WHERE item_id IN ('item-2', 'item-3')"
        )
        connection.commit()
    store.plan_remaining_first_assignments(
        "audit", labels_per_item=2, primary_participant_count=2, seed="published-seed"
    )
    store.seal_validation("audit")
    store.set_study_status("audit", "open")
    tokens = [_session_and_consent(store, invite) for invite in invites]

    def finish_two(token: str) -> None:
        completed_images = 0
        while completed_images < 2:
            result = store.fetch_next_task(token)
            payload = CAPTION_YES if result["phase"] == "caption" else IMAGE_YES
            store.save_annotation(token, result["task"]["assignment_id"], payload)
            completed_images += result["phase"] == "image"

    finish_two(tokens[0])
    finish_two(tokens[1])

    with sqlite3.connect(store.path) as connection:
        panels = connection.execute(
            """
            SELECT i.stratum, a.item_id, COUNT(DISTINCT a.participant_id) labels
            FROM assignments AS a
            JOIN items AS i ON i.study_id = a.study_id AND i.item_id = a.item_id
            WHERE a.phase = 'image' AND a.status = 'completed' AND i.repeat_of IS NULL
            GROUP BY i.stratum, a.item_id
            ORDER BY i.stratum, a.item_id
            """
        ).fetchall()
    assert len(panels) == 2
    assert {row[0] for row in panels} == {"ours/object", "reference/object"}
    assert all(row[2] == 2 for row in panels)


def test_optional_extension_grants_bounded_persistent_batches(tmp_path: Path) -> None:
    store, invites = _draft_store(
        tmp_path,
        item_count=10,
        participant_count=4,
        target_items_per_participant=2,
        optional_extension_batch_size=2,
        optional_extension_max_items=5,
    )
    store.plan_remaining_first_assignments(
        "audit",
        labels_per_item=2,
        primary_participant_count=4,
        seed="published-seed",
    )
    store.seal_validation("audit")
    store.set_study_status("audit", "open")
    token = _session_and_consent(store, invites[0])

    with pytest.raises(ConflictError, match="not currently available"):
        store.extend_remaining_first_work(token)

    def finish_pending() -> dict[str, Any]:
        while True:
            result = store.fetch_next_task(token)
            if result["status"] != "task":
                return result
            payload = CAPTION_YES if result["phase"] == "caption" else IMAGE_YES
            store.save_annotation(token, result["task"]["assignment_id"], payload)

    complete = finish_pending()
    assert complete["work_extension"] == {
        "batch_size": 2,
        "max_items": 5,
        "assigned_items": 2,
        "can_extend": True,
    }
    assert store.extend_remaining_first_work(token)["granted_items"] == 2
    assert store.fetch_next_task(token)["status"] == "task"
    complete = finish_pending()
    assert complete["work_extension"]["assigned_items"] == 4
    assert complete["work_extension"]["batch_size"] == 1
    assert store.extend_remaining_first_work(token)["granted_items"] == 1
    complete = finish_pending()
    assert complete["work_extension"] == {
        "batch_size": 0,
        "max_items": 5,
        "assigned_items": 5,
        "can_extend": False,
    }
    with pytest.raises(ConflictError, match="not currently available"):
        store.extend_remaining_first_work(token)


def test_repeat_assignments_do_not_consume_base_target(tmp_path: Path) -> None:
    store, invites = _draft_store(
        tmp_path,
        item_count=2,
        participant_count=2,
        target_items_per_participant=2,
    )
    repeated = json.loads(json.dumps(_item(0)))
    repeated.update({"item_id": "repeat-0", "repeat_of": "item-0", "weight": 0.0, "sample_probability": None})
    store.add_items("audit", [repeated])
    store.plan_remaining_first_assignments(
        "audit", labels_per_item=2, primary_participant_count=2, seed="published-seed"
    )
    store.seal_validation("audit")
    store.set_study_status("audit", "open")
    token = _session_and_consent(store, invites[0])

    while True:
        result = store.fetch_next_task(token)
        if result["status"] != "task":
            break
        payload = CAPTION_YES if result["phase"] == "caption" else IMAGE_YES
        store.save_annotation(token, result["task"]["assignment_id"], payload)
    with sqlite3.connect(store.path) as connection:
        base_caption_count = connection.execute(
            """
            SELECT COUNT(*) FROM assignments AS a
            JOIN items AS i ON i.study_id = a.study_id AND i.item_id = a.item_id
            WHERE a.phase = 'caption' AND i.repeat_of IS NULL
            """
        ).fetchone()[0]
    assert base_caption_count == 2


def test_bounded_self_enrollment_is_atomic_and_resumable(tmp_path: Path) -> None:
    store, invites = _draft_store(
        tmp_path,
        item_count=2,
        participant_count=3,
        target_items_per_participant=2,
        self_enrollment_limit=3,
    )
    store.plan_remaining_first_assignments(
        "audit",
        labels_per_item=2,
        primary_participant_count=3,
        seed="published-seed",
    )
    store.seal_validation("audit")
    store.set_study_status("audit", "open")
    codes = tuple(invite["invite_code"] for invite in invites)

    with ThreadPoolExecutor(max_workers=3) as executor:
        issued = list(
            executor.map(
                lambda _: store.claim_preprovisioned_participant_session(
                    "audit",
                    codes,
                ),
                range(3),
            )
        )

    assert len({row["participant_code"] for row in issued}) == 3
    assert len({row["participant_id"] for row in issued}) == 3
    assert all(row["session_scope"] == SESSION_SCOPE_FULL for row in issued)
    with pytest.raises(ConflictError, match="no self-enrollment slots remain"):
        store.claim_preprovisioned_participant_session("audit", codes)

    resumed = store.create_session(issued[0]["participant_code"], study_id="audit")
    assert resumed["participant_id"] == issued[0]["participant_id"]
    with sqlite3.connect(store.path) as connection:
        counts = connection.execute(
            "SELECT use_count FROM participant_invites ORDER BY invite_id"
        ).fetchall()
    assert sorted(row[0] for row in counts) == [1, 1, 2]
    assert store.status_summary("audit")["self_enrollment"] == {
        "limit": 3,
        "available": 0,
        "issued": 3,
        "revoked": 0,
    }


def _session_and_consent(
    store: HumanCBUStore,
    invite: dict[str, Any],
    *,
    profile: dict[str, Any] | None = None,
) -> str:
    session = store.create_session(invite["invite_code"])
    token = session["session_token"]
    store.record_consent_profile(
        token,
        consented=True,
        consent_version="consent-v1",
        profile=profile
        or {
            "age_band": "25-34",
            "author_status": "non_author",
            "recruitment_source": "academic_pool",
            "t2i_experience": "some",
        },
    )
    return token


def _complete_phase(
    store: HumanCBUStore,
    token: str,
    phase: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    saved: list[dict[str, Any]] = []
    while True:
        next_result = store.fetch_next_task(token)
        if next_result["status"] != "task" or next_result["phase"] != phase:
            break
        saved.append(
            store.save_annotation(
                token,
                next_result["task"]["assignment_id"],
                payload,
            )
        )
    return saved


def _v1_disagreement_store(
    tmp_path: Path,
    *,
    close: bool,
    item_count: int = 1,
) -> tuple[HumanCBUStore, list[dict[str, Any]], str]:
    store = HumanCBUStore.initialize(tmp_path / "human-cbu-v1.sqlite3", clock=MutableClock())
    store.create_study(
        "audit",
        title="CBU calibration",
        protocol_version="human_cbu_v1",
        consent_version="consent-v1",
        metadata=_v1_metadata(),
    )
    store.add_items("audit", [_item(index) for index in range(item_count)])
    invites = store.create_participant_invites("audit", count=2)
    store.assign_items("audit", labels_per_item=2, seed="published-seed")
    provisioned = store.provision_adjudicator("audit")
    assert store.seal_validation("audit")["valid"]
    store.set_study_status("audit", "open")
    tokens = [_session_and_consent(store, invite) for invite in invites]
    caption_payloads = (
        CAPTION_YES,
        {
            "caption_licensed": "no",
            "atomic_visual_claim": "not_visual",
            "category_check": "not_applicable",
        },
    )
    image_payloads = ({"image_support": "yes"}, {"image_support": "no"})
    for token, payload in zip(tokens, caption_payloads, strict=True):
        _complete_phase(store, token, "caption", payload)
    for token, payload in zip(tokens, image_payloads, strict=True):
        _complete_phase(store, token, "image", payload)
    if close:
        store.set_study_status("audit", "closed")
    return store, invites, provisioned["credential"]


def _v1_repeat_pair() -> list[dict[str, Any]]:
    original = _item(0)
    repeated = json.loads(json.dumps(original))
    repeated.update(
        {
            "item_id": "repeat-0",
            "repeat_of": original["item_id"],
            "weight": 0.0,
            "sample_probability": None,
        }
    )
    return [original, repeated]


def _prepared_v1_store(
    tmp_path: Path,
    *,
    items: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> tuple[HumanCBUStore, list[dict[str, Any]]]:
    store = HumanCBUStore.initialize(tmp_path / "prepared-v1.sqlite3", clock=MutableClock())
    store.create_study(
        "audit",
        title="CBU calibration",
        protocol_version="human_cbu_v1",
        consent_version="consent-v1",
        metadata=metadata
        or _v1_metadata(expected_repeat_items=sum(item.get("repeat_of") is not None for item in items)),
    )
    store.add_items("audit", items)
    invites = store.create_participant_invites("audit", count=2)
    store.assign_items("audit", labels_per_item=2, seed="published-seed")
    store.provision_adjudicator("audit")
    return store, invites


def _write_private_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _adjudication_packet_response(
    tmp_path: Path,
    *,
    name: str,
    packet: dict[str, Any],
    decision: dict[str, Any],
    image_bytes: bytes | None = None,
) -> tuple[Path, Path]:
    phase_root = tmp_path / name
    phase_root.mkdir(mode=0o700, exist_ok=True)
    packets_root = phase_root / "packets"
    packets_root.mkdir(mode=0o700, exist_ok=True)
    packet_dir = packets_root / packet["packet_id"]
    packet_dir.mkdir(mode=0o700)
    packet_file = packet_dir / "packet.json"
    _write_private_json(packet_file, packet)
    review_file = packet_dir / "review.html"
    review_file.write_text(
        render_adjudication_html(
            packet,
            semantic_categories=list(SEMANTIC_VISUAL_CLAIM_TYPES),
        ),
        encoding="utf-8",
    )
    review_file.chmod(0o600)
    if packet["phase"] == "image":
        image_path = packet_dir / packet["evidence"]["image_file"]
        image_path.write_bytes(image_bytes if image_bytes is not None else b"image-0")
        image_path.chmod(0o600)
    response = packet_dir / "response.json"
    _write_private_json(
        response,
        {
            "response_version": ADJUDICATION_RESPONSE_VERSION,
            "review_version": ADJUDICATION_REVIEW_VERSION,
            "packet_file_sha256": hashlib.sha256(packet_file.read_bytes()).hexdigest(),
            "packet": packet,
            "decision": decision,
        },
    )
    return packet_dir, response


def _submit_packet_response(
    store: HumanCBUStore,
    credential: str,
    packet_dir: Path,
    *,
    phase: str,
) -> dict[str, Any]:
    return store.adjudicate_packet_response(
        "audit",
        phase,
        packet_dir=packet_dir,
        adjudicator_credential=credential,
    )


def test_schema_v3_persists_fail_closed_session_scope(tmp_path: Path) -> None:
    store, invites, _ = _open_store(tmp_path)
    session = store.create_session(invites[0]["invite_code"])

    assert SCHEMA_VERSION == 3
    assert session["session_scope"] == SESSION_SCOPE_FULL
    assert store.authenticate_session(session["session_token"])["session_scope"] == SESSION_SCOPE_FULL
    with sqlite3.connect(store.path) as connection:
        connection.row_factory = sqlite3.Row
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert connection.execute("SELECT value FROM schema_metadata WHERE key = 'schema_version'").fetchone()[
            0
        ] == str(SCHEMA_VERSION)
        scope_column = next(
            row for row in connection.execute("PRAGMA table_info(bearer_sessions)") if row["name"] == "scope"
        )
        assert scope_column["notnull"] == 1
        assert scope_column["dflt_value"] == "'withdrawal_only'"
        assert connection.execute("SELECT scope FROM bearer_sessions").fetchone()["scope"] == SESSION_SCOPE_FULL


def test_schema_v2_database_is_rejected_instead_of_silently_extended(
    tmp_path: Path,
) -> None:
    store = HumanCBUStore.initialize(tmp_path / "schema-v2.sqlite3")
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE schema_metadata SET value = '2' WHERE key = 'schema_version'")
        connection.execute("PRAGMA user_version=2")

    with pytest.raises(HumanCBUStoreError, match=r"version '2'.*expected 3"):
        HumanCBUStore.open(store.path)


def test_digest_only_credentials_wal_and_coarse_profile(tmp_path: Path) -> None:
    store, invites, clock = _open_store(tmp_path)
    invite = invites[0]
    session = store.create_session(invite["invite_code"], ttl_seconds=60)

    assert len(invite["invite_code"]) >= 40
    assert len(session["session_token"]) >= 40
    with sqlite3.connect(store.path) as connection:
        connection.row_factory = sqlite3.Row
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        invite_row = connection.execute("SELECT invite_digest FROM participant_invites").fetchone()
        session_row = connection.execute("SELECT token_digest FROM bearer_sessions").fetchone()
        assert len(invite_row["invite_digest"]) == 64
        assert len(session_row["token_digest"]) == 64
        assert invite["invite_code"] not in invite_row["invite_digest"]
        assert session["session_token"] not in session_row["token_digest"]

        all_columns: set[str] = set()
        for table in (
            "participants",
            "participant_invites",
            "bearer_sessions",
            "annotations",
        ):
            all_columns.update(row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall())
        assert {"email", "name", "ip", "user_agent"}.isdisjoint(all_columns)

    with pytest.raises(ValidationError, match="unsupported"):
        store.record_consent_profile(
            session["session_token"],
            consented=True,
            consent_version="consent-v1",
            profile={"email": "rater@example.test"},
        )
    with pytest.raises(ValidationError, match="allowed coarse category"):
        store.record_consent_profile(
            session["session_token"],
            consented=True,
            consent_version="consent-v1",
            profile={"domain_experience": "rater@example.test"},
        )

    clock.advance(seconds=61)
    with pytest.raises(AuthenticationError, match="expired"):
        store.authenticate_session(session["session_token"])


def test_default_participant_session_lasts_twelve_hours(tmp_path: Path) -> None:
    clock = MutableClock()
    store, invites, _ = _open_store(tmp_path, clock=clock)
    session = store.create_session(invites[0]["invite_code"])

    assert DEFAULT_SESSION_TTL_SECONDS == 12 * 60 * 60
    assert datetime.fromisoformat(session["expires_at"]) == clock.value + timedelta(
        seconds=DEFAULT_SESSION_TTL_SECONDS
    )


def test_expected_study_rejects_cross_study_mutations_before_commit(
    tmp_path: Path,
) -> None:
    store, _, _ = _open_store(tmp_path)
    store.create_study(
        "other",
        title="Other study",
        protocol_version="protocol-v1",
        consent_version="consent-v1",
        metadata={"required_labels_per_item": 2, **_launch_metadata()},
    )
    store.add_item("other", **_item(100))
    other_invites = store.create_participant_invites("other", count=2)
    store.assign_items("other", labels_per_item=2, seed="other-seed")
    store.seal_validation("other")
    store.set_study_status("other", "open")
    other_token = _session_and_consent(store, other_invites[0])
    other_task = store.fetch_next_task(other_token, study_id="other")["task"]

    with sqlite3.connect(store.path) as connection:
        participant_before = connection.execute(
            """
            SELECT status, consented, consent_version
            FROM participants
            WHERE study_id = 'other' AND participant_id = ?
            """,
            (other_invites[0]["participant_id"],),
        ).fetchone()
        consent_events_before = connection.execute(
            "SELECT COUNT(*) FROM consent_events WHERE study_id = 'other'"
        ).fetchone()[0]
        presentation_before = connection.execute(
            "SELECT presentation_count FROM assignments WHERE assignment_id = ?",
            (other_task["assignment_id"],),
        ).fetchone()[0]

    with pytest.raises(AuthenticationError, match="not valid for this study"):
        store.record_consent_profile(
            other_token,
            study_id="audit",
            consented=False,
            consent_version="consent-v1",
        )
    with pytest.raises(AuthenticationError, match="not valid for this study"):
        store.save_annotation(
            other_token,
            other_task["assignment_id"],
            CAPTION_YES,
            study_id="audit",
        )

    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute(
                """
                SELECT status, consented, consent_version
                FROM participants
                WHERE study_id = 'other' AND participant_id = ?
                """,
                (other_invites[0]["participant_id"],),
            ).fetchone()
            == participant_before
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM consent_events WHERE study_id = 'other'").fetchone()[0]
            == consent_events_before
        )
        assert connection.execute("SELECT COUNT(*) FROM annotations WHERE study_id = 'other'").fetchone()[0] == 0
        assignment_after = connection.execute(
            """
            SELECT status, presentation_count
            FROM assignments
            WHERE assignment_id = ?
            """,
            (other_task["assignment_id"],),
        ).fetchone()
        assert assignment_after == ("pending", presentation_before)


def test_paused_reauthentication_scope_never_upgrades_after_reopen(
    tmp_path: Path,
) -> None:
    store, invites, _ = _open_store(tmp_path)
    full_token = _session_and_consent(store, invites[0])
    presented_task = store.fetch_next_task(full_token, study_id="audit")["task"]
    store.set_study_status("audit", "paused")

    recovery_session = store.create_session(
        invites[0]["invite_code"],
        study_id="audit",
    )
    recovery_token = recovery_session["session_token"]
    assert recovery_session["session_scope"] == SESSION_SCOPE_WITHDRAWAL_ONLY
    assert (
        store.authenticate_session(recovery_token, study_id="audit")["session_scope"] == SESSION_SCOPE_WITHDRAWAL_ONLY
    )

    store.set_study_status("audit", "open")
    with sqlite3.connect(store.path) as connection:
        presentation_before = connection.execute(
            "SELECT presentation_count FROM assignments WHERE assignment_id = ?",
            (presented_task["assignment_id"],),
        ).fetchone()[0]
        consent_events_before = connection.execute(
            "SELECT COUNT(*) FROM consent_events WHERE study_id = 'audit'"
        ).fetchone()[0]

    with pytest.raises(AuthorizationError, match="limited to withdrawal"):
        store.fetch_next_task(recovery_token, study_id="audit")
    with pytest.raises(AuthorizationError, match="limited to withdrawal"):
        store.authorized_image_asset(
            recovery_token,
            presented_task["assignment_id"],
            study_id="audit",
        )
    with pytest.raises(AuthorizationError, match="limited to withdrawal"):
        store.save_annotation(
            recovery_token,
            presented_task["assignment_id"],
            CAPTION_YES,
            study_id="audit",
        )
    with pytest.raises(AuthorizationError, match="limited to withdrawal"):
        store.record_consent_profile(
            recovery_token,
            study_id="audit",
            consented=True,
            consent_version="consent-v1",
        )

    with sqlite3.connect(store.path) as connection:
        assignment = connection.execute(
            """
            SELECT status, presentation_count
            FROM assignments
            WHERE assignment_id = ?
            """,
            (presented_task["assignment_id"],),
        ).fetchone()
        assert assignment == ("pending", presentation_before)
        assert connection.execute("SELECT COUNT(*) FROM annotations WHERE study_id = 'audit'").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM consent_events WHERE study_id = 'audit'").fetchone()[0]
            == consent_events_before
        )

    assert store.get_progress(recovery_token, study_id="audit")["current_phase"] == "caption"
    assert store.authenticate_session(full_token, study_id="audit")["session_scope"] == SESSION_SCOPE_FULL
    store.save_annotation(
        full_token,
        presented_task["assignment_id"],
        CAPTION_YES,
        study_id="audit",
    )
    withdrawn = store.record_consent_profile(
        recovery_token,
        study_id="audit",
        consented=False,
        consent_version="consent-v1",
    )
    assert withdrawn["status"] == "withdrawn"


def test_participant_summary_reports_flow_and_marginals_without_joint_profiles(
    tmp_path: Path,
) -> None:
    store, invites, _ = _open_store(
        tmp_path,
        participant_count=2,
        labels_per_item=2,
    )
    tokens = [
        _session_and_consent(
            store,
            invites[0],
            profile={
                "age_band": "25-34",
                "author_status": "non_author",
                "recruitment_source": "academic_pool",
                "t2i_experience": "some",
            },
        ),
        _session_and_consent(
            store,
            invites[1],
            profile={
                "age_band": "35-44",
                "author_status": "non_author",
                "recruitment_source": "professional_platform",
                "t2i_experience": "professional",
            },
        ),
    ]
    for token in tokens:
        _complete_phase(store, token, "caption", CAPTION_YES)
        _complete_phase(store, token, "image", IMAGE_YES)
    store.record_consent_profile(
        tokens[1],
        consented=False,
        consent_version="consent-v1",
    )

    summary = store.participant_summary("audit")
    assert summary["direct_identifiers_collected"] is False
    assert summary["flow"] == {
        "invited_total": 2,
        "current_status": {
            "invited": 0,
            "active": 1,
            "declined": 0,
            "withdrawn": 1,
        },
        "consented_current": 1,
        "completed": 1,
        "procedurally_completed_total": 2,
        "analyzed": 1,
    }
    profiles = summary["profile_marginals"]
    assert profiles["n_participants"] == 1
    assert profiles["joint_profiles_released"] is False
    assert profiles["fields"]["age_band"]["counts"] == [{"value": "25-34", "count": 1}]
    assert "participant_id" not in str(summary)
    assert "participant_pseudonym" not in str(summary)


def test_phase_projection_is_blinded_and_caption_gate_is_global(tmp_path: Path) -> None:
    store, invites, _ = _open_store(tmp_path, item_count=2)
    token = _session_and_consent(store, invites[0])

    caption_result = store.fetch_next_task(token)
    assert caption_result["status"] == "task"
    assert caption_result["phase"] == "caption"
    caption_task = caption_result["task"]
    assert set(caption_task) == {
        "assignment_id",
        "phase",
        "position",
        "caption",
        "unit",
        "span",
        "target",
        "category",
    }
    assert caption_task["span"] == "A red cat"
    assert {
        "surface",
        "qwen_answer",
        "gemma_answer",
        "image_locator",
        "image_asset",
        "asset_status",
        "item_id",
        "item_group",
        "repeat_of",
    }.isdisjoint(caption_task)

    with sqlite3.connect(store.path) as connection:
        image_assignment_id = connection.execute(
            """
            SELECT assignment_id FROM assignments
            WHERE participant_id = ? AND phase = 'image'
            ORDER BY position LIMIT 1
            """,
            (invites[0]["participant_id"],),
        ).fetchone()[0]
    with pytest.raises(AuthorizationError, match="caption-phase"):
        store.save_annotation(token, image_assignment_id, IMAGE_YES)
    with pytest.raises(AuthorizationError, match="matching caption"):
        store.authorized_image_asset(token, image_assignment_id)

    caption_annotations = _complete_phase(store, token, "caption", CAPTION_YES)
    assert len(caption_annotations) == 2
    image_result = store.fetch_next_task(token)
    assert image_result["status"] == "task"
    assert image_result["phase"] == "image"
    image_task = image_result["task"]
    assert set(image_task) == {
        "assignment_id",
        "phase",
        "position",
        "unit",
        "target",
        "category",
        "image_url",
    }
    assert {
        "caption",
        "span",
        "surface",
        "qwen_answer",
        "gemma_answer",
        "image_locator",
        "image_asset",
        "asset_status",
        "item_id",
        "item_group",
        "repeat_of",
    }.isdisjoint(image_task)
    authorized = store.authorized_image_asset(token, image_task["assignment_id"])
    assert authorized["assignment_id"] == image_task["assignment_id"]
    assert authorized["asset_ref"].startswith("assets/item-")
    coverage = store.authorized_coverage_caption(token, image_task["assignment_id"])
    assert coverage["assignment_id"] == image_task["assignment_id"]
    assert coverage["caption"].startswith("A red cat number ")
    assert coverage["caption"].endswith(" sits on a mat.")
    with pytest.raises(StudyStateError, match="pending assignments"):
        store.set_study_status("audit", "closed")
    with pytest.raises(AuthorizationError, match="frozen"):
        store.save_annotation(
            token,
            caption_annotations[0]["assignment_id"],
            CAPTION_YES,
        )


def test_resume_and_append_only_revision_events(tmp_path: Path) -> None:
    store, invites, clock = _open_store(tmp_path, item_count=2)
    token = _session_and_consent(store, invites[0])
    first = store.fetch_next_task(token)["task"]
    first_saved = store.save_annotation(token, first["assignment_id"], CAPTION_YES)
    assert first_saved["revision"] == 1

    reopened = HumanCBUStore.open(store.path, clock=clock)
    resumed = reopened.fetch_next_task(token)
    assert resumed["status"] == "task"
    assert resumed["phase"] == "caption"
    assert resumed["task"]["assignment_id"] != first["assignment_id"]
    progress = reopened.get_progress(token)
    assert progress["caption"] == {"total": 2, "completed": 1, "remaining": 1}

    revised_payload = {
        "caption_licensed": "no",
        "atomic_visual_claim": "yes",
        "category_check": "incorrect",
        "corrected_category": "relation",
        "reason_tags": ["not_explicit"],
        "confidence": 4,
    }
    revision = reopened.save_annotation(token, first["assignment_id"], revised_payload)
    assert revision["revision"] == 2
    events = reopened.annotation_events("audit", assignment_id=first["assignment_id"])
    assert [event["revision"] for event in events] == [1, 2]
    assert events[0]["payload"]["caption_licensed"] == "yes"
    assert events[1]["payload"]["caption_licensed"] == "no"

    with sqlite3.connect(store.path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "UPDATE annotation_events SET revision = 99 WHERE event_id = ?",
                (events[0]["event_id"],),
            )
    exported = [row for row in reopened.export_rows("audit") if row["assignment_id"] == first["assignment_id"]]
    assert len(exported) == 1
    assert exported[0]["revision"] == 2
    assert exported[0]["caption_licensed"] == "no"
    assert exported[0]["sampling_weight"] == pytest.approx(2.5)
    assert "created_at" not in exported[0]
    assert exported[0]["created_date"] == "2026-07-27"
    exact = reopened.export_rows("audit", include_exact_timestamps=True)
    assert "created_at" in exact[0]


@pytest.mark.parametrize(
    ("phase", "payload", "message"),
    [
        (
            "caption",
            {
                "caption_licensed": "partial",
                "atomic_visual_claim": "yes",
                "category_check": "correct",
            },
            "caption_licensed",
        ),
        (
            "caption",
            {
                "caption_licensed": "yes",
                "atomic_visual_claim": "grammatical",
                "category_check": "correct",
            },
            "atomic_visual_claim",
        ),
        (
            "caption",
            {
                "caption_licensed": "yes",
                "atomic_visual_claim": "yes",
                "category_check": "correct",
                "image_support": "yes",
            },
            "not allowed",
        ),
        ("image", {"image_support": "partial"}, "image_support"),
        (
            "image",
            {"image_support": "yes", "control_usefulness": 6},
            "control_usefulness",
        ),
        (
            "image",
            {"image_support": "yes", "salient_coverage": 6},
            "salient_coverage",
        ),
        (
            "image",
            {"image_support": "yes", "note": "contact rater@example.test"},
            "identifiers",
        ),
    ],
)
def test_invalid_annotation_labels_are_rejected(
    phase: str,
    payload: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        HumanCBUStore.validate_annotation_payload(phase, payload)


def test_control_usefulness_is_only_valid_for_supported_images() -> None:
    for image_support in (
        "no",
        "uncertain",
        "not_visual",
        "image_unavailable",
        "prefer_not_to_answer",
    ):
        with pytest.raises(ValidationError, match="unless image_support is 'yes'"):
            HumanCBUStore.validate_annotation_payload(
                "image",
                {"image_support": image_support, "control_usefulness": 3},
            )

    omitted = HumanCBUStore.validate_annotation_payload(
        "image",
        {"image_support": "yes"},
    )
    assert omitted["control_usefulness"] is None
    cannot_judge = HumanCBUStore.validate_annotation_payload(
        "image",
        {"image_support": "yes", "control_usefulness": "cannot_judge"},
    )
    assert cannot_judge["control_usefulness"] == "cannot_judge"


def test_salient_coverage_is_valid_for_viewable_images_only() -> None:
    for image_support in ("yes", "no", "uncertain", "not_visual"):
        normalized = HumanCBUStore.validate_annotation_payload(
            "image",
            {"image_support": image_support, "salient_coverage": 3},
        )
        assert normalized["salient_coverage"] == 3
    for image_support in ("image_unavailable", "prefer_not_to_answer"):
        with pytest.raises(ValidationError, match="must be null"):
            HumanCBUStore.validate_annotation_payload(
                "image",
                {"image_support": image_support, "salient_coverage": 3},
            )


def test_disagreement_key_includes_correction_but_not_optional_usefulness() -> None:
    caption_a = {
        "caption_licensed": "yes",
        "atomic_visual_claim": "yes",
        "category_check": "incorrect",
        "corrected_category": "object",
    }
    caption_b = {**caption_a, "corrected_category": "relation"}
    assert HumanCBUStore._agreement_key("caption", caption_a) != HumanCBUStore._agreement_key("caption", caption_b)
    image_a = {"image_support": "yes", "control_usefulness": 1}
    image_b = {"image_support": "yes", "control_usefulness": 5}
    assert HumanCBUStore._agreement_key("image", image_a) == HumanCBUStore._agreement_key("image", image_b)


def test_elapsed_time_sensitive_skip_and_not_applicable_are_explicit(
    tmp_path: Path,
) -> None:
    store, invites, _ = _open_store(tmp_path)
    token = _session_and_consent(store, invites[0])
    caption_task = store.fetch_next_task(token)["task"]
    caption = store.save_annotation(
        token,
        caption_task["assignment_id"],
        {
            "caption_licensed": "no",
            "atomic_visual_claim": "not_visual",
            "category_check": "not_applicable",
        },
        elapsed_ms=1234,
    )
    assert caption["payload"]["elapsed_ms"] == 1234
    image_task = store.fetch_next_task(token)["task"]
    image = store.save_annotation(
        token,
        image_task["assignment_id"],
        {"image_support": "prefer_not_to_answer", "control_usefulness": None},
        duration_ms=2345,
    )
    assert image["payload"]["elapsed_ms"] == 2345
    assert image["payload"]["image_support"] == "prefer_not_to_answer"
    summary = store.status_summary("audit")
    assert summary["analysis_redacted"] is True
    assert "label_counts" not in summary
    store.set_study_status("audit", "closed", allow_incomplete=True)
    analysis = store.status_summary("audit", include_analysis=True)
    assert analysis["label_counts"]["image_support"]["prefer_not_to_answer"] == 1
    events = store.annotation_events("audit")
    assert [event["payload"]["elapsed_ms"] for event in events] == [1234, 2345]
    with pytest.raises(ValidationError, match="elapsed_ms"):
        HumanCBUStore.validate_annotation_payload(
            "image",
            {"image_support": "yes", "elapsed_ms": 3_600_001},
        )


def test_withdrawn_participant_cannot_reactivate_and_is_excluded_from_export(
    tmp_path: Path,
) -> None:
    store, invites, _ = _open_store(tmp_path)
    token = _session_and_consent(store, invites[0])
    task = store.fetch_next_task(token)["task"]
    store.save_annotation(token, task["assignment_id"], CAPTION_YES)
    withdrawn = store.record_consent_profile(
        token,
        consented=False,
        consent_version="consent-v1",
        profile={"age_band": "prefer_not_to_say"},
    )
    assert withdrawn["status"] == "withdrawn"
    repeated = store.record_consent_profile(
        token,
        consented=False,
        consent_version="consent-v1",
    )
    assert repeated["status"] == "withdrawn"
    assert store.export_rows("audit") == []
    participant_flow = store.participant_summary("audit")["flow"]
    assert participant_flow["current_status"]["withdrawn"] == 1
    assert participant_flow["current_status"]["declined"] == 0
    with pytest.raises(AuthorizationError, match="cannot be reactivated"):
        store.record_consent_profile(
            token,
            consented=True,
            consent_version="consent-v1",
            profile={"age_band": "25-34"},
        )


def test_withdrawal_while_paused_is_terminal_and_blocks_task_access(
    tmp_path: Path,
) -> None:
    store, invites, _ = _open_store(tmp_path)
    token = _session_and_consent(store, invites[0])
    store.set_study_status("audit", "paused")
    withdrawn = store.record_consent_profile(
        token,
        consented=False,
        consent_version="consent-v1",
    )
    assert withdrawn["status"] == "withdrawn"
    with pytest.raises(StudyStateError, match="not open"):
        store.fetch_next_task(token)


@pytest.mark.parametrize("study_status", ["paused", "closed"])
def test_current_consent_participant_can_reauthenticate_only_to_withdraw(
    tmp_path: Path,
    study_status: str,
) -> None:
    store, invites, _ = _open_store(tmp_path, participant_count=2)
    _session_and_consent(store, invites[0])
    if study_status == "paused":
        store.set_study_status("audit", "paused")
    else:
        store.set_study_status("audit", "closed", allow_incomplete=True)

    replacement_session = store.create_session(invites[0]["invite_code"])
    replacement_token = replacement_session["session_token"]
    assert replacement_session["session_scope"] == SESSION_SCOPE_WITHDRAWAL_ONLY
    assert store.authenticate_session(replacement_token)["session_scope"] == SESSION_SCOPE_WITHDRAWAL_ONLY
    assert store.get_progress(replacement_token)["current_phase"] == "caption"
    with pytest.raises(AuthorizationError, match="limited to withdrawal"):
        store.fetch_next_task(replacement_token)
    with sqlite3.connect(store.path) as connection:
        assignment_id = connection.execute(
            """
            SELECT assignment_id FROM assignments
            WHERE study_id = 'audit' AND participant_id = ?
            ORDER BY position LIMIT 1
            """,
            (invites[0]["participant_id"],),
        ).fetchone()[0]
    with pytest.raises(AuthorizationError, match="limited to withdrawal"):
        store.authorized_image_asset(replacement_token, assignment_id)
    with pytest.raises(AuthorizationError, match="limited to withdrawal"):
        store.save_annotation(replacement_token, assignment_id, CAPTION_YES)
    with pytest.raises(AuthorizationError, match="limited to withdrawal"):
        store.record_consent_profile(
            replacement_token,
            consented=True,
            consent_version="consent-v1",
        )

    withdrawn = store.record_consent_profile(
        replacement_token,
        consented=False,
        consent_version="consent-v1",
    )
    assert withdrawn["status"] == "withdrawn"

    with pytest.raises(StudyStateError, match="limited to active current-consent"):
        store.create_session(invites[1]["invite_code"])


def test_close_requires_exact_current_panel_after_completed_rater_withdraws(
    tmp_path: Path,
) -> None:
    store, invites = _draft_store(
        tmp_path,
        item_count=1,
        participant_count=3,
    )
    store.assign_items(
        "audit",
        labels_per_item=2,
        primary_participant_count=2,
        reserve_participant_count=1,
        seed="close-eligibility-v1",
    )
    store.seal_validation("audit")
    store.set_study_status("audit", "open")
    tokens = [_session_and_consent(store, invite) for invite in invites]
    for token in tokens[:2]:
        _complete_phase(store, token, "caption", CAPTION_YES)
        _complete_phase(store, token, "image", IMAGE_YES)

    store.record_consent_profile(
        tokens[1],
        consented=False,
        consent_version="consent-v1",
    )
    with pytest.raises(StudyStateError, match="pause and replace"):
        store.set_study_status("audit", "closed")
    assert store.get_study("audit")["status"] == "open"

    store.set_study_status("audit", "paused")
    replacement = store.replace_participant(
        "audit",
        source_participant_id=invites[1]["participant_id"],
        replacement_participant_id=invites[2]["participant_id"],
    )
    assert replacement["replacement"]["fresh_rerating_assignment_count"] == 2
    store.set_study_status("audit", "open")
    _complete_phase(store, tokens[2], "caption", CAPTION_YES)
    _complete_phase(store, tokens[2], "image", IMAGE_YES)
    closed = store.set_study_status("audit", "closed")
    assert closed["status"] == "closed"
    analysis = store.status_summary("audit", include_analysis=True)
    assert analysis["agreement"]["caption"]["eligible_panels_complete"] == 1
    assert analysis["agreement"]["caption"]["eligible_panel_shortfalls"] == 0
    assert analysis["agreement"]["image"]["eligible_panels_complete"] == 1
    assert analysis["agreement"]["image"]["eligible_panel_shortfalls"] == 0
    assert analysis["stored_latest_annotations"] == 6
    assert analysis["latest_annotations"] == 4


def test_emergency_close_preserves_eligible_panel_shortfall(
    tmp_path: Path,
) -> None:
    store, invites, _ = _open_store(
        tmp_path,
        participant_count=2,
        labels_per_item=2,
    )
    tokens = [_session_and_consent(store, invite) for invite in invites]
    for token in tokens:
        _complete_phase(store, token, "caption", CAPTION_YES)
        _complete_phase(store, token, "image", IMAGE_YES)
    store.record_consent_profile(
        tokens[1],
        consented=False,
        consent_version="consent-v1",
    )

    with pytest.raises(StudyStateError, match="pause and replace"):
        store.set_study_status("audit", "closed")
    closed = store.set_study_status(
        "audit",
        "closed",
        allow_incomplete=True,
    )
    assert closed["status"] == "closed"
    analysis = store.status_summary("audit", include_analysis=True)
    for phase in ("caption", "image"):
        assert analysis["agreement"][phase]["eligible_panels_complete"] == 0
        assert analysis["agreement"][phase]["eligible_panel_shortfalls"] == 1
        assert analysis["assignments"][phase] == {
            "pending": 0,
            "completed": 1,
            "total": 1,
        }
        assert analysis["stored_assignments"][phase] == {
            "pending": 0,
            "completed": 2,
            "total": 2,
        }


def test_post_close_withdrawal_invalidates_panel_and_suppresses_adjudication(
    tmp_path: Path,
) -> None:
    store, invites, _ = _open_store(
        tmp_path,
        participant_count=2,
        labels_per_item=2,
    )
    tokens = [_session_and_consent(store, invite) for invite in invites]
    caption_payloads = (
        CAPTION_YES,
        {
            "caption_licensed": "no",
            "atomic_visual_claim": "not_visual",
            "category_check": "not_applicable",
        },
    )
    for token, caption_payload in zip(tokens, caption_payloads, strict=True):
        _complete_phase(store, token, "caption", caption_payload)
        _complete_phase(store, token, "image", IMAGE_YES)
    store.set_study_status("audit", "closed")
    store.adjudicate(
        "audit",
        "item-0",
        "caption",
        CAPTION_YES,
        adjudicator_pseudonym="adjudicator-a",
    )

    withdrawn = store.record_consent_profile(
        tokens[1],
        consented=False,
        consent_version="consent-v1",
    )
    assert withdrawn["status"] == "withdrawn"
    analysis = store.status_summary("audit", include_analysis=True)
    assert analysis["label_counts"]["caption_licensed"]["yes"] == 1
    assert analysis["label_counts"]["caption_licensed"]["no"] == 0
    caption_agreement = analysis["agreement"]["caption"]
    assert caption_agreement["items_with_two_or_more_labels"] == 0
    assert caption_agreement["disagreements"] == 0
    assert caption_agreement["adjudicated"] == 0
    assert caption_agreement["eligible_panel_shortfalls"] == 1
    assert caption_agreement["eligible_panels_complete"] == 0
    with pytest.raises(ValidationError, match=r"1/2 labels available"):
        store.adjudicate("audit", "item-0", "caption", CAPTION_YES)

    exported = store.export_rows("audit")
    assert {row["participant_pseudonym"] for row in exported} == {invites[0]["pseudonym"]}
    assert all(row["adjudication"] is None for row in exported)


def test_asset_validation_seal_freezes_manifest_and_backup_is_consistent(
    tmp_path: Path,
) -> None:
    clock = MutableClock()
    store = HumanCBUStore.initialize(tmp_path / "assets.sqlite3", clock=clock)
    store.create_study(
        "audit",
        title="Asset validation",
        protocol_version="v1",
        consent_version="c1",
        metadata={"required_labels_per_item": 2},
    )
    store.add_item(
        "audit",
        **_item(0, image_asset=None, asset_status="pending"),
    )
    invite = store.create_participant_invites("audit", count=2)[0]
    store.assign_items("audit", labels_per_item=2)
    report = store.validation_report("audit")
    assert not report["valid"]
    assert report["asset_failures"] == [{"item_id": "item-0", "status": "pending"}]
    with pytest.raises(ValidationError, match="cannot be sealed"):
        store.seal_validation("audit", require_assets=False)
    with pytest.raises(ValidationError, match="image assets"):
        store.seal_validation("audit")
    with pytest.raises(ValidationError, match="safe relative"):
        store.set_asset_status(
            "audit",
            "item-0",
            "available",
            image_asset={"path": "https://example.test/private.jpg"},
        )

    status = store.set_asset_status(
        "audit",
        "item-0",
        "available",
        image_asset={"route": "assets/item-0.jpg", "sha256": "abc"},
    )
    assert status["has_asset"]
    store.seal_validation("audit")
    with pytest.raises(StudyStateError):
        store.set_asset_status("audit", "item-0", "invalid")
    with pytest.raises(StudyStateError):
        store.add_item("audit", **_item(1))
    with pytest.raises(StudyStateError):
        store.assign_items("audit", labels_per_item=1, seed="different")

    backup_path = store.backup(tmp_path / "exports" / "human-cbu-backup.sqlite3")
    backup = HumanCBUStore.open(backup_path, clock=clock)
    assert backup.status_summary("audit")["items"] == 1
    assert backup.get_asset_status("audit", "item-0")["status"] == "available"
    with sqlite3.connect(backup_path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    with pytest.raises(FileExistsError):
        store.backup(backup_path)
    assert invite["invite_code"] not in backup_path.read_bytes().decode("utf-8", errors="ignore")


def test_open_rechecks_launch_metadata_and_same_origin_notice_route(
    tmp_path: Path,
) -> None:
    store = HumanCBUStore.initialize(tmp_path / "launch-gate.sqlite3")
    store.create_study(
        "audit",
        title="Launch gate",
        protocol_version="v1",
        consent_version="consent-v1",
        metadata={"required_labels_per_item": 2},
    )
    store.add_item("audit", **_item(0))
    store.create_participant_invites("audit", count=2)
    store.assign_items("audit", labels_per_item=2)
    store.seal_validation("audit")

    with pytest.raises(StudyStateError, match="institutional"):
        store.set_study_status("audit", "open")
    assert store.get_study("audit")["status"] == "ready"

    # Sealed metadata is immutable, so a valid study must have passed the gate
    # before sealing. This second study verifies the direct-store happy path.
    valid_metadata = _launch_metadata()
    valid_metadata["participant_information"]["information_sheet_url"] = "/study-information"
    valid = HumanCBUStore.initialize(tmp_path / "launch-valid.sqlite3")
    valid.create_study(
        "audit",
        title="Launch gate",
        protocol_version="v1",
        consent_version="consent-v1",
        metadata={"required_labels_per_item": 2, **valid_metadata},
    )
    valid.add_item("audit", **_item(0))
    valid.create_participant_invites("audit", count=2)
    valid.assign_items("audit", labels_per_item=2)
    valid.seal_validation("audit")
    assert valid.set_study_status("audit", "open")["status"] == "open"

    invalid_metadata = _launch_metadata()
    invalid_metadata["participant_information"]["information_sheet_url"] = "/other-page"
    invalid = HumanCBUStore.initialize(tmp_path / "launch-invalid-url.sqlite3")
    invalid.create_study(
        "audit",
        title="Launch gate",
        protocol_version="v1",
        consent_version="consent-v1",
        metadata={"required_labels_per_item": 2, **invalid_metadata},
    )
    invalid.add_item("audit", **_item(0))
    invalid.create_participant_invites("audit", count=2)
    invalid.assign_items("audit", labels_per_item=2)
    invalid.seal_validation("audit")
    with pytest.raises(StudyStateError, match="credential-free HTTPS URL"):
        invalid.set_study_status("audit", "open")


@pytest.mark.parametrize(
    ("required_labels", "assigned_labels", "expected_error"),
    [
        (2, 1, "labels_per_item must be at least 2"),
        (3, 2, "does not match frozen required_labels_per_item"),
    ],
)
def test_validation_and_seal_reject_unsafe_or_mismatched_label_panels(
    tmp_path: Path,
    required_labels: int,
    assigned_labels: int,
    expected_error: str,
) -> None:
    store = HumanCBUStore.initialize(
        tmp_path / f"labels-{required_labels}-{assigned_labels}.sqlite3",
        clock=MutableClock(),
    )
    store.create_study(
        "audit",
        title="Label panel freeze",
        protocol_version="v1",
        consent_version="c1",
        metadata={"required_labels_per_item": required_labels},
    )
    store.add_item("audit", **_item(0))
    store.create_participant_invites("audit", count=required_labels)
    store.assign_items("audit", labels_per_item=required_labels)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE studies SET labels_per_item = ? WHERE study_id = 'audit'",
            (assigned_labels,),
        )

    report = store.validation_report("audit")
    assert report["valid"] is False
    assert report["required_labels_per_item"] == required_labels
    assert any(expected_error in error for error in report["errors"])
    with pytest.raises(ValidationError, match=expected_error):
        store.seal_validation("audit")


def test_assignment_rejects_frozen_label_count_mismatch_without_writes(
    tmp_path: Path,
) -> None:
    store, _ = _draft_store(tmp_path, participant_count=2)
    with pytest.raises(
        ValidationError,
        match="does not match frozen required_labels_per_item",
    ):
        store.assign_items("audit", labels_per_item=1)
    summary = store.status_summary("audit")
    assert all(phase["total"] == 0 for phase in summary["assignments"].values())


def test_human_cbu_v1_assignment_requires_exactly_two_without_writes(
    tmp_path: Path,
) -> None:
    store = HumanCBUStore.initialize(tmp_path / "v1-three-labels.sqlite3")
    store.create_study(
        "audit",
        title="Exact initial panel",
        protocol_version="human_cbu_v1",
        consent_version="consent-v1",
        metadata=_v1_metadata(required_labels_per_item=3),
    )
    store.add_item("audit", **_item(0))
    store.create_participant_invites("audit", count=3)

    with pytest.raises(
        ValidationError,
        match="human_cbu_v1 requires exactly 2 initial labels per item",
    ):
        store.assign_items("audit", labels_per_item=3)

    assert store.get_study("audit")["labels_per_item"] is None
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0] == 0


def test_human_cbu_v1_seal_rechecks_exact_frozen_two_label_plan(
    tmp_path: Path,
) -> None:
    store = HumanCBUStore.initialize(tmp_path / "v1-seal-labels.sqlite3")
    store.create_study(
        "audit",
        title="Exact initial panel",
        protocol_version="human_cbu_v1",
        consent_version="consent-v1",
        metadata=_v1_metadata(),
    )
    store.add_item("audit", **_item(0))
    store.create_participant_invites("audit", count=2)
    store.assign_items("audit", labels_per_item=2)
    store.provision_adjudicator("audit")
    tampered_metadata = dict(store.get_study("audit")["metadata"])
    tampered_metadata["required_labels_per_item"] = 3
    store.update_study_metadata("audit", metadata=tampered_metadata)

    report = store.validation_report("audit")
    assert report["valid"] is False
    assert "human_cbu_v1 requires exactly 2 initial labels per item" in report["errors"]
    with pytest.raises(
        ValidationError,
        match="human_cbu_v1 requires exactly 2 initial labels per item",
    ):
        store.seal_validation("audit")
    assert store.get_study("audit")["status"] == "draft"


def test_human_cbu_v1_open_rechecks_exact_two_after_seal(
    tmp_path: Path,
) -> None:
    store = HumanCBUStore.initialize(tmp_path / "v1-open-labels.sqlite3")
    store.create_study(
        "audit",
        title="Exact initial panel",
        protocol_version="human_cbu_v1",
        consent_version="consent-v1",
        metadata=_v1_metadata(),
    )
    store.add_item("audit", **_item(0))
    store.create_participant_invites("audit", count=2)
    store.assign_items("audit", labels_per_item=2)
    store.provision_adjudicator("audit")
    store.seal_validation("audit")
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE studies SET labels_per_item = 3 WHERE study_id = 'audit'")

    with pytest.raises(
        StudyStateError,
        match="human_cbu_v1 requires exactly 2 initial labels per item",
    ):
        store.set_study_status("audit", "open")
    assert store.get_study("audit")["status"] == "ready"


def test_human_cbu_v1_accepts_frozen_repeat_payload_at_validation_seal_and_open(
    tmp_path: Path,
) -> None:
    store, _ = _prepared_v1_store(tmp_path, items=_v1_repeat_pair())

    report = store.validation_report("audit")
    assert report["valid"] is True
    assert report["expected_repeat_items"] == 1
    assert report["repeat_items"] == 1
    assert report["repeat_payload_failures"] == []
    assert report["semantic_visual_claim_types"] == list(SEMANTIC_VISUAL_CLAIM_TYPES)
    assert store.seal_validation("audit")["valid"] is True
    assert store.set_study_status("audit", "open")["status"] == "open"


@pytest.mark.parametrize(
    ("column", "value", "expected_field"),
    [
        ("caption", "A blue dog sits elsewhere.", "caption"),
        ("surface", "tampered", "surface"),
        (
            "image_locator_json",
            json.dumps({"kind": "cc12m", "url_hash": "different"}),
            "image_locator",
        ),
        ("qwen_answer_json", json.dumps({"support": "no"}), "qwen_answer"),
        ("gemma_answer_json", json.dumps({"support": "no"}), "gemma_answer"),
        ("stratum", "relation", "stratum"),
        ("weight", 1.0, "weight"),
        ("sample_probability", 0.25, "sample_probability"),
    ],
)
def test_human_cbu_v1_direct_repeat_tampering_blocks_validation_and_seal(
    tmp_path: Path,
    column: str,
    value: Any,
    expected_field: str,
) -> None:
    store, _ = _prepared_v1_store(tmp_path, items=_v1_repeat_pair())
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            f"UPDATE items SET {column} = ? WHERE study_id = 'audit' AND item_id = 'repeat-0'",
            (value,),
        )

    report = store.validation_report("audit")
    assert report["valid"] is False
    assert any(failure["field"].startswith(expected_field) for failure in report["repeat_payload_failures"])
    with pytest.raises(ValidationError, match="hidden repeats"):
        store.seal_validation("audit")
    assert store.get_study("audit")["status"] == "draft"


def test_human_cbu_v1_expected_repeat_count_blocks_validation_and_seal(
    tmp_path: Path,
) -> None:
    store, _ = _prepared_v1_store(
        tmp_path,
        items=[_item(0)],
        metadata=_v1_metadata(expected_repeat_items=1),
    )

    report = store.validation_report("audit")
    assert report["valid"] is False
    assert report["expected_repeat_items"] == 1
    assert report["repeat_items"] == 0
    assert any("hidden-repeat count" in error for error in report["errors"])
    with pytest.raises(ValidationError, match="hidden-repeat count"):
        store.seal_validation("audit")


def test_human_cbu_v1_malformed_repeat_json_fails_closed_without_crashing(
    tmp_path: Path,
) -> None:
    store, _ = _prepared_v1_store(tmp_path, items=_v1_repeat_pair())
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE items SET qwen_answer_json = '{'
            WHERE study_id = 'audit' AND item_id = 'repeat-0'
            """
        )

    report = store.validation_report("audit")
    assert report["valid"] is False
    assert any("malformed stored JSON" in error for error in report["errors"])
    with pytest.raises(ValidationError, match="malformed stored JSON"):
        store.seal_validation("audit")


def test_human_cbu_v1_open_rechecks_repeat_identity_after_seal(
    tmp_path: Path,
) -> None:
    store, _ = _prepared_v1_store(tmp_path, items=_v1_repeat_pair())
    assert store.seal_validation("audit")["valid"] is True
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE items SET qwen_answer_json = ?
            WHERE study_id = 'audit' AND item_id = 'repeat-0'
            """,
            (json.dumps({"support": "no"}),),
        )

    with pytest.raises(StudyStateError, match="frozen human_cbu_v1 design"):
        store.set_study_status("audit", "open")
    assert store.get_study("audit")["status"] == "ready"


@pytest.mark.parametrize(
    "semantic_types",
    [
        None,
        [],
        "attribute",
        ["attribute", "attribute"],
        ["attribute", "unsupported_type"],
        [1],
    ],
)
def test_human_cbu_v1_rejects_invalid_frozen_semantic_type_vocabulary(
    tmp_path: Path,
    semantic_types: Any,
) -> None:
    metadata = _v1_metadata()
    metadata["semantic_visual_claim_types"] = semantic_types
    store, _ = _prepared_v1_store(
        tmp_path,
        items=[_item(0)],
        metadata=metadata,
    )

    report = store.validation_report("audit")
    assert report["valid"] is False
    assert any("semantic_visual_claim_types" in error for error in report["errors"])
    with pytest.raises(ValidationError, match="semantic_visual_claim_types"):
        store.seal_validation("audit")


def test_human_cbu_v1_rejects_item_category_outside_frozen_vocabulary(
    tmp_path: Path,
) -> None:
    metadata = _v1_metadata()
    metadata["semantic_visual_claim_types"] = ["object"]
    store, _ = _prepared_v1_store(
        tmp_path,
        items=[_item(0)],
        metadata=metadata,
    )

    report = store.validation_report("audit")
    assert report["valid"] is False
    assert any("item categories are outside" in error for error in report["errors"])
    with pytest.raises(ValidationError, match="item categories are outside"):
        store.seal_validation("audit")


def test_human_cbu_v1_open_rechecks_frozen_semantic_vocabulary_after_seal(
    tmp_path: Path,
) -> None:
    store, _ = _prepared_v1_store(tmp_path, items=[_item(0)])
    assert store.seal_validation("audit")["valid"] is True
    metadata = store.get_study("audit")["metadata"]
    metadata["semantic_visual_claim_types"] = []
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE studies SET metadata_json = ? WHERE study_id = 'audit'",
            (json.dumps(metadata),),
        )

    with pytest.raises(StudyStateError, match="semantic_visual_claim_types"):
        store.set_study_status("audit", "open")
    assert store.get_study("audit")["status"] == "ready"


def test_human_cbu_v1_caption_correction_is_item_aware_and_frozen(
    tmp_path: Path,
) -> None:
    store, invites = _prepared_v1_store(tmp_path, items=[_item(0)])
    store.seal_validation("audit")
    store.set_study_status("audit", "open")
    token = _session_and_consent(store, invites[0])
    task = store.fetch_next_task(token)["task"]
    assert task["category"] == "attribute"
    incorrect = {
        "caption_licensed": "yes",
        "atomic_visual_claim": "yes",
        "category_check": "incorrect",
    }

    with pytest.raises(
        ValidationError,
        match="frozen semantic visual-claim types",
    ):
        store.save_annotation(
            token,
            task["assignment_id"],
            {**incorrect, "corrected_category": "unsupported_type"},
        )
    with pytest.raises(
        ValidationError,
        match="must differ from the proposed category",
    ):
        store.save_annotation(
            token,
            task["assignment_id"],
            {**incorrect, "corrected_category": "attribute"},
        )
    with pytest.raises(
        ValidationError,
        match="only allowed when category_check is 'incorrect'",
    ):
        store.save_annotation(
            token,
            task["assignment_id"],
            {**CAPTION_YES, "corrected_category": None},
        )
    assert store.annotation_events("audit") == []

    saved = store.save_annotation(
        token,
        task["assignment_id"],
        {**incorrect, "corrected_category": "object"},
    )
    assert saved["payload"]["corrected_category"] == "object"
    assert len(store.annotation_events("audit")) == 1


def test_non_human_cbu_protocol_can_freeze_three_initial_labels(
    tmp_path: Path,
) -> None:
    store = HumanCBUStore.initialize(tmp_path / "generic-three-labels.sqlite3")
    store.create_study(
        "audit",
        title="Generic three-person panel",
        protocol_version="protocol-v1",
        consent_version="consent-v1",
        metadata={"required_labels_per_item": 3, **_launch_metadata()},
    )
    store.add_item("audit", **_item(0))
    store.create_participant_invites("audit", count=3)
    store.assign_items("audit", labels_per_item=3)
    assert store.seal_validation("audit")["valid"] is True
    assert store.set_study_status("audit", "open")["status"] == "open"


def test_assignment_rejects_frozen_seed_mismatch_without_writes(
    tmp_path: Path,
) -> None:
    store = HumanCBUStore.initialize(tmp_path / "frozen-seed.sqlite3")
    store.create_study(
        "audit",
        title="Frozen assignment seed",
        protocol_version="v1",
        consent_version="c1",
        metadata={
            "required_labels_per_item": 2,
            "assignment_seed": "1477",
        },
    )
    store.add_item("audit", **_item(0))
    store.create_participant_invites("audit", count=2)

    with pytest.raises(ValidationError, match="seed does not match frozen assignment_seed"):
        store.assign_items("audit", labels_per_item=2, seed="other")
    summary = store.status_summary("audit")["assignments"]
    assert all(phase["total"] == 0 for phase in summary.values())


def test_initialize_rejects_unknown_database_without_mutating_it(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE legacy_only(value TEXT)")
        connection.execute("INSERT INTO legacy_only VALUES ('preserve-me')")
    with pytest.raises(HumanCBUStoreError, match="without schema metadata"):
        HumanCBUStore.initialize(path)
    with sqlite3.connect(path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert tables == {"legacy_only"}
        assert connection.execute("SELECT value FROM legacy_only").fetchone()[0] == "preserve-me"


def test_hidden_repeats_are_deterministically_spaced_when_feasible(
    tmp_path: Path,
) -> None:
    store = HumanCBUStore.initialize(tmp_path / "repeat-spacing.sqlite3", clock=MutableClock())
    store.create_study(
        "audit",
        title="Repeat spacing",
        protocol_version="v1",
        consent_version="c1",
    )
    items = [_item(0), _item(1, repeat_of="item-0", group="image-0")]
    items.extend(_item(index) for index in range(2, 30))
    store.add_items("audit", items)
    store.create_participant_invite("audit")
    store.assign_items("audit", labels_per_item=1, seed="spacing-seed")
    report = store.validation_report("audit")
    assert report["repeat_spacing_failures"] == []
    assert len(report["repeat_spacing"]) == 2
    assert all(row["position_distance"] >= 10 for row in report["repeat_spacing"])


def test_deterministic_assignments_and_repeats_use_same_raters(tmp_path: Path) -> None:
    store = HumanCBUStore.initialize(tmp_path / "assign.sqlite3", clock=MutableClock())
    store.create_study(
        "audit",
        title="Assignment plan",
        protocol_version="v1",
        consent_version="c1",
    )
    store.add_items(
        "audit",
        [
            _item(0),
            _item(1, repeat_of="item-0", group="image-0"),
            _item(2),
        ],
    )
    store.create_participant_invites("audit", count=3)
    first_summary = store.assign_items(
        "audit",
        labels_per_item=2,
        seed="predeclared-seed",
    )
    second_summary = store.assign_items(
        "audit",
        labels_per_item=2,
        seed="predeclared-seed",
    )
    assert first_summary == second_summary
    assert first_summary["assignments"] == {"caption": 6, "image": 6}

    with sqlite3.connect(store.path) as connection:
        connection.row_factory = sqlite3.Row
        for phase in ("caption", "image"):
            original = {
                row["participant_id"]
                for row in connection.execute(
                    """
                    SELECT participant_id FROM assignments
                    WHERE study_id = 'audit' AND item_id = 'item-0' AND phase = ?
                    """,
                    (phase,),
                )
            }
            repeated = {
                row["participant_id"]
                for row in connection.execute(
                    """
                    SELECT participant_id FROM assignments
                    WHERE study_id = 'audit' AND item_id = 'item-1' AND phase = ?
                    """,
                    (phase,),
                )
            }
            assert original == repeated
        duplicate_positions = connection.execute(
            """
            SELECT participant_id, phase, position, COUNT(*) AS n
            FROM assignments
            GROUP BY participant_id, phase, position
            HAVING n > 1
            """
        ).fetchall()
        assert duplicate_positions == []

    with sqlite3.connect(store.path) as connection:
        all_participants = [
            row[0] for row in connection.execute("SELECT participant_id FROM participants ORDER BY participant_id")
        ]
        assigned_participants = {
            row[0] for row in connection.execute("SELECT DISTINCT participant_id FROM assignments")
        }
    if len(assigned_participants) == 3:
        changed_plan = all_participants[:2]
    else:
        omitted = next(
            participant_id for participant_id in all_participants if participant_id not in assigned_participants
        )
        changed_plan = [next(iter(assigned_participants)), omitted]
    with pytest.raises(ConflictError, match="different plan"):
        store.assign_items(
            "audit",
            labels_per_item=2,
            participant_ids=changed_plan,
            seed="predeclared-seed",
        )
    with pytest.raises(StudyStateError, match="invitations are frozen"):
        store.create_participant_invite("audit")


def test_predeclared_reserve_replaces_withdrawn_primary_and_rerates_full_plan(
    tmp_path: Path,
) -> None:
    store, invites = _draft_store(
        tmp_path,
        item_count=3,
        participant_count=4,
    )
    plan = store.assign_items(
        "audit",
        labels_per_item=2,
        primary_participant_count=3,
        reserve_participant_count=1,
        seed="reserve-plan-v1",
    )
    assert plan["planned_primary_participants"] == 3
    assert plan["reserve_participants"] == 1
    reserve_id = invites[-1]["participant_id"]
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM assignments WHERE participant_id = ?",
                (reserve_id,),
            ).fetchone()[0]
            == 0
        )
        source_id = connection.execute(
            """
            SELECT participant_id
            FROM assignments
            WHERE participant_id != ?
            GROUP BY participant_id
            ORDER BY COUNT(*) DESC, participant_id
            LIMIT 1
            """,
            (reserve_id,),
        ).fetchone()[0]
    source_invite = next(invite for invite in invites if invite["participant_id"] == source_id)
    reserve_invite = invites[-1]

    store.seal_validation("audit")
    store.set_study_status("audit", "open")
    source_token = _session_and_consent(store, source_invite)
    reserve_session = store.create_session(reserve_invite["invite_code"])
    first_source_task = store.fetch_next_task(source_token)["task"]
    assert first_source_task["phase"] == "caption"
    store.save_annotation(
        source_token,
        first_source_task["assignment_id"],
        CAPTION_YES,
    )
    store.record_consent_profile(
        source_token,
        consented=False,
        consent_version="consent-v1",
    )
    with sqlite3.connect(store.path) as connection:
        source_total_before = connection.execute(
            "SELECT COUNT(*) FROM assignments WHERE participant_id = ?",
            (source_id,),
        ).fetchone()[0]
        source_completed_before = connection.execute(
            """
            SELECT COUNT(*) FROM assignments
            WHERE participant_id = ? AND status = 'completed'
            """,
            (source_id,),
        ).fetchone()[0]
    assert source_completed_before == 1

    store.set_study_status("audit", "paused")
    with pytest.raises(ValidationError, match="active with the study's current consent"):
        store.replace_participant(
            "audit",
            source_participant_id=source_id,
            replacement_participant_id=reserve_id,
        )
    store.set_study_status("audit", "open")
    store.record_consent_profile(
        reserve_session["session_token"],
        consented=True,
        consent_version="consent-v1",
        profile={"author_status": "non_author"},
    )
    store.set_study_status("audit", "paused")

    result = store.replace_participant(
        "audit",
        source_participant_id=source_id,
        replacement_participant_id=reserve_id,
    )
    event = result["replacement"]
    assert result["idempotent"] is False
    assert event["source_assignment_count"] == source_total_before
    assert event["source_completed_assignment_count"] == 1
    assert event["fresh_rerating_assignment_count"] == 1
    assert event["transferred_pending_assignment_count"] == source_total_before - 1
    assert event["remaining_reserve_participants"] == 0
    assert len(event["event_digest"]) == 64
    assert result["assignment_plan"]["planned_primary_participants"] == 3
    assert result["assignment_plan"]["reserve_participants"] == 0
    assert result["assignment_plan"]["assignments"] == {
        "caption": 6,
        "image": 6,
    }
    assert result["assignment_plan"]["stored_assignments"] == {
        "caption": 7,
        "image": 6,
    }

    with sqlite3.connect(store.path) as connection:
        connection.row_factory = sqlite3.Row
        source_rows = connection.execute(
            "SELECT * FROM assignments WHERE participant_id = ?",
            (source_id,),
        ).fetchall()
        replacement_rows = connection.execute(
            "SELECT * FROM assignments WHERE participant_id = ?",
            (reserve_id,),
        ).fetchall()
        assert len(source_rows) == source_completed_before
        assert all(row["status"] == "completed" for row in source_rows)
        assert len(replacement_rows) == source_total_before
        assert all(row["status"] == "pending" for row in replacement_rows)
        assert all(row["first_presented_at"] is None for row in replacement_rows)
        assert all(row["presentation_count"] == 0 for row in replacement_rows)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []

    validation = store.validation_report("audit")
    assert validation["valid"] is True
    assert validation["assignment_failures"] == []
    status = store.status_summary("audit")
    assert status["assignments"]["caption"] == {
        "pending": 6,
        "completed": 0,
        "total": 6,
    }
    assert status["stored_assignments"]["caption"] == {
        "pending": 6,
        "completed": 1,
        "total": 7,
    }
    assert status["latest_annotations"] == 0
    assert status["stored_latest_annotations"] == 1
    repeated = store.replace_participant(
        "audit",
        source_participant_id=source_id,
        replacement_participant_id=reserve_id,
    )
    assert repeated["idempotent"] is True
    assert repeated["replacement"]["event_digest"] == event["event_digest"]

    store.set_study_status("audit", "open")
    _complete_phase(
        store,
        reserve_session["session_token"],
        "caption",
        CAPTION_YES,
    )
    _complete_phase(
        store,
        reserve_session["session_token"],
        "image",
        IMAGE_YES,
    )
    exported = store.export_rows("audit")
    reserve_pseudonym = reserve_invite["pseudonym"]
    assert sum(row["participant_pseudonym"] == reserve_pseudonym for row in exported) == source_total_before
    assert all(row["participant_pseudonym"] != source_invite["pseudonym"] for row in exported)
    assert store.participant_summary("audit")["flow"]["procedurally_completed_total"] == 1


def test_participant_replacement_rejects_nonreserve_and_requires_pause(
    tmp_path: Path,
) -> None:
    store, invites = _draft_store(
        tmp_path,
        item_count=1,
        participant_count=3,
    )
    store.assign_items(
        "audit",
        labels_per_item=2,
        primary_participant_count=2,
        reserve_participant_count=1,
    )
    source_id = invites[0]["participant_id"]
    reserve_id = invites[2]["participant_id"]
    store.seal_validation("audit")
    store.set_study_status("audit", "open")
    source_token = _session_and_consent(store, invites[0])
    _session_and_consent(store, invites[1])
    _session_and_consent(store, invites[2])
    store.record_consent_profile(
        source_token,
        consented=False,
        consent_version="consent-v1",
    )
    with pytest.raises(StudyStateError, match="paused"):
        store.replace_participant(
            "audit",
            source_participant_id=source_id,
            replacement_participant_id=reserve_id,
        )
    store.set_study_status("audit", "paused")
    with pytest.raises(ValidationError, match="predeclared reserve"):
        store.replace_participant(
            "audit",
            source_participant_id=source_id,
            replacement_participant_id=invites[1]["participant_id"],
        )


def test_two_rater_disagreement_requires_and_resolves_adjudication(tmp_path: Path) -> None:
    store, invites, _ = _open_store(
        tmp_path,
        participant_count=2,
        labels_per_item=2,
    )
    tokens = [_session_and_consent(store, invite) for invite in invites]
    first_payload = CAPTION_YES
    second_payload = {
        "caption_licensed": "no",
        "atomic_visual_claim": "not_visual",
        "category_check": "not_applicable",
    }
    for token, payload in zip(tokens, (first_payload, second_payload), strict=True):
        task = store.fetch_next_task(token)["task"]
        assert task["phase"] == "caption"
        store.save_annotation(token, task["assignment_id"], payload)
    for token, payload in zip(
        tokens,
        (
            {"image_support": "yes", "control_usefulness": 1},
            {"image_support": "no"},
        ),
        strict=True,
    ):
        task = store.fetch_next_task(token)["task"]
        assert task["phase"] == "image"
        store.save_annotation(token, task["assignment_id"], payload)

    open_summary = store.status_summary("audit")
    assert open_summary["analysis_redacted"] is True
    assert "agreement" not in open_summary
    with pytest.raises(StudyStateError, match="unavailable until the study is closed"):
        store.status_summary("audit", include_analysis=True)
    with pytest.raises(StudyStateError):
        store.adjudicate("audit", "item-0", "caption", CAPTION_YES)

    store.set_study_status("audit", "closed")
    pre_adjudication = store.status_summary("audit", include_analysis=True)
    caption_agreement = pre_adjudication["agreement"]["caption"]
    assert caption_agreement == {
        "items_with_two_or_more_labels": 1,
        "disagreements": 1,
        "adjudicated": 0,
        "unresolved_disagreements": 1,
        "eligible_panel_shortfalls": 0,
        "eligible_panels_complete": 1,
        "required_labels_per_item": 2,
    }
    assert pre_adjudication["agreement"]["image"]["disagreements"] == 1
    with pytest.raises(
        ValidationError,
        match="must differ from every eligible initial rater pseudonym",
    ):
        store.adjudicate(
            "audit",
            "item-0",
            "caption",
            CAPTION_YES,
            adjudicator_pseudonym=invites[0]["pseudonym"],
        )
    assert store.adjudication_events("audit") == []
    decision = store.adjudicate(
        "audit",
        "item-0",
        "caption",
        CAPTION_YES,
        adjudicator_pseudonym="adjudicator-a",
    )
    assert decision["revision"] == 1
    revised_decision = store.adjudicate(
        "audit",
        "item-0",
        "caption",
        CAPTION_YES,
        adjudicator_pseudonym="adjudicator-b",
    )
    assert revised_decision["revision"] == 2
    image_decision = store.adjudicate(
        "audit",
        "item-0",
        "image",
        {"image_support": "yes", "control_usefulness": 3},
        adjudicator_pseudonym="adjudicator-a",
    )
    assert image_decision["revision"] == 1
    events = store.adjudication_events("audit")
    assert [event["revision"] for event in events] == [1, 2, 1]
    assert [event["adjudicator_pseudonym"] for event in events[:2]] == [
        "adjudicator-a",
        "adjudicator-b",
    ]
    closed_summary = store.status_summary("audit", include_analysis=True)
    assert closed_summary["agreement"]["caption"]["adjudicated"] == 1
    assert closed_summary["agreement"]["caption"]["unresolved_disagreements"] == 0
    assert closed_summary["agreement"]["image"]["adjudicated"] == 1
    assert closed_summary["adjudication_events"] == 3
    exported = store.export_rows("audit")
    caption_exports = [row for row in exported if row["phase"] == "caption"]
    assert {row["adjudication"]["caption_licensed"] for row in caption_exports} == {"yes"}
    assert {row["adjudication"]["revision"] for row in caption_exports} == {2}


def test_v1_adjudication_packets_are_deterministic_phase_safe_and_hashed(
    tmp_path: Path,
) -> None:
    store, _, credential = _v1_disagreement_store(tmp_path, close=False)
    with pytest.raises(StudyStateError):
        store.adjudication_packets("audit", "caption")
    store.set_study_status("audit", "closed")

    caption_packets = store.adjudication_packets("audit", "caption")
    assert caption_packets == store.adjudication_packets("audit", "caption")
    caption_packet_inputs = store.adjudication_packet_bundle_inputs("audit", "caption")
    assert caption_packet_inputs == {
        "packets": caption_packets,
        "image_sources": {},
    }
    assert len(caption_packets) == 1
    caption = caption_packets[0]
    with pytest.raises(StudyStateError, match="caption disagreement"):
        store.adjudication_packets("audit", "image")
    caption_packet_dir, _ = _adjudication_packet_response(
        tmp_path,
        name="caption-phase",
        packet=caption,
        decision=CAPTION_YES,
    )
    assert caption_packet_dir.parent.name == "packets"
    assert caption_packet_dir.name == caption["packet_id"]
    assert caption_packet_dir.stat().st_mode & 0o777 == 0o700
    assert {path.name for path in caption_packet_dir.iterdir()} == {
        "packet.json",
        "review.html",
        "response.json",
    }
    for filename in ("packet.json", "review.html", "response.json"):
        assert (caption_packet_dir / filename).stat().st_mode & 0o777 == 0o600
    assert not (caption_packet_dir.parent.parent / "packets.json").exists()
    response_payload = json.loads((caption_packet_dir / "response.json").read_text(encoding="utf-8"))
    assert set(response_payload) == {
        "response_version",
        "review_version",
        "packet_file_sha256",
        "packet",
        "decision",
    }
    assert response_payload["packet"] == caption
    assert response_payload["decision"] == CAPTION_YES
    caption_decision = _submit_packet_response(
        store,
        credential,
        caption_packet_dir,
        phase="caption",
    )
    assert caption_decision["packet_id"] == caption["packet_id"]
    assert caption_decision["idempotent_replay"] is False
    image_packets = store.adjudication_packets("audit", "image")
    assert len(image_packets) == 1
    image = image_packets[0]
    image_packet_inputs = store.adjudication_packet_bundle_inputs("audit", "image")
    assert image_packet_inputs["packets"] == image_packets
    assert image_packet_inputs["image_sources"] == {
        image["packet_id"]: "assets/item-0.jpg",
    }

    for packet in [caption, image]:
        assert packet["packet_id"].startswith("ap_")
        projection = {key: packet[key] for key in ("packet_version", "packet_id", "phase", "evidence")}
        expected_hash = hashlib.sha256(
            json.dumps(
                projection,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        assert packet["packet_sha256"] == expected_hash
        assert set(packet) == {
            "packet_version",
            "packet_id",
            "packet_sha256",
            "phase",
            "evidence",
        }

    assert set(caption["evidence"]) == {
        "unit",
        "target",
        "category",
        "caption",
        "span",
    }
    assert set(image["evidence"]) == {
        "unit",
        "target",
        "category",
        "image_file",
        "image_sha256",
    }
    assert image["evidence"]["image_file"] == "image.jpg"
    assert image["evidence"]["image_sha256"] == hashlib.sha256(b"image-0").hexdigest()
    serialized_caption = json.dumps(caption, sort_keys=True)
    serialized_image = json.dumps(image, sort_keys=True)
    for forbidden in (
        "ours",
        "must-not-leak",
        "qwen",
        "gemma",
        "population",
        "sample_probability",
        "weight",
        "participant",
    ):
        assert forbidden not in serialized_caption
        assert forbidden not in serialized_image
    assert "item-0" not in serialized_caption
    assert "assets/item-0.jpg" not in serialized_caption
    assert "assets/item-0.jpg" not in serialized_image
    assert "A red cat number 0 sits on a mat." not in serialized_image


def test_v1_image_packet_submission_recomputes_frozen_asset_projection(
    tmp_path: Path,
) -> None:
    store, _, credential = _v1_disagreement_store(tmp_path, close=True)
    caption_packet = store.adjudication_packets("audit", "caption")[0]
    caption_packet_dir, _ = _adjudication_packet_response(
        tmp_path,
        name="caption-before-image",
        packet=caption_packet,
        decision=CAPTION_YES,
    )
    _submit_packet_response(
        store,
        credential,
        caption_packet_dir,
        phase="caption",
    )
    original_image_packet = store.adjudication_packets("audit", "image")[0]
    image_packet_dir, _ = _adjudication_packet_response(
        tmp_path,
        name="frozen-image-phase",
        packet=original_image_packet,
        decision={"image_support": "yes"},
        image_bytes=b"image-0",
    )
    changed_asset = {
        "route": "assets/item-0.jpg",
        "sha256": "0" * 64,
        "private_provenance": {"surface": "must-not-leak"},
    }
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE items
            SET image_asset_json = ?
            WHERE study_id = 'audit' AND item_id = 'item-0'
            """,
            (json.dumps(changed_asset),),
        )
    changed_image_packet = store.adjudication_packets("audit", "image")[0]
    assert changed_image_packet["packet_id"] == original_image_packet["packet_id"]
    assert changed_image_packet["packet_sha256"] != original_image_packet["packet_sha256"]
    assert changed_image_packet["evidence"]["image_sha256"] == "0" * 64
    with pytest.raises(ValidationError, match="frozen database projection"):
        _submit_packet_response(
            store,
            credential,
            image_packet_dir,
            phase="image",
        )
    assert len(store.adjudication_events("audit")) == 1


def test_v1_image_packet_requires_frozen_materialization_sha256(
    tmp_path: Path,
) -> None:
    store, _, credential = _v1_disagreement_store(tmp_path, close=True)
    caption_packet = store.adjudication_packets("audit", "caption")[0]
    packet_dir, _ = _adjudication_packet_response(
        tmp_path,
        name="caption-for-missing-sha",
        packet=caption_packet,
        decision=CAPTION_YES,
    )
    _submit_packet_response(
        store,
        credential,
        packet_dir,
        phase="caption",
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE items
            SET image_asset_json = ?
            WHERE study_id = 'audit' AND item_id = 'item-0'
            """,
            (json.dumps({"route": "assets/item-0.jpg"}),),
        )
    with pytest.raises(
        ValidationError,
        match="frozen materialization SHA-256",
    ):
        store.adjudication_packets("audit", "image")


def test_v1_adjudication_requires_predeclared_identity_and_untampered_packet(
    tmp_path: Path,
) -> None:
    store, _, credential = _v1_disagreement_store(tmp_path, close=True)
    caption = store.adjudication_packets("audit", "caption")[0]
    packet_dir, response = _adjudication_packet_response(
        tmp_path,
        name="identity-phase",
        packet=caption,
        decision=CAPTION_YES,
    )

    with pytest.raises(ValidationError, match="human_cbu_v1 requires"):
        store.adjudicate(
            "audit",
            "item-0",
            "caption",
            CAPTION_YES,
            adjudicator_pseudonym="human-adjudicator-01",
            packet_id=caption["packet_id"],
            packet_sha256=caption["packet_sha256"],
            adjudicator_credential=credential,
        )
    with pytest.raises(ValidationError, match="phase does not match the response"):
        _submit_packet_response(
            store,
            credential,
            packet_dir,
            phase="image",
        )
    with pytest.raises(AuthenticationError, match="credential is invalid"):
        _submit_packet_response(
            store,
            "wrong-credential",
            packet_dir,
            phase="caption",
        )
    outside = tmp_path / caption["packet_id"]
    outside.mkdir(mode=0o700)
    with pytest.raises(ValidationError, match="directly inside"):
        _submit_packet_response(
            store,
            credential,
            outside,
            phase="caption",
        )

    decision = _submit_packet_response(
        store,
        credential,
        packet_dir,
        phase="caption",
    )
    assert decision["packet_id"] == caption["packet_id"]
    assert "item_id" not in decision
    event = store.adjudication_events("audit")[0]
    assert event["packet_version"] == ADJUDICATION_PACKET_VERSION
    assert event["packet_id"] == caption["packet_id"]
    assert event["packet_sha256"] == caption["packet_sha256"]
    assert event["packet_file_sha256"] == decision["packet_file_sha256"]
    assert event["packet_file_sha256"] == hashlib.sha256((packet_dir / "packet.json").read_bytes()).hexdigest()
    assert event["response_version"] == ADJUDICATION_RESPONSE_VERSION
    assert event["response_sha256"] == decision["response_sha256"]
    assert event["image_sha256"] is None

    replay = _submit_packet_response(
        store,
        credential,
        packet_dir,
        phase="caption",
    )
    assert replay["idempotent_replay"] is True
    assert replay["adjudication_id"] == decision["adjudication_id"]
    assert len(store.adjudication_events("audit")) == 1

    response_payload = json.loads(response.read_text(encoding="utf-8"))
    response_payload["decision"] = {
        "caption_licensed": "uncertain",
        "atomic_visual_claim": "yes",
        "category_check": "correct",
    }
    _write_private_json(response, response_payload)
    with pytest.raises(ConflictError, match="different adjudication response"):
        _submit_packet_response(
            store,
            credential,
            packet_dir,
            phase="caption",
        )
    assert len(store.adjudication_events("audit")) == 1

    image = store.adjudication_packets("audit", "image")[0]
    assert image["phase"] == "image"


def test_v1_adjudication_rejects_coherent_packet_and_review_tamper(
    tmp_path: Path,
) -> None:
    store, _, credential = _v1_disagreement_store(tmp_path, close=True)
    packet = store.adjudication_packets("audit", "caption")[0]
    packet_dir, response = _adjudication_packet_response(
        tmp_path,
        name="tampered-evidence-phase",
        packet=packet,
        decision=CAPTION_YES,
    )
    packet_file = packet_dir / "packet.json"
    tampered_packet = json.loads(packet_file.read_text(encoding="utf-8"))
    tampered_packet["evidence"]["caption"] = "A forged caption."
    projection = {key: tampered_packet[key] for key in ("packet_version", "packet_id", "phase", "evidence")}
    tampered_packet["packet_sha256"] = hashlib.sha256(
        json.dumps(
            projection,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    _write_private_json(packet_file, tampered_packet)
    review_file = packet_dir / "review.html"
    review_file.write_text(
        render_adjudication_html(
            tampered_packet,
            semantic_categories=list(SEMANTIC_VISUAL_CLAIM_TYPES),
        ),
        encoding="utf-8",
    )
    review_file.chmod(0o600)
    response_payload = json.loads(response.read_text(encoding="utf-8"))
    response_payload["packet_file_sha256"] = hashlib.sha256(packet_file.read_bytes()).hexdigest()
    response_payload["packet"] = tampered_packet
    _write_private_json(response, response_payload)

    with pytest.raises(ValidationError, match="frozen database projection"):
        _submit_packet_response(
            store,
            credential,
            packet_dir,
            phase="caption",
        )
    assert store.adjudication_events("audit") == []
    assert store.adjudication_packets("audit", "caption") == [packet]


def test_v1_adjudication_rejects_response_copied_from_packet_a_to_packet_b(
    tmp_path: Path,
) -> None:
    store, _, credential = _v1_disagreement_store(
        tmp_path,
        close=True,
        item_count=2,
    )
    packets = store.adjudication_packets("audit", "caption")
    assert len(packets) == 2
    selected, substituted = packets
    packet_a_dir, response_a = _adjudication_packet_response(
        tmp_path,
        name="cross-packet-phase",
        packet=selected,
        decision=CAPTION_YES,
    )
    packet_b_dir, response_b = _adjudication_packet_response(
        tmp_path,
        name="cross-packet-phase",
        packet=substituted,
        decision=CAPTION_YES,
    )
    response_b.write_bytes(response_a.read_bytes())
    response_b.chmod(0o600)

    with pytest.raises(ValidationError, match="exact packet sidecar"):
        _submit_packet_response(
            store,
            credential,
            packet_b_dir,
            phase="caption",
        )
    assert store.adjudication_events("audit") == []
    assert store.adjudication_packets("audit", "caption") == packets

    # Even a whole A context copied into B remains bound to B's directory
    # selector and therefore cannot adjudicate either packet.
    for filename in ("packet.json", "review.html", "response.json"):
        destination = packet_b_dir / filename
        destination.write_bytes((packet_a_dir / filename).read_bytes())
        destination.chmod(0o600)
    with pytest.raises(ValidationError, match="directory name does not match packet_id"):
        _submit_packet_response(
            store,
            credential,
            packet_b_dir,
            phase="caption",
        )
    assert store.adjudication_events("audit") == []


def test_v1_adjudication_rejects_review_page_swapped_between_packets(
    tmp_path: Path,
) -> None:
    store, _, credential = _v1_disagreement_store(
        tmp_path,
        close=True,
        item_count=2,
    )
    packet_a, packet_b = store.adjudication_packets("audit", "caption")
    packet_a_dir, _ = _adjudication_packet_response(
        tmp_path,
        name="swapped-review-phase",
        packet=packet_a,
        decision=CAPTION_YES,
    )
    packet_b_dir, _ = _adjudication_packet_response(
        tmp_path,
        name="swapped-review-phase",
        packet=packet_b,
        decision=CAPTION_YES,
    )
    review_b = packet_b_dir / "review.html"
    review_b.write_bytes((packet_a_dir / "review.html").read_bytes())
    review_b.chmod(0o600)

    with pytest.raises(ValidationError, match="does not deterministically match"):
        _submit_packet_response(
            store,
            credential,
            packet_b_dir,
            phase="caption",
        )
    assert store.adjudication_events("audit") == []


def test_v1_image_adjudication_rejects_same_size_image_byte_tamper(
    tmp_path: Path,
) -> None:
    store, _, credential = _v1_disagreement_store(tmp_path, close=True)
    caption = store.adjudication_packets("audit", "caption")[0]
    caption_packet_dir, _ = _adjudication_packet_response(
        tmp_path,
        name="caption-before-image-tamper",
        packet=caption,
        decision=CAPTION_YES,
    )
    _submit_packet_response(
        store,
        credential,
        caption_packet_dir,
        phase="caption",
    )
    image = store.adjudication_packets("audit", "image")[0]
    image_packet_dir, _ = _adjudication_packet_response(
        tmp_path,
        name="image-byte-tamper",
        packet=image,
        decision={"image_support": "yes"},
        image_bytes=b"image-0",
    )
    image_path = image_packet_dir / image["evidence"]["image_file"]
    assert image_path.parent == image_packet_dir
    assert image_path.stat().st_mode & 0o777 == 0o600
    assert len(b"forged!") == len(b"image-0")
    image_path.write_bytes(b"forged!")
    image_path.chmod(0o600)

    with pytest.raises(ValidationError, match="image SHA-256"):
        _submit_packet_response(
            store,
            credential,
            image_packet_dir,
            phase="image",
        )
    assert len(store.adjudication_events("audit")) == 1
    assert store.adjudication_packets("audit", "image") == [image]


def test_v1_image_adjudication_rejects_symlinked_packet_local_image(
    tmp_path: Path,
) -> None:
    store, _, credential = _v1_disagreement_store(tmp_path, close=True)
    caption = store.adjudication_packets("audit", "caption")[0]
    caption_packet_dir, _ = _adjudication_packet_response(
        tmp_path,
        name="caption-before-symlink",
        packet=caption,
        decision=CAPTION_YES,
    )
    _submit_packet_response(
        store,
        credential,
        caption_packet_dir,
        phase="caption",
    )
    image = store.adjudication_packets("audit", "image")[0]
    image_packet_dir, _ = _adjudication_packet_response(
        tmp_path,
        name="symlinked-image-phase",
        packet=image,
        decision={"image_support": "yes"},
        image_bytes=b"image-0",
    )
    image_path = image_packet_dir / image["evidence"]["image_file"]
    external_images = tmp_path / "external-images"
    external_images.mkdir(mode=0o700)
    external_image = external_images / image_path.name
    external_image.write_bytes(image_path.read_bytes())
    external_image.chmod(0o600)
    image_path.unlink()
    image_path.symlink_to(external_image)

    with pytest.raises(ValidationError, match="non-symlink file"):
        _submit_packet_response(
            store,
            credential,
            image_packet_dir,
            phase="image",
        )
    assert len(store.adjudication_events("audit")) == 1


def test_v1_adjudication_rehashes_held_files_immediately_before_insert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _, credential = _v1_disagreement_store(tmp_path, close=True)
    packet = store.adjudication_packets("audit", "caption")[0]
    packet_dir, _ = _adjudication_packet_response(
        tmp_path,
        name="last-moment-rehash-phase",
        packet=packet,
        decision=CAPTION_YES,
    )
    packet_file = packet_dir / "packet.json"
    original_candidates = store._adjudication_packet_candidates

    # Ignore timestamps in the signature for this test so only the second
    # byte-level digest can detect an in-place, same-inode, same-size edit.
    monkeypatch.setattr(
        store_module,
        "_file_signature",
        lambda value: (value.st_dev, value.st_ino, value.st_size),
    )

    def mutate_after_initial_read(
        connection: sqlite3.Connection,
        study: sqlite3.Row,
    ) -> list[dict[str, Any]]:
        candidates = original_candidates(connection, study)
        raw = packet_file.read_bytes()
        marker = b"A red cat"
        assert marker in raw
        packet_file.write_bytes(raw.replace(marker, b"B red cat", 1))
        packet_file.chmod(0o600)
        return candidates

    monkeypatch.setattr(
        store,
        "_adjudication_packet_candidates",
        mutate_after_initial_read,
    )
    with pytest.raises(ValidationError, match="bytes changed"):
        _submit_packet_response(
            store,
            credential,
            packet_dir,
            phase="caption",
        )
    assert store.adjudication_events("audit") == []


def test_v1_invalid_category_correction_rolls_back_and_remains_retryable(
    tmp_path: Path,
) -> None:
    store, _, credential = _v1_disagreement_store(tmp_path, close=True)
    packet = store.adjudication_packets("audit", "caption")[0]
    packet_dir, response = _adjudication_packet_response(
        tmp_path,
        name="category-correction-phase",
        packet=packet,
        decision={
            "caption_licensed": "yes",
            "atomic_visual_claim": "yes",
            "category_check": "incorrect",
        },
    )

    invalid_decisions = [
        {
            "caption_licensed": "yes",
            "atomic_visual_claim": "yes",
            "category_check": "incorrect",
        },
        {
            "caption_licensed": "yes",
            "atomic_visual_claim": "yes",
            "category_check": "incorrect",
            "corrected_category": "unknown_type",
        },
        {
            "caption_licensed": "yes",
            "atomic_visual_claim": "yes",
            "category_check": "incorrect",
            "corrected_category": packet["evidence"]["category"],
        },
    ]
    for invalid in invalid_decisions:
        response_payload = json.loads(response.read_text(encoding="utf-8"))
        response_payload["decision"] = invalid
        _write_private_json(response, response_payload)
        with pytest.raises(ValidationError):
            _submit_packet_response(
                store,
                credential,
                packet_dir,
                phase="caption",
            )
        assert store.adjudication_events("audit") == []
        assert store.adjudication_packets("audit", "caption") == [packet]
        with pytest.raises(StudyStateError, match="caption disagreement"):
            store.adjudication_packets("audit", "image")

    response_payload = json.loads(response.read_text(encoding="utf-8"))
    response_payload["decision"] = {
        "caption_licensed": "yes",
        "atomic_visual_claim": "yes",
        "category_check": "incorrect",
        "corrected_category": "object",
    }
    _write_private_json(response, response_payload)
    result = _submit_packet_response(
        store,
        credential,
        packet_dir,
        phase="caption",
    )
    assert result["payload"]["corrected_category"] == "object"
    assert len(store.adjudication_events("audit")) == 1
    assert len(store.adjudication_packets("audit", "image")) == 1


@pytest.mark.parametrize("control_usefulness", [None, 5, "cannot_judge"])
def test_v1_image_adjudication_rejects_exploratory_control_usefulness(
    tmp_path: Path,
    control_usefulness: Any,
) -> None:
    store, _, credential = _v1_disagreement_store(tmp_path, close=True)
    caption_packet = store.adjudication_packets("audit", "caption")[0]
    caption_packet_dir, _ = _adjudication_packet_response(
        tmp_path,
        name="caption-before-image-usefulness",
        packet=caption_packet,
        decision=CAPTION_YES,
    )
    _submit_packet_response(
        store,
        credential,
        caption_packet_dir,
        phase="caption",
    )
    image_packet = store.adjudication_packets("audit", "image")[0]
    image_packet_dir, _ = _adjudication_packet_response(
        tmp_path,
        name="image-usefulness-rejected",
        packet=image_packet,
        decision={
            "image_support": "yes",
            "control_usefulness": control_usefulness,
        },
    )

    with pytest.raises(
        ValidationError,
        match="control_usefulness is not allowed in image adjudication",
    ):
        _submit_packet_response(
            store,
            credential,
            image_packet_dir,
            phase="image",
        )
    assert len(store.adjudication_events("audit")) == 1
    assert store.adjudication_packets("audit", "image") == [image_packet]


def test_v1_seal_requires_one_time_adjudicator_provisioning(tmp_path: Path) -> None:
    store = HumanCBUStore.initialize(tmp_path / "unprovisioned.sqlite3", clock=MutableClock())
    store.create_study(
        "audit",
        title="CBU calibration",
        protocol_version="human_cbu_v1",
        consent_version="consent-v1",
        metadata=_v1_metadata(),
    )
    store.add_item("audit", **_item(0))
    store.create_participant_invites("audit", count=2)
    store.assign_items("audit", labels_per_item=2, seed="published-seed")
    report = store.validation_report("audit")
    assert "predeclared adjudicator credential has not been provisioned" in report["errors"]
    with pytest.raises(ValidationError, match="credential has not been provisioned"):
        store.seal_validation("audit")

    provisioned = store.provision_adjudicator("audit")
    assert provisioned["adjudicator_pseudonym"] == "human-adjudicator-01"
    assert len(provisioned["credential"]) >= 40
    assert "credential" not in store.get_study("audit")
    changed_metadata = dict(store.get_study("audit")["metadata"])
    changed_metadata["adjudicator_pseudonym"] = "changed-adjudicator"
    with pytest.raises(ValidationError, match="frozen at init-db"):
        store.update_study_metadata("audit", metadata=changed_metadata)
    with sqlite3.connect(store.path) as connection:
        credential_digest = connection.execute(
            "SELECT adjudicator_credential_digest FROM studies WHERE study_id = 'audit'"
        ).fetchone()[0]
    projected = json.dumps(
        {
            "study": store.get_study("audit"),
            "status": store.status_summary("audit"),
        },
        sort_keys=True,
    )
    assert provisioned["credential"] not in projected
    assert credential_digest not in projected
    with pytest.raises(ConflictError, match="already been provisioned"):
        store.provision_adjudicator("audit")
    assert store.seal_validation("audit")["valid"]


def test_v1_packet_and_submission_fail_closed_if_provisioning_is_lost(
    tmp_path: Path,
) -> None:
    store, _, credential = _v1_disagreement_store(tmp_path, close=True)
    packet = store.adjudication_packets("audit", "caption")[0]
    packet_dir, _ = _adjudication_packet_response(
        tmp_path,
        name="lost-provisioning-phase",
        packet=packet,
        decision=CAPTION_YES,
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE studies
            SET adjudicator_credential_digest = NULL
            WHERE study_id = 'audit'
            """
        )
    with pytest.raises(ValidationError, match="provisioned adjudicator credential"):
        store.adjudication_packets("audit", "caption")
    with pytest.raises(ValidationError, match="provisioned adjudicator credential"):
        _submit_packet_response(
            store,
            credential,
            packet_dir,
            phase="caption",
        )
