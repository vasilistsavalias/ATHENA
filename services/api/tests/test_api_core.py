from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db.base import Base
from app.db.session import SessionLocal, engine
from app.main import create_app
from app.models import BlockAItem, BlockBItem, Campaign


def _reset_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def _build_pairwise_pack_zip(zip_path: Path) -> None:
    pack_root = zip_path.parent / "pack_payload"
    practice_items = []
    scored_items = []
    private_practice_items = []
    private_scored_items = []

    for idx in range(20):
        sample_id = f"boot{idx:02d}"
        image_dir = pack_root / "images" / sample_id
        image_dir.mkdir(parents=True, exist_ok=True)
        for name in ["input.png", "A.png", "B.png"]:
            image_path = image_dir / name
            image_path.write_bytes(
                bytes.fromhex("89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE0000000A49444154789C6360000000020001E221BC330000000049454E44AE426082")
            )

        item = {
            "sample_id": sample_id,
            "input": f"images/{sample_id}/input.png",
            "A": f"images/{sample_id}/A.png",
            "B": f"images/{sample_id}/B.png",
            "mask_type": "edge",
            "mask_coverage": 0.18,
            "coverage_bin": "10-25%",
            "is_anchor": idx in {4, 8, 12},
            "is_practice": idx < 3,
        }
        if idx < 3:
            practice_items.append(item)
            private_practice_items.append(
                {
                    "sample_id": sample_id,
                    "mapping": {"A": "Ground-Truth", "B": "FT-SD+TTA"},
                }
            )
        else:
            scored_items.append(item)
            private_scored_items.append(
                {
                    "sample_id": sample_id,
                    "mapping": {"A": "FT-SD+TTA", "B": "Ground-Truth"},
                }
            )

    manifest = {
        "pack_id": "Website_Expert_Pack_Final_V8",
        "study_mode": "pairwise_only",
        "block_a_target_count": 0,
        "block_b_target_count": 20,
        "practice_items": practice_items,
        "items": scored_items,
    }
    (pack_root / "manifest_public.json").write_text(json.dumps(manifest), encoding="utf-8")
    (pack_root / "manifest_private.json").write_text(
        json.dumps({"practice_items": private_practice_items, "items": private_scored_items}),
        encoding="utf-8",
    )

    with zipfile.ZipFile(zip_path, mode="w") as archive:
        for path in pack_root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(pack_root))


def _seed_campaign_items():
    with SessionLocal() as db:
        campaign = Campaign(
            name="Test Campaign",
            seed=42,
            is_active=True,
            block_a_target_count=25,
            block_b_target_count=15,
        )
        db.add(campaign)
        db.flush()
        for idx in range(40):
            db.add(
                BlockAItem(
                    campaign_id=campaign.id,
                    sample_id=f"a_{idx:03d}",
                    image_url=f"/static/a_{idx:03d}.png",
                    source_label="generated",
                    mask_type="rect",
                    mask_coverage_bin="10-25%",
                )
            )
        for idx in range(30):
            db.add(
                BlockBItem(
                    campaign_id=campaign.id,
                    sample_id=f"b_{idx:03d}",
                    input_url=f"/static/in_{idx:03d}.png",
                    option_a_url=f"/static/a_{idx:03d}.png",
                    option_b_url=f"/static/b_{idx:03d}.png",
                    mask_type="edge",
                    mask_coverage_bin="25-50%",
                )
            )
        db.commit()


def _invite(client: TestClient, invite_code: str = "athena-invite"):
    response = client.post(
        "/api/v1/auth/invite",
        json={"invite_code": invite_code},
    )
    return response


def _login(client: TestClient):
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "archaeologist", "password": "change-me"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _complete_block_a(client: TestClient):
    while True:
        next_response = client.get("/api/v1/block-a/next")
        assert next_response.status_code == 200
        payload = next_response.json()
        if payload["done"]:
            break
        item = payload["item"]
        submit = client.post(
            "/api/v1/block-a/submit",
            json={
                "assignment_id": item["assignment_id"],
                "authenticity_likelihood": 4,
                "archaeological_plausibility": 4,
                "confidence": 4,
                "comment": "Observed restoration cues on this sample.",
                "response_time_ms": 1200,
            },
        )
        assert submit.status_code == 200, submit.text


def _submit_profile(
    client: TestClient,
    *,
    discipline: str = "Archaeology",
    name: str = "Validator Test",
    institution: str = "University of Macedonia",
):
    response = client.put(
        "/api/v1/session/profile",
        json={
            "name": name,
            "institution": institution,
            "discipline": discipline,
            "discipline_other": "",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_profile_requires_name_and_institution():
    _reset_db()
    _seed_campaign_items()
    client = TestClient(create_app())

    invite = _invite(client)
    assert invite.status_code == 200

    missing_name = client.put(
        "/api/v1/session/profile",
        json={
            "name": "  ",
            "institution": "University of Macedonia",
            "discipline": "Archaeology",
            "discipline_other": "",
        },
    )
    assert missing_name.status_code == 422
    assert missing_name.json()["detail"] == "Name is required."

    missing_institution = client.put(
        "/api/v1/session/profile",
        json={
            "name": "Validator Test",
            "institution": " ",
            "discipline": "Archaeology",
            "discipline_other": "",
        },
    )
    assert missing_institution.status_code == 422
    assert missing_institution.json()["detail"] == "Institution is required."


def _submit_stage_feedback(client: TestClient, block: str, comment: str):
    response = client.put(f"/api/v1/session/feedback/{block}", json={"comment": comment})
    assert response.status_code == 200, response.text
    return response.json()


def _submit_block_b_comprehension(client: TestClient, selected_option: str):
    response = client.post("/api/v1/session/block-b-comprehension", json={"selected_option": selected_option})
    assert response.status_code == 200, response.text
    return response.json()


def _unlock_block_b_scored(client: TestClient):
    response = _submit_block_b_comprehension(client, "spot_machine")
    assert response["passed"] is True
    return response


def test_login_and_assignment_counts():
    _reset_db()
    _seed_campaign_items()
    client = TestClient(create_app())

    invite = _invite(client)
    assert invite.status_code == 200, invite.text
    invite_payload = invite.json()
    assert invite_payload["participant_public_id"] == "R0001"
    assert invite_payload["progress"]["block_a_total"] == 25
    assert invite_payload["progress"]["block_b_total"] == 15
    assert invite_payload["progress"]["profile_completed"] is False
    assert invite_payload["progress"]["block_a_feedback_completed"] is False
    assert invite_payload["progress"]["block_b_feedback_completed"] is False

    next_a = client.get("/api/v1/block-a/next")
    assert next_a.status_code == 409

    next_b = client.get("/api/v1/block-b/next")
    assert next_b.status_code == 409

    progress = client.get("/api/v1/progress")
    assert progress.status_code == 200
    payload = progress.json()
    assert payload["block_a_total"] == 25
    assert payload["block_b_total"] == 15
    assert payload["block_a_completed"] == 0
    assert payload["block_b_completed"] == 0
    assert payload["profile_completed"] is False


def test_profile_roundtrip_and_block_a_unlock():
    _reset_db()
    _seed_campaign_items()
    client = TestClient(create_app())

    invite = _invite(client)
    assert invite.status_code == 200, invite.text

    profile = client.get("/api/v1/session/profile")
    assert profile.status_code == 200
    assert profile.json()["profile_completed"] is False

    update = _submit_profile(client, discipline="Philology / History / Archaeology")
    assert update["profile_completed"] is True
    assert update["discipline"] == "Philology / History / Archaeology"

    profile = client.get("/api/v1/session/profile")
    assert profile.status_code == 200
    assert profile.json()["profile_completed"] is True

    next_a = client.get("/api/v1/block-a/next")
    assert next_a.status_code == 200
    assert next_a.json()["done"] is False


def test_health_db_endpoint():
    _reset_db()
    client = TestClient(create_app())
    response = client.get("/health/db")
    assert response.status_code == 200
    assert response.json()["database"] == "reachable"


def test_duplicate_submission_blocked():
    _reset_db()
    _seed_campaign_items()
    client = TestClient(create_app())
    invite = _invite(client)
    assert invite.status_code == 200, invite.text
    _submit_profile(client)

    item = client.get("/api/v1/block-a/next").json()["item"]
    submit_payload = {
        "assignment_id": item["assignment_id"],
        "authenticity_likelihood": 4,
        "archaeological_plausibility": 4,
        "confidence": 3,
        "comment": "Looks plausible.",
        "response_time_ms": 2500,
    }
    first = client.post("/api/v1/block-a/submit", json=submit_payload)
    assert first.status_code == 200
    second = client.post("/api/v1/block-a/submit", json=submit_payload)
    assert second.status_code == 409


def test_block_a_submit_requires_non_empty_comment():
    _reset_db()
    _seed_campaign_items()
    client = TestClient(create_app())
    invite = _invite(client)
    assert invite.status_code == 200, invite.text
    _submit_profile(client)

    item = client.get("/api/v1/block-a/next").json()["item"]
    missing_comment = client.post(
        "/api/v1/block-a/submit",
        json={
            "assignment_id": item["assignment_id"],
            "authenticity_likelihood": 4,
            "archaeological_plausibility": 4,
            "confidence": 3,
            "response_time_ms": 1500,
        },
    )
    assert missing_comment.status_code == 422

    whitespace_comment = client.post(
        "/api/v1/block-a/submit",
        json={
            "assignment_id": item["assignment_id"],
            "authenticity_likelihood": 4,
            "archaeological_plausibility": 4,
            "confidence": 3,
            "comment": "   ",
            "response_time_ms": 1500,
        },
    )
    assert whitespace_comment.status_code == 422

    valid_comment = client.post(
        "/api/v1/block-a/submit",
        json={
            "assignment_id": item["assignment_id"],
            "authenticity_likelihood": 4,
            "archaeological_plausibility": 4,
            "confidence": 3,
            "comment": "Useful restoration baseline for this vessel.",
            "response_time_ms": 1500,
        },
    )
    assert valid_comment.status_code == 200, valid_comment.text


def test_block_b_submit_requires_non_empty_comment():
    _reset_db()
    _seed_campaign_items()
    client = TestClient(create_app())
    invite = _invite(client)
    assert invite.status_code == 200, invite.text
    _submit_profile(client)
    _complete_block_a(client)
    _submit_stage_feedback(client, "A", "Block A complete.")
    _unlock_block_b_scored(client)

    item = client.get("/api/v1/block-b/next").json()["item"]
    missing_comment = client.post(
        "/api/v1/block-b/submit",
        json={
            "assignment_id": item["assignment_id"],
            "choice": "A",
            "confidence": 4,
            "response_time_ms": 1400,
        },
    )
    assert missing_comment.status_code == 422

    whitespace_comment = client.post(
        "/api/v1/block-b/submit",
        json={
            "assignment_id": item["assignment_id"],
            "choice": "A",
            "confidence": 4,
            "comment": "   ",
            "response_time_ms": 1400,
        },
    )
    assert whitespace_comment.status_code == 422

    valid_comment = client.post(
        "/api/v1/block-b/submit",
        json={
            "assignment_id": item["assignment_id"],
            "choice": "A",
            "confidence": 4,
            "comment": "A aligns better with the damaged reference context.",
            "response_time_ms": 1400,
        },
    )
    assert valid_comment.status_code == 200, valid_comment.text


def test_admin_export_endpoints():
    _reset_db()
    _seed_campaign_items()
    client = TestClient(create_app())
    invite = _invite(client)
    assert invite.status_code == 200, invite.text
    _submit_profile(client, discipline="Museum / Curatorial")

    block_a_item = client.get("/api/v1/block-a/next").json()["item"]
    client.post(
        "/api/v1/block-a/submit",
        json={
            "assignment_id": block_a_item["assignment_id"],
            "authenticity_likelihood": 3,
            "archaeological_plausibility": 4,
            "confidence": 4,
            "comment": "Structure appears consistent with the damaged input.",
            "response_time_ms": 1900,
        },
    )

    _complete_block_a(client)
    block_b_locked = client.get("/api/v1/block-b/next")
    assert block_b_locked.status_code == 409
    _submit_stage_feedback(client, "A", "This is a valid Block A expert note.")
    block_b_item = client.get("/api/v1/block-b/next").json()["item"]
    client.post(
        "/api/v1/block-b/submit",
        json={
            "assignment_id": block_b_item["assignment_id"],
            "choice": "A",
            "confidence": 4,
            "comment": "Candidate A preserves visual continuity better.",
            "response_time_ms": 1600,
        },
    )
    while True:
        next_response = client.get("/api/v1/block-b/next")
        assert next_response.status_code == 200, next_response.text
        payload = next_response.json()
        if payload["done"]:
            break
        item = payload["item"]
        submit = client.post(
            "/api/v1/block-b/submit",
            json={
                "assignment_id": item["assignment_id"],
                "choice": "A",
                "confidence": 4,
                "comment": "Candidate A remains more plausible after zoom.",
                "response_time_ms": 1600,
            },
        )
        assert submit.status_code == 200, submit.text
    _submit_stage_feedback(client, "B", "This is a valid Block B expert note.")

    headers = {"x-admin-secret": "change-me-admin-secret"}
    csv_res = client.get("/api/v1/admin/export/responses.csv", headers=headers)
    assert csv_res.status_code == 200
    assert "participant_id" in csv_res.text
    assert "block_a_stage_comment" in csv_res.text
    assert "discipline" in csv_res.text

    json_res = client.get("/api/v1/admin/export/responses.json", headers=headers)
    assert json_res.status_code == 200
    body = json_res.json()
    assert "participant_level" in body
    assert "stage_feedback" in body
    assert "item_level" in body
    assert "pairwise_preference" in body
    assert "quality_flags" in body
    assert body["participant_level"][0]["discipline"] == "Museum / Curatorial"
    assert body["participant_level"][0]["block_a_stage_comment"] is not None
    assert body["participant_level"][0]["block_b_stage_comment"] is not None

    quality = client.get("/api/v1/admin/export/quality_report.json", headers=headers)
    assert quality.status_code == 200
    quality_body = quality.json()
    assert quality_body["participants_total"] >= 1
    assert quality_body["profiles_completed_total"] >= 1
    assert "reliability" in quality_body
    assert "block_a" in quality_body["reliability"]
    assert "block_b" in quality_body["reliability"]


def test_admin_cookie_login_and_dashboard():
    _reset_db()
    _seed_campaign_items()
    client = TestClient(create_app())

    unauthorized = client.get("/api/v1/admin/dashboard")
    assert unauthorized.status_code == 401

    login = client.post("/api/v1/admin/auth/login", json={"password": "change-me-admin-secret"})
    assert login.status_code == 200, login.text
    assert login.json()["authenticated"] is True
    assert login.cookies.get("arch_eval_admin_session")

    session = client.get("/api/v1/admin/session")
    assert session.status_code == 200, session.text
    assert session.json()["auth_mode"] == "cookie"

    dashboard = client.get("/api/v1/admin/dashboard")
    assert dashboard.status_code == 200, dashboard.text
    body = dashboard.json()
    assert body["campaign"]["name"] == "Test Campaign"
    assert body["stats"]["participants_total"] == 0
    assert isinstance(body["discipline_breakdown"], list)

    logout = client.post("/api/v1/admin/auth/logout")
    assert logout.status_code == 200, logout.text

    relock = client.get("/api/v1/admin/dashboard")
    assert relock.status_code == 401


def test_admin_runtime_reset_requires_auth_and_confirm_phrase():
    _reset_db()
    _seed_campaign_items()
    client = TestClient(create_app())

    unauthorized = client.post(
        "/api/v1/admin/runtime/reset",
        json={"confirm_phrase": "RESET_STUDY_RUNTIME"},
    )
    assert unauthorized.status_code == 401

    headers = {"x-admin-secret": "change-me-admin-secret"}
    invalid_phrase = client.post(
        "/api/v1/admin/runtime/reset",
        headers=headers,
        json={"confirm_phrase": "WRONG"},
    )
    assert invalid_phrase.status_code == 400


def test_admin_runtime_reset_clears_runtime_and_resets_first_invite_id():
    _reset_db()
    _seed_campaign_items()
    settings = get_settings()

    client_a = TestClient(create_app())
    client_b = TestClient(create_app())

    first_invite = _invite(client_a)
    assert first_invite.status_code == 200
    assert first_invite.json()["participant_public_id"] == "R0001"

    second_invite = _invite(client_b)
    assert second_invite.status_code == 200
    assert second_invite.json()["participant_public_id"] == "R0002"

    headers = {"x-admin-secret": "change-me-admin-secret"}
    reset_response = client_a.post(
        "/api/v1/admin/runtime/reset",
        headers=headers,
        json={"confirm_phrase": settings.admin_reset_confirm_phrase, "all_campaigns": True},
    )
    assert reset_response.status_code == 200, reset_response.text
    reset_payload = reset_response.json()
    assert reset_payload["deleted_counts"]["participants"] >= 2
    assert reset_payload["identity_reset"] is True
    assert reset_payload["next_participant_public_id"] == "R0001"

    export_after_reset = client_a.get("/api/v1/admin/export/responses.json", headers=headers)
    assert export_after_reset.status_code == 200, export_after_reset.text
    assert export_after_reset.json()["item_level"] == []

    client_after_reset = TestClient(create_app())
    invite_after_reset = _invite(client_after_reset)
    assert invite_after_reset.status_code == 200, invite_after_reset.text
    assert invite_after_reset.json()["participant_public_id"] == "R0001"


def test_admin_import_pack_roundtrip(tmp_path: Path):
    _reset_db()
    client = TestClient(create_app())

    pack_dir = tmp_path / "Expert_Pack_v2"
    items = []
    for idx in range(40):
        sample_id = f"s{idx:02d}"
        images_dir = pack_dir / "images" / sample_id
        images_dir.mkdir(parents=True, exist_ok=True)
        for name in ["input.png", "A.png", "B.png"]:
            (images_dir / name).write_bytes(b"fake")
        items.append(
            {
                "sample_id": sample_id,
                "input": f"images/{sample_id}/input.png",
                "A": f"images/{sample_id}/A.png",
                "B": f"images/{sample_id}/B.png",
                "mask_type": "rect",
                "mask_coverage": 0.2,
                "coverage_bin": "10-25%",
            }
        )
    manifest = {"pack_id": "Expert_Pack_v2", "items": items}
    (pack_dir / "manifest_public.json").write_text(json.dumps(manifest), encoding="utf-8")

    headers = {"x-admin-secret": "change-me-admin-secret"}
    import_response = client.post(
        "/api/v1/admin/import-pack",
        headers=headers,
        json={
            "pack_dir": str(pack_dir),
            "campaign_name": "Imported Campaign",
            "seed": 11,
            "activate": True,
        },
    )
    assert import_response.status_code == 200, import_response.text

    with SessionLocal() as db:
        campaign = db.query(Campaign).filter(Campaign.name == "Imported Campaign").first()
        assert campaign is not None
        block_b_sample_ids = {
            row.sample_id for row in db.query(BlockBItem).filter(BlockBItem.campaign_id == campaign.id).all()
        }
        generated_a_items = (
            db.query(BlockAItem)
            .filter(BlockAItem.campaign_id == campaign.id, BlockAItem.source_label == "generated")
            .all()
        )
        origin_ids = {
            (row.metadata_json or {}).get("origin_sample_id")
            for row in generated_a_items
            if (row.metadata_json or {}).get("origin_sample_id")
        }
        assert origin_ids.isdisjoint(block_b_sample_ids)

    invite = _invite(client)
    assert invite.status_code == 200, invite.text
    _submit_profile(client)
    assert invite.json()["campaign"]["name"] == "Imported Campaign"
    next_a = client.get("/api/v1/block-a/next")
    next_b = client.get("/api/v1/block-b/next")
    assert next_a.status_code == 200 and next_a.json()["done"] is False
    assert next_b.status_code == 409


def test_admin_import_pack_rejects_gt_50_mask_bins(tmp_path: Path):
    _reset_db()
    client = TestClient(create_app())

    pack_dir = tmp_path / "Expert_Pack_bad_bins"
    items = []
    for idx in range(40):
        sample_id = f"s{idx:02d}"
        images_dir = pack_dir / "images" / sample_id
        images_dir.mkdir(parents=True, exist_ok=True)
        for name in ["input.png", "A.png", "B.png"]:
            (images_dir / name).write_bytes(b"fake")
        items.append(
            {
                "sample_id": sample_id,
                "input": f"images/{sample_id}/input.png",
                "A": f"images/{sample_id}/A.png",
                "B": f"images/{sample_id}/B.png",
                "mask_type": "rect",
                "mask_coverage": 0.61,
                "coverage_bin": ">50%" if idx == 0 else "10-25%",
            }
        )
    manifest = {"pack_id": "Expert_Pack_bad_bins", "items": items}
    (pack_dir / "manifest_public.json").write_text(json.dumps(manifest), encoding="utf-8")

    headers = {"x-admin-secret": "change-me-admin-secret"}
    import_response = client.post(
        "/api/v1/admin/import-pack",
        headers=headers,
        json={
            "pack_dir": str(pack_dir),
            "campaign_name": "Bad Campaign",
            "seed": 11,
            "activate": True,
        },
    )

    assert import_response.status_code == 400
    assert "out-of-policy mask coverage bins" in import_response.json()["detail"]


def test_admin_import_pairwise_only_pack_roundtrip(tmp_path: Path):
    _reset_db()
    client = TestClient(create_app())

    pack_dir = tmp_path / "Website_Expert_Pack_Final_V8"
    practice_items = []
    scored_items = []
    private_practice_items = []
    private_scored_items = []
    for idx in range(20):
        sample_id = f"s{idx:02d}"
        images_dir = pack_dir / "images" / sample_id
        images_dir.mkdir(parents=True, exist_ok=True)
        for name in ["input.png", "A.png", "B.png"]:
            (images_dir / name).write_bytes(b"fake")
        item = {
            "sample_id": sample_id,
            "input": f"images/{sample_id}/input.png",
            "A": f"images/{sample_id}/A.png",
            "B": f"images/{sample_id}/B.png",
            "mask_type": "edge" if idx % 2 == 0 else "rect",
            "mask_coverage": 0.18,
            "coverage_bin": "10-25%",
            "is_anchor": idx in {3, 7, 11},
            "is_practice": idx < 3,
        }
        if idx < 3:
            practice_items.append(item)
            private_practice_items.append(
                {"sample_id": sample_id, "mapping": {"A": "Ground-Truth", "B": "FT-SD+TTA"}}
            )
        else:
            scored_items.append(item)
            private_scored_items.append(
                {"sample_id": sample_id, "mapping": {"A": "FT-SD+TTA", "B": "Ground-Truth"}}
            )

    manifest = {
        "pack_id": "Website_Expert_Pack_Final_V8",
        "study_mode": "pairwise_only",
        "block_a_target_count": 0,
        "block_b_target_count": 20,
        "practice_items": practice_items,
        "items": scored_items,
    }
    (pack_dir / "manifest_public.json").write_text(json.dumps(manifest), encoding="utf-8")
    (pack_dir / "manifest_private.json").write_text(
        json.dumps({"practice_items": private_practice_items, "items": private_scored_items}),
        encoding="utf-8",
    )

    headers = {"x-admin-secret": "change-me-admin-secret"}
    import_response = client.post(
        "/api/v1/admin/import-pack",
        headers=headers,
        json={
            "pack_dir": str(pack_dir),
            "campaign_name": "Pairwise Only Campaign",
            "seed": 7,
            "activate": True,
        },
    )
    assert import_response.status_code == 200, import_response.text

    invite = _invite(client)
    assert invite.status_code == 200, invite.text
    progress = invite.json()["progress"]
    assert progress["block_a_total"] == 0
    assert progress["block_b_total"] == 20
    _submit_profile(client)

    next_b = client.get("/api/v1/block-b/next")
    assert next_b.status_code == 200, next_b.text
    item = next_b.json()["item"]
    assert item["is_practice"] is True
    assert item["is_anchor"] is False

    for index in range(20):
        current = client.get("/api/v1/block-b/next")
        assert current.status_code == 200, current.text
        payload = current.json()
        if payload["done"]:
            break
        submit = client.post(
            "/api/v1/block-b/submit",
            json={
                "assignment_id": payload["item"]["assignment_id"],
                "choice": "A",
                "confidence": 4,
                "comment": "Practice pair judged with contextual plausibility.",
                "response_time_ms": 900,
            },
        )
        assert submit.status_code == 200, submit.text

    feedback = client.put("/api/v1/session/feedback/B", json={"comment": "Pairwise-only validation was coherent and usable."})
    assert feedback.status_code == 200, feedback.text

    export_headers = {"x-admin-secret": "change-me-admin-secret"}
    json_res = client.get("/api/v1/admin/export/responses.json", headers=export_headers)
    assert json_res.status_code == 200, json_res.text
    item_rows = json_res.json()["item_level"]
    assert any(row["is_practice"] for row in item_rows if row["block"] == "B")
    assert any(row["is_anchor"] for row in item_rows if row["block"] == "B")


def test_block_b_scored_pairs_continue_without_comprehension_gate(tmp_path: Path):
    _reset_db()
    client = TestClient(create_app())

    pack_dir = tmp_path / "Website_Expert_Pack_Final_V8"
    practice_items = []
    scored_items = []
    private_practice_items = []
    private_scored_items = []
    for idx in range(20):
        sample_id = f"gate{idx:02d}"
        images_dir = pack_dir / "images" / sample_id
        images_dir.mkdir(parents=True, exist_ok=True)
        for name in ["input.png", "A.png", "B.png"]:
            (images_dir / name).write_bytes(b"fake")
        item = {
            "sample_id": sample_id,
            "input": f"images/{sample_id}/input.png",
            "A": f"images/{sample_id}/A.png",
            "B": f"images/{sample_id}/B.png",
            "mask_type": "edge",
            "mask_coverage": 0.18,
            "coverage_bin": "10-25%",
            "is_anchor": idx in {5, 9, 14},
            "is_practice": idx < 3,
        }
        if idx < 3:
            practice_items.append(item)
            private_practice_items.append({"sample_id": sample_id, "mapping": {"A": "Ground-Truth", "B": "FT-SD+TTA"}})
        else:
            scored_items.append(item)
            private_scored_items.append({"sample_id": sample_id, "mapping": {"A": "FT-SD+TTA", "B": "Ground-Truth"}})

    (pack_dir / "manifest_public.json").write_text(
        json.dumps(
            {
                "pack_id": "Website_Expert_Pack_Final_V8",
                "study_mode": "pairwise_only",
                "block_a_target_count": 0,
                "block_b_target_count": 20,
                "practice_items": practice_items,
                "items": scored_items,
            }
        ),
        encoding="utf-8",
    )
    (pack_dir / "manifest_private.json").write_text(
        json.dumps({"practice_items": private_practice_items, "items": private_scored_items}),
        encoding="utf-8",
    )

    headers = {"x-admin-secret": "change-me-admin-secret"}
    import_response = client.post(
        "/api/v1/admin/import-pack",
        headers=headers,
        json={"pack_dir": str(pack_dir), "campaign_name": "Comprehension Campaign", "seed": 17, "activate": True},
    )
    assert import_response.status_code == 200, import_response.text

    invite = _invite(client)
    assert invite.status_code == 200, invite.text
    _submit_profile(client)

    for practice_index in range(3):
        next_b = client.get("/api/v1/block-b/next")
        assert next_b.status_code == 200, next_b.text
        submit = client.post(
            "/api/v1/block-b/submit",
            json={
                "assignment_id": next_b.json()["item"]["assignment_id"],
                "choice": "A",
                "confidence": 4,
                "comment": "Practice response logged with rationale.",
                "response_time_ms": 800,
            },
        )
        assert submit.status_code == 200, submit.text

    unlocked = client.get("/api/v1/block-b/next")
    assert unlocked.status_code == 200, unlocked.text
    session = client.get("/api/v1/session/me")
    assert session.status_code == 200, session.text
    assert session.json()["block_b_comprehension_passed"] is False
    assert session.json()["comprehension_risk"] is False


def test_pairwise_export_includes_protocol_and_exceedance_report(tmp_path: Path):
    _reset_db()
    client = TestClient(create_app())

    pack_dir = tmp_path / "Website_Expert_Pack_Final_V8"
    practice_items = []
    scored_items = []
    private_practice_items = []
    private_scored_items = []
    for idx in range(20):
        sample_id = f"exp{idx:02d}"
        images_dir = pack_dir / "images" / sample_id
        images_dir.mkdir(parents=True, exist_ok=True)
        for name in ["input.png", "A.png", "B.png"]:
            (images_dir / name).write_bytes(b"fake")
        item = {
            "sample_id": sample_id,
            "input": f"images/{sample_id}/input.png",
            "A": f"images/{sample_id}/A.png",
            "B": f"images/{sample_id}/B.png",
            "mask_type": "edge",
            "mask_coverage": 0.18,
            "coverage_bin": "10-25%",
            "is_anchor": idx in {4, 8, 12},
            "is_practice": idx < 3,
        }
        if idx < 3:
            practice_items.append(item)
            private_practice_items.append({"sample_id": sample_id, "mapping": {"A": "Ground-Truth", "B": "FT-SD+TTA"}})
        else:
            scored_items.append(item)
            private_scored_items.append({"sample_id": sample_id, "mapping": {"A": "FT-SD+TTA", "B": "Ground-Truth"}})

    (pack_dir / "manifest_public.json").write_text(
        json.dumps(
            {
                "pack_id": "Website_Expert_Pack_Final_V8",
                "study_mode": "pairwise_only",
                "block_a_target_count": 0,
                "block_b_target_count": 20,
                "practice_items": practice_items,
                "items": scored_items,
            }
        ),
        encoding="utf-8",
    )
    (pack_dir / "manifest_private.json").write_text(
        json.dumps({"practice_items": private_practice_items, "items": private_scored_items}),
        encoding="utf-8",
    )

    headers = {"x-admin-secret": "change-me-admin-secret"}
    import_response = client.post(
        "/api/v1/admin/import-pack",
        headers=headers,
        json={"pack_dir": str(pack_dir), "campaign_name": "Export Campaign", "seed": 19, "activate": True},
    )
    assert import_response.status_code == 200, import_response.text

    invite = _invite(client)
    assert invite.status_code == 200, invite.text
    _submit_profile(client)

    for practice_index in range(3):
        current = client.get("/api/v1/block-b/next")
        assert current.status_code == 200, current.text
        submit = client.post(
            "/api/v1/block-b/submit",
            json={
                "assignment_id": current.json()["item"]["assignment_id"],
                "choice": "A",
                "confidence": 5,
                "comment": "Practice pair selected for coherence.",
                "response_time_ms": 700,
            },
        )
        assert submit.status_code == 200, submit.text

    comprehension = _submit_block_b_comprehension(client, "spot_machine")
    assert comprehension["passed"] is True
    assert comprehension["comprehension_risk"] is False

    while True:
        current = client.get("/api/v1/block-b/next")
        assert current.status_code == 200, current.text
        payload = current.json()
        if payload["done"]:
            break
        submit = client.post(
            "/api/v1/block-b/submit",
            json={
                "assignment_id": payload["item"]["assignment_id"],
                "choice": "A",
                "confidence": 5,
                "comment": "Scored pair favors restoration usability.",
                "response_time_ms": 700,
            },
        )
        assert submit.status_code == 200, submit.text
        if submit.json()["done"]:
            break

    _submit_stage_feedback(client, "B", "Final review note for pairwise export coverage.")

    responses = client.get("/api/v1/admin/export/responses.json", headers=headers)
    assert responses.status_code == 200, responses.text
    responses_payload = responses.json()
    participant_row = responses_payload["participant_level"][0]
    assert participant_row["protocol_version"] == "ATHENA Expert Protocol v1.1"
    assert participant_row["block_b_comprehension_passed"] is True
    assert participant_row["comprehension_risk"] is False

    block_b_rows = [row for row in responses_payload["item_level"] if row["block"] == "B" and not row["is_practice"]]
    non_anchor_rows = [row for row in block_b_rows if not row["is_anchor"]]
    assert len(non_anchor_rows) == 12
    assert all(row["restoration_choice"] == "A" for row in non_anchor_rows)
    assert all(row["restoration_preferred"] is True for row in non_anchor_rows)

    quality = client.get("/api/v1/admin/export/quality_report.json", headers=headers)
    assert quality.status_code == 200, quality.text
    quality_payload = quality.json()
    exceedance = quality_payload["expert_plausibility_exceedance"]
    assert exceedance["tier_1_threshold"] == 0.4
    assert exceedance["tier_2_threshold"] == 3
    assert exceedance["robustness_rule"] == "Tier 2 must hold both with and without comprehension-risk participants."
    assert exceedance["full_cohort"]["participant_count"] == 1
    assert exceedance["full_cohort"]["tier_2_met"] is False
    assert exceedance["excluding_comprehension_risk"]["participant_count"] == 1
    assert exceedance["excluding_comprehension_risk"]["tier_2_met"] is False
    assert exceedance["findings_robust"] is False
    assert exceedance["participant_signals"][0]["non_anchor_scored_trials"] == 12
    assert exceedance["participant_signals"][0]["restoration_preference_rate"] == 1.0
    assert exceedance["participant_signals"][0]["tier_1_met"] is True


def test_admin_import_pack_upload_roundtrip(tmp_path: Path):
    _reset_db()
    client = TestClient(create_app())

    pack_dir = tmp_path / "Website_Expert_Pack_Final_V8"
    practice_items = []
    scored_items = []
    private_practice_items = []
    private_scored_items = []
    for idx in range(20):
        sample_id = "zip00_Vase funéraire en pâte claire" if idx == 0 else f"zip{idx:02d}"
        images_dir = pack_dir / "images" / sample_id
        images_dir.mkdir(parents=True, exist_ok=True)
        for name in ["input.png", "A.png", "B.png"]:
            (images_dir / name).write_bytes(b"fake")
        item = {
            "sample_id": sample_id,
            "input": f"images/{sample_id}/input.png",
            "A": f"images/{sample_id}/A.png",
            "B": f"images/{sample_id}/B.png",
            "mask_type": "edge",
            "mask_coverage": 0.18,
            "coverage_bin": "10-25%",
            "is_anchor": idx in {4, 8, 12},
            "is_practice": idx < 3,
        }
        if idx < 3:
            practice_items.append(item)
            private_practice_items.append(
                {"sample_id": sample_id, "mapping": {"A": "Ground-Truth", "B": "FT-SD+TTA"}}
            )
        else:
            scored_items.append(item)
            private_scored_items.append(
                {"sample_id": sample_id, "mapping": {"A": "FT-SD+TTA", "B": "Ground-Truth"}}
            )

    manifest = {
        "pack_id": "Website_Expert_Pack_Final_V8",
        "study_mode": "pairwise_only",
        "block_a_target_count": 0,
        "block_b_target_count": 20,
        "practice_items": practice_items,
        "items": scored_items,
    }
    (pack_dir / "manifest_public.json").write_text(json.dumps(manifest), encoding="utf-8")
    (pack_dir / "manifest_private.json").write_text(
        json.dumps({"practice_items": private_practice_items, "items": private_scored_items}),
        encoding="utf-8",
    )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w") as archive:
        for path in pack_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(pack_dir))
    zip_buffer.seek(0)

    headers = {"x-admin-secret": "change-me-admin-secret"}
    response = client.post(
        "/api/v1/admin/import-pack-upload",
        headers=headers,
        data={
            "campaign_name": "Uploaded Pairwise Campaign",
            "seed": "9",
            "activate": "true",
            "disjoint_blocks": "true",
        },
        files={"pack": ("Website_Expert_Pack_Final_V8.zip", zip_buffer.getvalue(), "application/zip")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["campaign_name"] == "Uploaded Pairwise Campaign"
    assert payload["is_active"] is True

    invite = _invite(client)
    assert invite.status_code == 200, invite.text
    progress = invite.json()["progress"]
    assert progress["block_a_total"] == 0
    assert progress["block_b_total"] == 20

    _submit_profile(client)
    next_b = client.get("/api/v1/block-b/next")
    assert next_b.status_code == 200, next_b.text
    item = next_b.json()["item"]
    assert item["input_url"].startswith("/static/")
    assert item["option_a_url"].startswith("/static/")
    assert item["option_b_url"].startswith("/static/")
    assert "funéraire" not in item["input_url"]

    asset_response = client.get(item["input_url"])
    assert asset_response.status_code == 200

    probe = client.get(
        "/api/v1/admin/debug/asset-probe",
        headers=headers,
        params={
            "campaign_id": payload["campaign_id"],
            "relative_path": item["input_url"].split(f"/static/{payload['campaign_id']}/", 1)[1],
        },
    )
    assert probe.status_code == 200, probe.text
    assert probe.json()["exists"] is True


def test_startup_bootstrap_imports_and_recovers_missing_assets(tmp_path: Path, monkeypatch):
    _reset_db()

    zip_path = tmp_path / "bootstrap_pack.zip"
    _build_pairwise_pack_zip(zip_path)

    monkeypatch.setenv("BOOTSTRAP_PACK_ON_STARTUP", "true")
    monkeypatch.setenv("BOOTSTRAP_PACK_ZIP_PATH", str(zip_path))
    monkeypatch.setenv("BOOTSTRAP_CAMPAIGN_NAME", "Bootstrap Campaign")
    monkeypatch.setenv("BOOTSTRAP_CAMPAIGN_SEED", "3030")
    monkeypatch.setenv("BOOTSTRAP_ACTIVATE", "true")
    monkeypatch.setenv("BOOTSTRAP_DISJOINT_BLOCKS", "true")
    monkeypatch.setenv("BOOTSTRAP_STRICT", "true")
    get_settings.cache_clear()

    with TestClient(create_app()) as client:
        invite = _invite(client)
        assert invite.status_code == 200, invite.text
        first_campaign_id = invite.json()["campaign"]["id"]

    with SessionLocal() as db:
        campaign = db.query(Campaign).filter(Campaign.id == first_campaign_id).first()
        assert campaign is not None
        first_asset = db.query(BlockBItem).filter(BlockBItem.campaign_id == first_campaign_id).first()
        assert first_asset is not None
        relative = first_asset.input_url.split(f"/static/{first_campaign_id}/", 1)[1]
        asset_path = get_settings().storage_root / "campaigns" / str(first_campaign_id) / relative
        assert asset_path.exists()
        os.remove(asset_path)
        assert not asset_path.exists()

    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        invite = _invite(client)
        assert invite.status_code == 200, invite.text
        second_campaign_id = invite.json()["campaign"]["id"]
        assert second_campaign_id != first_campaign_id

    with SessionLocal() as db:
        active_campaign = db.query(Campaign).filter(Campaign.is_active.is_(True)).first()
        assert active_campaign is not None
        assert active_campaign.id == second_campaign_id

    monkeypatch.delenv("BOOTSTRAP_PACK_ON_STARTUP")
    monkeypatch.delenv("BOOTSTRAP_PACK_ZIP_PATH")
    monkeypatch.delenv("BOOTSTRAP_CAMPAIGN_NAME")
    monkeypatch.delenv("BOOTSTRAP_CAMPAIGN_SEED")
    monkeypatch.delenv("BOOTSTRAP_ACTIVATE")
    monkeypatch.delenv("BOOTSTRAP_DISJOINT_BLOCKS")
    monkeypatch.delenv("BOOTSTRAP_STRICT")
    get_settings.cache_clear()


def test_legacy_relative_asset_urls_are_normalized_for_block_a_b_and_c():
    _reset_db()
    client = TestClient(create_app())

    with SessionLocal() as db:
        campaign = Campaign(
            name="Legacy Asset Campaign",
            seed=7,
            is_active=True,
            block_a_target_count=1,
            block_b_target_count=2,
        )
        db.add(campaign)
        db.flush()
        db.add(
            BlockAItem(
                campaign_id=campaign.id,
                sample_id="blkA_legacy_sample_A",
                image_url="A.png",
                source_label="generated",
                mask_type="rect",
                mask_coverage_bin="10-25%",
                metadata_json={"origin_sample_id": "legacy_sample", "origin_variant": "A"},
            )
        )
        db.add(
            BlockBItem(
                campaign_id=campaign.id,
                sample_id="legacy_sample",
                input_url="input.png",
                option_a_url="A.png",
                option_b_url="B.png",
                mask_type="edge",
                mask_coverage_bin="10-25%",
                metadata_json={"study_mode": "pairwise_only"},
            )
        )
        db.add(
            BlockBItem(
                campaign_id=campaign.id,
                sample_id="legacy_block_c_sample",
                input_url="input.png",
                option_a_url="A.png",
                option_b_url="B.png",
                mask_type="edge",
                mask_coverage_bin="10-25%",
                metadata_json={
                    "study_mode": "three_part",
                    "block_part": "C",
                    "option_c_url": "C.png",
                    "option_d_url": "D.png",
                },
            )
        )
        db.commit()

    invite = _invite(client)
    assert invite.status_code == 200, invite.text
    _submit_profile(client)

    next_a = client.get("/api/v1/block-a/next")
    assert next_a.status_code == 200, next_a.text
    block_a_item = next_a.json()["item"]
    assert block_a_item["image_url"] == "/static/1/images/legacy_sample/A.png"

    submit_a = client.post(
        "/api/v1/block-a/submit",
        json={
            "assignment_id": block_a_item["assignment_id"],
            "authenticity_likelihood": 4,
            "archaeological_plausibility": 4,
            "confidence": 4,
            "comment": "Legacy image path still produces interpretable output.",
            "response_time_ms": 1000,
        },
    )
    assert submit_a.status_code == 200, submit_a.text
    _submit_stage_feedback(client, "A", "Legacy assets validated.")

    next_b = client.get("/api/v1/block-b/next")
    assert next_b.status_code == 200, next_b.text
    block_b_item = next_b.json()["item"]
    assert block_b_item["input_url"] == "/static/1/images/legacy_sample/input.png"
    assert block_b_item["option_a_url"] == "/static/1/images/legacy_sample/A.png"
    assert block_b_item["option_b_url"] == "/static/1/images/legacy_sample/B.png"

    submit_b = client.post(
        "/api/v1/block-b/submit",
        json={
            "assignment_id": block_b_item["assignment_id"],
            "choice": "A",
            "confidence": 4,
            "comment": "Legacy pair still resolves to static assets.",
            "response_time_ms": 1000,
        },
    )
    assert submit_b.status_code == 200, submit_b.text

    _submit_stage_feedback(client, "B", "Legacy pairwise wrap-up is complete.")

    next_c = client.get("/api/v1/block-c/next")
    assert next_c.status_code == 200, next_c.text
    block_c_item = next_c.json()["item"]
    assert block_c_item["input_url"] == "/static/1/images/legacy_block_c_sample/input.png"
    assert block_c_item["option_a_url"] == "/static/1/images/legacy_block_c_sample/A.png"
    assert block_c_item["option_b_url"] == "/static/1/images/legacy_block_c_sample/B.png"
    assert block_c_item["option_c_url"] == "/static/1/images/legacy_block_c_sample/C.png"
    assert block_c_item["option_d_url"] == "/static/1/images/legacy_block_c_sample/D.png"

    submit_c = client.post(
        "/api/v1/block-c/submit",
        json={
            "assignment_id": block_c_item["assignment_id"],
            "choice": "C",
            "confidence": 4,
            "comment": "Candidate C keeps the vessel geometry more coherent.",
            "response_time_ms": 1000,
        },
    )
    assert submit_c.status_code == 200, submit_c.text

    progress = client.get("/api/v1/progress")
    assert progress.status_code == 200, progress.text
    assert progress.json()["block_c_feedback_completed"] is False

    submit_c_feedback = client.put(
        "/api/v1/session/feedback/C",
        json={"comment": "Model C is the most balanced option for the synthetic restoration set."},
    )
    assert submit_c_feedback.status_code == 200, submit_c_feedback.text

    final_progress = client.get("/api/v1/progress")
    assert final_progress.status_code == 200, final_progress.text
    assert final_progress.json()["block_c_feedback_completed"] is True
    assert final_progress.json()["is_complete"] is True


def test_invite_rejects_invalid_code():
    _reset_db()
    _seed_campaign_items()
    client = TestClient(create_app())
    invite = _invite(client, invite_code="wrong-code")
    assert invite.status_code == 401


def test_stale_session_for_inactive_campaign_is_rejected():
    _reset_db()
    _seed_campaign_items()
    client = TestClient(create_app())

    invite = _invite(client)
    assert invite.status_code == 200, invite.text

    with SessionLocal() as db:
        active_campaign = db.query(Campaign).filter(Campaign.is_active.is_(True)).first()
        assert active_campaign is not None
        active_campaign.is_active = False
        replacement = Campaign(
            name="Replacement Campaign",
            seed=99,
            is_active=True,
            block_a_target_count=1,
            block_b_target_count=1,
        )
        db.add(replacement)
        db.commit()

    me = client.get("/api/v1/session/me")
    assert me.status_code == 401
    assert me.json()["detail"] == "Participant session is stale."


def test_completed_session_blocks_new_submissions():
    _reset_db()
    _seed_campaign_items()
    client = TestClient(create_app())
    invite = _invite(client)
    assert invite.status_code == 200, invite.text
    _submit_profile(client)

    _complete_block_a(client)
    _submit_stage_feedback(client, "A", "Block A was coherent and archaeologically plausible.")
    while True:
        next_response = client.get("/api/v1/block-b/next")
        assert next_response.status_code == 200
        payload = next_response.json()
        if payload["done"]:
            break
        item = payload["item"]
        submit = client.post(
            "/api/v1/block-b/submit",
            json={
                "assignment_id": item["assignment_id"],
                "choice": "A",
                "confidence": 4,
                "comment": "Final session run indicates preference for A.",
                "response_time_ms": 1100,
            },
        )
        assert submit.status_code == 200, submit.text

    _submit_stage_feedback(client, "B", "Block B comparisons were meaningful and informative.")
    finalize = client.post("/api/v1/session/complete")
    assert finalize.status_code == 200, finalize.text

    completed_next = client.get("/api/v1/block-a/next")
    assert completed_next.status_code == 200
    assert completed_next.json()["done"] is True

    submit_again = client.post(
        "/api/v1/block-a/submit",
        json={
            "assignment_id": 1,
            "authenticity_likelihood": 4,
            "archaeological_plausibility": 4,
            "confidence": 4,
            "comment": "Post-completion submission attempt.",
            "response_time_ms": 1200,
        },
    )
    assert submit_again.status_code == 409


def test_block_b_requires_stage_feedback():
    _reset_db()
    _seed_campaign_items()
    client = TestClient(create_app())
    invite = _invite(client)
    assert invite.status_code == 200, invite.text
    _submit_profile(client)

    _complete_block_a(client)
    blocked = client.get("/api/v1/block-b/next")
    assert blocked.status_code == 409
    assert "feedback" in blocked.json()["detail"].lower()

    short_feedback = client.put("/api/v1/session/feedback/A", json={"comment": "too short"})
    assert short_feedback.status_code == 422

    _submit_stage_feedback(client, "A", "This block revealed useful archaeological concerns.")
    next_b = client.get("/api/v1/block-b/next")
    assert next_b.status_code == 200
