from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional at runtime
    plt = None


REPO_ROOT = Path(__file__).resolve().parents[2]
API_ROOT = REPO_ROOT / "website" / "services" / "api"

if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db.session import init_db
from app.main import create_app
from app.services.ingest_service import import_pack
from app.db.session import SessionLocal


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hard-reset local website DB, import pack, run two simulated validators, and export analysis artifacts."
    )
    parser.add_argument(
        "--pack-zip",
        default=str(API_ROOT / "bootstrap" / "final_expert_pack.zip"),
        help="Path to website-ready expert pack zip.",
    )
    parser.add_argument(
        "--campaign-name",
        default="Two-Validator Simulation Campaign",
        help="Campaign name for the imported simulation campaign.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260323,
        help="Deterministic campaign seed.",
    )
    parser.add_argument(
        "--output-root",
        default=str(API_ROOT / "data" / "simulation_outputs"),
        help="Root directory for exported simulation artifacts.",
    )
    parser.add_argument(
        "--reset-only",
        action="store_true",
        help="Only reset and import campaign, do not run validators.",
    )
    return parser.parse_args()


def _run(cmd: list[str], cwd: Path) -> None:
    completed = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(cmd)}):\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def _reset_database_and_storage() -> None:
    settings = get_settings()
    db_file: Path | None = None
    if settings.database_url.startswith("sqlite:///"):
        db_file = Path(settings.database_url.replace("sqlite:///", "", 1))
        if db_file.exists():
            db_file.unlink()
    campaigns_dir = settings.storage_root / "campaigns"
    if campaigns_dir.exists():
        shutil.rmtree(campaigns_dir)
    campaigns_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run(["alembic", "upgrade", "head"], cwd=API_ROOT)
    except RuntimeError:
        if db_file and db_file.exists():
            db_file.unlink()
        init_db()


def _import_pack(pack_zip: Path, campaign_name: str, seed: int) -> int:
    if not pack_zip.exists():
        raise FileNotFoundError(f"Pack zip not found: {pack_zip}")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(pack_zip, "r") as archive:
            archive.extractall(tmp_path)
        with SessionLocal() as db:
            campaign = import_pack(
                db=db,
                pack_dir=tmp_path,
                campaign_name=campaign_name,
                seed=seed,
                activate=True,
                disjoint_blocks=True,
            )
            return int(campaign.id)


def _submit_profile(client: TestClient, name: str, institution: str, discipline: str) -> None:
    response = client.put(
        "/api/v1/session/profile",
        json={
            "name": name,
            "institution": institution,
            "discipline": discipline,
            "discipline_other": "",
        },
    )
    if response.status_code != 200:
        raise RuntimeError(f"Profile submit failed: {response.status_code} {response.text}")


def _submit_block_a_all(client: TestClient, comment_prefix: str) -> None:
    block_a_done = False
    index = 0
    while not block_a_done:
        nxt = client.get("/api/v1/block-a/next")
        if nxt.status_code != 200:
            raise RuntimeError(f"Block A next failed: {nxt.status_code} {nxt.text}")
        payload = nxt.json()
        if payload.get("done"):
            block_a_done = True
            break
        item = payload["item"]
        rating = 3 + (index % 3 > 0)
        response = client.post(
            "/api/v1/block-a/submit",
            json={
                "assignment_id": item["assignment_id"],
                "authenticity_likelihood": rating,
                "archaeological_plausibility": min(5, rating + 1),
                "confidence": 2 + (index % 4),
                "comment": f"{comment_prefix} A-item {index + 1}: surface continuity and iconographic plausibility checked.",
                "response_time_ms": 1200 + (index * 33),
            },
        )
        if response.status_code != 200:
            raise RuntimeError(f"Block A submit failed: {response.status_code} {response.text}")
        index += 1

    feedback = client.put(
        "/api/v1/session/feedback/A",
        json={"comment": f"{comment_prefix} Block A summary: ratings completed with mandatory item comments."},
    )
    if feedback.status_code != 200:
        raise RuntimeError(f"Block A feedback failed: {feedback.status_code} {feedback.text}")


def _handle_comprehension_gate(client: TestClient, pass_gate: bool) -> None:
    if pass_gate:
        response = client.post("/api/v1/session/block-b-comprehension", json={"selected_option": "restoration_guide"})
        if response.status_code != 200:
            raise RuntimeError(f"Comprehension submit failed: {response.status_code} {response.text}")
        return

    for option in ("spot_machine", "guess_original"):
        response = client.post("/api/v1/session/block-b-comprehension", json={"selected_option": option})
        if response.status_code != 200:
            raise RuntimeError(f"Comprehension submit failed: {response.status_code} {response.text}")


def _submit_block_b_all(client: TestClient, choice_cycle: list[str], comment_prefix: str, pass_gate: bool) -> None:
    gate_handled = False
    index = 0
    while True:
        nxt = client.get("/api/v1/block-b/next")
        if nxt.status_code == 409:
            detail = nxt.json().get("detail", "")
            if "comprehension check required" in detail and not gate_handled:
                _handle_comprehension_gate(client, pass_gate=pass_gate)
                gate_handled = True
                continue
            raise RuntimeError(f"Block B next blocked: {nxt.status_code} {nxt.text}")
        if nxt.status_code != 200:
            raise RuntimeError(f"Block B next failed: {nxt.status_code} {nxt.text}")
        payload = nxt.json()
        if payload.get("done"):
            break
        item = payload["item"]
        choice = choice_cycle[index % len(choice_cycle)]
        confidence = 2 + (index % 4)
        response = client.post(
            "/api/v1/block-b/submit",
            json={
                "assignment_id": item["assignment_id"],
                "choice": choice,
                "confidence": confidence,
                "comment": f"{comment_prefix} B-item {index + 1}: chose {choice} based on restoration guide utility.",
                "response_time_ms": 1400 + (index * 45),
            },
        )
        if response.status_code == 409:
            detail = response.json().get("detail", "")
            if "comprehension check required" in detail and not gate_handled:
                _handle_comprehension_gate(client, pass_gate=pass_gate)
                gate_handled = True
                continue
        if response.status_code != 200:
            raise RuntimeError(f"Block B submit failed: {response.status_code} {response.text}")
        index += 1

    feedback = client.put(
        "/api/v1/session/feedback/B",
        json={"comment": f"{comment_prefix} Block B summary: pairwise comparisons completed with mandatory item comments."},
    )
    if feedback.status_code != 200:
        raise RuntimeError(f"Block B feedback failed: {feedback.status_code} {feedback.text}")


def _complete_session(client: TestClient) -> None:
    response = client.post("/api/v1/session/complete")
    if response.status_code != 200:
        raise RuntimeError(f"Session completion failed: {response.status_code} {response.text}")


def _simulate_validator(
    client: TestClient,
    *,
    invite_code: str,
    name: str,
    institution: str,
    discipline: str,
    comment_prefix: str,
    block_b_choices: list[str],
    pass_comprehension: bool,
) -> str:
    invite = client.post("/api/v1/auth/invite", json={"invite_code": invite_code})
    if invite.status_code != 200:
        raise RuntimeError(f"Invite failed: {invite.status_code} {invite.text}")
    participant_id = invite.json().get("participant_public_id", "UNKNOWN")
    _submit_profile(client, name=name, institution=institution, discipline=discipline)

    progress = client.get("/api/v1/progress")
    if progress.status_code != 200:
        raise RuntimeError(f"Progress load failed: {progress.status_code} {progress.text}")
    block_a_total = int(progress.json().get("block_a_total", 0))
    if block_a_total > 0:
        _submit_block_a_all(client, comment_prefix=comment_prefix)

    _submit_block_b_all(
        client,
        choice_cycle=block_b_choices,
        comment_prefix=comment_prefix,
        pass_gate=pass_comprehension,
    )
    _complete_session(client)
    return participant_id


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _plot_counter(counter: Counter[str], title: str, output_path: Path) -> None:
    if plt is None:
        return
    labels = list(counter.keys()) or ["none"]
    values = [counter[k] for k in labels] if counter else [0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Count")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_response_time(rows: list[dict[str, Any]], output_path: Path) -> None:
    if plt is None:
        return
    per_block: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        block = str(row.get("block", "NA"))
        per_block[block].append(int(row.get("response_time_ms", 0)))

    labels = sorted(per_block.keys())
    values = [sum(per_block[label]) / max(1, len(per_block[label])) for label in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_title("Mean response time by block")
    ax.set_ylabel("Milliseconds")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _analysis_from_exports(
    responses_json: dict[str, Any],
    quality_json: dict[str, Any],
    output_dir: Path,
) -> None:
    item_rows = list(responses_json.get("item_level", []))

    item_comment_total = len(item_rows)
    item_comment_non_empty = sum(1 for row in item_rows if str(row.get("comment", "")).strip())
    item_comment_completion = (
        (item_comment_non_empty / item_comment_total) * 100.0 if item_comment_total > 0 else 0.0
    )

    block_b_rows = [row for row in item_rows if row.get("block") == "B"]
    choice_counter = Counter(str(row.get("choice", "NA")) for row in block_b_rows)
    confidence_counter = Counter(str(row.get("confidence", "NA")) for row in block_b_rows)

    exceedance = quality_json.get("expert_plausibility_exceedance", {})
    summary_lines = [
        "# Two-validator simulation summary",
        "",
        f"- item comments non-empty: {item_comment_non_empty}/{item_comment_total} ({item_comment_completion:.1f}%)",
        f"- full cohort tier_2_met: {exceedance.get('full_cohort', {}).get('tier_2_met')}",
        f"- excluding comprehension risk tier_2_met: {exceedance.get('excluding_comprehension_risk', {}).get('tier_2_met')}",
        f"- findings robust: {exceedance.get('findings_robust')}",
    ]
    _write_text(output_dir / "analysis" / "summary.md", "\n".join(summary_lines) + "\n")

    completion_payload = {
        "item_comment_total": item_comment_total,
        "item_comment_non_empty": item_comment_non_empty,
        "item_comment_completion_percent": round(item_comment_completion, 4),
    }
    _write_text(output_dir / "analysis" / "comment_completion.json", json.dumps(completion_payload, indent=2))

    exceedance_table = io.StringIO()
    writer = csv.writer(exceedance_table)
    writer.writerow(["cohort", "participant_count", "tier_2_met"])
    writer.writerow(
        [
            "full_cohort",
            exceedance.get("full_cohort", {}).get("participant_count"),
            exceedance.get("full_cohort", {}).get("tier_2_met"),
        ]
    )
    writer.writerow(
        [
            "excluding_comprehension_risk",
            exceedance.get("excluding_comprehension_risk", {}).get("participant_count"),
            exceedance.get("excluding_comprehension_risk", {}).get("tier_2_met"),
        ]
    )
    _write_text(output_dir / "analysis" / "exceedance_summary.csv", exceedance_table.getvalue())

    _plot_counter(choice_counter, "Choice distribution (Block B)", output_dir / "analysis" / "choice_distribution.png")
    _plot_counter(
        confidence_counter,
        "Confidence distribution (Block B)",
        output_dir / "analysis" / "confidence_distribution.png",
    )
    _plot_response_time(item_rows, output_dir / "analysis" / "response_time_summary.png")


def _export_outputs(client: TestClient, output_dir: Path, admin_secret: str) -> tuple[dict[str, Any], dict[str, Any]]:
    headers = {"x-admin-secret": admin_secret}
    json_res = client.get("/api/v1/admin/export/responses.json", headers=headers)
    csv_res = client.get("/api/v1/admin/export/responses.csv", headers=headers)
    quality_res = client.get("/api/v1/admin/export/quality_report.json", headers=headers)
    if json_res.status_code != 200:
        raise RuntimeError(f"responses.json export failed: {json_res.status_code} {json_res.text}")
    if csv_res.status_code != 200:
        raise RuntimeError(f"responses.csv export failed: {csv_res.status_code} {csv_res.text}")
    if quality_res.status_code != 200:
        raise RuntimeError(f"quality_report export failed: {quality_res.status_code} {quality_res.text}")

    exports_dir = output_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    _write_text(exports_dir / "responses.json", json.dumps(json_res.json(), indent=2))
    _write_text(exports_dir / "responses.csv", csv_res.text)
    _write_text(exports_dir / "quality_report.json", json.dumps(quality_res.json(), indent=2))
    return json_res.json(), quality_res.json()


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    pack_zip = Path(args.pack_zip).resolve()
    output_root = Path(args.output_root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"two_validator_sim_{stamp}"

    _reset_database_and_storage()
    campaign_id = _import_pack(pack_zip=pack_zip, campaign_name=args.campaign_name, seed=args.seed)

    if args.reset_only:
        _write_text(
            output_dir / "reset_only.txt",
            f"Reset complete. Imported campaign_id={campaign_id}. Next participant ID starts at R0001.\n",
        )
        print(f"Reset complete. Campaign {campaign_id} imported. Output: {output_dir}")
        return

    app = create_app()
    client_a = TestClient(app)
    client_b = TestClient(app)
    validator_a = _simulate_validator(
        client_a,
        invite_code=settings.app_invite_code,
        name="Validator A",
        institution="University Lab",
        discipline="Archaeology",
        comment_prefix="Validator A",
        block_b_choices=["A", "B", "Tie", "A", "Unsure"],
        pass_comprehension=True,
    )
    validator_b = _simulate_validator(
        client_b,
        invite_code=settings.app_invite_code,
        name="Validator B",
        institution="Museum",
        discipline="Conservation / Restoration",
        comment_prefix="Validator B",
        block_b_choices=["B", "B", "A", "Unsure", "Tie"],
        pass_comprehension=False,
    )

    responses_json, quality_json = _export_outputs(
        client_a,
        output_dir=output_dir,
        admin_secret=settings.admin_export_secret,
    )
    _analysis_from_exports(responses_json=responses_json, quality_json=quality_json, output_dir=output_dir)

    _write_text(
        output_dir / "run_metadata.json",
        json.dumps(
            {
                "created_utc": stamp,
                "campaign_id": campaign_id,
                "protocol_version": settings.expert_protocol_version,
                "participants": [validator_a, validator_b],
                "expected_first_participant_id": "R0001",
            },
            indent=2,
        ),
    )
    print(f"Simulation complete. Participants: {validator_a}, {validator_b}. Output: {output_dir}")


if __name__ == "__main__":
    main()
