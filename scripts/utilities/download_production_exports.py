from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ATHENA production exports to a timestamped local folder.")
    parser.add_argument("--backend-url", default=os.getenv("ATHENA_BACKEND_URL"), required=os.getenv("ATHENA_BACKEND_URL") is None)
    parser.add_argument("--admin-secret-env", default="ADMIN_EXPORT_SECRET")
    parser.add_argument("--admin-ui-password-env", default="ADMIN_UI_PASSWORD")
    parser.add_argument("--campaign-id", type=int, default=None)
    parser.add_argument("--output-root", default="outputs/website_exports")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _build_admin_auth(
    *,
    backend_url: str,
    timeout_seconds: float,
    admin_secret_env: str,
    admin_ui_password_env: str,
) -> tuple[requests.Session, dict[str, str]]:
    session = requests.Session()
    admin_secret = os.getenv(admin_secret_env, "").strip()
    if admin_secret:
        return session, {"x-admin-secret": admin_secret}

    admin_ui_password = os.getenv(admin_ui_password_env, "").strip()
    if not admin_ui_password:
        raise RuntimeError(
            f"Provide {admin_secret_env} or {admin_ui_password_env} for admin authentication."
        )
    login_response = session.post(
        f"{backend_url}/api/v1/admin/auth/login",
        json={"password": admin_ui_password},
        timeout=timeout_seconds,
    )
    if login_response.status_code != 200:
        raise RuntimeError(f"Admin login failed: {login_response.status_code} {login_response.text}")
    return session, {}


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _download(session: requests.Session, url: str, headers: dict[str, str], timeout_seconds: float) -> requests.Response:
    response = session.get(url, headers=headers, timeout=timeout_seconds)
    if response.status_code != 200:
        raise RuntimeError(f"Export download failed for {url}: {response.status_code} {response.text}")
    return response


def main() -> None:
    args = _parse_args()
    backend_url = args.backend_url.rstrip("/")
    session, headers = _build_admin_auth(
        backend_url=backend_url,
        timeout_seconds=args.timeout_seconds,
        admin_secret_env=args.admin_secret_env,
        admin_ui_password_env=args.admin_ui_password_env,
    )
    query = f"?campaign_id={args.campaign_id}" if args.campaign_id else ""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_root).resolve() / f"prod_exports_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    responses_json = _download(
        session,
        f"{backend_url}/api/v1/admin/export/responses.json{query}",
        headers=headers,
        timeout_seconds=args.timeout_seconds,
    )
    responses_csv = _download(
        session,
        f"{backend_url}/api/v1/admin/export/responses.csv{query}",
        headers=headers,
        timeout_seconds=args.timeout_seconds,
    )
    quality_report = _download(
        session,
        f"{backend_url}/api/v1/admin/export/quality_report.json{query}",
        headers=headers,
        timeout_seconds=args.timeout_seconds,
    )

    _write_text(output_dir / "responses.json", json.dumps(responses_json.json(), indent=2))
    _write_text(output_dir / "responses.csv", responses_csv.text)
    _write_text(output_dir / "quality_report.json", json.dumps(quality_report.json(), indent=2))
    _write_text(
        output_dir / "metadata.json",
        json.dumps(
            {
                "created_utc": stamp,
                "backend_url": backend_url,
                "campaign_id": args.campaign_id,
                "files": ["responses.json", "responses.csv", "quality_report.json"],
            },
            indent=2,
        ),
    )
    print(f"Exports downloaded to: {output_dir}")


if __name__ == "__main__":
    main()
