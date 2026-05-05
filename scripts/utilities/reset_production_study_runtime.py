from __future__ import annotations

import argparse
import os

import requests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset production ATHENA study runtime rows and verify next invite starts from R0001."
    )
    parser.add_argument("--backend-url", default=os.getenv("ATHENA_BACKEND_URL"), required=os.getenv("ATHENA_BACKEND_URL") is None)
    parser.add_argument("--admin-secret-env", default="ADMIN_EXPORT_SECRET")
    parser.add_argument("--admin-ui-password-env", default="ADMIN_UI_PASSWORD")
    parser.add_argument("--invite-code-env", default="APP_INVITE_CODE")
    parser.add_argument("--confirm-phrase-env", default="ADMIN_RESET_CONFIRM_PHRASE")
    parser.add_argument("--campaign-id", type=int, default=None)
    parser.add_argument("--all-active-campaigns", action="store_true")
    parser.add_argument("--all-campaigns", action="store_true")
    parser.add_argument("--remove-assets", action="store_true")
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


def main() -> None:
    args = _parse_args()
    backend_url = args.backend_url.rstrip("/")
    invite_code = _required_env(args.invite_code_env)
    confirm_phrase = _required_env(args.confirm_phrase_env)

    session, headers = _build_admin_auth(
        backend_url=backend_url,
        timeout_seconds=args.timeout_seconds,
        admin_secret_env=args.admin_secret_env,
        admin_ui_password_env=args.admin_ui_password_env,
    )
    reset_payload = {
        "confirm_phrase": confirm_phrase,
        "campaign_id": args.campaign_id,
        "all_active_campaigns": args.all_active_campaigns,
        "all_campaigns": args.all_campaigns,
        "remove_assets": args.remove_assets,
    }
    reset_response = session.post(
        f"{backend_url}/api/v1/admin/runtime/reset",
        headers=headers,
        json=reset_payload,
        timeout=args.timeout_seconds,
    )
    if reset_response.status_code != 200:
        raise RuntimeError(f"Reset failed: {reset_response.status_code} {reset_response.text}")
    reset_report = reset_response.json()

    invite_response = requests.post(
        f"{backend_url}/api/v1/auth/invite",
        json={"invite_code": invite_code},
        timeout=args.timeout_seconds,
    )
    if invite_response.status_code != 200:
        raise RuntimeError(f"Invite verification failed: {invite_response.status_code} {invite_response.text}")
    participant_public_id = invite_response.json().get("participant_public_id", "")
    if participant_public_id != "R0001":
        raise RuntimeError(f"Expected first participant to be R0001, got {participant_public_id!r}")

    print("Runtime reset succeeded.")
    print(f"Campaign IDs: {reset_report.get('campaign_ids')}")
    print(f"Deleted counts: {reset_report.get('deleted_counts')}")
    print(f"Identity reset: {reset_report.get('identity_reset')}")
    print("Verification invite returned participant_public_id=R0001")


if __name__ == "__main__":
    main()
