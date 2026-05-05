from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


CURRENT_FILE = Path(__file__).resolve()


def _resolve_api_root(current_file: Path) -> Path:
    for parent in current_file.parents:
        if (parent / "alembic.ini").exists() and (parent / "app").is_dir():
            return parent

    website_root = next((parent for parent in current_file.parents if parent.name == "website"), None)
    if website_root is not None:
        return website_root / "services" / "api"

    parent_index = 2 if len(current_file.parents) > 2 else len(current_file.parents) - 1
    return current_file.parents[parent_index]


API_ROOT = _resolve_api_root(CURRENT_FILE)
WEBSITE_ROOT = next((parent for parent in CURRENT_FILE.parents if parent.name == "website"), API_ROOT)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Archaeologist Evaluation API"
    app_env: str = "development"
    app_invite_code: str = "athena-invite"
    app_shared_username: str = "archaeologist"
    app_shared_password: str = "change-me"
    session_secret: str = "change-me-session-secret"
    admin_export_secret: str = "change-me-admin-secret"
    admin_ui_password: str = ""
    admin_reset_confirm_phrase: str = "RESET_STUDY_RUNTIME"
    session_cookie_name: str = "arch_eval_session"
    admin_cookie_name: str = "arch_eval_admin_session"
    session_cookie_samesite: str = "lax"
    session_cookie_secure: bool | None = None  # None = auto (True when app_env==production)
    session_cookie_domain: str = ""
    session_ttl_hours: int = 12
    admin_session_ttl_hours: int = 12
    allowed_origins: str = "http://localhost:3000"
    database_url: str = f"sqlite:///{(API_ROOT / 'data' / 'website.db').as_posix()}"
    storage_root: Path = API_ROOT / "data"
    static_mount_path: str = "/static"
    default_campaign_seed: int = 42
    block_a_target_count: int = 25
    block_b_target_count: int = 15
    block_a_attention_checks: int = 2
    block_b_attention_checks: int = 1
    expert_protocol_version: str = "ATHENA Expert Protocol v1.1"
    bootstrap_pack_on_startup: bool = False
    bootstrap_pack_zip_path: Path = API_ROOT / "bootstrap" / "final_expert_pack.zip"
    bootstrap_campaign_name: str = "ATHENA Final V8 Expert Validation"
    bootstrap_campaign_seed: int = 2026
    bootstrap_activate: bool = True
    bootstrap_disjoint_blocks: bool = True
    bootstrap_strict: bool = False

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse comma-separated ALLOWED_ORIGINS env var into a list."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def effective_cookie_secure(self) -> bool:
        """Resolve cookie Secure flag: explicit setting wins, otherwise True in production."""
        if self.session_cookie_secure is not None:
            return self.session_cookie_secure
        return self.app_env == "production"

    @property
    def effective_cookie_samesite(self) -> str:
        """In production the frontend and backend are on different domains, so we need
        SameSite=None (with Secure=True) for the session cookie to be sent cross-site.
        In development, SameSite=Lax is fine."""
        if self.session_cookie_samesite != "lax":
            return self.session_cookie_samesite  # explicit override wins
        return "none" if self.app_env == "production" else "lax"

    @property
    def effective_cookie_domain(self) -> str | None:
        """Return cookie domain or None if blank (browser default)."""
        return self.session_cookie_domain.strip() or None

    @property
    def effective_admin_ui_password(self) -> str:
        """Allow a dedicated admin UI password while preserving backward compatibility."""
        return self.admin_ui_password or self.admin_export_secret


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    return settings
