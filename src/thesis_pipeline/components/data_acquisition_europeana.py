import json
import logging
import re
import time
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


@dataclass
class EuropeanaAcquisitionSummary:
    source: str
    attempted: int
    downloaded: int
    skipped_no_full_image: int
    skipped_no_rights: int
    skipped_non_copyright_free: int
    failed_downloads: int
    errors: list[str]


class EuropeanaOpenAcquirer:
    """Downloader for Europeana Search API (open-only items).

    Important: Europeana provides CC0 metadata, but images are licensed per item.
    We only query `reusability=open` and still record per-item rights in metadata.
    """

    API_URL = "https://api.europeana.eu/record/v2/search.json"
    DEFAULT_QUALITY_QF = [
        "IMAGE_COLOR:true",
        "IMAGE_GRAYSCALE:false",
    ]

    def __init__(self, *, api_key: str, session: requests.Session | None = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = (api_key or "").strip()
        self.session = session or requests.Session()
        self.session.headers.update(
            {"User-Agent": "MasterThesisResearchBot/1.0 (research; contact: vtsav) requests"}
        )

    def download(
        self,
        *,
        query: str,
        output_dir: Path,
        metadata_dir: Path,
        limit: int = 10_000,
        rows: int = 100,
        reusability: str = "open",
        qf_filters: list[str] | None = None,
        copyright_free_only: bool = True,
        additional_copyright_free_rights: list[str] | None = None,
        sleep_s: float = 0.05,
        state_file: Path | None = None,
    ) -> EuropeanaAcquisitionSummary:
        if not self.api_key:
            return EuropeanaAcquisitionSummary(
                source="europeana",
                attempted=0,
                downloaded=0,
                skipped_no_full_image=0,
                skipped_no_rights=0,
                skipped_non_copyright_free=0,
                failed_downloads=0,
                errors=["Missing EUROPEANA_API_KEY (env var or .env)."],
            )

        output_dir = Path(output_dir)
        metadata_dir = Path(metadata_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        state_path = Path(state_file) if state_file else (output_dir / "europeana_state.json")
        state: dict[str, Any] = {"cursor": "*", "downloaded_ids": []}
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                state = {"cursor": "*", "downloaded_ids": []}

        cursor = str(state.get("cursor") or "*")
        downloaded_ids: set[str] = set(str(x) for x in state.get("downloaded_ids", []))

        attempted = 0
        downloaded = 0
        skipped_no_full_image = 0
        skipped_no_rights = 0
        skipped_non_copyright_free = 0
        failed_downloads = 0
        errors: list[str] = []

        while downloaded < int(limit):
            try:
                data = self._search(
                    query=query,
                    cursor=cursor,
                    rows=int(rows),
                    reusability=reusability,
                    qf_filters=qf_filters,
                )
            except Exception as exc:
                errors.append(f"search failed: {exc}")
                break

            items = data.get("items") or []
            if not items:
                break

            next_cursor = data.get("nextCursor")
            for item in items:
                if downloaded >= int(limit):
                    break

                rec_id = str(item.get("id") or item.get("identifier") or "").strip()
                if not rec_id:
                    continue
                if rec_id in downloaded_ids:
                    continue

                attempted += 1

                rights = self._first_str(item.get("rights"))
                if not rights:
                    skipped_no_rights += 1
                    downloaded_ids.add(rec_id)
                    continue

                if copyright_free_only and not self._is_copyright_free_rights(
                    rights,
                    additional_allowed=additional_copyright_free_rights,
                ):
                    skipped_non_copyright_free += 1
                    downloaded_ids.add(rec_id)
                    continue

                # Prefer full-res / direct image URLs; skip thumbnails (edmPreview).
                download_url = self._pick_full_image_url(item)
                if not download_url:
                    skipped_no_full_image += 1
                    downloaded_ids.add(rec_id)
                    continue

                title = self._first_str(item.get("title")) or "europeana_object"
                safe_title = re.sub(r'[\\/*?:"<>|]', "", title)[:80]
                rec_hash = md5(rec_id.encode("utf-8", errors="ignore")).hexdigest()[:10]
                filename_stem = f"eur_{rec_hash}_{safe_title}".strip("_")

                # Resolve extension via URL or content-type fallback.
                ext = self._guess_ext(download_url)
                filename = output_dir / f"{filename_stem}{ext}"
                meta_filename = metadata_dir / f"{filename_stem}.json"

                ok = self._download_file(download_url, filename)
                if not ok:
                    failed_downloads += 1
                    downloaded_ids.add(rec_id)
                    continue

                # Normalize formats: keep pipeline-friendly raster types.
                # Stage 02b/03 expect jpg/png; convert other image formats to jpg.
                try:
                    if filename.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                        from PIL import Image

                        with Image.open(filename) as img:
                            img = img.convert("RGB")
                            jpg_path = filename.with_suffix(".jpg")
                            img.save(jpg_path, format="JPEG", quality=95)
                        try:
                            filename.unlink(missing_ok=True)
                        except Exception:
                            pass
                        filename = jpg_path
                except Exception:
                    # If conversion fails, keep original; downstream may skip it.
                    pass

                meta = {
                    "source": "europeana",
                    "source_id": rec_id,
                    "source_page_url": str(item.get("guid") or ""),
                    "download_url": download_url,
                    "license": rights,
                    "title": title,
                    "metadata": {},
                    "raw_metadata": item,
                }
                try:
                    meta_filename.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                except Exception as exc:
                    errors.append(f"record={rec_id} metadata write failed: {exc}")

                downloaded += 1
                downloaded_ids.add(rec_id)

                if sleep_s:
                    time.sleep(float(sleep_s))

                if downloaded % 50 == 0:
                    self._write_state(state_path, cursor=cursor, downloaded_ids=downloaded_ids)

            if not next_cursor or next_cursor == cursor:
                break
            cursor = str(next_cursor)
            self._write_state(state_path, cursor=cursor, downloaded_ids=downloaded_ids)

        self._write_state(state_path, cursor=cursor, downloaded_ids=downloaded_ids)

        return EuropeanaAcquisitionSummary(
            source="europeana",
            attempted=attempted,
            downloaded=downloaded,
            skipped_no_full_image=skipped_no_full_image,
            skipped_no_rights=skipped_no_rights,
            skipped_non_copyright_free=skipped_non_copyright_free,
            failed_downloads=failed_downloads,
            errors=errors,
        )

    def _write_state(self, state_path: Path, *, cursor: str, downloaded_ids: set[str]) -> None:
        try:
            state_path.write_text(
                json.dumps({"cursor": cursor, "downloaded_ids": sorted(downloaded_ids)}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            return

    def _search(
        self,
        *,
        query: str,
        cursor: str,
        rows: int,
        reusability: str,
        qf_filters: list[str] | None = None,
    ) -> dict[str, Any]:
        filters = ["TYPE:IMAGE"] + list(self.DEFAULT_QUALITY_QF)
        if qf_filters:
            filters.extend([str(f).strip() for f in qf_filters if str(f).strip()])
        params = {
            "wskey": self.api_key,
            "query": query,
            "rows": str(int(rows)),
            "cursor": cursor,
            "profile": "rich",
            "reusability": reusability,
            "media": "true",
            "qf": filters,
        }
        resp = self.session.get(self.API_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _is_copyright_free_rights(rights: str, additional_allowed: list[str] | None = None) -> bool:
        r = str(rights or "").strip().lower()
        if not r:
            return False

        base_allowed_fragments = (
            "creativecommons.org/publicdomain/zero",
            "creativecommons.org/publicdomain/mark",
            "rightsstatements.org/vocab/noc-",
        )
        if any(fragment in r for fragment in base_allowed_fragments):
            return True

        if additional_allowed:
            extra = [str(x).strip().lower() for x in additional_allowed if str(x).strip()]
            if any(fragment in r for fragment in extra):
                return True

        return False

    def _download_file(self, url: str, filename: Path) -> bool:
        try:
            with self.session.get(url, stream=True, timeout=60) as r:
                if r.status_code != 200:
                    return False
                content_type = (r.headers.get("Content-Type") or "").lower()
                if "image/" not in content_type and not self._looks_like_image_url(url):
                    return False
                filename.parent.mkdir(parents=True, exist_ok=True)
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception:
            return False

    @staticmethod
    def _first_str(value: Any) -> str:
        if isinstance(value, list) and value:
            return str(value[0])
        if isinstance(value, str):
            return value
        return ""

    def _pick_full_image_url(self, item: dict[str, Any]) -> str:
        """Pick the best candidate URL for a full image download.

        Do not require a file extension up-front because many providers expose
        direct image endpoints without an extension. We rank candidates and let
        `_download_file` verify by HTTP content-type.
        """
        candidates: list[tuple[int, str]] = []
        field_weights = {
            "edmIsShownBy": 100,  # usually direct media URL
            "isShownBy": 95,
            "edmIsShownAt": 40,   # often landing page; keep as fallback
        }

        for field, base_score in field_weights.items():
            for url in self._iter_urls(item.get(field)):
                if not self._is_http_url(url):
                    continue
                score = base_score + self._image_url_score(url)
                candidates.append((score, url))

        if not candidates:
            return ""
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    @staticmethod
    def _iter_urls(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(v) for v in value if isinstance(v, str)]
        return []

    @staticmethod
    def _is_http_url(url: str) -> bool:
        try:
            parsed = urlparse(url)
            return parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except Exception:
            return False

    @classmethod
    def _image_url_score(cls, url: str) -> int:
        u = str(url).lower()
        score = 0
        if cls._looks_like_image_url(u):
            score += 20
        if any(hint in u for hint in ("iiif", "/full/", "download", "image", "images")):
            score += 8
        if any(hint in u for hint in ("thumbnail", "thumb", "/small/", "/preview/")):
            score -= 6
        return score

    @staticmethod
    def _looks_like_image_url(url: str) -> bool:
        url_l = str(url).lower()
        return any(url_l.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"))

    @staticmethod
    def _guess_ext(url: str) -> str:
        url_l = str(url).lower()
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"):
            if url_l.endswith(ext):
                return ext
        return ".jpg"
