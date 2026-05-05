import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class MetAcquisitionSummary:
    source: str
    attempted: int
    downloaded: int
    skipped_no_images: int
    skipped_not_public_domain: int
    skipped_not_greek: int
    skipped_not_vase: int
    failed_downloads: int
    errors: list[str]


class MetOpenAccessAcquirer:
    """Downloader for The Met Collection API (Open Access / public domain).

    API docs: https://metmuseum.github.io/
    """

    BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"

    def __init__(self, session: requests.Session | None = None):
        self.logger = logging.getLogger(__name__)
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "MasterThesisResearchBot/1.0 (research; contact: vtsav) requests",
            }
        )

    def download(
        self,
        *,
        query: str,
        output_dir: Path,
        metadata_dir: Path,
        limit: int = 10_000,
        department_id: int = 13,
        sleep_s: float = 0.05,
        state_file: Path | None = None,
        max_checks_factor: int = 200,
        max_consecutive_403: int = 50,
        allow_department_list_fallback: bool = False,
    ) -> MetAcquisitionSummary:
        output_dir = Path(output_dir)
        metadata_dir = Path(metadata_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        state_path = Path(state_file) if state_file else (output_dir / "met_state.json")
        state: dict[str, Any] = {"processed_object_ids": []}
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                state = {"processed_object_ids": []}

        processed_ids: set[int] = set(int(x) for x in state.get("processed_object_ids", []) if str(x).isdigit())

        object_ids = self._search_object_ids(
            query=query,
            department_id=department_id,
            allow_department_list_fallback=allow_department_list_fallback,
        )
        if not object_ids:
            return MetAcquisitionSummary(
                source="met",
                attempted=0,
                downloaded=0,
                skipped_no_images=0,
                skipped_not_public_domain=0,
                skipped_not_greek=0,
                skipped_not_vase=0,
                failed_downloads=0,
                errors=[f"No object IDs returned for query='{query}' department_id={department_id}"],
            )

        attempted = 0
        downloaded = 0
        skipped_no_images = 0
        skipped_not_public_domain = 0
        skipped_not_greek = 0
        skipped_not_vase = 0
        failed_downloads = 0
        errors: list[str] = []

        # Guardrail: avoid pathological long scans when search returns huge pools
        # but strict vase filters accept very few items.
        max_checks = max(int(limit) * int(max_checks_factor), int(limit))
        checked = 0
        consecutive_403 = 0

        for object_id in object_ids:
            if downloaded >= int(limit):
                break
            if checked >= max_checks:
                errors.append(
                    f"Reached max candidate checks ({max_checks}) before hitting limit={int(limit)}"
                )
                break
            if object_id in processed_ids:
                continue

            checked += 1
            attempted += 1
            try:
                obj = self._get_object(object_id)
                consecutive_403 = 0
            except Exception as exc:
                errors.append(f"object_id={object_id} get_object failed: {exc}")
                if "403" in str(exc):
                    consecutive_403 += 1
                    if consecutive_403 >= int(max_consecutive_403):
                        errors.append(
                            f"Stopping early: {consecutive_403} consecutive 403 responses from Met objects endpoint."
                        )
                        break
                else:
                    consecutive_403 = 0
                processed_ids.add(object_id)
                continue

            # Hard filters
            if not obj.get("primaryImage"):
                skipped_no_images += 1
                processed_ids.add(object_id)
                continue
            if not bool(obj.get("isPublicDomain", False)):
                skipped_not_public_domain += 1
                processed_ids.add(object_id)
                continue

            culture = str(obj.get("culture") or "").lower()
            if "greek" not in culture:
                skipped_not_greek += 1
                processed_ids.add(object_id)
                continue

            if not self._looks_like_vase(obj):
                skipped_not_vase += 1
                processed_ids.add(object_id)
                continue

            # Download
            image_url = str(obj.get("primaryImage"))
            title = str(obj.get("title") or "met_object").strip()
            safe_title = re.sub(r'[\\/*?:"<>|]', "", title)[:80]
            filename_stem = f"met_{object_id}_{safe_title}".strip("_")
            filename = output_dir / f"{filename_stem}.jpg"
            meta_filename = metadata_dir / f"{filename_stem}.json"

            ok = self._download_file(image_url, filename)
            if not ok:
                failed_downloads += 1
                processed_ids.add(object_id)
                continue

            meta = {
                "source": "met",
                "source_id": object_id,
                "source_page_url": f"https://www.metmuseum.org/art/collection/search/{object_id}",
                "download_url": image_url,
                "license": "CC0 (Met Open Access; public domain)",
                "title": title,
                "metadata": {},
                "raw_metadata": obj,
            }
            try:
                meta_filename.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except Exception as exc:
                errors.append(f"object_id={object_id} metadata write failed: {exc}")

            downloaded += 1
            processed_ids.add(object_id)

            if sleep_s:
                time.sleep(float(sleep_s))

            # checkpoint state every 50 downloads
            if downloaded % 50 == 0:
                self._write_state(state_path, processed_ids)

        self._write_state(state_path, processed_ids)

        return MetAcquisitionSummary(
            source="met",
            attempted=attempted,
            downloaded=downloaded,
            skipped_no_images=skipped_no_images,
            skipped_not_public_domain=skipped_not_public_domain,
            skipped_not_greek=skipped_not_greek,
            skipped_not_vase=skipped_not_vase,
            failed_downloads=failed_downloads,
            errors=errors,
        )

    def _write_state(self, state_path: Path, processed_ids: set[int]) -> None:
        try:
            state_path.write_text(
                json.dumps({"processed_object_ids": sorted(processed_ids)}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            return

    def _search_object_ids(
        self,
        *,
        query: str,
        department_id: int,
        allow_department_list_fallback: bool = False,
    ) -> list[int]:
        def _do(params: dict) -> dict[str, Any]:
            url = f"{self.BASE_URL}/search"
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()

        def _list_by_department(dept_id: int) -> dict[str, Any]:
            # The Met API supports listing all object IDs, with optional department filtering.
            # Docs: https://metmuseum.github.io/ (Objects endpoint)
            url = f"{self.BASE_URL}/objects"
            resp = self.session.get(url, params={"departmentIds": str(int(dept_id))}, timeout=60)
            resp.raise_for_status()
            return resp.json()

        def _params_for(term: str) -> dict[str, str]:
            params = {"q": term, "hasImages": "true"}
            # If department_id <= 0, do not apply department filter.
            if int(department_id) > 0:
                params["departmentId"] = str(int(department_id))
            return params

        terms = [str(query or "").strip()]
        # Union multiple vase-centric terms to avoid tiny result sets from a single phrase query.
        terms.extend(["Greek", "vase", "amphora", "krater", "kylix", "hydria", "lekythos", "oinochoe", "skyphos"])
        seen_terms = set()
        object_ids: list[int] = []
        for term in terms:
            if not term:
                continue
            key = term.lower()
            if key in seen_terms:
                continue
            seen_terms.add(key)
            try:
                data = _do(_params_for(term))
                ids = data.get("objectIDs") or []
                if ids:
                    object_ids.extend(ids)
            except Exception:
                continue

        # Robustness: some queries are too "phrase-y" (e.g., "Greek vase") and return tiny lists.
        # If the query contains "greek", try an additional artistOrCulture search on "Greek"
        # and union results. We still apply vase-specific filtering later.
        try:
            if len(object_ids) < 50 and "greek" in str(query).lower():
                alt_params = {
                    "q": "Greek",
                    "artistOrCulture": "true",
                    "hasImages": "true",
                }
                if int(department_id) > 0:
                    alt_params["departmentId"] = str(int(department_id))
                alt = _do(alt_params)
                alt_ids = alt.get("objectIDs") or []
                object_ids = list({*object_ids, *alt_ids})
        except Exception:
            pass

        # Final fallback: if search still returns a tiny set, list the department and filter locally.
        # This is slower but far more reliable for building datasets.
        try:
            if allow_department_list_fallback and len(object_ids) < 50 and int(department_id) > 0:
                dept = _list_by_department(int(department_id))
                dept_ids = dept.get("objectIDs") or []
                if dept_ids:
                    object_ids = dept_ids
        except Exception:
            pass

        # Deduplicate while preserving order
        deduped: list[int] = []
        seen_ids: set[int] = set()
        for x in object_ids:
            if not str(x).isdigit():
                continue
            obj_id = int(x)
            if obj_id in seen_ids:
                continue
            seen_ids.add(obj_id)
            deduped.append(obj_id)
        return deduped

    def _get_object(self, object_id: int) -> dict[str, Any]:
        url = f"{self.BASE_URL}/objects/{int(object_id)}"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _download_file(self, url: str, filename: Path) -> bool:
        try:
            with self.session.get(url, stream=True, timeout=60) as r:
                if r.status_code != 200:
                    return False
                filename.parent.mkdir(parents=True, exist_ok=True)
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception:
            return False

    def _looks_like_vase(self, obj: dict[str, Any]) -> bool:
        object_name = str(obj.get("objectName") or "").lower()
        classification = str(obj.get("classification") or "").lower()
        medium = str(obj.get("medium") or "").lower()

        haystack = " ".join([object_name, classification, medium])
        keywords = [
            "vase",
            "pottery",
            "amphora",
            "krater",
            "kylix",
            "hydria",
            "lekythos",
            "oinochoe",
            "skyphos",
            "pelike",
            "aryballos",
        ]
        return any(k in haystack for k in keywords)
