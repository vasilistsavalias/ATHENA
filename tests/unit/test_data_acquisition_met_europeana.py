import json
import os
from pathlib import Path

import pytest

from thesis_pipeline.components.data_acquisition_europeana import EuropeanaOpenAcquirer
from thesis_pipeline.components.data_acquisition_met import MetOpenAccessAcquirer
from thesis_pipeline.utils.env_loader import load_env_file


def test_env_loader_supports_value_only(tmp_path, monkeypatch):
    monkeypatch.delenv("EUROPEANA_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("dummy_key_only_value\n", encoding="utf-8")
    loaded = load_env_file(env_path)
    assert "EUROPEANA_API_KEY" in loaded
    assert os.environ.get("EUROPEANA_API_KEY") == "dummy_key_only_value"


def test_met_acquirer_filters_and_downloads(tmp_path, monkeypatch):
    out_dir = tmp_path / "raw"
    meta_dir = tmp_path / "raw" / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    acq = MetOpenAccessAcquirer()

    # Three objects: only one should pass all filters.
    monkeypatch.setattr(
        acq,
        "_search_object_ids",
        lambda query, department_id, allow_department_list_fallback=False: [1, 2, 3],
    )

    def fake_get_object(object_id: int):
        if object_id == 1:
            return {
                "primaryImage": "http://img/1.jpg",
                "isPublicDomain": True,
                "culture": "Greek",
                "objectName": "Vase",
                "classification": "Vases",
                "medium": "Terracotta",
                "title": "Test Vase",
            }
        if object_id == 2:
            return {"primaryImage": "http://img/2.jpg", "isPublicDomain": False, "culture": "Greek", "objectName": "Vase"}
        return {"primaryImage": "http://img/3.jpg", "isPublicDomain": True, "culture": "Roman", "objectName": "Vase"}

    monkeypatch.setattr(acq, "_get_object", fake_get_object)

    def fake_download(url: str, filename: Path) -> bool:
        filename.write_bytes(b"img")
        return True

    monkeypatch.setattr(acq, "_download_file", fake_download)

    summary = acq.download(
        query="Greek vase",
        output_dir=out_dir,
        metadata_dir=meta_dir,
        limit=10,
        department_id=13,
        sleep_s=0,
        state_file=out_dir / "met_state.json",
    )

    assert summary.downloaded == 1
    assert summary.skipped_not_public_domain == 1
    assert summary.skipped_not_greek == 1

    metas = list(meta_dir.glob("met_*.json"))
    assert len(metas) == 1
    meta = json.loads(metas[0].read_text(encoding="utf-8"))
    assert meta["source"] == "met"
    assert meta["license"]


def test_europeana_acquirer_skips_thumbnails(tmp_path, monkeypatch):
    out_dir = tmp_path / "raw"
    meta_dir = tmp_path / "raw" / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    acq = EuropeanaOpenAcquirer(api_key="dummy")

    # Page 1: one item with rights but only preview (should skip), one item with shownBy image.
    page1 = {
        "items": [
            {
                "id": "/1",
                "rights": ["http://creativecommons.org/publicdomain/zero/1.0/"],
                "edmPreview": ["http://thumb.jpg"],
            },
            {
                "id": "/2",
                "rights": ["http://creativecommons.org/publicdomain/zero/1.0/"],
                "edmIsShownBy": ["http://full.jpg"],
                "title": ["Greek Vase"],
            },
        ],
        "nextCursor": "CUR2",
    }
    page2 = {"items": [], "nextCursor": "CUR2"}

    calls = {"n": 0}

    def fake_search(*, query, cursor, rows, reusability, qf_filters=None):
        calls["n"] += 1
        return page1 if calls["n"] == 1 else page2

    monkeypatch.setattr(acq, "_search", fake_search)

    def fake_download(url: str, filename: Path) -> bool:
        filename.write_bytes(b"img")
        return True

    monkeypatch.setattr(acq, "_download_file", fake_download)

    summary = acq.download(
        query="ancient greek vase",
        output_dir=out_dir,
        metadata_dir=meta_dir,
        limit=10,
        rows=100,
        reusability="open",
        sleep_s=0,
        state_file=out_dir / "europeana_state.json",
    )

    assert summary.downloaded == 1
    assert summary.skipped_no_full_image == 1
    assert summary.skipped_non_copyright_free == 0
    metas = list(meta_dir.glob("eur_*.json"))
    assert len(metas) == 1


def test_europeana_rights_filter_rejects_non_copyright_free(tmp_path, monkeypatch):
    out_dir = tmp_path / "raw"
    meta_dir = tmp_path / "raw" / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    acq = EuropeanaOpenAcquirer(api_key="dummy")

    page1 = {
        "items": [
            {
                "id": "/1",
                "rights": ["http://creativecommons.org/licenses/by/4.0/"],
                "edmIsShownBy": ["http://full1.jpg"],
                "title": ["Greek Vase 1"],
            },
            {
                "id": "/2",
                "rights": ["http://creativecommons.org/publicdomain/mark/1.0/"],
                "edmIsShownBy": ["http://full2.jpg"],
                "title": ["Greek Vase 2"],
            },
        ],
        "nextCursor": "CUR2",
    }
    page2 = {"items": [], "nextCursor": "CUR2"}

    calls = {"n": 0}

    def fake_search(*, query, cursor, rows, reusability, qf_filters=None):
        calls["n"] += 1
        return page1 if calls["n"] == 1 else page2

    monkeypatch.setattr(acq, "_search", fake_search)

    def fake_download(url: str, filename: Path) -> bool:
        filename.write_bytes(b"img")
        return True

    monkeypatch.setattr(acq, "_download_file", fake_download)

    summary = acq.download(
        query="ancient greek vase",
        output_dir=out_dir,
        metadata_dir=meta_dir,
        limit=10,
        rows=100,
        reusability="open",
        sleep_s=0,
        state_file=out_dir / "europeana_state.json",
    )

    assert summary.downloaded == 1
    assert summary.skipped_non_copyright_free == 1


def test_europeana_search_adds_default_qf_filters(monkeypatch):
    acq = EuropeanaOpenAcquirer(api_key="dummy")
    captured = {}

    class _Resp:
        def raise_for_status(self):
            return None

        @staticmethod
        def json():
            return {"items": [], "nextCursor": "*"}

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        return _Resp()

    monkeypatch.setattr(acq.session, "get", fake_get)

    acq._search(
        query="ancient greek vase",
        cursor="*",
        rows=10,
        reusability="open",
        qf_filters=["DATASET:my_collection"],
    )

    params = captured["params"]
    assert params["reusability"] == "open"
    assert "TYPE:IMAGE" in params["qf"]
    assert "IMAGE_COLOR:true" in params["qf"]
    assert "IMAGE_GRAYSCALE:false" in params["qf"]
    assert "DATASET:my_collection" in params["qf"]
