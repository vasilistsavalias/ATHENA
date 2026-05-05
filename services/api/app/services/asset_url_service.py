from __future__ import annotations


def normalize_block_a_image_url(*, campaign_id: int, sample_id: str, image_url: str, metadata_json: dict | None) -> str:
    if image_url.startswith("/"):
        return image_url

    origin_variant = str((metadata_json or {}).get("origin_variant") or "").strip()
    origin_sample_id = str((metadata_json or {}).get("origin_sample_id") or "").strip()

    if image_url in {"A.png", "B.png"} and origin_variant in {"A", "B"} and origin_sample_id:
        return f"/static/{campaign_id}/images/{origin_sample_id}/{origin_variant}.png"

    if image_url == "input.png":
        base_sample_id = origin_sample_id or sample_id
        return f"/static/{campaign_id}/images/{base_sample_id}/input.png"

    if "/" in image_url:
        return f"/static/{campaign_id}/{image_url.lstrip('/')}"

    return image_url


def normalize_block_b_image_url(*, campaign_id: int, sample_id: str, image_url: str) -> str:
    if image_url.startswith("/"):
        return image_url

    if image_url in {"input.png", "A.png", "B.png", "C.png", "D.png"}:
        return f"/static/{campaign_id}/images/{sample_id}/{image_url}"

    if "/" in image_url:
        return f"/static/{campaign_id}/{image_url.lstrip('/')}"

    return image_url
