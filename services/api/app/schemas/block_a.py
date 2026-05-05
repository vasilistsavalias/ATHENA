from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class BlockAItemPayload(BaseModel):
    assignment_id: int
    item_order: int
    total_items: int
    sample_id: str
    image_url: str
    mask_type: str | None = None
    mask_coverage_bin: str | None = None
    source_label: str
    is_attention_check: bool


class BlockANextResponse(BaseModel):
    done: bool
    item: BlockAItemPayload | None = None


class BlockASubmitRequest(BaseModel):
    assignment_id: int
    authenticity_likelihood: int = Field(ge=1, le=5)
    archaeological_plausibility: int = Field(ge=1, le=5)
    confidence: int = Field(ge=1, le=5)
    comment: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    response_time_ms: int = Field(ge=0, le=86_400_000)
