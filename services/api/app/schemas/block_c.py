from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class BlockCItemPayload(BaseModel):
    assignment_id: int
    item_order: int
    total_items: int
    sample_id: str
    input_url: str
    option_a_url: str
    option_b_url: str
    option_c_url: str
    option_d_url: str
    show_a_left: bool
    mask_type: str | None = None
    mask_coverage_bin: str | None = None
    is_anchor: bool = False
    is_attention_check: bool


class BlockCNextResponse(BaseModel):
    done: bool
    item: BlockCItemPayload | None = None


class BlockCSubmitRequest(BaseModel):
    assignment_id: int
    choice: str = Field(pattern=r"^(A|B|C|D)$")
    confidence: int = Field(ge=1, le=5)
    comment: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    response_time_ms: int = Field(ge=0, le=86_400_000)
