from __future__ import annotations

import re
from typing import Any


_PERIOD_PATTERNS = [
    r"\b(geometric|orientalizing|archaic|classical|hellenistic|roman)\b",
    r"\b(\d{1,2})(st|nd|rd|th)\s+century\b",
    r"\b(\d{3,4})\s*(bc|bce|ad|ce)\b",
]

_STYLE_PATTERNS = [
    r"\b(red-figure|black-figure|white-ground|attic)\b",
    r"\b(kylix|amphora|krater|hydria|lekythos|oinochoe|skyphos|kantharos|pelike)\b",
]


def _extract_tags(text: str) -> list[str]:
    t = text.lower()
    tags: list[str] = []

    for pat in _PERIOD_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            tags.append(m.group(0))

    for pat in _STYLE_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            tags.append(m.group(0))

    # De-dup while keeping order
    seen = set()
    out = []
    for x in tags:
        key = x.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(x.strip())
    return out[:6]


def _whitespace_trim(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def cap_prompt_to_token_budget(
    prompt: str,
    *,
    tokenizer: Any | None,
    max_tokens: int = 77,
) -> tuple[str, bool]:
    """
    Return (capped_prompt, was_truncated).

    Strategy:
    1. Extract high-value tags (period/style/vessel type) from anywhere in the prompt.
    2. Prepend those tags to the front so they survive truncation.
    3. Enforce the CLIP token budget using the provided tokenizer when available.

    Note: For SD 1.x inpainting, CLIP max length is effectively 77 tokens.
    """
    prompt = _whitespace_trim(prompt or "")
    if not prompt:
        return "", False

    tags = _extract_tags(prompt)
    if tags:
        tagged = _whitespace_trim(", ".join(tags) + ". " + prompt)
    else:
        tagged = prompt

    if tokenizer is None:
        # Fallback: cap by rough word budget (not exact tokens).
        words = tagged.split(" ")
        capped = " ".join(words[: max_tokens])
        return capped, len(words) > max_tokens

    # Token-accurate capping: tokenize with truncation, then decode back to text.
    encoded = tokenizer(
        tagged,
        padding=False,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][0]
    capped = tokenizer.decode(input_ids, skip_special_tokens=True)
    capped = _whitespace_trim(capped)

    # Determine truncation by comparing tokenized lengths.
    full_len = len(
        tokenizer(
            tagged,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )["input_ids"][0]
    )
    was_truncated = full_len > max_tokens
    return capped, was_truncated

