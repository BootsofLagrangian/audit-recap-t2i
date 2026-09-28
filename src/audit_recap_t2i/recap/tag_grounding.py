"""Helpers for grounding recap prompts with booru metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable

_CHARACTER_SUFFIX_RE = re.compile(r"_\([^)]+\)$")


def normalize_character_tag(tag: str) -> str:
    """Strip parenthetical qualifiers and convert to display name.

    ``gawr_gura_(1st_costume)`` → ``gawr gura``
    ``artoria_pendragon_(fate)`` → ``artoria pendragon``

    This gives the base identity for confirmed_characters.
    Costume/variant info like ``(1st costume)`` is separately available
    via the raw ``character_tags`` grounding field.
    """
    cleaned = _CHARACTER_SUFFIX_RE.sub("", tag.strip())
    return cleaned.replace("_", " ").strip()


def derive_confirmed_characters(
    general_tags: Iterable[str] | None,
    character_tags: Iterable[str] | None,
    *,
    require_solo: bool = True,
) -> list[str]:
    """Return base character names for solo images.

    Strips qualifiers and deduplicates so costume variants collapse:
    ``[gawr_gura, gawr_gura_(1st_costume)]`` → ``['gawr gura']``

    The VLM receives variant detail via the separate ``character_tags``
    grounding field which passes raw tags with all qualifiers intact.
    """

    general_tag_set = {
        tag.strip().lower()
        for tag in general_tags or []
        if isinstance(tag, str) and tag.strip()
    }
    if require_solo and "solo" not in general_tag_set:
        return []

    names: list[str] = []
    seen: set[str] = set()
    for raw_tag in character_tags or []:
        if not isinstance(raw_tag, str):
            continue
        name = normalize_character_tag(raw_tag)
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)

    return names
