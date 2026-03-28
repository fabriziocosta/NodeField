"""Naming helpers shared across persistence and runtime objects."""

from __future__ import annotations

import re


def sanitize_model_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return token or "gg"
