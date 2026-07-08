"""Parse KnotInfo cell values conservatively."""

from __future__ import annotations

import math
import re
from typing import Literal

OrderLabel = Literal["slice/order-1", "order-2", "order-4", "infinite-order", "unknown"]

# Bounds, intervals, and non-exact tokens -> missing.
BOUND_PATTERN = re.compile(
    r"^\s*(?:[<>=≤≥]|<=|>=|\?|unknown|not\s+computed|n/a|na)\s*",
    re.IGNORECASE,
)
VECTOR_PATTERN = re.compile(r"^\s*\[(.+)\]\s*$")
NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def parse_exact_numeric(cell: str | float | int | None) -> float | None:
    """
    Return a float only for exact numeric values.

    Empty, non-numeric, bounds, and intervals return ``None``.
    """
    if cell is None:
        return None
    if isinstance(cell, float):
        if math.isnan(cell):
            return None
        return cell
    if isinstance(cell, int):
        return float(cell)

    text = str(cell).strip()
    if not text:
        return None
    if BOUND_PATTERN.match(text):
        return None
    if VECTOR_PATTERN.match(text):
        return None
    if text.lower() in {"?", "unknown", "na", "n/a", "not computed", "none"}:
        return None

    # KnotInfo infinity tokens in numeric columns.
    if text.lower() in {"infty", "inf", "infinity", "infinite"}:
        return None

    if NUMBER_PATTERN.match(text):
        return float(text)

    return None


def parse_upsilon_components(cell: str | float | int | None) -> list[float] | None:
    """
    Parse upsilon sample vectors like ``[1;-1]`` or a single exact number.

    Returns ``None`` if the cell is missing or cannot be parsed exactly.
    """
    if cell is None:
        return None
    if isinstance(cell, (int, float)):
        if isinstance(cell, float) and math.isnan(cell):
            return None
        return [float(cell)]

    text = str(cell).strip()
    if not text:
        return None
    if BOUND_PATTERN.match(text):
        return None

    m = VECTOR_PATTERN.match(text)
    if m:
        parts = [p.strip() for p in m.group(1).split(";") if p.strip()]
        values: list[float] = []
        for part in parts:
            num = parse_exact_numeric(part)
            if num is None:
                return None
            values.append(num)
        return values if values else None

    num = parse_exact_numeric(text)
    return [num] if num is not None else None


def parse_concordance_order(cell: str | float | int | None) -> OrderLabel:
    """Map concordance-order column values to canonical label buckets."""
    if cell is None:
        return "unknown"
    text = str(cell).strip().lower()
    if not text or text in {"?", "unknown", "na", "n/a"}:
        return "unknown"
    if text in {"1", "order-1", "order 1", "slice"}:
        return "slice/order-1"
    if text in {"2", "order-2", "order 2"}:
        return "order-2"
    if text in {"4", "order-4", "order 4"}:
        return "order-4"
    if text in {"infty", "inf", "infinity", "infinite", "infinite-order"}:
        return "infinite-order"
    return "unknown"
