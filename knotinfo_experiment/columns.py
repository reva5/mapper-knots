"""Resolve KnotInfo CSV headers to internal invariant keys."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def normalize_header(header: str) -> str:
    """Lowercase, strip HTML, collapse punctuation to underscores."""
    text = re.sub(r"<[^>]+>", "", header).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


# Regex patterns tried in order for each internal scalar key.
SCALAR_KEY_PATTERNS: dict[str, tuple[str, ...]] = {
    "signature": (r"^signature$",),
    "smooth_4genus": (
        r"^genus_4d$",
        r"^smooth_4_genus$",
        r"^four_genus$",
        r"genus.*4d(?!.*top)",
    ),
    "topological_4genus": (
        r"genus.*4d.*top",
        r"^topological_4_genus$",
    ),
    "tau": (
        r"^tau$",
        r"ozsvath.*tau",
        r"^ozsvath_szabo_tau$",
    ),
    "rasmussen_s": (
        r"rasmussen",
    ),
    "concordance_genus": (
        r"^concordance_genus$",
        r"concordance_genus(?!.*top)",
    ),
    "arf": (
        r"^arf",
        r"arf_invariant",
    ),
    "crossing_number": (r"crossing",),
    "three_genus": (
        r"genus_3d",
        r"three_genus",
        r"seifert",
        r"^genus$",
    ),
    "bridge_number": (r"bridge",),
    "braid_index": (r"braid_index", r"^braid_index$"),
    "braid_length": (r"braid_length",),
    "determinant": (r"determinant",),
    "unknotting_number": (r"unknotting",),
    "concordance_order": (
        r"^concordance_order$",
        r"concordance_order(?!.*alg)(?!.*top)",
    ),
    "name": (r"^name$", r"^knot$", r"^knot_name$"),
    "family": (r"family", r"fibered", r"knot_family", r"type"),
}

# Columns whose values may be upsilon samples (scalar or vector-encoded).
UPSILON_COLUMN_PATTERNS: tuple[str, ...] = (
    r"upsilon",
    r"^nu$",
)


@dataclass
class ColumnMapping:
    """Maps internal keys to CSV column names."""

    scalar: dict[str, str] = field(default_factory=dict)
    upsilon_columns: list[str] = field(default_factory=list)
    upsilon_keys: list[str] = field(default_factory=list)
    unmatched_required: list[str] = field(default_factory=list)

    def column_for(self, key: str) -> str | None:
        if key in self.scalar:
            return self.scalar[key]
        return None

    def all_feature_keys(self) -> list[str]:
        keys = list(self.scalar.keys())
        keys = [k for k in keys if k not in ("name", "family", "concordance_order")]
        keys.extend(self.upsilon_keys)
        return sorted(set(keys))


def _match_pattern(norm: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, norm) for p in patterns)


def resolve_columns(headers: list[str]) -> ColumnMapping:
    """
    Match CSV headers to internal keys without hardcoding exact KnotInfo names.

    Raises
    ------
    ValueError
        If a required identity / label column cannot be resolved.
    """
    normalized = {h: normalize_header(h) for h in headers}
    used: set[str] = set()
    mapping = ColumnMapping()

    # Scalar keys (first match wins).
    for key, patterns in SCALAR_KEY_PATTERNS.items():
        for header, norm in normalized.items():
            if header in used:
                continue
            if _match_pattern(norm, patterns):
                mapping.scalar[key] = header
                used.add(header)
                break

    # Upsilon / Nu sample columns.
    for header, norm in normalized.items():
        if header in used:
            continue
        if any(re.search(p, norm) for p in UPSILON_COLUMN_PATTERNS):
            mapping.upsilon_columns.append(header)
            used.add(header)

    # Expand upsilon columns into internal keys (may become multi-dimensional later).
    for col in mapping.upsilon_columns:
        norm = normalize_header(col)
        if norm == "nu":
            mapping.upsilon_keys.append("upsilon_nu")
        else:
            suffix = norm.replace("upsilon", "").strip("_") or "sample"
            mapping.upsilon_keys.append(f"upsilon_{suffix}")

    # Required columns for the experiment.
    for required in ("name", "concordance_order"):
        if required not in mapping.scalar:
            mapping.unmatched_required.append(required)

    return mapping


def print_column_mapping(mapping: ColumnMapping, headers: list[str]) -> None:
    """Print resolved mapping and unmatched pool keys."""
    print("=== Column mapping (internal key -> CSV column) ===")
    for key in sorted(mapping.scalar):
        print(f"  {key:24s} -> {mapping.scalar[key]}")
    for col, ukey in zip(mapping.upsilon_columns, mapping.upsilon_keys):
        print(f"  {ukey:24s} -> {col} (upsilon sample column)")

    if mapping.unmatched_required:
        print("\n*** UNMATCHED REQUIRED COLUMNS ***")
        for key in mapping.unmatched_required:
            print(f"  {key}")

    all_internal = set(SCALAR_KEY_PATTERNS) - {"name", "family", "concordance_order"}
    all_internal.update(mapping.upsilon_keys)
    matched = set(mapping.scalar) | set(mapping.upsilon_keys)
    unmatched_pool = sorted(all_internal - matched - {"name", "family", "concordance_order"})
    if unmatched_pool:
        print("\n=== Pool keys with no CSV column (skipped) ===")
        for key in unmatched_pool:
            print(f"  {key}")

    print("\n=== CSV columns not mapped to any internal key ===")
    mapped_headers = set(mapping.scalar.values()) | set(mapping.upsilon_columns)
    for h in headers:
        if h not in mapped_headers:
            print(f"  {h}")
