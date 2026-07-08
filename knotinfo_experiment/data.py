"""Load KnotInfo CSV into a typed knot table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from knotinfo_experiment.columns import ColumnMapping, resolve_columns, print_column_mapping
from knotinfo_experiment.config import ExperimentConfig
from knotinfo_experiment.parse import (
    parse_concordance_order,
    parse_exact_numeric,
    parse_upsilon_components,
)


CHIRAL_KEYS = frozenset({"signature", "tau", "rasmussen_s"})


@dataclass(frozen=True)
class KnotTable:
    """Parsed knot data with internal feature columns."""

    frame: pd.DataFrame
    mapping: ColumnMapping
    feature_keys: tuple[str, ...]

    @property
    def names(self) -> pd.Series:
        return self.frame["name"]

    def feature_matrix(self, subset: tuple[str, ...], row_mask: np.ndarray) -> np.ndarray:
        cols = [self.frame[k].to_numpy(dtype=float) for k in subset]
        return np.column_stack(cols)[row_mask]


def _require_columns(mapping: ColumnMapping) -> None:
    missing = list(mapping.unmatched_required)
    if missing:
        raise ValueError(
            "CSV is missing required mapped columns: "
            + ", ".join(missing)
            + ". Fix the export or extend COLUMN_ALIASES in columns.py."
        )


def load_knot_table(csv_path: Path, config: ExperimentConfig) -> KnotTable:
    """Read CSV, resolve columns, and build internal feature columns."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"KnotInfo CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    headers = list(raw.columns)
    mapping = resolve_columns(headers)
    print_column_mapping(mapping, headers)
    _require_columns(mapping)

    scalar_features = [
        k
        for k in mapping.scalar
        if k not in {"name", "family", "concordance_order"}
    ]

    # Accumulate rows, expanding multi-component upsilon vectors into extra keys.
    pending_rows: list[dict[str, object]] = []

    order_col = mapping.scalar["concordance_order"]

    for _, record in raw.iterrows():
        row: dict[str, object] = {
            "name": record[mapping.scalar["name"]],
            "family": record[mapping.scalar["family"]] if "family" in mapping.scalar else "unknown",
            "concordance_order_raw": record[order_col],
            "label_order": parse_concordance_order(record[order_col]),
        }

        for key in scalar_features:
            row[key] = parse_exact_numeric(record[mapping.scalar[key]])

        for col, ukey in zip(mapping.upsilon_columns, mapping.upsilon_keys):
            components = parse_upsilon_components(record[col])
            if components is None:
                row[ukey] = np.nan
            else:
                row[ukey] = components[0]
                for idx, val in enumerate(components[1:], start=1):
                    row[f"{ukey}_{idx}"] = val

        pending_rows.append(row)

    frame = pd.DataFrame(pending_rows)

    # Extend upsilon keys discovered from multi-component cells.
    dynamic_upsilon = sorted(k for k in frame.columns if k.startswith("upsilon_"))
    feature_keys = tuple(sorted(set(scalar_features) | set(dynamic_upsilon)))

    # Apply chirality / absolute-value transform.
    if config.use_absolute_values:
        for key in feature_keys:
            base = key.split("_")[0] if key.startswith("upsilon_") else key
            if key in CHIRAL_KEYS or base in {"upsilon"}:
                frame[key] = frame[key].abs()

    return KnotTable(frame=frame, mapping=mapping, feature_keys=feature_keys)


def available_pool_keys(table: KnotTable, config: ExperimentConfig) -> tuple[list[str], list[str]]:
    """Return concordance and non-concordance keys present in the table."""
    present = set(table.feature_keys)
    concordance = [k for k in config.concordance_pool if k in present]
    concordance.extend(k for k in table.feature_keys if k.startswith("upsilon_"))
    concordance = sorted(set(concordance))

    control = [k for k in config.non_concordance_pool if k in present]
    return concordance, control


def completeness_mask(table: KnotTable, subset: tuple[str, ...]) -> np.ndarray:
    """True for rows where every feature in *subset* is present."""
    if not subset:
        raise ValueError("subset must be non-empty")
    sub = table.frame[list(subset)]
    return sub.notna().all(axis=1).to_numpy()


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each column; return standardized matrix, means, stds."""
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0, ddof=0)
    stds_safe = np.where(stds == 0, 1.0, stds)
    return (X - means) / stds_safe, means, stds


def label_distribution(labels: pd.Series) -> dict[str, int]:
    return labels.value_counts().to_dict()
