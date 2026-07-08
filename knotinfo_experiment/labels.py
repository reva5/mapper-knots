"""Ground-truth label construction."""

from __future__ import annotations

import warnings

import pandas as pd

from knotinfo_experiment.config import ExperimentConfig
from knotinfo_experiment.data import KnotTable


def build_labels(
    table: KnotTable,
    config: ExperimentConfig,
    row_mask: pd.Series | None = None,
) -> pd.Series:
    """
    Build ground-truth labels for rows.

    Returns a Series indexed like ``table.frame`` (optionally masked later).
    """
    frame = table.frame if row_mask is None else table.frame.loc[row_mask]

    if config.label_mode == "order":
        return frame["label_order"].astype(str)

    # Tuple mode: exact tuple of concordance invariants.
    parts: list[pd.Series] = []
    for key in config.label_tuple:
        if key not in table.frame.columns:
            raise ValueError(f"label_tuple key not in table: {key}")
        parts.append(frame[key].map(lambda v: "NA" if pd.isna(v) else str(v)))
    if not parts:
        raise ValueError("label_tuple is empty")
    label = parts[0]
    for part in parts[1:]:
        label = label + "|" + part
    return label


def scoring_mask(labels: pd.Series) -> pd.Series:
    """Exclude unknown-order knots from scoring."""
    return labels != "unknown"


def assert_no_label_leakage(
    subset: tuple[str, ...],
    config: ExperimentConfig,
) -> bool:
    """
    Return True if subset is safe for scoring under tuple label mode.

    Emits a warning and returns False when a label invariant appears in features.
    """
    if config.label_mode != "tuple":
        return True
    overlap = set(subset) & set(config.label_tuple)
    if overlap:
        warnings.warn(
            f"Skipping subset {subset}: features overlap label_tuple keys {overlap}",
            stacklevel=2,
        )
        return False
    return True
