"""Extract mathematically interesting artifacts from top-ranked runs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ball_mapper import BallMapperResult
from knotinfo_experiment.data import KnotTable


@dataclass(frozen=True)
class InequalitySpec:
    name: str
    left_key: str
    right_key: str
    left_transform: str  # "abs", "abs_half", "identity"
    right_transform: str
    chain: tuple[str, ...] = ()  # optional middle keys for chains


STANDARD_INEQUALITIES: tuple[InequalitySpec, ...] = (
    InequalitySpec("tau_le_smooth_4genus", "tau", "smooth_4genus", "abs", "identity"),
    InequalitySpec(
        "rasmussen_half_le_smooth_4genus",
        "rasmussen_s",
        "smooth_4genus",
        "abs_half",
        "identity",
    ),
    InequalitySpec(
        "signature_half_le_topological_4genus",
        "signature",
        "topological_4genus",
        "abs_half",
        "identity",
    ),
    InequalitySpec(
        "topological_le_smooth_4genus",
        "topological_4genus",
        "smooth_4genus",
        "identity",
        "identity",
    ),
)


def _transform(value: float, kind: str) -> float:
    if kind == "abs":
        return abs(value)
    if kind == "abs_half":
        return abs(value) / 2.0
    return value


def find_collisions(
    table: KnotTable,
    row_indices: np.ndarray,
    labels: np.ndarray,
    result: BallMapperResult,
    X_std: np.ndarray,
    subset: tuple[str, ...],
    tolerance: float,
) -> pd.DataFrame:
    """
    Find knots sharing a ball or near-identical standardized vectors with
    different ground-truth labels.
    """
    names = table.frame.iloc[row_indices]["name"].to_numpy()
    records: list[dict] = []

    # Ball collisions.
    for node, info in result.node_info.items():
        pop = info["indices"]
        if len(pop) < 2:
            continue
        node_labels = labels[pop]
        if len(set(node_labels)) <= 1:
            continue
        for i in pop:
            rec = {
                "collision_type": "shared_ball",
                "ball_node": node,
                "name": names[i],
                "label": labels[i],
            }
            for feat in subset:
                rec[feat] = table.frame.iloc[row_indices[i]][feat]
            records.append(rec)

    # Near-duplicate standardized vectors.
    n = len(row_indices)
    for i, j in combinations(range(n), 2):
        if labels[i] == labels[j]:
            continue
        dist = np.linalg.norm(X_std[i] - X_std[j])
        if dist <= tolerance:
            for idx in (i, j):
                rec = {
                    "collision_type": "near_duplicate_vector",
                    "ball_node": "",
                    "vector_distance": dist,
                    "name": names[idx],
                    "label": labels[idx],
                }
                for feat in subset:
                    rec[feat] = table.frame.iloc[row_indices[idx]][feat]
                records.append(rec)

    return pd.DataFrame(records)


def find_extremal_knots(
    table: KnotTable,
    row_indices: np.ndarray,
    subset: tuple[str, ...],
    tolerance: float,
) -> pd.DataFrame:
    """Flag knots saturating standard inequalities among selected invariants."""
    subset_set = set(subset)
    names = table.frame.iloc[row_indices]["name"].to_numpy()
    records: list[dict] = []

    for spec in STANDARD_INEQUALITIES:
        keys_needed = {spec.left_key, spec.right_key}
        if not keys_needed <= subset_set:
            continue

        for local_i, global_i in enumerate(row_indices):
            row = table.frame.iloc[global_i]
            left_val = row[spec.left_key]
            right_val = row[spec.right_key]
            if pd.isna(left_val) or pd.isna(right_val):
                continue
            left = _transform(float(left_val), spec.left_transform)
            right = _transform(float(right_val), spec.right_transform)
            gap = right - left
            if gap < -tolerance:
                status = "violation"
            elif abs(gap) <= tolerance:
                status = "saturated"
            else:
                continue
            records.append(
                {
                    "inequality": spec.name,
                    "status": status,
                    "name": names[local_i],
                    "left": left,
                    "right": right,
                    "gap": gap,
                    **{k: row[k] for k in subset},
                }
            )

    return pd.DataFrame(records)


def write_artifacts(
    collisions: pd.DataFrame,
    extremal: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    collisions.to_csv(output_dir / "collisions.csv", index=False)
    extremal.to_csv(output_dir / "extremal.csv", index=False)
