"""Experiment configuration and CLI argument parsing."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


LabelMode = Literal["order", "tuple"]
SubsetKind = Literal["concordance", "control"]


@dataclass(frozen=True)
class ExperimentConfig:
    """All knobs for the KnotInfo Ball Mapper experiment."""

    csv_path: Path = Path("data/knotinfo.csv")
    output_dir: Path = Path("output")

    # Ground-truth labeling
    label_mode: LabelMode = "order"
    label_tuple: tuple[str, ...] = ()  # used when label_mode == "tuple"

    # Chirality
    use_absolute_values: bool = True
    chiral_keys: tuple[str, ...] = ("signature", "tau", "rasmussen_s")

    # Invariant pools (internal keys; resolved to CSV columns at runtime)
    concordance_pool: tuple[str, ...] = (
        "signature",
        "smooth_4genus",
        "topological_4genus",
        "tau",
        "rasmussen_s",
        "concordance_genus",
        "arf",
    )
    non_concordance_pool: tuple[str, ...] = (
        "crossing_number",
        "three_genus",
        "bridge_number",
        "braid_index",
        "braid_length",
        "determinant",
        "unknotting_number",
    )

    # Subset search
    k_max: int = 4
    max_subsets: int = 500
    explicit_subsets: tuple[tuple[str, ...], ...] | None = None

    # Ball Mapper
    epsilon_grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    bm_method: Literal["greedy", "maxmin"] = "greedy"
    min_points_per_node: int = 1

    # Artifacts / plotting
    top_n_plots: int = 3
    collision_vector_tolerance: float = 1e-6
    extremal_tolerance: float = 1e-9
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.label_mode == "tuple" and not self.label_tuple:
            raise ValueError("label_tuple must be non-empty when label_mode='tuple'")


def _parse_subset_list(raw: str) -> tuple[tuple[str, ...], ...]:
    """Parse JSON like '[["signature","tau"],["crossing_number"]'."""
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("explicit_subsets must be a JSON list of feature lists")
    return tuple(tuple(s) for s in parsed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run KnotInfo Ball Mapper invariant-subset experiment.",
    )
    parser.add_argument("--csv", type=Path, default=ExperimentConfig.csv_path)
    parser.add_argument("--output-dir", type=Path, default=ExperimentConfig.output_dir)
    parser.add_argument(
        "--label-mode",
        choices=("order", "tuple"),
        default="order",
    )
    parser.add_argument(
        "--label-tuple",
        type=str,
        default="",
        help="Comma-separated internal keys for tuple labeling.",
    )
    parser.add_argument(
        "--no-absolute-values",
        action="store_true",
        help="Disable |.| for chiral invariants.",
    )
    parser.add_argument("--k-max", type=int, default=4)
    parser.add_argument("--max-subsets", type=int, default=500)
    parser.add_argument(
        "--explicit-subsets",
        type=str,
        default="",
        help='JSON list of feature subsets, e.g. \'[["signature","tau"]]\'',
    )
    parser.add_argument(
        "--epsilon-grid",
        type=str,
        default="0.5,1.0,1.5,2.0,2.5,3.0",
    )
    parser.add_argument("--top-n-plots", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def config_from_args(args: argparse.Namespace | None = None) -> ExperimentConfig:
    """Build :class:`ExperimentConfig` from CLI args or defaults."""
    if args is None:
        return ExperimentConfig()

    label_tuple: tuple[str, ...] = ()
    if args.label_tuple.strip():
        label_tuple = tuple(k.strip() for k in args.label_tuple.split(",") if k.strip())

    explicit: tuple[tuple[str, ...], ...] | None = None
    if args.explicit_subsets.strip():
        explicit = _parse_subset_list(args.explicit_subsets)

    epsilon_grid = tuple(float(x) for x in args.epsilon_grid.split(",") if x.strip())

    return ExperimentConfig(
        csv_path=args.csv,
        output_dir=args.output_dir,
        label_mode=args.label_mode,
        label_tuple=label_tuple,
        use_absolute_values=not args.no_absolute_values,
        k_max=args.k_max,
        max_subsets=args.max_subsets,
        explicit_subsets=explicit,
        epsilon_grid=epsilon_grid,
        top_n_plots=args.top_n_plots,
        random_seed=args.seed,
    )
