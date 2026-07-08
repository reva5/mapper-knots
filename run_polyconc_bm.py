#!/usr/bin/env python3
"""Build Ball Mapper graphs on Alexander, Jones, and HOMFLYPT coefficient spaces."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from knotinfo_experiment.polyconc import PolyConcConfig, run_polyconc_bm


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run three separate Ball Mapper graphs on knot polynomial coefficients "
            "(Alexander, Jones, HOMFLYPT), coloring each ball by the mode "
            "concordance order of its knots."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/knotpolyconc.csv"),
        help="KnotInfo export with Alexander, Jones, HOMFLY, and concordance order",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for figures and parse-failure report",
    )
    parser.add_argument(
        "--eps-alexander",
        type=float,
        default=1.5,
        help="Ball radius ε for the Alexander graph (standardized units by default)",
    )
    parser.add_argument(
        "--eps-jones",
        type=float,
        default=2.0,
        help="Ball radius ε for the Jones graph",
    )
    parser.add_argument(
        "--eps-homfly",
        type=float,
        default=2.5,
        help="Ball radius ε for the HOMFLYPT graph",
    )
    parser.add_argument(
        "--standardize",
        dest="standardize",
        action="store_true",
        default=True,
        help="Z-score each coefficient column before Ball Mapper (default)",
    )
    parser.add_argument(
        "--no-standardize",
        dest="standardize",
        action="store_false",
        help="Use raw coefficient values (changes geometry and ε scale)",
    )
    parser.add_argument(
        "--max-knots",
        type=int,
        default=None,
        help="Optional cap on rows read from the CSV (head only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for nerve graph layout",
    )
    parser.add_argument(
        "--method",
        choices=("greedy", "maxmin"),
        default="greedy",
        help="ε-net center selection method",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = PolyConcConfig(
        csv_path=args.csv,
        output_dir=args.output_dir,
        eps_alexander=args.eps_alexander,
        eps_jones=args.eps_jones,
        eps_homfly=args.eps_homfly,
        standardize=args.standardize,
        seed=args.seed,
        max_knots=args.max_knots,
        method=args.method,
    )
    run_polyconc_bm(config)


if __name__ == "__main__":
    main()
