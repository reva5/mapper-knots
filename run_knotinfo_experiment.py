#!/usr/bin/env python3
"""CLI entry point for the KnotInfo Ball Mapper experiment."""

from __future__ import annotations

from knotinfo_experiment.config import build_arg_parser, config_from_args
from knotinfo_experiment.runner import run_experiment


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    run_experiment(config)


if __name__ == "__main__":
    main()
