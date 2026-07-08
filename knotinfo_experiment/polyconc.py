"""Ball Mapper on knot polynomial coefficient spaces colored by concordance order."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ball_mapper import BallMapperResult
from knotinfo_experiment.columns import normalize_header
from knotinfo_experiment.data import standardize as zscore_features
from knotinfo_experiment.parse import parse_concordance_order
from knotinfo_experiment.polynomials import (
    ExponentGrid1D,
    ExponentGrid2D,
    PolyKind,
    build_grid_1d,
    build_grid_2d,
    parse_alexander,
    parse_jones,
    parse_homfly,
    vectorize_1d,
    vectorize_2d,
)
from knotinfo_experiment.scoring import run_ball_mapper
from knotinfo_experiment.visualize import plot_polyconc_combined, plot_polyconc_graph

POLY_COLUMNS: dict[PolyKind, str] = {
    "alexander": "Alexander",
    "jones": "Jones",
    "homfly": "HOMFLY",
}

POLY_PARSERS = {
    "alexander": parse_alexander,
    "jones": parse_jones,
    "homfly": parse_homfly,
}

POLY_TITLES = {
    "alexander": "Alexander polynomial",
    "jones": "Jones polynomial",
    "homfly": "HOMFLYPT (HOMFLY column)",
}


@dataclass(frozen=True)
class PolyConcConfig:
    """Configuration for the three polynomial Ball Mapper graphs."""

    csv_path: Path = Path("data/knotpolyconc.csv")
    output_dir: Path = Path("output")
    eps_alexander: float = 1.5
    eps_jones: float = 2.0
    eps_homfly: float = 2.5
    standardize: bool = True
    seed: int = 0
    max_knots: int | None = None
    method: str = "greedy"


@dataclass(frozen=True)
class ParseFailure:
    name: str
    graph: PolyKind
    reason: str
    raw_value: str


@dataclass(frozen=True)
class PolyGraphRun:
    kind: PolyKind
    bm_result: BallMapperResult
    labels: np.ndarray
    names: np.ndarray
    feature_dim: int
    eps: float
    n_input: int
    n_used: int
    failures: tuple[ParseFailure, ...]
    grid: ExponentGrid1D | ExponentGrid2D


def resolve_concordance_column(headers: list[str]) -> str:
    """Find a concordance-order column (accepts ``Concordance Order (Alg.)``)."""
    matches = [
        header
        for header in headers
        if re.search(r"concordance_order", normalize_header(header))
    ]
    if not matches:
        raise ValueError(
            "CSV has no concordance order column. Expected a header matching "
            "'concordance order' (e.g. 'Concordance Order' or "
            "'Concordance Order (Alg.)')."
        )
    for header in matches:
        if normalize_header(header) == "concordance_order":
            return header
    return sorted(matches)[0]


def _require_poly_columns(headers: list[str]) -> None:
    missing = [col for col in POLY_COLUMNS.values() if col not in headers]
    if missing:
        raise ValueError(
            "CSV is missing polynomial columns: "
            + ", ".join(missing)
            + ". Expected Name, Alexander, Jones, HOMFLY, and a concordance-order column."
        )


def load_polyconc_frame(csv_path: Path, *, max_knots: int | None = None) -> pd.DataFrame:
    """Load ``knotpolyconc.csv`` and attach parsed concordance labels."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Polynomial CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    headers = list(raw.columns)
    _require_poly_columns(headers)
    order_col = resolve_concordance_column(headers)
    print(f"Concordance order column: {order_col!r}")

    if max_knots is not None and max_knots < len(raw):
        raw = raw.iloc[:max_knots].copy()

    frame = pd.DataFrame(
        {
            "name": raw["Name"],
            "concordance_order_raw": raw[order_col],
            "label_order": [
                parse_concordance_order(v) for v in raw[order_col]
            ],
            "alexander_raw": raw["Alexander"],
            "jones_raw": raw["Jones"],
            "homfly_raw": raw["HOMFLY"],
        }
    )
    return frame


def _missing_reason(raw: str) -> str | None:
    text = raw.strip()
    if not text or text.lower() in {"?", "unknown", "na", "n/a"}:
        return "missing"
    return None


def _build_graph_run(
    frame: pd.DataFrame,
    kind: PolyKind,
    *,
    eps: float,
    use_standardize: bool,
    method: str,
) -> PolyGraphRun:
    raw_col = f"{kind}_raw" if kind != "homfly" else "homfly_raw"
    parser = POLY_PARSERS[kind]
    failures: list[ParseFailure] = []
    parsed_rows: list[dict[str, object]] = []

    for _, row in frame.iterrows():
        name = str(row["name"])
        raw_value = str(row[raw_col])
        missing = _missing_reason(raw_value)
        if missing:
            failures.append(
                ParseFailure(name=name, graph=kind, reason=missing, raw_value=raw_value)
            )
            continue
        coeffs = parser(raw_value)
        if coeffs is None:
            failures.append(
                ParseFailure(
                    name=name,
                    graph=kind,
                    reason="parse_error",
                    raw_value=raw_value,
                )
            )
            continue
        parsed_rows.append(
            {
                "name": name,
                "label_order": row["label_order"],
                "coeffs": coeffs,
            }
        )

    if not parsed_rows:
        raise ValueError(f"No knots parsed successfully for {kind} polynomial graph.")

    coeffs_list = [row["coeffs"] for row in parsed_rows]
    if kind == "homfly":
        grid = build_grid_2d(coeffs_list)  # type: ignore[arg-type]
        X = np.array(
            [vectorize_2d(c, grid) for c in coeffs_list],  # type: ignore[arg-type]
            dtype=float,
        )
    else:
        grid = build_grid_1d(coeffs_list)  # type: ignore[arg-type]
        X = np.array(
            [vectorize_1d(c, grid) for c in coeffs_list],  # type: ignore[arg-type]
            dtype=float,
        )

    if use_standardize:
        X, _, _ = zscore_features(X)

    bm_result = run_ball_mapper(X, eps, method=method)
    labels = np.array([row["label_order"] for row in parsed_rows], dtype=str)
    names = np.array([row["name"] for row in parsed_rows], dtype=str)

    return PolyGraphRun(
        kind=kind,
        bm_result=bm_result,
        labels=labels,
        names=names,
        feature_dim=X.shape[1],
        eps=eps,
        n_input=len(frame),
        n_used=len(parsed_rows),
        failures=tuple(failures),
        grid=grid,
    )


def write_parse_failures(failures: list[ParseFailure], output_path: Path) -> None:
    if not failures:
        return
    rows = [
        {
            "name": f.name,
            "graph": f.graph,
            "reason": f.reason,
            "raw_value": f.raw_value,
        }
        for f in failures
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def run_polyconc_bm(config: PolyConcConfig) -> list[PolyGraphRun]:
    """Build Alexander, Jones, and HOMFLYPT Ball Mapper graphs."""
    frame = load_polyconc_frame(config.csv_path, max_knots=config.max_knots)
    eps_by_kind = {
        "alexander": config.eps_alexander,
        "jones": config.eps_jones,
        "homfly": config.eps_homfly,
    }

    runs: list[PolyGraphRun] = []
    all_failures: list[ParseFailure] = []

    for kind in ("alexander", "jones", "homfly"):
        run = _build_graph_run(
            frame,
            kind,
            eps=eps_by_kind[kind],
            use_standardize=config.standardize,
            method=config.method,
        )
        runs.append(run)
        all_failures.extend(run.failures)

        print(
            f"\n=== {POLY_TITLES[kind]} ==="
            f"\n  feature dim: {run.feature_dim}"
            f"\n  knots used: {run.n_used}/{run.n_input}"
            f"\n  parse/missing failures: {len(run.failures)}"
            f"\n  ε: {run.eps} ({'standardized' if config.standardize else 'raw'} units)"
            f"\n  nerve nodes: {run.bm_result.num_nodes}"
            f"\n  nerve edges: {len(run.bm_result.edges())}"
        )
        if run.failures:
            sample = run.failures[:5]
            for failure in sample:
                print(
                    f"    - {failure.name}: {failure.reason} "
                    f"({failure.raw_value[:60]!r})"
                )
            if len(run.failures) > 5:
                print(f"    ... and {len(run.failures) - 5} more")

    figures_dir = config.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for run in runs:
        plot_polyconc_graph(
            run.bm_result,
            run.labels,
            title=f"{POLY_TITLES[run.kind]}  ε={run.eps}",
            output_path=figures_dir / f"polyconc_{run.kind}_eps{run.eps}.png",
            seed=config.seed,
        )

    plot_polyconc_combined(
        runs,
        output_path=figures_dir / "polyconc_combined.png",
        seed=config.seed,
    )

    write_parse_failures(all_failures, config.output_dir / "polyconc_parse_failures.csv")
    if all_failures:
        print(f"\nWrote parse/missing report: {config.output_dir / 'polyconc_parse_failures.csv'}")
    print(f"\nFigures saved under: {figures_dir}")

    return runs
