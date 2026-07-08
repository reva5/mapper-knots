"""Main experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ball_mapper import BallMapperResult
from knotinfo_experiment.artifacts import find_collisions, find_extremal_knots, write_artifacts
from knotinfo_experiment.config import ExperimentConfig
from knotinfo_experiment.data import (
    KnotTable,
    available_pool_keys,
    completeness_mask,
    label_distribution,
    load_knot_table,
    standardize,
)
from knotinfo_experiment.labels import assert_no_label_leakage, build_labels, scoring_mask
from knotinfo_experiment.scoring import run_ball_mapper, score_clustering
from knotinfo_experiment.subsets import enumerate_subsets
from knotinfo_experiment.visualize import plot_top_subsets


@dataclass(frozen=True)
class RunRecord:
    subset: str
    kind: str
    epsilon: float
    n: int
    n_scored: int
    ari: float
    nmi: float
    purity: float
    base_rate: float
    n_components: int
    mean_ball_entropy: float
    label_distribution: str
    matched_concordance: str


@dataclass
class BestConcordanceRun:
    subset: tuple[str, ...]
    subset_name: str
    epsilon: float
    ari: float
    bm_result: BallMapperResult
    labels: np.ndarray
    families: np.ndarray
    X_std: np.ndarray
    row_indices: np.ndarray


def _format_distribution(dist: dict[str, int]) -> str:
    return ";".join(f"{k}:{v}" for k, v in sorted(dist.items()))


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """Execute the full invariant-subset Ball Mapper experiment."""
    table = load_knot_table(config.csv_path, config)
    concordance_pool, control_pool = available_pool_keys(table, config)
    print(f"\nConcordance pool available: {concordance_pool}")
    print(f"Control pool available: {control_pool}")

    subset_specs = enumerate_subsets(concordance_pool, control_pool, config)
    print(f"\nEvaluating {len(subset_specs)} feature subsets")

    records: list[RunRecord] = []
    best_runs: dict[tuple[str, ...], BestConcordanceRun] = {}

    for spec in subset_specs:
        if not assert_no_label_leakage(spec.features, config):
            continue

        complete = completeness_mask(table, spec.features)
        row_indices = np.flatnonzero(complete)
        n = len(row_indices)
        if n < 3:
            print(f"  SKIP {spec.name}: only {n} complete knots")
            continue

        labels = build_labels(table, config).iloc[row_indices].to_numpy(dtype=str)
        families = table.frame.iloc[row_indices]["family"].to_numpy(dtype=str)
        score_mask = scoring_mask(pd.Series(labels)).to_numpy()
        n_scored = int(score_mask.sum())
        dist = label_distribution(pd.Series(labels))

        X_raw = table.feature_matrix(spec.features, complete)
        X_std, _, _ = standardize(X_raw)

        best_ari = -np.inf
        best_eps = config.epsilon_grid[0]
        best_bm: BallMapperResult | None = None

        for eps in config.epsilon_grid:
            bm_result = run_ball_mapper(
                X_std,
                eps,
                method=config.bm_method,
                min_points_per_node=config.min_points_per_node,
            )
            metrics = score_clustering(bm_result, labels, score_mask)

            records.append(
                RunRecord(
                    subset=spec.name,
                    kind=spec.kind,
                    epsilon=eps,
                    n=n,
                    n_scored=n_scored,
                    ari=metrics.ari,
                    nmi=metrics.nmi,
                    purity=metrics.purity,
                    base_rate=metrics.base_rate,
                    n_components=metrics.n_components,
                    mean_ball_entropy=metrics.mean_ball_entropy,
                    label_distribution=_format_distribution(dist),
                    matched_concordance=(
                        "+".join(spec.matched_concordance) if spec.matched_concordance else ""
                    ),
                )
            )

            if spec.kind == "concordance" and metrics.ari > best_ari:
                best_ari = metrics.ari
                best_eps = eps
                best_bm = bm_result

        if spec.kind == "concordance" and best_bm is not None:
            best_runs[spec.features] = BestConcordanceRun(
                subset=spec.features,
                subset_name=spec.name,
                epsilon=best_eps,
                ari=best_ari,
                bm_result=best_bm,
                labels=labels,
                families=families,
                X_std=X_std,
                row_indices=row_indices,
            )

    results_df = pd.DataFrame([r.__dict__ for r in records])
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results_df.sort_values(["kind", "ari", "nmi"], ascending=[True, False, False]).to_csv(
        config.output_dir / "results_ranked.csv",
        index=False,
    )

    _print_summary(results_df, config)

    if not best_runs:
        return results_df

    top_run = max(best_runs.values(), key=lambda r: r.ari)
    collisions = find_collisions(
        table,
        top_run.row_indices,
        top_run.labels,
        top_run.bm_result,
        top_run.X_std,
        top_run.subset,
        config.collision_vector_tolerance,
    )
    extremal = find_extremal_knots(
        table,
        top_run.row_indices,
        top_run.subset,
        config.extremal_tolerance,
    )
    write_artifacts(collisions, extremal, config.output_dir)

    plot_specs = _top_n_for_plots(results_df, best_runs, config)
    plot_top_subsets(plot_specs, config.output_dir / "figures", config.random_seed)

    print(
        f"\nTop subset artifacts ({top_run.subset_name}, ε={top_run.epsilon}): "
        f"{len(collisions)} collision rows, {len(extremal)} extremal rows"
    )
    print(f"Outputs in {config.output_dir.resolve()}")
    return results_df


def _top_n_for_plots(
    results_df: pd.DataFrame,
    best_runs: dict[tuple[str, ...], BestConcordanceRun],
    config: ExperimentConfig,
) -> list[tuple[str, float, BallMapperResult, np.ndarray, np.ndarray]]:
    conc = results_df[results_df["kind"] == "concordance"].copy()
    conc = conc.sort_values(["ari", "nmi"], ascending=False)

    plot_specs: list[tuple[str, float, BallMapperResult, np.ndarray, np.ndarray]] = []
    seen: set[str] = set()
    for _, row in conc.iterrows():
        name = row["subset"]
        if name in seen:
            continue
        features = tuple(name.split("+"))
        run = best_runs.get(features)
        if run is None:
            continue
        plot_specs.append((name, run.epsilon, run.bm_result, run.labels, run.families))
        seen.add(name)
        if len(plot_specs) >= config.top_n_plots:
            break
    return plot_specs


def _print_summary(results_df: pd.DataFrame, config: ExperimentConfig) -> None:
    if results_df.empty:
        print("\nNo results to summarize.")
        return

    conc = results_df[results_df["kind"] == "concordance"].sort_values(
        ["ari", "nmi"], ascending=False
    )
    ctrl = results_df[results_df["kind"] == "control"]

    best = conc.iloc[0]
    print("\n=== SUMMARY ===")
    print(
        f"Best concordance subset: {best['subset']}  "
        f"(ε={best['epsilon']}, n={best['n']}, ARI={best['ari']:.4f}, NMI={best['nmi']:.4f})"
    )
    print(
        f"  Context: purity={best['purity']:.4f}, majority base rate={best['base_rate']:.4f}"
    )

    matched = ctrl[ctrl["matched_concordance"] == best["subset"]]
    if not matched.empty:
        best_ctrl = matched.sort_values(["ari", "nmi"], ascending=False).iloc[0]
        gap = best["ari"] - best_ctrl["ari"]
        print(
            f"Matched control: {best_ctrl['subset']}  "
            f"(ARI={best_ctrl['ari']:.4f}, gap={gap:+.4f})"
        )
        if gap <= 0:
            print("  ** WARNING: concordance subset does NOT beat its matched control **")
    else:
        print("  (no matched control found for best concordance subset)")

    print("\n=== Concordance vs control gaps ===")
    for subset in conc["subset"].unique():
        c_best = conc[conc["subset"] == subset].sort_values("ari", ascending=False).iloc[0]
        m = ctrl[ctrl["matched_concordance"] == subset]
        if m.empty:
            continue
        k_best = m.sort_values("ari", ascending=False).iloc[0]
        gap = c_best["ari"] - k_best["ari"]
        flag = "" if gap > 0 else "  ** FAIL **"
        print(
            f"  {subset:40s}  ARI gap={gap:+.4f}  "
            f"(conc={c_best['ari']:.3f} vs ctrl={k_best['subset']} {k_best['ari']:.3f})"
            f"{flag}"
        )

    print(f"\nFull results: {config.output_dir / 'results_ranked.csv'}")
