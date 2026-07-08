#!/usr/bin/env python3
"""
Interactive Ball Mapper explorer for knot numeric invariants.

Loads ``data/knotnumericinvariants.csv`` (or a path you pass), lets you pick
numeric invariant columns, set ε, and redraw the nerve graph.

Controls (matplotlib window):
  - Checkboxes: select 1–4 numeric invariants for the feature vector
  - ε slider: ball radius in standardized units
  - Max knots slider: subsample cap (full CSV can be 10k+ rows)
  - Color: uniform (gray), Fibered, or L-space (blue = yes/Y, orange = no/N)
  - |chiral|: use absolute values for signature, τ, Rasmussen s, Nu samples
  - Compute: run Ball Mapper and refresh the graph
  - Save PNG: export the current view
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from ball_mapper import BallMapper
from knotinfo_experiment.columns import normalize_header
from knotinfo_experiment.data import standardize
from knotinfo_experiment.parse import parse_exact_numeric, parse_upsilon_components
from knotinfo_experiment.visualize import _node_sizes

DEFAULT_CSV = Path("data/knotnumericinvariants.csv")
META_COLUMNS = frozenset({"name", "fibered", "l_space"})
CHIRAL_KEYS = frozenset({"signature", "ozsvath_szabo_tau", "rasmussen_s"})
CHIRAL_PREFIX = "nu_"
MAX_INVARIANTS = 4


@dataclass(frozen=True)
class NumericTable:
    """Parsed numeric invariant table."""

    frame: pd.DataFrame
    numeric_keys: tuple[str, ...]
    labels: dict[str, str]  # internal key -> display label


def _column_key(header: str) -> str:
    return normalize_header(header)


def _is_upsilon_column(key: str) -> bool:
    """Only Nu / upsilon columns use vector [a;b] sample notation."""
    return key == "nu" or key.startswith("upsilon")


def load_numeric_table(csv_path: Path) -> NumericTable:
    """Load CSV and parse numeric invariant columns."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    rows: list[dict[str, object]] = []
    labels: dict[str, str] = {}

    for _, record in raw.iterrows():
        row: dict[str, object] = {}
        for header in raw.columns:
            key = _column_key(header)
            value = record[header]
            if key == "name":
                row["name"] = value
                labels["name"] = header
            elif key == "fibered":
                row["fibered"] = value.strip() or "unknown"
            elif key in {"l_space", "lspace"}:
                row["l_space"] = value.strip() or "unknown"
            else:
                if _is_upsilon_column(key):
                    components = parse_upsilon_components(value)
                    if components is not None:
                        for idx, comp in enumerate(components):
                            col = f"nu_{idx}" if key == "nu" else f"{key}_{idx}"
                            row[col] = comp
                            labels.setdefault(col, f"{header}[{idx}]")
                else:
                    # Scalar invariants: exact numbers only. KnotInfo intervals
                    # like "[2;3]" are bounds/uncertainty — treat as missing.
                    row[key] = parse_exact_numeric(value)
                    labels[key] = header
        rows.append(row)

    frame = pd.DataFrame(rows)
    numeric_keys = tuple(
        sorted(
            k
            for k in frame.columns
            if k not in META_COLUMNS and k != "name"
        )
    )
    return NumericTable(frame=frame, numeric_keys=numeric_keys, labels=labels)


def _apply_chiral(frame: pd.DataFrame, keys: tuple[str, ...], use_abs: bool) -> pd.DataFrame:
    if not use_abs:
        return frame
    out = frame.copy()
    for key in keys:
        if key in CHIRAL_KEYS or key.startswith(CHIRAL_PREFIX):
            out[key] = out[key].abs()
    return out


def _complete_mask(frame: pd.DataFrame, keys: tuple[str, ...]) -> np.ndarray:
    return frame[list(keys)].notna().all(axis=1).to_numpy()


def _subsample(mask: np.ndarray, max_knots: int, seed: int) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if len(indices) <= max_knots:
        return mask
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=max_knots, replace=False)
    out = np.zeros_like(mask, dtype=bool)
    out[chosen] = True
    return out


def _dominant(values: np.ndarray, indices: list[int]) -> str:
    if not indices:
        return "?"
    return Counter(values[i] for i in indices).most_common(1)[0][0]


def _normalize_yes_no(value: str) -> str:
    """Map Fibered / L-space labels to yes, no, or unknown."""
    text = str(value).strip().lower()
    if text in {"y", "yes"}:
        return "yes"
    if text in {"n", "no"}:
        return "no"
    return "unknown"


YES_NO_COLORS = {
    "yes": "#1f77b4",  # blue
    "no": "#ff7f0e",   # orange
    "unknown": "#6c757d",
}


def _node_colors(
    result,
    color_values: np.ndarray | None,
) -> list:
    if color_values is None:
        return [YES_NO_COLORS["unknown"]] * len(result.node_info)
    return [
        YES_NO_COLORS[_normalize_yes_no(_dominant(color_values, result.node_info[n]["indices"]))]
        for n in result.node_info
    ]


class NumericBMExplorer:
    """Matplotlib UI for exploring Ball Mapper on numeric invariants."""

    def __init__(
        self,
        table: NumericTable,
        *,
        max_knots: int = 800,
        seed: int = 0,
        eps: float = 1.5,
    ) -> None:
        self.table = table
        self.max_knots = max_knots
        self.seed = seed
        self.eps = eps
        self.pos_cache: dict[frozenset[str], dict] = {}

        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title("Numeric Ball Mapper Explorer")
        self.ax_graph = self.fig.add_axes((0.38, 0.12, 0.58, 0.82))
        self.ax_checks = self.fig.add_axes((0.02, 0.24, 0.32, 0.70))
        self.ax_chiral = self.fig.add_axes((0.02, 0.19, 0.32, 0.04))
        self.ax_color = self.fig.add_axes((0.02, 0.02, 0.32, 0.15))
        self.ax_eps = self.fig.add_axes((0.38, 0.06, 0.35, 0.03))
        self.ax_max = self.fig.add_axes((0.74, 0.06, 0.22, 0.03))
        self.ax_btn = self.fig.add_axes((0.38, 0.01, 0.10, 0.04))
        self.ax_save = self.fig.add_axes((0.50, 0.01, 0.10, 0.04))
        self.ax_stats = self.fig.add_axes((0.62, 0.01, 0.34, 0.04))
        self.ax_stats.axis("off")

        default_on = {"signature", "genus_4d", "ozsvath_szabo_tau"} & set(table.numeric_keys)
        if not default_on:
            default_on = set(table.numeric_keys[: min(3, len(table.numeric_keys))])

        check_labels = [table.labels.get(k, k) for k in table.numeric_keys]
        check_active = [k in default_on for k in table.numeric_keys]
        self.check = CheckButtons(self.ax_checks, check_labels, check_active)
        self.ax_checks.set_title("Numeric invariants", fontsize=10, loc="left")

        self.chiral_check = CheckButtons(self.ax_chiral, ["|chiral|"], [True])
        self.ax_chiral.axis("off")

        self.color_radio = RadioButtons(
            self.ax_color,
            ("uniform", "fibered", "l_space"),
            active=1,
        )
        self.ax_color.set_title("Node color", fontsize=10, loc="left")

        self.eps_slider = Slider(
            self.ax_eps,
            "ε (std)",
            0.2,
            4.0,
            valinit=eps,
            valstep=0.1,
        )
        self.max_slider = Slider(
            self.ax_max,
            "max knots",
            100,
            len(table.frame),
            valinit=min(max_knots, len(table.frame)),
            valstep=1,
        )

        self.compute_btn = Button(self.ax_btn, "Compute")
        self.save_btn = Button(self.ax_save, "Save PNG")

        self.compute_btn.on_clicked(lambda _event: self.compute())
        self.save_btn.on_clicked(lambda _event: self.save())
        self.stats_text = self.ax_stats.text(
            0.0, 0.5, "Select invariants and click Compute.", fontsize=9, va="center"
        )

        self._last_result = None
        self.compute()

    def _selected_keys(self) -> tuple[str, ...]:
        status = self.check.get_status()
        keys = tuple(k for k, on in zip(self.table.numeric_keys, status) if on)
        if len(keys) > MAX_INVARIANTS:
            raise ValueError(f"Select at most {MAX_INVARIANTS} invariants (got {len(keys)}).")
        if not keys:
            raise ValueError("Select at least one numeric invariant.")
        return keys

    def _color_values(self, row_mask: np.ndarray) -> np.ndarray | None:
        mode = self.color_radio.value_selected
        sub = self.table.frame.loc[row_mask]
        if mode == "fibered":
            return sub["fibered"].to_numpy(dtype=str)
        if mode == "l_space":
            return sub["l_space"].to_numpy(dtype=str)
        return None

    def compute(self) -> None:
        try:
            keys = self._selected_keys()
        except ValueError as exc:
            self.stats_text.set_text(str(exc))
            self.fig.canvas.draw_idle()
            return

        use_abs = self.chiral_check.get_status()[0]
        eps = float(self.eps_slider.val)
        max_knots = int(self.max_slider.val)

        mask = _complete_mask(self.table.frame, keys)
        n_complete = int(mask.sum())
        if n_complete < 3:
            self.stats_text.set_text(f"Only {n_complete} knots with all selected invariants.")
            self.fig.canvas.draw_idle()
            return

        mask = _subsample(mask, max_knots, self.seed)
        n = int(mask.sum())
        sub = _apply_chiral(self.table.frame.loc[mask], keys, use_abs)
        X = np.column_stack([sub[k].to_numpy(dtype=float) for k in keys])
        X_std, _, _ = standardize(X)

        bm = BallMapper(eps=eps, min_points_per_node=1)
        result = bm.fit(X_std)
        self._last_result = result

        self.ax_graph.clear()
        G = nx.Graph()
        G.add_nodes_from(result.node_info.keys())
        G.add_edges_from(result.edges())

        if G.number_of_nodes() == 0:
            self.stats_text.set_text("Ball Mapper returned no nodes for this ε.")
            self.fig.canvas.draw_idle()
            return

        layout_key = (frozenset(keys), eps, max_knots)
        pos = self.pos_cache.get(layout_key)
        if pos is None or not set(G.nodes()).issubset(pos):
            pos = nx.spring_layout(G, seed=self.seed)
            self.pos_cache[layout_key] = pos

        colors = _node_colors(result, self._color_values(mask))
        sizes = _node_sizes(result, list(G.nodes()))
        nx.draw_networkx(
            G,
            pos=pos,
            node_color=colors,
            node_size=sizes,
            with_labels=False,
            ax=self.ax_graph,
        )

        inv_label = ", ".join(self.table.labels.get(k, k) for k in keys)
        self.ax_graph.set_title(
            f"Ball Mapper: {inv_label}\nε={eps:.1f} (standardized), n={n} knots, "
            f"{result.num_nodes} balls, {len(result.edges())} edges",
            fontsize=10,
        )
        self.ax_graph.axis("off")

        sampled = " (subsampled)" if n < n_complete else ""
        self.stats_text.set_text(
            f"Complete: {n_complete}  Plotted: {n}{sampled}  "
            f"Balls: {result.num_nodes}  Edges: {len(result.edges())}  "
            f"Node size ∝ ball population"
        )
        self.fig.canvas.draw_idle()

    def save(self) -> None:
        if self._last_result is None:
            return
        path = Path("output") / "numeric_bm_interactive.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        self.stats_text.set_text(f"Saved {path.resolve()}")
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive Ball Mapper for numeric knot invariants.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Numeric invariants CSV path.")
    parser.add_argument("--max-knots", type=int, default=800, help="Default subsample cap.")
    parser.add_argument("--eps", type=float, default=1.5, help="Initial ε in standardized units.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling and layout.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    table = load_numeric_table(args.csv)
    print(f"Loaded {len(table.frame)} knots, {len(table.numeric_keys)} numeric columns:")
    for key in table.numeric_keys:
        n_present = table.frame[key].notna().sum()
        print(f"  {table.labels.get(key, key):30s}  ({n_present} values)")

    explorer = NumericBMExplorer(
        table,
        max_knots=args.max_knots,
        seed=args.seed,
        eps=args.eps,
    )
    explorer.show()


if __name__ == "__main__":
    main()
