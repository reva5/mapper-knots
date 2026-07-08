"""Visualize Ball Mapper graphs for KnotInfo runs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ball_mapper import BallMapperResult

# NetworkX node_size is in points^2; keep absolute sizes modest while preserving ratios.
MIN_NODE_SIZE = 35.0
MAX_NODE_SIZE = 160.0


def _node_sizes(result: BallMapperResult, nodes: list) -> list[float]:
    """Map ball populations to drawable node sizes (linear, min-max normalized)."""
    populations = [result.node_info[n]["size"] for n in nodes]
    if not populations:
        return []
    min_pop = min(populations)
    max_pop = max(populations)
    if max_pop == min_pop:
        mid = (MIN_NODE_SIZE + MAX_NODE_SIZE) / 2
        return [mid] * len(populations)
    scale = (MAX_NODE_SIZE - MIN_NODE_SIZE) / (max_pop - min_pop)
    return [MIN_NODE_SIZE + (p - min_pop) * scale for p in populations]


def _dominant_label(labels: np.ndarray, indices: list[int]) -> str:
    if not indices:
        return "empty"
    counts = Counter(labels[i] for i in indices)
    return counts.most_common(1)[0][0]


def _dominant_family(families: np.ndarray, indices: list[int]) -> str:
    if not indices:
        return "unknown"
    counts = Counter(families[i] for i in indices)
    return counts.most_common(1)[0][0]


def _label_color_map(labels: np.ndarray) -> dict[str, tuple[float, float, float, float]]:
    unique = sorted(set(labels))
    cmap = plt.cm.tab10
    return {lab: cmap(i % 10) for i, lab in enumerate(unique)}


def plot_ball_mapper_graph(
    result: BallMapperResult,
    labels: np.ndarray,
    families: np.ndarray,
    *,
    color_by: str,
    title: str,
    output_path: Path,
    seed: int = 0,
) -> None:
    """Save a Ball Mapper nerve graph PNG."""
    G = nx.Graph()
    G.add_nodes_from(result.node_info.keys())
    G.add_edges_from(result.edges())

    if G.number_of_nodes() == 0:
        return

    # Size nodes by ball population (relative scale, modest absolute sizes).
    sizes = _node_sizes(result, list(G.nodes()))
    pos = nx.spring_layout(G, seed=seed, weight=None)

    if color_by == "label":
        colors = [
            _label_color_map(labels)[_dominant_label(labels, result.node_info[n]["indices"])]
            for n in G.nodes()
        ]
    else:
        families_unique = sorted(set(families))
        fam_cmap = {f: plt.cm.Set2(i % 8) for i, f in enumerate(families_unique)}
        colors = [
            fam_cmap[_dominant_family(families, result.node_info[n]["indices"])]
            for n in G.nodes()
        ]

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx(
        G,
        pos=pos,
        node_color=colors,
        node_size=sizes,
        with_labels=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_subsets(
    results: list[tuple[str, float, BallMapperResult, np.ndarray, np.ndarray]],
    output_dir: Path,
    seed: int,
) -> None:
    """Write label- and family-colored PNGs for each top subset."""
    for subset_name, eps, bm_result, labels, families in results:
        safe = subset_name.replace("/", "_")
        plot_ball_mapper_graph(
            bm_result,
            labels,
            families,
            color_by="label",
            title=f"{subset_name}  ε={eps}  (by concordance order)",
            output_path=output_dir / f"bm_{safe}_eps{eps}_by_label.png",
            seed=seed,
        )
        plot_ball_mapper_graph(
            bm_result,
            labels,
            families,
            color_by="family",
            title=f"{subset_name}  ε={eps}  (by family)",
            output_path=output_dir / f"bm_{safe}_eps{eps}_by_family.png",
            seed=seed,
        )
