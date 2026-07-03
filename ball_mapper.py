#!/usr/bin/env python3
"""
Ball Mapper algorithm using GUDHI.

Implements the Ball Mapper construction from Dłotko (arXiv:1901.07410):
  1. Build an ε-net of landmark centers on the point cloud.
  2. Form a cover by balls of radius ε around each center.
  3. Compute the nerve of that cover with GUDHI's NerveComplex.

References:
  - Ball mapper: a shape summary for topological data analysis (1901.07410)
  - GUDHI cover complex: https://gudhi.inria.fr/python/latest/cover_complex_sklearn_user.html
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

import numpy as np
from gudhi.cover_complex import NerveComplex
from scipy.spatial.distance import cdist


NetMethod = Literal["greedy", "maxmin"]


def greedy_epsilon_net(distances: np.ndarray, eps: float) -> list[int]:
    """
    Algorithm 1 from the Ball Mapper paper: greedy ε-net.

    Returns indices of landmark points in the order they are selected.
    """
    covered = np.zeros(distances.shape[0], dtype=bool)
    centers: list[int] = []

    while not covered.all():
        center = int(np.flatnonzero(~covered)[0])
        centers.append(center)
        covered |= distances[center] <= eps

    return centers


def maxmin_epsilon_net(distances: np.ndarray, eps: float) -> list[int]:
    """
    Algorithm 2 from the Ball Mapper paper: max-min ε-net.

    Returns indices of landmark points.
    """
    centers = [0]
    min_dist = distances[0].copy()

    while True:
        min_dist = np.minimum(min_dist, distances[centers[-1]])
        farthest = int(np.argmax(min_dist))
        if min_dist[farthest] <= eps:
            break
        centers.append(farthest)

    return centers


def build_cover_assignments(
    distances: np.ndarray,
    centers: list[int],
    eps: float,
) -> list[list[int]]:
    """
    Build the cover vector B(X, ε): for each point, the list of ball-center indices
    whose ε-balls contain that point.
    """
    center_to_id = {c: i for i, c in enumerate(centers)}
    assignments: list[list[int]] = []

    for point_idx in range(distances.shape[0]):
        covering = [
            center_to_id[c]
            for c in centers
            if distances[point_idx, c] <= eps
        ]
        assignments.append(covering)

    return assignments


@dataclass
class BallMapperResult:
    """Output of a Ball Mapper fit."""

    centers: np.ndarray
    center_indices: list[int]
    assignments: list[list[int]]
    nerve: NerveComplex
    simplex_tree: object
    node_info: dict

    @property
    def num_nodes(self) -> int:
        return len(self.node_info)

    def edges(self) -> list[tuple[int, int]]:
        """1-skeleton edges of the Ball Mapper nerve graph."""
        st = self.simplex_tree
        edges: list[tuple[int, int]] = []
        for simplex, _ in st.get_simplices():
            if len(simplex) == 2:
                edges.append((simplex[0], simplex[1]))
        return edges

    def simplices(self) -> list[tuple[list[int], float]]:
        return list(self.simplex_tree.get_simplices())


class BallMapper:
    """
    Ball Mapper: nerve of an ε-ball cover built from an ε-net.

    Parameters
    ----------
    eps : float
        Ball radius (must be > 0).
    method : {"greedy", "maxmin"}
        Strategy for selecting ε-net centers (Algorithms 1 and 2 in the paper).
    min_points_per_node : int
        Drop nerve nodes covering fewer than this many data points (GUDHI mask).
    metric : str
        Distance metric passed to ``scipy.spatial.distance.cdist``.
    verbose : bool
        Print GUDHI progress messages.
    """

    def __init__(
        self,
        eps: float,
        *,
        method: NetMethod = "greedy",
        min_points_per_node: int = 0,
        metric: str = "euclidean",
        verbose: bool = False,
    ) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive")
        if method not in ("greedy", "maxmin"):
            raise ValueError("method must be 'greedy' or 'maxmin'")

        self.eps = eps
        self.method = method
        self.min_points_per_node = min_points_per_node
        self.metric = metric
        self.verbose = verbose

    def fit(
        self,
        X: np.ndarray,
        *,
        color: np.ndarray | None = None,
    ) -> BallMapperResult:
        """
        Run Ball Mapper on a point cloud.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input point cloud.
        color : array-like, shape (n_samples,) or (n_samples, n_colors), optional
            Functions used to color nerve nodes (averaged per node by GUDHI).

        Returns
        -------
        BallMapperResult
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")

        distances = cdist(X, X, metric=self.metric)
        if self.method == "greedy":
            center_indices = greedy_epsilon_net(distances, self.eps)
        else:
            center_indices = maxmin_epsilon_net(distances, self.eps)

        assignments = build_cover_assignments(distances, center_indices, self.eps)

        nerve = NerveComplex(
            input_type="point cloud",
            min_points_per_node=self.min_points_per_node,
            verbose=self.verbose,
        )
        nerve.fit(X, assignments=assignments, color=color)

        return BallMapperResult(
            centers=X[center_indices],
            center_indices=center_indices,
            assignments=assignments,
            nerve=nerve,
            simplex_tree=nerve.simplex_tree_,
            node_info=nerve.node_info_,
        )


def sample_circle(n: int, radius: float = 1.0, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Sample n points from a circle (optionally with Gaussian noise)."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    X = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    if noise > 0:
        X += rng.normal(scale=noise, size=X.shape)
    return X


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ball Mapper on a synthetic point cloud using GUDHI.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.35,
        help="Ball radius ε (default: 0.35)",
    )
    parser.add_argument(
        "--method",
        choices=("greedy", "maxmin"),
        default="greedy",
        help="ε-net construction method (default: greedy)",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=200,
        help="Number of points to sample from a unit circle (default: 200)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.02,
        help="Gaussian noise scale for the circle sample (default: 0.02)",
    )
    parser.add_argument(
        "--min-points-per-node",
        type=int,
        default=0,
        help="Minimum points per nerve node (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the Ball Mapper graph (requires matplotlib and networkx)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    X = sample_circle(args.n_points, noise=args.noise, seed=args.seed)

    bm = BallMapper(
        eps=args.eps,
        method=args.method,
        min_points_per_node=args.min_points_per_node,
    )
    result = bm.fit(X, color=X[:, 0])

    print(f"Point cloud: {X.shape[0]} points in R^{X.shape[1]}")
    print(f"ε-net ({args.method}): {len(result.center_indices)} centers")
    print(f"Nerve: {result.num_nodes} nodes, {len(result.edges())} edges")
    print(f"Simplices (up to dimension {result.simplex_tree.dimension()}):")
    for simplex, filtration in result.simplices():
        print(f"  {simplex}  (filtration={filtration})")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as exc:
            raise SystemExit(
                "Plotting requires matplotlib and networkx: pip install matplotlib networkx"
            ) from exc

        G = nx.Graph()
        G.add_nodes_from(result.node_info.keys())
        G.add_edges_from(result.edges())

        pos = nx.spring_layout(G, seed=args.seed)
        node_colors = [result.node_info[n]["colors"][0] for n in G.nodes()]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(X[:, 0], X[:, 1], s=8, c="lightgray", label="data")
        axes[0].scatter(
            result.centers[:, 0],
            result.centers[:, 1],
            s=40,
            c="crimson",
            label="ε-net centers",
        )
        axes[0].set_title("Point cloud and ε-net")
        axes[0].set_aspect("equal")
        axes[0].legend()

        nx.draw_networkx(
            G,
            pos=pos,
            node_color=node_colors,
            cmap=plt.cm.viridis,
            with_labels=False,
            ax=axes[1],
        )
        axes[1].set_title("Ball Mapper nerve graph")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
