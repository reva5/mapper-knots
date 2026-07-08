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
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from gudhi.cover_complex import NerveComplex
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

NetMethod = Literal["greedy", "maxmin"]

MIN_NODE_SIZE = 35.0
MAX_NODE_SIZE = 160.0


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


def _nerve_node_sizes(result: BallMapperResult, nodes: list) -> list[float]:
    """Map ball populations to networkx node_size values."""
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


def _nerve_node_colors(result: BallMapperResult) -> list:
    """Node colors from GUDHI's per-node averaged color values."""
    import matplotlib.pyplot as plt

    values = [result.node_info[n]["colors"][0] for n in result.node_info]
    vmin, vmax = min(values), max(values)
    if vmin == vmax:
        return ["#4c72b0"] * len(values)
    cmap = plt.cm.viridis
    return [cmap((v - vmin) / (vmax - vmin)) for v in values]


@dataclass
class BallMapperResult:
    """Output of a Ball Mapper fit."""

    centers: np.ndarray
    center_indices: list[int]
    assignments: list[list[int]]
    nerve: NerveComplex
    simplex_tree: object
    node_info: dict
    data: np.ndarray
    eps: float
    point_colors: np.ndarray | None = None

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

    def plot(
        self,
        *,
        ax: matplotlib.axes.Axes | None = None,
        figsize: tuple[float, float] = (10.0, 4.0),
        seed: int = 0,
        show: bool = False,
        savepath: str | Path | None = None,
        draw_balls: bool = True,
        title: str | None = None,
    ) -> tuple[matplotlib.figure.Figure, np.ndarray]:
        """
        Plot the input data and Ball Mapper nerve graph.

        For 2D data, draws the point cloud, ε-net centers, optional covering
        balls, and the nerve graph side by side. For higher-dimensional data,
        only the nerve graph is shown (using the first two coordinates for a
        scatter preview when *ax* is not passed).

        Parameters
        ----------
        ax
            If given, draw only the nerve graph on this axes. Otherwise create
            a side-by-side data + nerve figure (or nerve-only when *d > 2*).
        figsize
            Figure size when creating a new figure.
        seed
            Random seed for the nerve graph spring layout.
        show
            Call ``plt.show()`` before returning.
        savepath
            If set, save the figure to this path.
        draw_balls
            Draw ε-radius circles around centers when data is 2D.
        title
            Optional suptitle for the figure.

        Returns
        -------
        fig, axes
            The matplotlib figure and axes array.
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib.patches import Circle

        G = nx.Graph()
        G.add_nodes_from(self.node_info.keys())
        G.add_edges_from(self.edges())

        pos = nx.spring_layout(G, seed=seed) if G.number_of_nodes() else {}
        node_sizes = _nerve_node_sizes(self, list(G.nodes()))
        node_colors = _nerve_node_colors(self)

        def _draw_nerve(target_ax: matplotlib.axes.Axes) -> None:
            if G.number_of_nodes() == 0:
                target_ax.set_title("Ball Mapper nerve graph (empty)")
                target_ax.axis("off")
                return
            nx.draw_networkx(
                G,
                pos=pos,
                node_color=node_colors,
                node_size=node_sizes,
                with_labels=False,
                ax=target_ax,
            )
            target_ax.set_title(
                f"Nerve graph ({self.num_nodes} balls, {len(self.edges())} edges)"
            )
            target_ax.axis("off")

        if ax is not None:
            fig = ax.figure
            _draw_nerve(ax)
            if title:
                fig.suptitle(title)
            if savepath:
                fig.savefig(savepath, dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            return fig, np.array([ax])

        d = self.data.shape[1]
        show_data_panel = d <= 2

        if show_data_panel:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            data_ax, nerve_ax = axes
        else:
            fig, nerve_ax = plt.subplots(figsize=(figsize[0] / 2, figsize[1]))
            axes = np.array([nerve_ax])

        if show_data_panel:
            if d == 1:
                data_ax.scatter(self.data[:, 0], np.zeros(len(self.data)), s=8, c="lightgray")
                data_ax.scatter(
                    self.centers[:, 0],
                    np.zeros(len(self.centers)),
                    s=40,
                    c="crimson",
                    label="ε-net centers",
                )
            else:
                point_c = self.point_colors if self.point_colors is not None else "lightgray"
                data_ax.scatter(
                    self.data[:, 0], self.data[:, 1], s=8, c=point_c, alpha=0.7
                )
                data_ax.scatter(
                    self.centers[:, 0],
                    self.centers[:, 1],
                    s=40,
                    c="crimson",
                    label="ε-net centers",
                    zorder=3,
                )
                if draw_balls and d == 2:
                    for center in self.centers:
                        data_ax.add_patch(
                            Circle(
                                (center[0], center[1]),
                                radius=self.eps,
                                fill=False,
                                edgecolor="crimson",
                                linewidth=0.8,
                                alpha=0.35,
                            )
                        )
            data_ax.set_title(f"Point cloud (n={len(self.data)}, ε={self.eps})")
            data_ax.set_aspect("equal", adjustable="datalim")
            data_ax.legend(loc="best")

        _draw_nerve(nerve_ax)

        if title:
            fig.suptitle(title)
        fig.tight_layout()

        if savepath:
            path = Path(savepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig, axes


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
            data=X,
            eps=self.eps,
            point_colors=np.asarray(color) if color is not None else None,
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
            result.plot(seed=args.seed, show=True)
        except ImportError as exc:
            raise SystemExit(
                "Plotting requires matplotlib and networkx: pip install matplotlib networkx"
            ) from exc


if __name__ == "__main__":
    main()
