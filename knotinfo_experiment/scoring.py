"""Cluster Ball Mapper covers and score against ground truth."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import networkx as nx
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ball_mapper import BallMapper, BallMapperResult


@dataclass(frozen=True)
class ClusteringResult:
    """Ball Mapper clustering and diagnostic metrics."""

    component_ids: np.ndarray
    n_components: int
    ari: float
    nmi: float
    purity: float
    base_rate: float
    mean_ball_entropy: float
    multi_component_points: int


def ball_mapper_clustering(
    result: BallMapperResult,
    n_points: int,
) -> tuple[np.ndarray, int, int]:
    """
    Assign each point a connected-component id from the nerve graph.

    Points covered by multiple balls in different components receive the
    majority component; ties go to the smallest component id.
    """
    G = nx.Graph()
    G.add_nodes_from(result.node_info.keys())
    G.add_edges_from(result.edges())
    components = list(nx.connected_components(G))
    ball_to_component = {}
    for comp_id, nodes in enumerate(components):
        for node in nodes:
            ball_to_component[node] = comp_id

    point_components: list[list[int]] = [[] for _ in range(n_points)]
    for point_idx, balls in enumerate(result.assignments):
        for ball in balls:
            if ball in ball_to_component:
                point_components[point_idx].append(ball_to_component[ball])

    component_ids = np.full(n_points, -1, dtype=int)
    multi = 0
    for i, comps in enumerate(point_components):
        if not comps:
            continue
        counts = Counter(comps)
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        component_ids[i] = best
        if len(set(comps)) > 1:
            multi += 1

    return component_ids, len(components), multi


def _purity(labels: np.ndarray, clusters: np.ndarray) -> float:
    correct = 0
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        if not mask.any():
            continue
        majority = Counter(labels[mask]).most_common(1)[0][1]
        correct += majority
    return correct / len(labels)


def _majority_base_rate(labels: np.ndarray) -> float:
    counts = Counter(labels)
    return counts.most_common(1)[0][1] / len(labels)


def _ball_label_entropy(
    result: BallMapperResult,
    labels: np.ndarray,
    score_mask: np.ndarray | None = None,
) -> float:
    """Mean label entropy over balls (optionally restricted to scorable points)."""
    entropies: list[float] = []
    for _node, info in result.node_info.items():
        idx = list(info["indices"])
        if not idx:
            continue
        if score_mask is not None:
            idx = [i for i in idx if score_mask[i]]
        if not idx:
            continue
        node_labels = labels[idx]
        counts = Counter(node_labels)
        total = sum(counts.values())
        ent = 0.0
        for c in counts.values():
            p = c / total
            ent -= p * np.log2(p)
        entropies.append(ent)
    return float(np.mean(entropies)) if entropies else 0.0


def run_ball_mapper(
    X: np.ndarray,
    eps: float,
    *,
    method: str = "greedy",
    min_points_per_node: int = 1,
) -> BallMapperResult:
    bm = BallMapper(
        eps=eps,
        method=method,  # type: ignore[arg-type]
        min_points_per_node=min_points_per_node,
    )
    return bm.fit(X)


def score_clustering(
    result: BallMapperResult,
    labels: np.ndarray,
    score_mask: np.ndarray,
) -> ClusteringResult:
    """Score BM connected-component clustering against ground-truth labels."""
    component_ids, n_components, multi = ball_mapper_clustering(result, len(labels))

    valid = score_mask & (component_ids >= 0)
    y_true = labels[valid]
    y_pred = component_ids[valid]

    if len(y_true) < 2 or len(np.unique(y_pred)) < 1:
        ari = 0.0
        nmi = 0.0
    else:
        ari = float(adjusted_rand_score(y_true, y_pred))
        nmi = float(normalized_mutual_info_score(y_true, y_pred))

    purity = _purity(y_true, y_pred) if len(y_true) else 0.0
    base_rate = _majority_base_rate(y_true) if len(y_true) else 0.0
    entropy = _ball_label_entropy(result, labels, score_mask)

    return ClusteringResult(
        component_ids=component_ids,
        n_components=n_components,
        ari=ari,
        nmi=nmi,
        purity=purity,
        base_rate=base_rate,
        mean_ball_entropy=entropy,
        multi_component_points=multi,
    )
