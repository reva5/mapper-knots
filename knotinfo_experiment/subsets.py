"""Feature-subset enumeration with matched controls."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from knotinfo_experiment.config import ExperimentConfig, SubsetKind


@dataclass(frozen=True)
class SubsetSpec:
    """A feature subset with metadata."""

    features: tuple[str, ...]
    kind: SubsetKind
    matched_concordance: tuple[str, ...] | None = None

    @property
    def name(self) -> str:
        return "+".join(self.features)


def _concordance_subsets(pool: list[str], k_max: int) -> list[tuple[str, ...]]:
    subsets: list[tuple[str, ...]] = []
    for k in range(1, min(k_max, len(pool)) + 1):
        subsets.extend(itertools.combinations(pool, k))
    return subsets


def matched_control_subset(
    concordance_subset: tuple[str, ...],
    control_pool: list[str],
) -> tuple[str, ...] | None:
    """Same-cardinality control subset using the first available control features."""
    k = len(concordance_subset)
    if len(control_pool) < k:
        return None
    return tuple(control_pool[:k])


def enumerate_subsets(
    concordance_pool: list[str],
    control_pool: list[str],
    config: ExperimentConfig,
) -> list[SubsetSpec]:
    """
    Enumerate concordance subsets up to k_max plus matched control counterparts.

    If ``config.explicit_subsets`` is set, it overrides automatic enumeration.
    """
    if config.explicit_subsets is not None:
        specs: list[SubsetSpec] = []
        for subset in config.explicit_subsets:
            is_control = any(f in control_pool for f in subset) and not all(
                f in concordance_pool for f in subset
            )
            kind: SubsetKind = "control" if is_control else "concordance"
            specs.append(SubsetSpec(features=subset, kind=kind))
        return specs[: config.max_subsets]

    specs = []
    concordance_subsets = _concordance_subsets(concordance_pool, config.k_max)

    for subset in concordance_subsets:
        specs.append(SubsetSpec(features=subset, kind="concordance"))
        control = matched_control_subset(subset, control_pool)
        if control is not None:
            specs.append(
                SubsetSpec(
                    features=control,
                    kind="control",
                    matched_concordance=subset,
                )
            )

    return specs[: config.max_subsets]
