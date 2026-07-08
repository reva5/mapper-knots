# mapper-knots

Ball Mapper for knot theory: a Python implementation of the Ball Mapper algorithm (via GUDHI), plus a reproducible experiment pipeline that searches concordance-invariant subsets on a local [KnotInfo](https://knotinfo.math.indiana.edu/) export and scores how well the resulting covers align with concordance order.

## Overview

Ball Mapper summarizes a point cloud by building an overlapping cover with balls of fixed radius ε, then taking the **nerve** of that cover as a graph (or simplicial complex). Unlike classical Mapper, it needs no lens function—only a distance and ε.

This repository has four components:

1. **`ball_mapper.py`** — a standalone Ball Mapper implementation using GUDHI's `NerveComplex`.
2. **`knotinfo_experiment/`** — an experiment runner that:
   - loads a KnotInfo CSV you export locally (no network access),
   - builds feature vectors from candidate invariant subsets,
   - runs Ball Mapper at several ε values (in standardized coordinates),
   - scores alignment with concordance-order labels (ARI / NMI),
   - compares concordance subsets against matched complexity controls,
   - extracts collisions, extremal knots, and Ball Mapper figures.
3. **`plot_numeric_bm.py`** — an interactive matplotlib explorer for Ball Mapper on numeric knot invariants (see below).
4. **`run_polyconc_bm.py`** — three separate Ball Mapper graphs on Alexander, Jones, and HOMFLYPT polynomial coefficients, colored by concordance order (see below).

Reference: Dłotko, [*Ball mapper: a shape summary for topological data analysis*](https://arxiv.org/abs/1901.07410) (arXiv:1901.07410).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+ and: `gudhi`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `networkx`.

## Quick start

### 1. Export KnotInfo data

Download a CSV from [knotinfo.org](https://knotinfo.math.indiana.edu/) with at least:

- knot **Name**
- **Concordance Order** (for ground-truth labels)
- numeric invariants you care about (signature, 4-genus, τ, Rasmussen *s*, etc.)

Place it at:

```
data/knotinfo.csv
```

On startup the experiment prints a **column mapping** (internal key → CSV header). Fix your export or extend patterns in `knotinfo_experiment/columns.py` if required columns are unmatched.

### 2. Run the experiment

```bash
python run_knotinfo_experiment.py --output-dir output
```

Typical options:

```bash
python run_knotinfo_experiment.py \
  --csv data/knotinfo.csv \
  --output-dir output \
  --k-max 4 \
  --epsilon-grid 0.5,1.0,1.5,2.0,2.5,3.0 \
  --top-n-plots 3
```

The run is fully offline once the CSV is in place.

## Standalone Ball Mapper

Use `ball_mapper.py` directly on any point cloud:

```python
import numpy as np
from ball_mapper import BallMapper

X = np.random.randn(500, 3)
bm = BallMapper(eps=1.0, method="greedy")
result = bm.fit(X, color=X[:, 0])

print(result.num_nodes, len(result.edges()))
for simplex, _ in result.simplices():
    print(simplex)

# Plot: for 2D data, point cloud + ε-balls + nerve graph; otherwise nerve only
result.plot(show=True)
result.plot(savepath="output/ball_mapper.png", show=False)
```

Demo on a noisy circle (same plot via CLI):

```bash
python ball_mapper.py --eps 0.35 --n-points 200 --plot
```

**Algorithm (greedy ε-net):**

1. Select landmark centers until every point lies within ε of some center.
2. Assign each point to the balls that cover it.
3. Build the nerve with GUDHI's `NerveComplex`.

## Interactive numeric explorer

Explore Ball Mapper on scalar invariants from a KnotInfo numeric export:

```
data/knotnumericinvariants.csv
```

Run:

```bash
python plot_numeric_bm.py
# or
python plot_numeric_bm.py --csv data/knotnumericinvariants.csv --max-knots 800 --eps 1.5
```

The matplotlib window lets you:

- pick **1–4 numeric invariants** (Signature, Genus-4D, τ, Nu samples, Epsilon, etc.),
- set **ε** (ball radius in standardized units) and **max knots** (subsample cap; slider goes up to the full CSV size),
- color nerve nodes by **Fibered** or **L-space** (blue = yes/Y, orange = no/N; uniform = gray),
- toggle **|chiral|** for `|signature|`, `|τ|`, `|Rasmussen s|`, and Nu samples,
- click **Compute** to refresh the graph, **Save PNG** to export.

**Parsing notes:**

- Exact numeric cells are used as features; KnotInfo **intervals** like `[2;3]` on scalar columns (Unknotting Number, Genus-4D, etc.) are treated as **missing**, not split into components.
- Only the **Nu** column is parsed as a two-sample vector (`Nu[0]`, `Nu[1]`).
- **Epsilon** in the CSV is Hom's knot Floer invariant ϵ ∈ {−1, 0, 1} — not the Ball Mapper radius slider.

**Reading the graph:**

- Each **node** is one ε-ball (landmark), not one knot; **size** ∝ knots in that ball.
- **Layout** is an abstract spring layout of the nerve, not positions in invariant space.
- Large ε on thousands of knots yields few landmarks — expected Ball Mapper behavior.

## Polynomial concordance graphs

Build three **separate** Ball Mapper nerve graphs — one per polynomial — from a KnotInfo export with Alexander, Jones, and the **HOMFLY** column (HOMFLYPT in variables `v`, `z`):

```
data/knotpolyconc.csv
```

Each knot is a point in that polynomial's coefficient space (shared fixed exponent grid, zero-padded). Balls are colored by the **mode** concordance order among knots they cover. The same concordance color map is used across all three figures.

Run:

```bash
python run_polyconc_bm.py
# or tune ε per graph (standardized units by default):
python run_polyconc_bm.py \
  --eps-alexander 1.5 \
  --eps-jones 2.0 \
  --eps-homfly 2.5 \
  --output-dir output
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--csv` | `data/knotpolyconc.csv` | KnotInfo export path |
| `--eps-alexander` | `1.5` | ε for Alexander graph |
| `--eps-jones` | `2.0` | ε for Jones graph |
| `--eps-homfly` | `2.5` | ε for HOMFLYPT graph |
| `--standardize` / `--no-standardize` | standardize on | Z-score coefficient columns before Ball Mapper |
| `--max-knots` | (all rows) | Optional head cap for quick tests |
| `--seed` | `0` | Nerve layout seed |

**Parsing notes:**

- Uses the CSV **HOMFLY** column only (not any separate homflypt/links column).
- Alexander, Jones, and the `v`-part of HOMFLYPT are **Laurent** polynomials (negative exponents, division form like `4/t^7`).
- Alexander is symmetrized to Δ(t)=Δ(1/t) before coefficients are extracted (KnotInfo gives Δ up to ±t^k).
- Each polynomial has its **own** exponent grid; features are **not** concatenated across polynomials.
- Concordance labels come from **Concordance Order (Alg.)** when that is the exported column (or any header matching `concordance order`).

**Outputs** (`output/figures/`):

| File | Description |
|---|---|
| `polyconc_alexander_eps*.png` | Alexander coefficient Ball Mapper |
| `polyconc_jones_eps*.png` | Jones coefficient Ball Mapper |
| `polyconc_homfly_eps*.png` | HOMFLYPT coefficient Ball Mapper |
| `polyconc_combined.png` | Side-by-side comparison |

Knots that fail to parse or have missing polynomial values are excluded **per graph** and listed in `output/polyconc_parse_failures.csv` when any failures occur.

## Experiment design

### Feature pools

| Pool | Internal keys (examples) | Role |
|---|---|---|
| Concordance | `signature`, `smooth_4genus`, `topological_4genus`, `tau`, `rasmussen_s`, `concordance_genus`, `arf`, `upsilon_*` | Signal — should track concordance if informative |
| Control | `crossing_number`, `three_genus`, `bridge_number`, `braid_index`, `braid_length`, `determinant`, `unknotting_number` | Confound — complexity proxies |

Subset search (default): all non-empty concordance subsets up to size `k_max`, each paired with a **matched control** of the same dimensionality (first *k* control features in fixed order).

Override with explicit subsets:

```bash
python run_knotinfo_experiment.py \
  --explicit-subsets '[["signature","tau"],["crossing_number","three_genus"]]'
```

### Ground-truth labels

**`label_mode=order`** (default): buckets from the concordance-order column:

`slice/order-1`, `order-2`, `order-4`, `infinite-order`, `unknown`

Knots labeled `unknown` are **excluded from scoring** but still appear in plots.

**`label_mode=tuple`**: label = exact tuple of invariants (comma-separated via `--label-tuple`). Subsets that overlap the label tuple are skipped to avoid circular scoring.

### Parsing and chirality

- Empty cells, bounds (`< 4`, `≥ 2`), and non-exact values are treated as **missing** — never coerced to numbers.
- With `use_absolute_values=True` (default), `|signature|`, `|tau|`, `|rasmussen_s|`, and `|upsilon|` are used so mirror choices in KnotInfo do not split mirror-symmetric classes.

### Per-subset pipeline

For each subset *S*:

1. Keep only knots with **all** features in *S* present (completeness filter recomputed per subset).
2. **Z-score** each coordinate (ε is reported in standardized units).
3. Run Ball Mapper for each ε in the grid.

### Scoring

Ball Mapper returns a **cover**, not a partition. Scoring uses **connected components** of the nerve graph:

- each knot inherits the component of its covering ball(s) (majority vote if needed),
- primary metrics: **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)** vs. concordance order,
- purity and majority-class base rate are reported for context only (purity is inflated by class imbalance among low-crossing knots),
- per-ball label entropy is a secondary diagnostic.

The summary flags concordance subsets that **fail to beat their matched control** — that gap is the meaningful signal, not the raw score alone.

## Outputs

After a run, `output/` contains:

| File | Description |
|---|---|
| `results_ranked.csv` | All (subset, ε, n, ARI, NMI, purity, base rate, components, …) |
| `collisions.csv` | Knots sharing a ball or nearly identical feature vectors with **different** labels |
| `extremal.csv` | Knots saturating standard inequalities among selected invariants |
| `figures/bm_*_by_label.png` | Nerve graph, nodes colored by dominant concordance order |
| `figures/bm_*_by_family.png` | Same graph, colored by knot family (if exported) |

### Reading the figures

- Each **node is one ball** (ε-net landmark), not one knot. Node count is always much smaller than knot count.
- **Node size** = number of knots covered by that ball (min–max normalized for visibility).
- **Node color** = dominant label (or family) among knots in the ball.
- **Layout** is an abstract spring layout of the nerve graph, not positions in feature space.
- **Edges** connect balls that share at least one knot (nonempty intersection of cover elements).

With large ε in standardized space, few landmarks can cover thousands of knots — that is expected Ball Mapper behavior, not a bug.

## Configuration reference

CLI flags map to `ExperimentConfig` in `knotinfo_experiment/config.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--csv` | `data/knotinfo.csv` | KnotInfo export path |
| `--output-dir` | `output` | Results directory |
| `--label-mode` | `order` | `order` or `tuple` |
| `--label-tuple` | (empty) | Invariants for tuple labels |
| `--no-absolute-values` | off | Keep sign of chiral invariants |
| `--k-max` | `4` | Max concordance subset size |
| `--max-subsets` | `500` | Cap on subsets evaluated |
| `--explicit-subsets` | (none) | JSON list overriding search |
| `--epsilon-grid` | `0.5,…,3.0` | ε values in standardized units |
| `--top-n-plots` | `3` | Ball Mapper PNGs for top concordance subsets |
| `--seed` | `0` | Random seed for graph layout |

Programmatic use:

```python
from knotinfo_experiment.config import ExperimentConfig
from knotinfo_experiment.runner import run_experiment

config = ExperimentConfig(k_max=3, epsilon_grid=(0.5, 1.0, 1.5))
run_experiment(config)
```

## Project layout

```
mapper-knots/
├── ball_mapper.py              # Ball Mapper + circle demo CLI
├── plot_numeric_bm.py          # Interactive explorer for numeric invariants
├── run_polyconc_bm.py          # Alexander / Jones / HOMFLYPT polynomial graphs
├── run_knotinfo_experiment.py  # Experiment entry point
├── knotinfo_experiment/
│   ├── config.py               # ExperimentConfig and argparse
│   ├── columns.py              # Fuzzy CSV header → internal key mapping
│   ├── parse.py                # Conservative numeric / order parsing
│   ├── polynomials.py          # Laurent polynomial parsing + grids
│   ├── polyconc.py             # Polynomial Ball Mapper pipeline
│   ├── data.py                 # Load table, completeness, standardization
│   ├── labels.py               # Ground-truth construction
│   ├── subsets.py              # Subset enumeration + matched controls
│   ├── scoring.py              # BM clustering metrics (ARI, NMI, …)
│   ├── artifacts.py            # Collisions and extremal knots
│   ├── visualize.py            # Nerve graph figures
│   └── runner.py               # Full experiment orchestration
├── data/
│   ├── knotinfo.csv            # KnotInfo export for concordance experiment
│   ├── knotnumericinvariants.csv  # Numeric invariant export for interactive explorer
│   └── knotpolyconc.csv        # Polynomial + concordance export for polyconc graphs
├── output/                     # Generated results (gitignored)
└── requirements.txt
```

## Extending column mapping

KnotInfo headers vary between exports. Patterns live in `knotinfo_experiment/columns.py` (`SCALAR_KEY_PATTERNS`, `UPSILON_COLUMN_PATTERNS`). After each run, check the printed mapping for unmatched pool keys or unmapped CSV columns.

Required columns: **name**, **concordance_order**. Missing required mappings raise a clear error rather than silently dropping data.

## References

- P. Dłotko, [Ball mapper: a shape summary for topological data analysis](https://arxiv.org/abs/1901.07410), arXiv:1901.07410.
- [GUDHI cover complex documentation](https://gudhi.inria.fr/python/latest/cover_complex_sklearn_user.html).
- [KnotInfo](https://knotinfo.math.indiana.edu/) — knot invariant tables.
