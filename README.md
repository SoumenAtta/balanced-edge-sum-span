# Computational artifacts

This directory contains the implementation and solver outputs used for the
small exact computations reported in the manuscript.

## Contents

- `scripts/compute_beta_gurobi.py`: Gurobi implementation of the integer
  program for the balanced edge-sum span.
- `solver_outputs/beta_table.csv`: summary of all reported instances.
- `solver_outputs/beta_width4_gap.csv`: summary of the width-four instances
  `P_4 x P_n` for `5 <= n <= 8`.
- `solver_outputs/json/`: full machine-readable solutions, including the
  vertex labeling, all edge sums, objective value, lower bound, runtime, and
  final mixed-integer programming gap.

## Requirements

- Python 3.9 or later
- Gurobi Optimizer with a valid license
- the `gurobipy` Python package

## Reproduce the complete table

Run from this directory:

```bash
python3 scripts/compute_beta_gurobi.py batch \
  --cases grid:3:3 grid:3:4 grid:4:4 grid:4:5 grid:4:6 grid:4:7 grid:4:8 \
          torus:3:3 torus:4:4 torus:5:5 \
  --csv solver_outputs/beta_table.csv \
  --json-dir solver_outputs/json \
  --quiet
```

## Reproduce the width-four certifications

```bash
python3 scripts/compute_beta_gurobi.py batch \
  --cases grid:4:5 grid:4:6 grid:4:7 grid:4:8 \
  --csv solver_outputs/beta_width4_gap.csv \
  --json-dir solver_outputs/json \
  --time-limit 1800 \
  --quiet
```

A reported value is exact only when the status is `OPTIMAL`.  In the supplied
outputs, `beta` equals `best_bound` and `mip_gap` is zero for every
instance.
