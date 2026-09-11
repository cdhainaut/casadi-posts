# Validation for the second post

The article shows what sharing a design block across conditions does to the
sparsity of the Lagrangian Hessian. This folder holds the measurements taken while
establishing it.

| file | what it does |
|---|---|
| `figures.py` | draws the cost figure from `results.json` |
| `results.json` | layouts, sizes, nonzeros, graph nodes and evaluation times |
| `patterns.npz` | raw sparsity patterns |
| `figures/` | the cost figure |

```bash
python figures.py
```

The three layouts measured there include a frozen design — the case where each
condition stands alone — which the article discusses but does not plot.
