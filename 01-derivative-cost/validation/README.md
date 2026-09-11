# Validation for the first post

Scripts and raw numbers behind the tables of the first article.

| file | what it does |
|---|---|
| `check_equivalence.py` | builds the dense kernel two ways and compares values, gradient and Hessian over four design points |
| `writings.py` | the same solve written five ways — elimination, closure constraints, implicit rootfinder, frozen design, linear output — with their graph sizes and evaluation times |
| `scaling.py` | sweeps the system size from `N = 12` to `192` with three control objects and writes `scaling.json` |
| `figures.py` | draws the cost and pattern figures from the JSON files |
| `results.json`, `scaling.json`, `equivalence.json` | raw numbers |
| `patterns.npz` | raw sparsity patterns |
| `figures/` | the figures produced from those numbers |

```bash
python check_equivalence.py   # ~10 s
python writings.py            # ~10 s
python scaling.py             # ~5 min
python figures.py             # ~30 s
```

Times vary by tens of percent between runs on a shared machine; graph sizes and
nonzero counts do not.
