# Validation for the first post

Scripts and raw numbers behind the tables of the first article.

| file | what it does |
|---|---|
| `check_equivalence.py` | builds the dense kernel two ways and compares values, gradient and Hessian over four design points |
| `writings.py` | elimination, closure constraints and rootfinder, plus frozen-design and linear-output ablations |
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

Times depend on the machine and load. These scripts overwrite the adjacent
JSON/NPZ files. Run a copied directory to preserve published measurements.
The lifted Lagrangian has additional variables and multipliers; its Hessian is
not the reduced Hessian without a corresponding elimination.
