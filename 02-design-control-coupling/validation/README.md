# Validation for the second post

Scripts and raw numbers behind the tables of the second article.

| file | what it does |
|---|---|
| `figures.py` | draws the cost figure from `results.json` |
| `results.json` | layouts, sizes, nonzeros, graph nodes and evaluation times |
| `patterns.npz` | raw sparsity patterns |
| `figures/` | the cost figure |

```bash
python figures.py
```

The three layouts measured there include a frozen design, the case where each
condition stands alone.
