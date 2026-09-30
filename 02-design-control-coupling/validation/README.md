# Records for the three design layouts

The current records use N=16 and K=5. `example.py` is their generator, including
the frozen-design case with no inactive design variables.

- `results.json`: dimensions, sparsity counts, graph sizes, primal points,
  multipliers and raw evaluation-time samples.
- `patterns.npz`: lower-triangle Hessian patterns and contiguous block metadata.
- `provenance.json`: parameters, versions and measured generator hash.
- `figures.py`: redraws derivative-pattern and cost figures from these records.

Times are medians of 20 warmed evaluations; these are derivative measurements,
not NLP solves. The reference execution used one numerical thread, nice 10,
2 GiB address space, 64 MiB stack and disabled core dumps. No between-run
variation is estimated.

From `02-design-control-coupling/`:

```bash
python -m pytest -q
python validation/figures.py
(cd validation && sha256sum -c sha256sums.txt)
```

To regenerate the numerical records, use `example.py --output` and a fresh
path as in the [main README](../README.md#reproduce). Do not overwrite this
published snapshot. `legacy/` preserves the earlier K=3 artifacts separately.
