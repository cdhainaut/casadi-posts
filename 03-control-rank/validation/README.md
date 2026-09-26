# Preserved measurements and validation

`results.json` contains the **16 J0/J1 cases**, imported from campaign
`joris_controls_j0_j1_20260926T092000Z`. These are existing measurements,
**not a new campaign executed while organizing this repository**.

## Provenance

- Numerical `result` objects were preserved exactly, without rounding,
  precision changes or metric renaming.
- The export removes only machine-specific command paths and redundant child
  return codes. Statuses, resource envelope, checks and results are retained.
- `provenance.json` records the source archive identifier and SHA-256 hashes
  of its campaign manifest, executed sources and 16 individual JSON files.
- `sha256sums.txt` verifies the data files shared here. Complete raw logs remain
  in the original archive; they are not needed to run the reproducer.
- `run_joris_controls.py` is byte-for-byte identical to the measured driver.
  `bench_joris_controls.py` differs only in a corrected documentation link.
  Tests also verify this correspondence.
- The source and JSON received from Joris are unchanged in `../reference/`.

```bash
cd validation
sha256sum -c sha256sums.txt
```

## Checks fixed before measurement

- `Solve_Succeeded` status for every case.
- `LK−I` error ≤ 1e-10 at three admissible designs.
- Initial primal violation ≤ 1e-8.
- Final unscaled residuals: primal ≤ 1e-6, stationarity ≤ 1e-4,
  complementarity ≤ 1e-4. These are external checks, not IPOPT settings.
- Objective agreement at fixed rank: `rtol=1e-8`, `atol=1e-7`.
- At J0: agreement with Joris's archived objective, dimensions and J/H nonzero
  counts. Timings and graph node counts were not acceptance criteria.

All 16 cases passed. Final primal residuals reach about 2e-7 and complementarity
about 4.2e-5, reflecting IPOPT's bound relaxation among other effects. Do not
present them as below `tol=1e-8`. Equal objectives establish neither design
uniqueness nor global optimality.

## Reading the metrics

- `solve_seconds`: Python `opti.solve()` time, including preparation;
  `model_build_seconds` separately measures model construction.
- `hessian_triangle_nnz`: the triangle supplied to IPOPT, not the full symmetric
  Hessian.
- `hessian_mean_seconds`: mean H evaluation time during the solve, using each
  trajectory's own points and multipliers.
- `hessian_initial_seconds`: three warmed calls at the initial point, objective
  factor 1 and all constraint multipliers set to 1. External Python/CasADi call
  overhead is included; this is not the same measurement as the IPOPT mean.
- `peak_rss_kib`: Linux peak RSS of the entire process, including post-solve
  checks and measurements. Neither peak virtual address space nor factor size.
- `solution`: objective, parameters, station-wise controls and KKT residuals
  computed using the actual solver functions.

KKT factorization is not timed separately. The difference between total time
and Hessian evaluation time must not be attributed to it.

## Revalidate without a new campaign

From `03-control-rank/`:

```bash
python -m pytest tests/test_reference_results.py -q
```

These tests reread the data, apply the original acceptance criteria, check that
all 16 expected cases are present and verify reference-source hashes.
Use a fresh `runs/` directory for a new benchmark; do not replace the published
measurements with a new execution.
