# Measurements and provenance

`results.json` contains complete result objects, including solver status, KKT
residuals, derivative timings and validation errors. `provenance.json` identifies
the campaigns, measured source hashes and export decisions. Checksums verify
the files shipped here; they are not a proof of scientific validity.

## Correction of the same-chain comparison

The previous 70x Hessian claim is withdrawn. Its RBF Reynolds inputs were around
1.4 million while the centres were near zero. The forward outputs, Jacobian
and Hessian were all numerically zero. Timed JAX calls were also unsynchronized.
`invalidated_results.json` preserves the original published record exactly.
Its homotopy results are not invalidated by these two same-chain defects.

The current same-chain record comes from a new run with normalized RBF inputs,
active controls, synchronized evaluations and three-point parity. The optional
C comparison checks the same Hessian before timing. Raw timing samples, CPU,
affinity, versions, resource limits and measured-source hash are recorded.
The measured Hessian is 1.354 ms in CasADi and 1.380 ms in JAX; no advantage is
established on this CPU example. Neither the old gains nor their alleged
mechanism are retained as conclusions.

## Historical field-size campaign

The exported campaign `field_ratio_20260929T160316Z` contains 32 validated H
cases, three J cases and nine T cases including the stopping case. H is the
field-size sweep, J the batched dense AD comparison, T the condition-count
sweep. H completed, but the combined campaign stopped at T=32 when dense
eliminated exceeded the complementarity limit. J at T=32 records a JAX memory
failure, not a valid timing.

`homotopy`, `batched_dense` and `additional_conditions` retain these outcomes
separately. Flags for different stationary points at z=64/128 are preserved.
Machine-specific command paths are omitted; numerical result objects are not
rounded. The measured source snapshots are in `sources/`, identified by SHA.
They are historical evidence, not alternative maintained implementations.

The envelope was one numerical thread, 2 GiB address space, 64 MiB stack,
core dumps disabled, nice 10 and PYTHONHASHSEED=0. The new same-chain run records
its actual envelope and CPU affinity. Times are machine-dependent and there
is one run per case; within-run samples do not establish between-run variation.

## Revalidation and new runs

From `04-field-ratio/`, use the envelope and commands in the
[main README](../README.md#reproduce). New output paths belong in `runs/`.
Do not run benchmarks inside `validation/`, and do not overwrite these records.

```bash
python -m pytest tests/test_published_records.py -q
(cd validation && sha256sum -c sha256sums.txt)
```

These checks revalidate archived records and source hashes without starting a
new timing campaign. Regenerate `crossover.png` with `python plot_results.py`.
