# Published measurements and provenance

`results.json` holds the measurements quoted in the README. Both come from
campaigns run under the documented envelope: one thread, 2 GiB address space,
64 MiB stack, core dumps disabled, nice 10, PYTHONHASHSEED=0.

## Provenance

- `homotopy` — 32 cells of the H group from campaign
  `runs/field_ratio_20260929T160316Z`, one process per cell, stop at first
  failure. Each cell is an independent IPOPT solve with an exact Hessian; the
  four writings of one field ratio share the same optimum by construction.
- `same_chain` — one run of `jax_same_chain.py --mechanism`, both frameworks in
  the same process on the same seeded rig, parity gated at 1e-9 before timing.

Machine-dependent timings drift between runs: the Hessian speedup of the
same-chain bench was measured at 57x to 111x across runs on a shared machine,
with the same parity. The structure of the result is stable; the last digit is
not. Reproduce with:

```bash
python jax_same_chain.py --mechanism --output results.json
```

`sha256sums.txt` verifies the files shared here.
