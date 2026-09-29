# CasADi — derivative cost, shared design and sparse operators

Small, reproducible experiments on exact derivatives and nonlinear optimisation.
Each study is standalone: its own README, its own code, its own measurements.

| Study | Question |
|---|---|
| [01 — Derivative cost](01-derivative-cost/README.md) | Why is a cheap dense solve expensive to differentiate twice? |
| [02 — Shared design](02-design-control-coupling/README.md) | How does a shared design block connect otherwise separate conditions? |
| [03 — Control rank](03-control-rank/README.md) | Does lifting the operator pay, as the number of controls varies? |
| [04 — Field ratio](04-field-ratio/README.md) | When does it stop paying, and what is a compiled backend worth? |

Coming from Joris Gillis's sparse-operator note: start with
[04-field-ratio](04-field-ratio/README.md). It runs the homotopy he proposed,
from his kernel to a field that no longer fits, and times the same lifting-line
chain in CasADi and in JAX with parity gated first.

## Running

Python ≥ 3.11, CasADi ≥ 3.7 with IPOPT/MUMPS, NumPy. Studies 01/02 add
Matplotlib; study 04 adds JAX for its comparison benches. Versions used for the
published numbers are pinned in the `requirements.txt` of studies 03 and 04.

```bash
(cd 01-derivative-cost && python example.py)
(cd 02-design-control-coupling && python example.py)
```

Studies 03 and 04 document their own reproduction commands. Benchmarks run in a
fixed envelope: one thread, 2 GiB, nice 10. Published numbers live in each
study's `validation/`, new runs in `runs/`; a campaign never overwrites an old
one, and nothing runs merely by importing a module.

## Attribution

Study 03 reproduces **Joris Gillis's "Lifting the operator, not the solve"**. The
files received from him are preserved unchanged and credited in
[reference/README.md](03-control-rank/reference/README.md).
