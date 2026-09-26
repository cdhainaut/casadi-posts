# Control rank — a single-scenario sanity check

Reproduction of **Joris Gillis's** example, followed by a change in the number
of independent controls within one scenario. This is a standalone mathematical
benchmark with no domain-model dependency.

**All 16 cases converged and passed the predefined checks.** The benefit of the
sparse operator is reproduced. Reducing the control rank reduces the additional
benefit of lifting in this example. The structural explanation is elementary:
a smaller reduced decision space has a smaller reduced Hessian. This experiment
is a harness check and a quantitative ablation, not a new theoretical result or
a representative multipoint benchmark.

See the [discussion and next questions](DISCUSSION.md) and
[measurement provenance](validation/README.md).

## 1. What stays the same, what changes

Joris's exponential kernel is exactly the inverse of a tridiagonal operator:
`K(p)=L(p)⁻¹`. There are four design parameters `p`, a scalar state `v`, spatial
controls `u`, a response `y`, and its observation `z`:

```text
D = diag(ω w(p) / v)
b = ω w(p) ⊙ (u + 0.1 s)
(I + D K) y = b
z = K y
```

The nonlinear local objective, bounds and initial design/state values follow
the [original source](reference/sparse_operator.py) and
[note](reference/sparse_operator.pdf). Only the controls become `u=B_r a`:

- `r=N`: one control per station, `B_N=I`;
- `r=4`: the first four Legendre polynomials evaluated at the stations;
- `r=1`: one spatially constant control.

The bounds `−1 ≤ u_i ≤ 1` still apply at the stations. Reducing the rank changes
the feasible set; it is not just a coordinate transformation. Lifted responses
are initialized by solving at the common initial design/state/control point.

| Formulation | Response added to the NLP | Equation |
|---|---|---|
| Dense eliminated | none | `y=solve(I+DK,b)`, then `z=Ky` |
| Dense lifted | `y` | `(I+DK)y=b`, then `z=Ky` |
| Sparse eliminated | none | `z=solve(L+D,b)` |
| Sparse lifted | `z` | `(L+D)z=b` |

The four formulations are algebraically equivalent **at a fixed control rank**.

## 2. J0 — reproduction with 192 stations and 192 controls

| Formulation | Variables | nnz H (triangle) | Mean H/call | `opti.solve()` time |
|---|---:|---:|---:|---:|
| Dense eliminated | 197 | 19,503 | 1,386.22 ms | 28.759 s |
| Dense lifted | 389 | 57,517 | 1,295.13 ms | 29.676 s |
| Sparse eliminated | 197 | 19,503 | 20.83 ms | 1.079 s |
| **Sparse lifted** | **389** | **2,119** | **0.142 ms** | **0.151 s** |

Common objective: **−4213.93302773**, consistent with Joris's reference.
Dimensions and J/H nonzero counts match his table. Graph node counts do not
all match his historical JSON; this was not a validation requirement.
J0 uses the adapted measurement harness, not an uninstrumented execution of
the original script.

## 3. J1 — 42 stations, then 42, 4 and 1 control

`opti.solve()` time, including solver preparation:

| Formulation | 42 controls | 4 controls | 1 control |
|---|---:|---:|---:|
| Dense eliminated | 0.231 s | 0.165 s | 0.257 s |
| Dense lifted | 0.257 s | 0.218 s | 0.318 s |
| Sparse eliminated | 0.141 s | 0.140 s | 0.123 s |
| Sparse lifted | 0.130 s | 0.108 s | 0.118 s |

To isolate **lifting**, compare sparse eliminated against sparse lifted:

| Controls | nnz H eliminated / lifted | Mean H/call eliminated / lifted | Mean H speedup |
|---|---:|---:|---:|
| 42 | 1,128 / 469 | 0.798 / 0.035 ms | ×22.7 |
| 4 | 45 / 449 | 0.143 / 0.060 ms | ×2.4 |
| 1 | 21 / 305 | 0.107 / 0.054 ms | ×2.0 |

With one control, the reduced Hessian has only 21 triangular entries: less
remains to be gained by exposing 42 additional response unknowns. Nevertheless,
even a small reduced Hessian can be expensive to evaluate through a dense solve.
The sparse operator remains useful: with one control, dense eliminated takes
0.257 s versus 0.123 s for sparse eliminated.

Times are not monotonic in control count: dense eliminated takes 12 iterations
with 42 controls, but 30 with one. The optimization problems differ across ranks.

## 4. What this does not say about multipoint problems

**One control per scenario is not one control for the entire problem.** With
`T` scenarios and `q` independent controls per scenario, there are `Tq` controls,
plus states and any shared design variables. The rank `r` tested here describes
spatial control freedom within a single scenario; every measured case has `T=1`.

Nor are `Tq` controls in separate scenarios equivalent to `Tq` controls acting
through one common spatial influence problem. For a separable multipoint model
with fixed design, the reduced Hessian has independent scenario blocks. Shared
optimized design introduces a common border; trajectory constraints introduce
additional coupling. The whole Hessian need not become dense.

A representative next comparison must retain actual local controls and states,
scenario count, geometry dependence and inter-scenario constraints. It must
measure total NLP cost and memory, not infer them from the Hessian size alone.

## 5. Measurement limits

- One run per case; the four formulations agree in objective at each rank,
  not necessarily in optimal design parameters.
- Mean H/call is measured along each solver trajectory. At the common initial
  point, sparse-eliminated/lifted ratios are instead ×8.0, ×1.4 and ×1.0
  (median of three warmed calls, fixed multiplier convention). These external
  timings include Python/CasADi call overhead.
- Millisecond differences between small solves are not statistically established
  advantages. `opti.solve()` includes preparation; KKT factorization alone is
  not instrumented.
- One field unknown per station and one scenario only. No larger auxiliary 2D
  field, multipoint problem or trajectory optimization has been measured here.
- These measurements do not establish a universal formulation choice or a
  crossover size.

## 6. Reproduce from this repository

From `03-control-rank/`, in a Python ≥ 3.11 environment:

```bash
python -m pip install -r requirements.txt
```

Measured versions: Python 3.11.15, NumPy 2.4.6, CasADi 3.7.2,
IPOPT 3.14.11 / MUMPS 5.4.1. Backend versions may depend on the installed
package; they are printed in each log.

Reference Linux resource envelope and full campaign:

```bash
ulimit -v 2097152
ulimit -s 65536
ulimit -c 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
nice -n 10 python run_joris_controls.py \
  --reference-source reference/sparse_operator.py \
  --reference-results reference/results.json \
  --output runs/reproduction_NEW
```

For one case only, under the same envelope:

```bash
nice -n 10 timeout 300s python bench_joris_controls.py \
  --stations 42 --controls 4 --kernel sparse --lifted \
  --output runs/single_NEW.json
```

Use fresh output paths. The driver starts a new process per case, applies a
300-second timeout, saves sources/logs/results and stops at the first failure
without retrying. Do not run campaigns in `reference/` or `validation/`: these
hold the preserved measurements. New `runs/` outputs are ignored by Git.
No path to another repository is required.

## 7. Quick checks

Install development tools if needed:

```bash
python -m pip install pytest ruff
# Same envelope; smoke tests start four small solves.
nice -n 10 timeout 180s python -m pytest -x -q
python -m ruff check .
```

**Import validation: 30 tests passed**, both in this repository and in an
isolated copy without another checkout or `PYTHONPATH`. Ruff and checksum
checks also passed. No new J0/J1 campaign was run during the import.

Tests cover control bases, original-source parity, reduced derivatives and
finite differences, all four CLI formulations outside this directory,
stop-on-first-failure behavior and consistency of the 16 imported results.
To avoid all solves: `python -m pytest -m 'not slow' -q`.

The harness preserves the measured campaign options and checks. The only
benchmark-code change during import was a docstring link. See
[validation/README.md](validation/README.md) for provenance and KKT criteria.
