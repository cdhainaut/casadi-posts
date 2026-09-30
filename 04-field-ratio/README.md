# Field size and exact derivative cost

How much does an auxiliary field cost when the response model stays unchanged?
This experiment enlarges a tridiagonal field while keeping its compressed
Green function exactly equal to the original dense kernel.

For the measured case, N=42 and one condition, sparse lifting loses its
whole-solve advantage between `z=8` and `z=16`. Its mean Hessian callback remains
cheaper until somewhere between `z=32` and `z=64`. These are observations from
one implementation and one run per case, not universal crossover sizes.

## The discrete model

The response model follows [Joris Gillis's example](../03-control-rank/reference/README.md):
four design parameters, one state, 42 controls and a nonlinear local output.
Only the auxiliary field size changes:

```text
M = z N
K(p) = S.T L(p)^-1 S,   L tridiagonal
D = diag(omega w(p) / v)
(I + D K)y = b
(L + S D S.T)f = S b,   observed response = S.T f
```

Choosing the field decay and scale as in `field_constants()` preserves the
compressed kernel. At `z=1`, the operator is the reference operator. Bounds on
design, controls and state are identical between formulations; lifted fields
have no extra bounds and start at the corresponding eliminated solution.
The numerical compression check at three admissible designs stays below 1e-10.

Four formulations are measured: dense or sparse operator, response/field
eliminated or lifted. They describe the same discrete problem, but their solver
trajectories and Lagrangian Hessians differ.

## Hessian cost is not solve time

![Hessian callback and complete solve against field ratio](crossover.png)

| z | H sparse lifted (ms) | H dense lifted (ms) | Solve sparse lifted (s) | Solve dense lifted (s) |
|---|---:|---:|---:|---:|
| 1 | 0.069 | 3.932 | 0.095 | 0.166 |
| 8 | 0.280 | 3.772 | 0.113 | 0.157 |
| 16 | 0.784 | 3.571 | 0.158 | 0.147 |
| 32 | 2.859 | 3.415 | 0.371 | 0.141 |
| 64 | 10.994 | 3.417 | 1.343 | 0.146 |
| 128 | 43.105 | 3.438 | 5.830 | 0.150 |

H is the mean callback time along each IPOPT trajectory. The solve time is the
wall time of `opti.solve()`, including preparation; KKT factorisation is not
isolated. Small differences between these single runs are not statistically
established advantages.

All four formulations reach an objective of about -921.79785 up to `z=32`.
At `z=64` and `128`, sparse lifted reaches different stationary points:

| z | Sparse-lifted iterations | Objective |
|---|---:|---:|
| 32 | 36 | -921.79785 |
| 64 | 56 | -834.37263 |
| 128 | 86 | -651.52440 |

The equations are equivalent; this non-convex solve does not necessarily reach
the same point in every formulation. Agreement in objective does not prove a
global optimum. The large-field timings therefore also reflect different
trajectories. No 3-D physical field or moving-geometry discretisation is tested.

## An active synthetic chain in CasADi and JAX

`jax_same_chain.py` implements the same geometry, regularised vortex influence,
dense solve and 250-centre Gaussian RBF map in both frameworks. Nine controls
feed six synthetic force/moment outputs. RBF coordinates are dimensionless;
the Reynolds feature is `Re/Re_reference - 1`.

Before timing, both implementations must have finite, nonzero outputs,
Jacobians and Hessians, nine active Jacobian columns, and relative parity at
1e-9 at three points. Tests also compare derivative columns with finite
differences. The Hessian is of the sum of the six outputs, not a complete NLP
Lagrangian. Each JAX evaluation is synchronized inside its timed interval.

| Quantity | CasADi (ms) | JAX (ms) | CasADi / JAX |
|---|---:|---:|---:|
| Forward | 0.156 | 0.0504 | 3.09 |
| Jacobian | 0.605 | 0.399 | 1.52 |
| Hessian | 1.354 | 1.380 | 0.98 |

These are medians of 15 warmed evaluations on one CPU core. The worst scaled
parity error is below 3e-14. This run does **not establish a Hessian speedup**.
The same CasADi Hessian compiled with `gcc -O2` takes 1.074 ms, with parity
checked before timing; compilation takes 19.7 s and is excluded from that row.
No fusion or interpretation-cost mechanism is inferred from these timings.

The chain is a computational example. Its seeded geometry and random RBF
coefficients are not validated aerodynamics, and there is no VPP solve here.
These results do not predict performance for another model or for many
conditions. See [measurement provenance and the correction notice](validation/README.md).

## Reproduce

From `04-field-ratio/`, on Linux with Python 3.11 or later:

```bash
python -m pip install -r requirements-jax.txt
ulimit -v 2097152
ulimit -s 65536
ulimit -c 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 MPLBACKEND=Agg
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

# Use a fresh directory/file each time. Replace CPU 0 if it is not allowed.
# One process per test module: JAX compilation can exhaust shared 2 GiB otherwise.
nice -n 10 timeout 240s taskset -c 0 python -m pytest tests/test_field_ratio.py -x -q
nice -n 10 timeout 240s taskset -c 0 python -m pytest tests/test_same_chain.py -x -q
nice -n 10 timeout 240s taskset -c 0 python -m pytest tests/test_published_records.py -x -q
nice -n 10 timeout 1200s taskset -c 0 python run_field_ratio.py --output runs/homotopy_NEW
nice -n 10 timeout 900s taskset -c 0 python jax_same_chain.py \
  --compile-c --output runs/same_chain_NEW.json
python plot_results.py
```

GCC is needed only for `--compile-c`; omit that flag to compare CasADi and JAX
alone. NumPy/CasADi suffice for the homotopy. `requirements.txt` also includes
Matplotlib for the figure; JAX is installed by `requirements-jax.txt`.

The campaign defaults to the H field-size sweep. Optional `--group J` runs the
batched dense derivative experiment; `--group T` runs the condition-count
experiment. The historical J/T results and failed cases are preserved in
`validation/results.json`, not promoted to a successful complete campaign.
In that campaign the full dense JAX Hessian ran out of memory at T=32, and the
T=32 dense-eliminated solve failed the complementarity gate. No general claim
against JAX AD follows from that implementation-specific result.
