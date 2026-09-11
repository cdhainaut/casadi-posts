# Sharing one design block across every condition

*A CasADi post about the structure that appears when design parameters and
trajectory variables live in the same nonlinear program.*

## The situation

A multidisciplinary optimisation problem usually looks like this:

- a handful of **design parameters** that describe the system;
- a number of **conditions** — operating points, time steps, load cases — each
  with its own **controls** and its own **state**;
- one model per condition whose matrix is dense and depends on both the shared
  design and the local variables.

Writing all of it as a single NLP is the natural thing to do: the optimiser
explores the design space and the trajectory at the same time. It also changes the
shape of the derivatives in a way that is easy to miss.

## The example

`example.py` reuses the model of the first post — a dense matrix built from
coordinates that depend on the design parameters, a response, and a non-linear
output — and repeats it over `K` conditions:

```python
coordinates = (
    locator
    + 0.12 * design_k[0] * (1.0 - locator**2)
    + 0.08 * design_k[1] * locator * (1.0 - locator**2)
)
span = cas.reshape(coordinates, N, 1)
difference = cas.repmat(span, 1, N) - cas.repmat(span.T, N, 1) + cas.MX.eye(N)
kernel = ((coordinates[1] - coordinates[0]) / (4.0 * np.pi)) / difference
kernel = kernel - cas.diag(cas.diag(kernel))
```

Three writings of the same problem are compared:

| writing | design parameters |
|---|---|
| `design shared` | one block, used by every condition |
| `design per condition` | one independent block per condition |
| `design frozen` | constants |

The first two have identical local models and therefore identical differentiation
work per condition. The only difference is the sharing.

## What the measurements say

![patterns](figures/hessian_patterns.png)

| writing | variables | Hessian nonzeros | density | Hessian graph | one evaluation |
|---|---:|---:|---:|---:|---:|
| design shared | 55 | 673 | 44 % | 7 934 | 0.69 ms |
| design per condition | 63 | 693 | 34 % | 7 944 | 0.68 ms |
| design frozen | 55 | 459 | 30 % | 2 488 | 0.34 ms |

Three observations:

1. **Sharing couples everything.** With one shared block, the design columns run
   through the whole Hessian: every condition is connected to every other one
   through the design. With one block per condition, the Hessian keeps its block
   structure. The nonzeros are almost the same; the connectivity is not — and
   connectivity is what a sparse factorisation pays for.
2. **Sharing does not cost more to differentiate.** Both writings walk the same
   graph (7 934 and 7 944 nodes), because the local model is the same. Sharing
   even removes eight variables. The structural coupling is not a hidden
   differentiation cost.
3. **Depending on the design is what costs.** Freezing the design parameters cuts
   the graph by 3.2 and the evaluation time by half. That is the same effect the
   first post isolated, seen from the other side.

![cost](figures/cost.png)

## What to take away

Putting the design in the same NLP as the trajectory is a structural choice, not
a performance trap:

- you gain a single joint problem, solved once, with the coupling handled by the
  KKT system;
- you pay a denser Lagrangian Hessian (44 % against 34 % here) and wide design
  columns. Order the variables so that the design block comes first or last, and
  the sparsity pattern stays as close to block-arrow as it can be;
- you do not pay extra differentiation: as long as the model depends on the
  design, the derivative work is the same whether the design is shared or local.

What is *not* measured here: how that connectivity translates into the cost of the
KKT factorisation inside a given NLP solver. The graph is only one of the two
costs; the linear algebra is the other.

## Where this shows up

This is the standard shape of multidisciplinary optimisation: one set of design
variables, many conditions, and a model that couples them.

- **Aerospace sizing.** Structural thicknesses shared by several load cases, each
  with its own aeroelastic state. The all-at-once (SAND) architecture keeps
  everything in one problem; the multidisciplinary-feasible (MDF) architecture
  solves each discipline in an inner loop instead.
- **Power systems.** Equipment ratings (design) with hourly dispatch profiles
  (controls) for a whole year of representative days.
- **Robot co-design.** Link lengths and actuator sizes optimised together with
  the trajectories the robot will execute.
- **Circuit design.** Component sizes tuned across several operating points —
  temperature, supply voltage, load — so the design block is shared by every
  corner.
- **Process design.** Reactor volume and feed temperatures chosen jointly with
  the operating profile over a batch.
- **Building energy.** Equipment sizing plus usage schedules, where the schedule
  is the controller's decision on top of a fixed design.

The naming is old and stable: optimising analysis variables and design variables
as one problem is *simultaneous analysis and design* (SAND), the alternative being
to solve the analysis inside an outer design loop.

## Reproduce

```bash
python example.py     # results.json + patterns.npz
python figures.py     # figures/hessian_patterns.png + figures/cost.png
```

Times are single-thread medians on a shared workstation and move by about 20 %
between runs; the graph sizes and the ranking do not.

Remember the first lesson of the companion post: write the dense kernel as a
matrix expression. On this model, the scalar-loop version measured 235 000 nodes
against 7 900 here — identical numbers, thirty times the work.

## References

- Haftka, R. T., "Simultaneous Analysis and Design," *AIAA Journal*, Vol. 23,
  No. 7, 1985, pp. 1099–1103. [doi:10.2514/3.9043](https://doi.org/10.2514/3.9043)
- Martins, J. R. R. A., and Lambe, A. B., "Multidisciplinary Design Optimization:
  A Survey of Architectures," *AIAA Journal*, Vol. 51, No. 9, 2013, pp. 2049–2075.
  [doi:10.2514/1.J051895](https://doi.org/10.2514/1.J051895)
