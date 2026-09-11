# A sparse Hessian can still be expensive

*A CasADi post about nested dense systems: what makes their derivatives costly,
and what only looks like it does.*

## The situation

Many models hide a small dense linear system:

```
A(p, v) y = b(p, u, v)
```

`A` is dense because it comes from some interaction law — an influence matrix, a
kernel, an integral operator — and it depends on the design parameters `p`, on the
controls `u` and on the state `v`. The outputs are then some non-linear function
of the response `y`.

Three things can make the derivatives of such a model expensive, and they are
easy to confuse:

1. **how the dense object is written** — a matrix expression or a scalar loop;
2. **how the inner solve is written** — elimination, closure constraints, implicit
   rootfinder;
3. **how much second-order coupling actually exists** — which is a property of the
   model, not of the code.

This post separates them with measurements.

## The example

`example.py` builds the model:

```python
coordinates = (
    locator
    + 0.12 * design[0] * (1.0 - locator**2)
    + 0.08 * design[1] * locator * (1.0 - locator**2)
)
span = cas.reshape(coordinates, N, 1)
difference = cas.repmat(span, 1, N) - cas.repmat(span.T, N, 1) + cas.MX.eye(N)
kernel = ((coordinates[1] - coordinates[0]) / (4.0 * np.pi)) / difference
kernel = kernel - cas.diag(cas.diag(kernel))
```

The kernel is a rational function of coordinates that depend on the design
parameters, the response feeds an argument `a = u + c (K y) / v`, and the outputs
pass through a non-linearity `g(a) = ω a − 5 a³`.

## Lesson 1: a matrix expression is not a scalar loop

The same kernel can be assembled entry by entry:

```python
kernel = cas.MX.zeros(N, N)
for i in range(N):
    for j in range(N):
        if i != j:
            kernel[i, j] = (x[1] - x[0]) / (4.0 * np.pi * (x[i] - x[j]))
```

The numbers are identical to the matrix form (agreement to `1e-17`), the graphs
are not:

| at `N = 24` | scalar loop | matrix expression |
|---|---:|---:|
| nodes for the kernel | 5 538 | 32 |
| nodes for the Lagrangian Hessian | 231 332 | **4 015** |
| one Hessian evaluation | 12.3 ms | **0.71 ms** |

With `MX`, a matrix operation stays one node; a Python loop with scalar
assignments creates one node per entry. This is pure representation: fixing it
costs nothing and changes the picture by a factor of about 60. Everything below
uses the matrix expression.

## Lesson 2: how you write the solve barely matters

![cost](figures/cost.png)

| writing | Hessian nonzeros | Hessian graph | one evaluation |
|---|---:|---:|---:|
| elimination | 435 | 3 947 | 0.76 ms |
| SAND (closure constraints) | 1 153 | 3 356 | 0.50 ms |
| rootfinder (implicit) | 435 | 3 212 | 1.68 ms |
| elimination, `A` design-frozen | 325 | 1 431 | 0.46 ms |
| elimination, `g` linear | 135 | 868 | 0.20 ms |

All five are the same problem. Elimination, closure constraints and the implicit
rootfinder land within a factor of three on graph size — and the rootfinder is the
slowest to evaluate, because it re-solves the system on every call. Nothing here
is a lever on the order of magnitude we saw above.

## Lesson 3: what is left is genuine second-order coupling

![scaling](figures/scaling.png)

Sweeping the size of the dense system, with the kernel written as a matrix
expression:

| `N` | gradient graph | Hessian graph | gradient | exact Hessian |
|---:|---:|---:|---:|---:|
| 12 | 177 | 2 243 | 0.02 ms | 0.13 ms |
| 24 | 189 | 3 947 | 0.05 ms | 0.76 ms |
| 48 | 213 | 8 219 | 0.25 ms | 7.0 ms |
| 96 | 261 | 20 229 | 1.5 ms | 86 ms |
| 192 | 357 | 58 073 | 11 ms | **1 210 ms** |

The gradient grows slowly — first-order derivatives of an implicit solve are
carried by a linear solve, not by an expanded graph. The exact Hessian does not:
at `N = 192` it costs a hundred times the gradient, and it is the only curve that
would dominate a solver's inner loop.

Freezing the design parameters inside the matrix helps at small sizes (×1.7 at
`N = 24`) and almost stops helping at large ones (×1.1 at `N = 192`). Beyond a
certain size the cost is carried by the dense operator itself, not by which
parameters it happens to depend on.

## Reading the pattern

![patterns](figures/patterns.png)

The assembled derivatives stay small — 720 nonzeros for the closure Jacobian,
1 153 for the Lagrangian Hessian. A solver sees well-structured matrices; the
work of producing them is somewhere else. Nonzeros count what is stored, nodes
count what is computed.

## What actually helps

- **Write matrix expressions where matrix expressions exist.** The single
  largest win in this study, and it is free.
- **Freeze what is genuinely constant.** NumPy values, not symbolic loops, for
  anything that does not depend on the decision variables.
- **Prefer first-order methods if the second-order model is not the bottleneck of
  your convergence.** The numbers above are the cost side of that trade.
- **Check before rewriting your solve.** On this structure, elimination, closure
  constraints and an implicit rootfinder are interchangeable in cost; changing
  them is not where the money is.

## Where this shows up

The pattern is always the same: a dense system whose entries depend on the
decision variables, followed by a non-linear reading of its solution.

- **Structural design.** A linear finite-element model `K(p) u = f`, with `p` the
  thicknesses or sections. Condense or substructure the model and `K` becomes
  dense; optimising the shape means differentiating `K(p)^-1 f`.
- **Circuit design.** Modified nodal analysis `G(p) v = i`, with `p` the component
  sizes, evaluated at several operating points.
- **Process engineering.** A thermodynamic equilibrium solved by Newton inside an
  energy balance. The gradient of the process model goes through the solver, not
  through the balance.
- **Gaussian-process models.** Fitting a kernel whose length scales are tuned by
  gradient: the gradient of the likelihood passes through a dense factorisation.
- **Boundary and panel methods.** A dense influence matrix built from a geometry
  that itself depends on shape parameters.
- **Robotics and control.** Kinematic or contact constraints solved at every step
  of a predictive controller.

## Reproduce

```bash
python example.py     # five writings at N = 24: results.json + patterns.npz
python scaling.py     # N = 12 to 192: scaling.json
python figures.py     # figures/cost.png, patterns.png, scaling.png
```

Times are single-thread medians on a shared workstation and move by about 20 %
between runs; the graph sizes do not.

## References

- Haftka, R. T., "Simultaneous Analysis and Design," *AIAA Journal*, Vol. 23,
  No. 7, 1985, pp. 1099–1103. [doi:10.2514/3.9043](https://doi.org/10.2514/3.9043)
- CasADi documentation, *The MX symbolics*:
  <https://web.casadi.org/docs/#the-mx-symbolics>
