# Exact Hessians of nested dense systems can be expensive

*A CasADi post about dense models: what makes their derivatives costly, and what
only looks like it does.*

## The situation

Many models hide a small dense linear system:

```
A(p, v) y = b(p, u, v)
```

`A` is dense because it comes from some interaction law — an influence matrix, a
kernel, an integral operator — and it depends on the design parameters `p`, on the
controls `u` and on the state `v`. The outputs are then some non-linear function
of the response `y`.

Three things can make the derivatives of such a model expensive, and they are easy
to confuse:

1. **how the dense object is written** — a matrix expression or a scalar loop;
2. **how the inner solve is written** — elimination, closure constraints, implicit
   rootfinder;
3. **how much second-order differentiation the model really demands**.

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
| nodes for the Hessian | 226 881 | **3 947** |
| one Hessian evaluation | 11.9 ms | **0.8 ms** |

Measured by `check_equivalence.py`, which also verifies that the two writings
agree to `7e-15` on the Hessian, `4e-15` on the gradient and `1e-17` on the
kernel itself, over four design points.

With `MX`, a matrix operation stays one node; a Python loop with scalar
assignments creates one node per entry. This is pure representation: fixing it
costs nothing and changes the picture by a factor of about 60. Everything below
uses the matrix expression.

## Lesson 2: how you write the solve barely matters

![cost](figures/cost.png)

| writing | Hessian nonzeros | Hessian graph | one evaluation |
|---|---:|---:|---:|
| elimination | 435 | 3 947 | 0.76 ms |
| SAND (closure constraints) | 1 153 | 3 356 | 0.49 ms |
| rootfinder (implicit) | 435 | 3 212 | 1.72 ms |
| elimination, `A` design-frozen | 325 | 1 431 | 0.42 ms |
| elimination, `g` linear | 135 | 868 | 0.19 ms |

All five are the same problem. Elimination, closure constraints and the implicit
rootfinder land within a factor of three on graph size — and the rootfinder is the
slowest to evaluate, because it re-solves the system on every call. Nothing here
is a lever on the order of magnitude we saw above.

## Lesson 3: what is left is second-order differentiation, not dense algebra

![scaling](figures/scaling.png)

The obvious objection is that the Hessian is dense, so of course it gets
expensive: at `N = 192` it has 19 503 nonzeros, exactly the full triangle. To
separate "a dense object is expensive" from "differentiating this model is
expensive", the sweep below adds two control objects of the same dimension:

- a **dense constant matrix**, the floor cost of materialising that many numbers;
- a **synthetic dense cubic** `sum((C x)^3)` with `C` dense and constant — a
  genuinely dense, genuinely variable Hessian with no model structure in it.

| `N` | nonzeros | gradient | `H·v` | exact Hessian | design frozen | dense cubic | dense constant |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 153 | 0.02 ms | 0.05 ms | 0.22 ms | 0.09 ms | 0.02 ms | 0.011 ms |
| 24 | 435 | 0.06 ms | 0.13 ms | 0.95 ms | 0.57 ms | 0.07 ms | 0.013 ms |
| 48 | 1 431 | 0.37 ms | 0.86 ms | 10.8 ms | 7.7 ms | 0.20 ms | 0.014 ms |
| 96 | 5 151 | 2.3 ms | 5.1 ms | 136 ms | 122 ms | 1.6 ms | 0.024 ms |
| 192 | 19 503 | 16 ms | 38 ms | **1 502 ms** | 1 313 ms | 7.7 ms | 0.032 ms |

Three readings:

1. **It is not the size of the object.** Materialising a dense matrix of the same
   shape costs 0.03 ms at `N = 192` — five orders of magnitude below the model.
2. **It is not dense algebra in general.** The synthetic cubic has a dense,
   variable Hessian of the same shape and costs 7.7 ms, 195 times less. The
   expensive part is specific to the model: differentiating the dense solve and
   the matrix products around it.
3. **Materialising the full second-order object has its own price.** The
   Hessian-vector product is 39 times cheaper than the complete Hessian at the
   same size. If a solver only needs directional second-order information, the
   full matrix is not the thing to build.

One observation the numbers force, and which deserves stating rather than
explaining away: freezing the design parameters cuts the graph by a factor of 5.6
at `N = 192` but the evaluation time by only 14 %. The two graphs share the part
that dominates — differentiating the dense solve, which still depends on the
state. Design dependence is no longer the driver at this size; the operator is.

## Reading the pattern

![patterns](figures/patterns.png)

The assembled derivatives stay small — 720 nonzeros for the closure Jacobian,
1 153 for the Lagrangian Hessian. A solver sees well-structured matrices; the
work of producing them is somewhere else. Nonzeros count what is stored, nodes
count what is computed.

## What actually helps

- **Write matrix expressions where matrix expressions exist.** The single largest
  win in this study, and it is free.
- **Freeze what is genuinely constant.** NumPy values, not symbolic loops, for
  anything that does not depend on the decision variables.
- **Ask whether you need the full Hessian.** Directional second-order information
  is an order of magnitude cheaper here; quasi-Newton updates are cheaper still.
- **Check before rewriting your solve.** On this structure, elimination, closure
  constraints and an implicit rootfinder are interchangeable in cost.

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

## Scope

Everything above measures a model in isolation: derivatives of its outputs, not a
solved NLP. Nothing here says how these costs translate into IPOPT iterations, a
KKT factorisation, or the wall time of a complete optimisation. The claim is
narrower: for this structure, the derivative cost lives in the second-order
differentiation of the model, and not in the dense object's size or its algebra.

## Reproduce

```bash
python check_equivalence.py   # scalar loop against matrix expression, values and derivatives
python example.py             # five writings at N = 24 -> results.json, patterns.npz
python scaling.py             # N = 12 to 192, with controls -> scaling.json
python figures.py             # figures/cost.png, patterns.png, scaling.png
```

Times are single-thread medians on a shared workstation and move by about 20 %
between runs; the graph sizes do not.

## References

- Haftka, R. T., "Simultaneous Analysis and Design," *AIAA Journal*, Vol. 23,
  No. 7, 1985, pp. 1099–1103. [doi:10.2514/3.9043](https://doi.org/10.2514/3.9043)
- CasADi documentation, *The MX symbolics*:
  <https://web.casadi.org/docs/#the-mx-symbolics>
