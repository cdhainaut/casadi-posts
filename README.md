# Two candidate CasADi blog posts

Each post is one short article and one flat, executable example — the kind of
script you can read from top to bottom like a MATLAB script. No framework, no
package, no benchmark harness.

| post | one line |
|---|---|
| [A dense solve is cheap to evaluate and expensive to differentiate](01-derivative-cost/README.md) | a dense `A(x) y = b(x)` costs little to evaluate and a lot to differentiate twice, and the cost is the dependence of `A` on `x` |
| [Sharing a few design parameters across many conditions makes a star](02-design-control-coupling/README.md) | one shared design block connects every condition to every other one, and the Hessian shows it |

Both use the same small model — a dense matrix built from coordinates that depend
on design parameters, a response, and a non-linear output — so the two notes can
be read in either order.

## Requirements

```
python >= 3.11
casadi >= 3.7
numpy
matplotlib
```

## Run

```bash
cd 01-derivative-cost && python example.py
cd 02-design-control-coupling && python example.py
```

Each script prints its table and writes its figure.

## Validation

Every number quoted in the two articles can be traced to a file in the repository.
The machinery that established them — equivalence checks between two ways of
writing the same object, a size sweep with synthetic control objects, raw JSON and
sparsity patterns — sits in a `validation/` folder inside each post, out of the
reader's way but available to anyone who wants to check or contest the results.

```bash
cd 01-derivative-cost/validation && python check_equivalence.py && python scaling.py && python figures.py
cd 02-design-control-coupling/validation && python figures.py
```
