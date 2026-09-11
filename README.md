# Two candidate CasADi blog posts

| post | one line |
|---|---|
| [When a dense solve becomes expensive to differentiate](01-derivative-cost/README.md) | a dense `A(x) y = b(x)` costs little to evaluate and a lot to differentiate twice, and the cost is the dependence of `A` on `x` |
| [Sharing a few design parameters across many conditions makes a star](02-design-control-coupling/README.md) | one shared design block connects every condition to every other one, and the Hessian shows it |

Both use the same small model — a dense matrix built from coordinates that depend
on design parameters, a response, and a non-linear output — so the two notes can
be read in either order.

## Contents

```
01-derivative-cost/
    README.md             the article
    example.py            the model, measured at N = 192
    hessian_cost.png      its figure
    validation/           derivation checks, the size sweep, raw numbers

02-design-control-coupling/
    README.md             the article
    example.py            the same problem with K = 5 conditions
    hessian_structure.png its figure
    validation/           the layouts measured, raw numbers
```

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

Each script prints its table and writes its figure. The scripts in `validation/`
reproduce the numbers quoted in the two articles.
