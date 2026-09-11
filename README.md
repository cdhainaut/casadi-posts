# CasADi posts

Two short, self-contained notes on the derivative structure of nonlinear programs
that hide a dense system. Each post is a handful of flat Python scripts, a figure
generator and a text, built on CasADi, NumPy and Matplotlib only.

| post | question |
|---|---|
| [01 — Exact Hessians of nested dense systems can be expensive](01-derivative-cost/README.md) | Where does the work of differentiating a nested dense system actually go, and which part of it is real? |
| [02 — Sharing one design block across every condition](02-design-control-coupling/README.md) | What changes when design parameters and trajectory variables share one NLP? |

Both posts use the same small model — a dense matrix built from coordinates that
depend on design parameters, a response and a non-linear output — so the two notes
can be read in either order.

## Requirements

```
python >= 3.11
casadi >= 3.7
numpy
matplotlib
```

Each script writes its own JSON next to itself, so every number quoted in the two
texts can be traced back to a file in the repository.

```bash
cd 01-derivative-cost
python check_equivalence.py && python example.py && python scaling.py && python figures.py
cd ../02-design-control-coupling
python example.py && python figures.py
