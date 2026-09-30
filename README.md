# CasADi derivative experiments

Standalone experiments on exact derivatives and nonlinear optimisation.
Each study documents its equations, measurement scope and reproduction.

| Study | Question |
|---|---|
| [01 Derivative cost](01-derivative-cost/README.md) | Why can a cheap dense solve be expensive to differentiate twice? |
| [02 Shared design](02-design-control-coupling/README.md) | What happens to the Hessian structure when conditions share a design? |
| [03 Control rank](03-control-rank/README.md) | How does spatial control rank affect operator lifting within one scenario? |
| [04 Field size](04-field-ratio/README.md) | How do auxiliary-field size and backend choice affect measured cost? |

Start with study 04 for the field-size comparison. It distinguishes mean
Hessian-callback cost from complete solve time. Its corrected active CasADi/JAX
comparison establishes no Hessian speedup; the previous 70x claim is withdrawn.
The [correction notice](04-field-ratio/validation/README.md) preserves the
invalidated record and explains the replacement measurement.

## Running

Python 3.11 or later, NumPy and CasADi with IPOPT/MUMPS. Matplotlib draws the
figures; JAX is optional for the comparisons in study 04. Studies 02, 03 and 04
pin their measured dependencies. Each README gives execution and validation
commands; run tests from the corresponding study directory.

Published records live in `validation/`. New benchmark outputs go to ignored
`runs/` paths, which the maintained drivers refuse to overwrite. Study 01 is a
flat script and its validation scripts write adjacent artifacts: run a copy
when preserving those files. Studies 02/03/04 do no computation on import.

The measurement envelope is documented per study. The field-size and corrected
backend comparisons use one numerical thread and 2 GiB address space. Timings
are machine-dependent; toy-model results do not establish a production
formulation choice.

## Attribution

Studies 03 and 04 build on Joris Gillis's "Lifting the operator, not the solve".
The received reference files are preserved unchanged and credited in
[reference/README.md](03-control-rank/reference/README.md).
