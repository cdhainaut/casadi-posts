# CasADi — derivative cost, shared design and sparse operators

Small, reproducible experiments on exact derivatives and nonlinear optimisation.
The first two notes study a dense influence model. The third follows Joris Gillis's
sparse-operator example and varies the number of independent controls within
one scenario. That ablation is a sanity check, not a representative multipoint
benchmark.

**For the current discussion, start with [03-control-rank](03-control-rank/README.md)**
and its [questions for Joris](03-control-rank/DISCUSSION.md).

| Study | Question | Scope |
|---|---|---|
| [01 — Derivative cost](01-derivative-cost/README.md) | Why can a cheap dense solve be expensive to differentiate twice? | Model and derivative evaluation, not a complete NLP solve |
| [02 — Shared design](02-design-control-coupling/README.md) | How does a shared design block connect otherwise separate conditions? | Derivative structure, not measured KKT factorisation |
| [03 — Control rank](03-control-rank/README.md) | How does spatial control rank affect lifting within one scenario? | Four equivalent NLP formulations; 16 converged single-scenario cases |

The first two notes are unchanged. Study 03 uses an exponential kernel whose
sparse inverse is known exactly; it is not a claim of equivalence to the kernel
of study 01. Its gain must not be extrapolated to larger auxiliary fields or
multiple scenarios without further measurements.

## Repository map

```text
01-derivative-cost/          first article, example and validation data
02-design-control-coupling/  second article, example and validation data
03-control-rank/
    README.md               equations, results, limitations and reproduction
    DISCUSSION.md           concise questions for Joris
    bench_joris_controls.py  one standalone benchmark case
    run_joris_controls.py    isolated J0/J1 campaign; stop at the first failure
    reference/              original source, results and note received from Joris
    validation/             published numerical records and provenance
    tests/                  algebra, derivative, CLI and provenance checks
notes/                      historical local working files, kept out of Git
```

## Requirements and execution

Studies 01/02 use Python ≥ 3.11, CasADi ≥ 3.7, NumPy and Matplotlib:

```bash
(cd 01-derivative-cost && python example.py)
(cd 02-design-control-coupling && python example.py)
```

Study 03 needs only NumPy and CasADi with IPOPT/MUMPS; its measured versions are
pinned in `03-control-rank/requirements.txt`. It has no dependency on a domain
model, another checkout or a private data directory. See its
[reproduction instructions](03-control-rank/README.md#6-reproduce-from-this-repository)
for the Linux resource envelope and fresh output directory.

Published measurements are kept separate from new runs. The 03 campaign never
overwrites an existing output directory; new `03-control-rank/runs/` directories
are ignored by Git. No benchmark is launched merely by importing its modules.

## Attribution

Study 03 is based on **Joris Gillis's “Lifting the operator, not the solve”**.
The received reference files are preserved unchanged and attributed in
[reference/README.md](03-control-rank/reference/README.md). The adaptations,
measurement conventions and open questions are identified separately.
