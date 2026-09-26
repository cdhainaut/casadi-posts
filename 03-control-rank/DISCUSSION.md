# Discussion with Joris: from reproduction to a representative problem

Thank you for “Lifting the operator, not the solve”. We reproduced your example
before changing the number of independent controls within one scenario. The
next useful question is how the operator formulation behaves with many operating
points, their own controls and states, and a larger auxiliary field.

## What has been checked

At N=192, all four formulations reach the reference objective with matching
NLP dimensions and J/H nonzero counts. `opti.solve()` takes 28.76 s for dense
eliminated and 0.15 s for sparse lifted on our machine.

At N=42, we retain your kernel, objective, four design parameters and scalar
state, then set `u=B_r a`, with 42, 4 and 1 independent control. Station-wise
bounds are unchanged. The reduced bases are fixed Legendre polynomials,
including the constant control.

The mean sparse-eliminated/lifted Hessian timing ratio falls from ×22.7 to
×2.4 and ×2.0. Small solve times are close, around 0.1 s including preparation;
one run per case does not establish their fine ranking. The sparse operator
remains useful even with a single control.

[Equations, tables and reproduction](README.md) —
[measurements and validation criteria](validation/README.md).

## What is already expected

A smaller reduced decision space has a smaller reduced Hessian. We regard the
control-rank experiment as a sanity check and quantitative ablation, not as a
new theoretical observation or evidence against operator lifting.

It is also not a model of an entire multipoint problem: one spatially constant
control **per operating point** still gives T independent controls over T points.
Those controls belong to separate spatial influence problems. With a shared
optimized design, the reduced Lagrangian Hessian has a common border and local
blocks, not necessarily a dense T-by-T control block. Trajectory constraints
introduce additional coupling.

The relevant trade-off therefore involves local decision count, number of
operating points, auxiliary-field size, geometry dependence and KKT fill-in.

## Proposed next comparison — not yet implemented or measured

Use N spatial stations and T genuinely distinct operating points, with local
controls/states at every point. Separate fixed design from shared optimized
design, then add local trajectory coupling as a distinct experiment.

For a chosen discrete field operator, keep an exact algebraic pair:

```text
L ψ = S y,   z = R ψ,   y + D z = b
K = R L⁻¹ S
(L + S D R) ψ = S b
```

Here the field has M unknowns, potentially M>N. Injection `S` and observation
`R` must preserve locality; forming `S D R` is not automatically sparse for
arbitrary maps. A general physical closure may require retaining both response
and field variables instead of using the final condensed equation.

Compare eliminated and lifted versions of this **same discrete operator**
before comparing its physical accuracy against an independent dense reference.
A field approximation and a different reference kernel are not an equal-accuracy
performance comparison. If the geometry depends on local controls, the operators
may differ across operating points: shared factorization must not be assumed.

## Questions where your advice would help

1. Is this exact discrete pair an appropriate bridge to a larger auxiliary
   field, or would you retain the block system even for the linear closure?
2. If the response identity term is absent, is the block formulation
   `[L, −S ; D R, 0]` the appropriate starting point, rather than forcing the
   same condensed form?
3. What factorization and memory measurements would best distinguish the
   intrinsic field cost from ordering, fill-in or implementation defects?
4. How would you structure local injection/observation for moving geometry
   while preserving differentiability and sparse support?

The current repository contains only the single-scenario reproduction and rank
ablation. No multipoint, larger-field or trajectory performance is claimed.
