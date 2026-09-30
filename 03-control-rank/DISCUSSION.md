# Follow-up to Joris Gillis

Thank you for "Lifting the operator, not the solve". I reproduced the example
before extending it. At N=192, the four formulations reach the reference
objective with matching NLP dimensions and Jacobian/Hessian nonzero counts.
Dense eliminated takes 28.76 s, sparse lifted 0.15 s on this machine.

Reducing the spatial control rank at N=42 makes the reduced Hessian smaller.
The mean sparse-eliminated/lifted Hessian ratio falls from 22.7 with 42 controls
to 2.4 with four and 2.0 with one. Those are single-scenario results, not a test
of many operating points sharing a design.

The proposed field-size homotopy is now in
[study 04](../04-field-ratio/README.md). The compressed Green function remains
exactly the original kernel, while the auxiliary chain grows from N to 128N.
At N=42, the mean Hessian-call crossover is between z=32 and z=64, but the
whole-solve crossover is already between z=8 and z=16. Sparse lifted also
reaches different stationary points at z=64/128. These are observations from
one implementation, not a universal limit on lifting.

The field can therefore cost more than the response saves, even for this exact
pair. For a physical nonlocal kernel, the next question is how to keep the
auxiliary representation small enough at a fixed error.

Would an operator on a surface or a screened kernel be a useful next comparison
against a volumetric field? For moving geometry, how would you preserve local
injection and observation without turning their support changes into
non-smooth derivatives?

[Equations and reproduction](README.md),
[measurement provenance](validation/README.md).
