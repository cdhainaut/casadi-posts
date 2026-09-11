"""Coupling design parameters with state and control variables in one NLP.

Generic numerical example. Several conditions live in the same optimisation
problem, each with its own controls and its own state, while the design
parameters are either shared by all conditions or repeated per condition:

    design parameters  p        shared, or one set per condition
    controls           u_k      one block per condition
    state              v_k      one block per condition
    response           y_k      A_k(p_k, v_k) y_k = b(p_k, u_k, v_k)

    objective          sum_k Phi_k(y_k)
    constraints        one closure equation per condition

Three writings of the same problem are measured:

    design per condition   each condition carries its own design parameters
    design shared          one design block used by every condition
    design frozen          design parameters are constants

The first two have exactly the same local model and the same derivative work per
condition; the only difference is that the shared block couples every condition.
That coupling is what fills the Lagrangian Hessian and inflates the graph.

Writes results.json and patterns.npz next to this file; figures.py draws them.

Run:  python example.py
"""

import json
import time
from pathlib import Path

import casadi as cas
import numpy as np

# --------------------------------------------------------------------- data
N = 16  # dense system size
K = 3  # number of conditions
OMEGA = 2.0 * np.pi  # gain of g
WEIGHT = 0.4  # output weighting constant
COUPLING = 0.3  # strength of the through-K argument
TARGET = 0.5  # closure target per condition
EVALUATIONS = 20

locator = np.cos(np.pi * (np.arange(N) + 0.5) / N)  # points in [-1, 1]
design_ref = np.array([0.5, 0.3, 1.0, 0.4])
controls_ref = np.full((N, K), 0.1)
state_ref = np.full(K, 12.0)

# ---------------------------------------------------------------- unknowns
shared_design = cas.MX.sym("design", 4)
local_designs = [cas.MX.sym(f"design_{k}", 4) for k in range(K)]
controls = cas.MX.sym("controls", N, K)
state = cas.MX.sym("state", K)

# -------------------------------------------------------- the three writings
writings = {
    "design per condition": local_designs,
    "design shared": [shared_design] * K,
    "design frozen": [cas.DM(design_ref)] * K,
}

print(f"{'writing':<22}{'nvars':>7}{'ncons':>7}{'nnz J':>8}{'nnz H':>8}"
      f"{'nodes J':>10}{'nodes H':>10}{'eval H (ms)':>13}")
results = []
patterns = {}
for label, designs in writings.items():
    if label == "design per condition":
        variables = cas.vertcat(cas.vertcat(*local_designs),
                                cas.reshape(controls, (N * K, 1)), state)
    else:
        variables = cas.vertcat(shared_design,
                                cas.reshape(controls, (N * K, 1)), state)

    objectives = []
    closures = []
    for k in range(K):
        design_k = designs[k]
        column = controls[:, k]
        coordinates = (
            locator
            + 0.12 * design_k[0] * (1.0 - locator**2)
            + 0.08 * design_k[1] * locator * (1.0 - locator**2)
        )
        weights = WEIGHT * design_k[2] * (1.0 - 0.4 * design_k[3] * locator)
        span = cas.reshape(coordinates, N, 1)
        difference = cas.repmat(span, 1, N) - cas.repmat(span.T, N, 1) + cas.MX.eye(N)
        kernel = ((coordinates[1] - coordinates[0]) / (4.0 * np.pi)) / difference
        kernel = kernel - cas.diag(cas.diag(kernel))
        matrix = cas.MX.eye(N) + cas.diag(OMEGA * weights / state[k]) @ kernel
        rhs = OMEGA * weights * (column + 0.1 * locator)
        response = cas.solve(matrix, rhs)
        argument = column + COUPLING * (kernel @ response) / state[k]
        transferred = OMEGA * argument - 5.0 * argument**3
        objectives.append(cas.sum1(transferred * weights))
        closures.append(cas.sum1(argument * weights) - TARGET)

    objective = sum(objectives)
    constraints = cas.vertcat(*closures)
    multipliers = cas.MX.sym("lambda", constraints.size1())
    lagrangian = objective + cas.dot(multipliers, constraints)

    timer = time.perf_counter()
    jacobian = cas.Function("jacobian_of", [variables],
                            [cas.jacobian(constraints, variables)])
    hessian = cas.Function(
        "hessian_of",
        [variables, multipliers],
        [cas.tril(cas.hessian(lagrangian, variables)[0], True)],
    )
    build = time.perf_counter() - timer

    point = np.concatenate(
        [
            np.tile(design_ref, K) if label == "design per condition" else design_ref,
            controls_ref.ravel(),
            state_ref,
        ]
    )
    for _ in range(2):
        hessian(point, np.ones(constraints.size1()))
    timings = []
    for _ in range(EVALUATIONS):
        start = time.perf_counter()
        hessian(point, np.ones(constraints.size1()))
        timings.append(time.perf_counter() - start)

    results.append({
        "label": label,
        "variables": int(variables.size1()),
        "constraints": int(constraints.size1()),
        "nnz_jacobian": int(jacobian.nnz_out(0)),
        "nnz_hessian": int(hessian.nnz_out(0)),
        "nodes_jacobian": int(jacobian.n_nodes()),
        "nodes_hessian": int(hessian.n_nodes()),
        "build_s": build,
        "hessian_ms": float(np.median(timings)) * 1e3,
    })
    jacobian_sparsity = jacobian.sparsity_out(0)
    hessian_sparsity = hessian.sparsity_out(0)
    patterns[label] = {
        "jacobian_rows": np.array(jacobian_sparsity.get_triplet()[0]),
        "jacobian_cols": np.array(jacobian_sparsity.get_triplet()[1]),
        "hessian_rows": np.array(hessian_sparsity.get_triplet()[0]),
        "hessian_cols": np.array(hessian_sparsity.get_triplet()[1]),
        "jacobian_shape": np.array(jacobian_sparsity.shape),
        "hessian_shape": np.array(hessian_sparsity.shape),
        "design_block": np.array(4 * K if label == "design per condition" else 4),
        "condition_block": np.array(N + 1),
    }
    print(f"{label:<22}{variables.size1():>7}{constraints.size1():>7}"
          f"{jacobian.nnz_out(0):>8}{hessian.nnz_out(0):>8}"
          f"{jacobian.n_nodes():>10}{hessian.n_nodes():>10}"
          f"{np.median(timings) * 1e3:>13.1f}")

Path(__file__).with_name("results.json").write_text(
    json.dumps(results, indent=2) + "\n", encoding="utf-8"
)
np.savez(
    Path(__file__).with_name("patterns.npz"),
    **{f"{label}|{key}": value
       for label, entry in patterns.items()
       for key, value in entry.items()},
)
print("\nwrote results.json and patterns.npz")
