"""Why a Hessian stays expensive even when the linear solve is cheap.

Generic numerical example. Dense linear system whose matrix is built from
coordinates that depend on the design parameters:

    design parameters  p        coordinates        x(p)
    controls           u        right-hand side    b(p, u, state)
    state              v        matrix             A(p, v) = I + diag(k(p, v)) K(x(p))

    response           y = A^-1 b
    argument           a = u + c (K y) / v
    outputs            Phi(y) = sum_i g(a_i) w_i(p)      (g non-linear)

Five equivalent writings of the same object:

    elimination            y = A \\ b
    SAND                   y is a variable, A y = b is a constraint
    rootfinder             y = rootfinder(...)
    elimination, A frozen  A stops depending on the design parameters
    elimination, g linear  g(a) = omega a

Writes results.json next to this file; make_figure.py turns it into the figure.

Run:  python dense_matrix_derivative_cost.py
"""

import json
import time
from pathlib import Path

import casadi as cas
import numpy as np

# --------------------------------------------------------------------- data
N = 24  # dense system size
OMEGA = 2.0 * np.pi  # gain of g
WEIGHT = 0.4  # output weighting constant
COUPLING = 0.3  # strength of the through-K argument
EVALUATIONS = 20

locator = np.cos(np.pi * (np.arange(N) + 0.5) / N)  # points in [-1, 1]
design_ref = np.array([0.5, 0.3, 1.0, 0.4])
controls_ref = np.full(N, 0.1)
state_ref = np.array([12.0])

# ---------------------------------------------------------------- unknowns
design = cas.MX.sym("design", 4)
controls = cas.MX.sym("controls", N)
state = cas.MX.sym("state")

# ------------------------------------------ dense system from the coordinates
coordinates = locator * (1.0 + 0.4 * design[1] * locator) * (0.5 + 0.2 * design[0])
weights = WEIGHT * design[2] * (1.0 - 0.4 * design[3] * locator)
K = cas.MX.zeros(N, N)
for i in range(N):
    for j in range(N):
        if i != j:  # rational in the coordinates: dense and non-linear in design
            K[i, j] = (coordinates[1] - coordinates[0]) / (
                4.0 * np.pi * (coordinates[i] - coordinates[j])
            )
A = cas.MX.eye(N) + cas.diag(OMEGA * weights / state) @ K
b = OMEGA * weights * (controls + 0.1 * locator)

# ------------------------------------------- same system, A independent of p
coordinates_frozen = locator * (1.0 + 0.4 * design_ref[1] * locator) * (
    0.5 + 0.2 * design_ref[0]
)
weights_frozen = WEIGHT * design_ref[2] * (1.0 - 0.4 * design[3] * locator)
K_frozen = cas.MX.zeros(N, N)
for i in range(N):
    for j in range(N):
        if i != j:
            K_frozen[i, j] = (coordinates_frozen[1] - coordinates_frozen[0]) / (
                4.0 * np.pi * (coordinates_frozen[i] - coordinates_frozen[j])
            )
A_frozen = cas.MX.eye(N) + cas.diag(OMEGA * weights_frozen / state) @ K_frozen
b_frozen = OMEGA * weights_frozen * (controls + 0.1 * locator)

y_rootfinder = cas.MX.sym("y_rootfinder", N)
rootfinder = cas.rootfinder(
    "rootfinder",
    "newton",
    {
        "x": y_rootfinder,
        "p": cas.vertcat(design, controls, state),
        "g": A @ y_rootfinder - b,
    },
)

# ------------------------------------------------------ the five writings
y_variable = cas.MX.sym("y", N)
cases = [
    ("elimination", cas.solve(A, b), cas.vertcat(design, controls, state), None, False, False),
    ("elimination, A frozen", cas.solve(A_frozen, b_frozen),
     cas.vertcat(design, controls, state), None, True, False),
    ("elimination, g linear", cas.solve(A, b),
     cas.vertcat(design, controls, state), None, False, True),
    ("SAND (closure constraints)", y_variable, cas.vertcat(design, controls, state, y_variable),
     A @ y_variable - b, False, False),
    ("rootfinder (implicit)", rootfinder(np.zeros(N), cas.vertcat(design, controls, state)),
     cas.vertcat(design, controls, state), None, False, False),
]

print(f"{'case':<30}{'vars':>5}{'nnz H':>7}{'nodes J':>10}{'nodes H':>10}"
      f"{'build H (s)':>12}{'eval H (ms)':>12}")
results = []
patterns = {}
for label, response, variables, closure, frozen, linear in cases:
    kernel_used = K_frozen if frozen else K
    weights_used = weights_frozen if frozen else weights
    argument = controls + COUPLING * (kernel_used @ response) / state
    transferred = OMEGA * argument if linear else OMEGA * argument - 5.0 * argument**3
    outputs = cas.vertcat(
        state * cas.sum1(transferred * weights_used), cas.sum1(argument * weights_used)
    )
    multipliers = cas.MX.sym("lambda", closure.size1()) if closure is not None else None
    lagrangian = cas.sum1(outputs)
    arguments = [variables]
    if closure is not None:
        lagrangian = lagrangian + cas.dot(multipliers, closure)
        arguments = [variables, multipliers]

    timer = time.perf_counter()
    jacobian = cas.Function("jacobian_of", arguments, [cas.jacobian(outputs, variables)])
    hessian = cas.Function(
        "hessian_of", arguments, [cas.tril(cas.hessian(lagrangian, variables)[0], True)]
    )
    build = time.perf_counter() - timer

    point = np.concatenate([design_ref, controls_ref, state_ref,
                            np.full(closure.size1(), 0.1) if closure is not None else []])
    values = [point] + ([np.ones(closure.size1())] if closure is not None else [])
    for _ in range(2):
        hessian(*values)
    timings = []
    for _ in range(EVALUATIONS):
        start = time.perf_counter()
        hessian(*values)
        timings.append(time.perf_counter() - start)

    closure_sparsity = (
        cas.jacobian(closure, variables).sparsity() if closure is not None else None
    )
    jacobian_sparsity = jacobian.sparsity_out(0)
    hessian_sparsity = hessian.sparsity_out(0)
    patterns[label] = {
        "jacobian_rows": np.array(jacobian_sparsity.get_triplet()[0]),
        "jacobian_cols": np.array(jacobian_sparsity.get_triplet()[1]),
        "hessian_rows": np.array(hessian_sparsity.get_triplet()[0]),
        "hessian_cols": np.array(hessian_sparsity.get_triplet()[1]),
        "jacobian_shape": np.array(jacobian_sparsity.shape),
        "hessian_shape": np.array(hessian_sparsity.shape),
        "variables": np.array(variables.size1()),
        "closure_rows": np.array(closure_sparsity.get_triplet()[0]) if closure_sparsity else np.array([], dtype=int),
        "closure_cols": np.array(closure_sparsity.get_triplet()[1]) if closure_sparsity else np.array([], dtype=int),
        "closure_shape": np.array(closure_sparsity.shape) if closure_sparsity else np.array([0, 0]),
        "blocks": np.array([4, N, 1, N if closure is not None else 0]),
    }
    results.append({
        "label": label,
        "variables": int(variables.size1()),
        "nnz_hessian": int(hessian.nnz_out(0)),
        "nodes_jacobian": int(jacobian.n_nodes()),
        "nodes_hessian": int(hessian.n_nodes()),
        "build_hessian_s": build,
        "hessian_ms": float(np.median(timings)) * 1e3,
    })
    print(f"{label:<30}{variables.size1():>5}{hessian.nnz_out(0):>7}"
          f"{jacobian.n_nodes():>10}{hessian.n_nodes():>10}{build:>12.2f}"
          f"{np.median(timings) * 1e3:>12.1f}")

Path(__file__).with_name("results.json").write_text(
    json.dumps(results, indent=2) + "\n", encoding="utf-8"
)
np.savez(
    Path(__file__).with_name("patterns.npz"),
    **{f"{label}|{key}": value
       for label, entry in patterns.items()
       for key, value in entry.items()},
)
print("\nwrote results.json")
