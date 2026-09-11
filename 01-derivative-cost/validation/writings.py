"""Why a Hessian stays expensive even when the linear solve is cheap.

A dense linear system whose matrix is built from coordinates that depend on the
design parameters:

    design parameters  p        coordinates        x(p)
    controls           u        right-hand side    b(p, u, state)
    state              v        matrix             A(p, v) = I + diag(k(p, v)) K(x(p))

    response           y = A^-1 b
    argument           a = u + c (K y) / v
    outputs            Phi(y) = sum_i g(a_i) w_i(p)      (g non-linear)

The kernel is written as a matrix expression, not assembled entry by entry: with
MX the two forms give the same numbers but very different graphs.

Five equivalent writings of the same object:

    elimination              y = A \\ b
    SAND                     y is a variable, A y = b is a constraint
    rootfinder               y = rootfinder(...)
    elimination, A design-frozen   A stops depending on the design parameters
    elimination, g linear          g(a) = omega a

Writes results.json and patterns.npz next to this file; figures.py draws them.
The growth of the cost with the system size is measured by scaling.py.

Run:  python writings.py
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

# ------------------------------------------ dense kernel, as a matrix expression
coordinates = (
    locator
    + 0.12 * design[0] * (1.0 - locator**2)
    + 0.08 * design[1] * locator * (1.0 - locator**2)
)
span = cas.reshape(coordinates, N, 1)
difference = cas.repmat(span, 1, N) - cas.repmat(span.T, N, 1) + cas.MX.eye(N)
kernel = ((coordinates[1] - coordinates[0]) / (4.0 * np.pi)) / difference
kernel = kernel - cas.diag(cas.diag(kernel))
weights = WEIGHT * design[2] * (1.0 - 0.4 * design[3] * locator)
matrix = cas.MX.eye(N) + cas.diag(OMEGA * weights / state) @ kernel
rhs = OMEGA * weights * (controls + 0.1 * locator)

# ------------------------------- same system with the design parameters frozen
coordinates_ref = (
    locator
    + 0.12 * design_ref[0] * (1.0 - locator**2)
    + 0.08 * design_ref[1] * locator * (1.0 - locator**2)
)
difference_ref = coordinates_ref[:, None] - coordinates_ref[None, :]
np.fill_diagonal(difference_ref, 1.0)
kernel_ref = ((coordinates_ref[1] - coordinates_ref[0]) / (4.0 * np.pi)) / difference_ref
np.fill_diagonal(kernel_ref, 0.0)
kernel_ref = cas.DM(kernel_ref)
weights_ref = cas.DM(WEIGHT * design_ref[2] * (1.0 - 0.4 * design_ref[3] * locator))
matrix_ref = cas.MX.eye(N) + cas.diag(OMEGA * weights_ref / state) @ kernel_ref
rhs_ref = OMEGA * weights_ref * (controls + 0.1 * locator)

# ------------------------------------------------------ the five writings
y_variable = cas.MX.sym("y", N)
y_rootfinder = cas.MX.sym("y_rootfinder", N)
rootfinder = cas.rootfinder(
    "rootfinder",
    "newton",
    {"x": y_rootfinder, "p": cas.vertcat(design, controls, state),
     "g": matrix @ y_rootfinder - rhs},
)
cases = [
    ("elimination", cas.solve(matrix, rhs),
     cas.vertcat(design, controls, state), None, False, False),
    ("elimination, A design-frozen", cas.solve(matrix_ref, rhs_ref),
     cas.vertcat(design, controls, state), None, True, False),
    ("elimination, g linear", cas.solve(matrix, rhs),
     cas.vertcat(design, controls, state), None, False, True),
    ("SAND (closure constraints)", y_variable,
     cas.vertcat(design, controls, state, y_variable), matrix @ y_variable - rhs,
     False, False),
    ("rootfinder (implicit)", rootfinder(np.zeros(N), cas.vertcat(design, controls, state)),
     cas.vertcat(design, controls, state), None, False, False),
]

print(f"{'case':<30}{'vars':>5}{'nnz H':>7}{'nodes primal':>13}{'nodes grad':>11}"
      f"{'nodes H':>10}{'eval H (ms)':>13}")
results = []
patterns = {}
for label, response, variables, closure, frozen, linear in cases:
    kernel_used = kernel_ref if frozen else kernel
    weights_used = weights_ref if frozen else weights
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
    prime = cas.Function("outputs_of", [variables], [outputs])
    gradient = cas.Function("gradient_of", [variables], [cas.gradient(cas.sum1(outputs), variables)])
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
        "jacobian_shape": np.array(jacobian_sparsity.shape),
        "hessian_rows": np.array(hessian_sparsity.get_triplet()[0]),
        "hessian_cols": np.array(hessian_sparsity.get_triplet()[1]),
        "hessian_shape": np.array(hessian_sparsity.shape),
        "closure_rows": (np.array(closure_sparsity.get_triplet()[0])
                         if closure_sparsity is not None else np.array([], dtype=int)),
        "closure_cols": (np.array(closure_sparsity.get_triplet()[1])
                         if closure_sparsity is not None else np.array([], dtype=int)),
        "closure_shape": (np.array(closure_sparsity.shape)
                          if closure_sparsity is not None else np.array([0, 0])),
        "blocks": np.array([4, N, 1, N if closure is not None else 0]),
    }
    results.append({
        "label": label,
        "variables": int(variables.size1()),
        "nnz_hessian": int(hessian.nnz_out(0)),
        "nodes_primal": int(prime.n_nodes()),
        "nodes_gradient": int(gradient.n_nodes()),
        "nodes_jacobian": int(jacobian.n_nodes()),
        "nodes_hessian": int(hessian.n_nodes()),
        "build_s": build,
        "hessian_ms": float(np.median(timings)) * 1e3,
    })
    print(f"{label:<30}{variables.size1():>5}{hessian.nnz_out(0):>7}"
          f"{prime.n_nodes():>13}{gradient.n_nodes():>11}{hessian.n_nodes():>10}"
          f"{np.median(timings) * 1e3:>13.2f}")

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
