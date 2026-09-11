"""Sharing a few design parameters across many conditions makes a star.

A common shape in multidisciplinary optimisation: one small set of design
parameters, many conditions, and one dense model per condition.

    design parameters p        shared by every condition
    controls u_k, state v_k    one block per condition
    A(p, v_k) y_k = b(p, u_k, v_k)

This script builds the same problem twice — once with a design block shared by
every condition, once with one independent design block per condition — and looks
at the structure of the Lagrangian Hessian. The local models are identical; only
the sharing differs.

Run:  python example.py
"""

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------ parameters
N = 16  # size of each dense system
K = 5  # number of conditions
OMEGA = 2.0 * np.pi
WEIGHT = 0.4
COUPLING = 0.3
TARGET = 0.5
design_value = np.array([0.5, 0.3, 1.0, 0.4])
locator = np.cos(np.pi * (np.arange(N) + 0.5) / N)

# --------------------------------------------------------------------- symbols
shared_design = cas.MX.sym("design", 4)
local_designs = [cas.MX.sym(f"design_{k}", 4) for k in range(K)]
controls = cas.MX.sym("controls", N, K)
state = cas.MX.sym("state", K)

# ------------------------------------------------------- the two design layouts
layouts = {
    "design shared": [shared_design] * K,
    "design per condition": local_designs,
}

print(f"{'layout':<22}{'variables':>10}{'nonzeros':>10}{'density':>9}{'graph':>8}")
structures = {}
for label, designs in layouts.items():
    if label == "design shared":
        variables = cas.vertcat(shared_design, cas.reshape(controls, (N * K, 1)), state)
    else:
        variables = cas.vertcat(
            cas.vertcat(*local_designs), cas.reshape(controls, (N * K, 1)), state
        )

    objective_terms = []
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
        warped = OMEGA * argument - 5.0 * argument**3
        objective_terms.append(cas.sum1(warped * weights))
        closures.append(cas.sum1(argument * weights) - TARGET)

    multipliers = cas.MX.sym("lambda", K)
    lagrangian = sum(objective_terms) + cas.dot(multipliers, cas.vertcat(*closures))
    hessian = cas.tril(cas.hessian(lagrangian, variables)[0], True)
    hessian_function = cas.Function("hessian_of", [variables, multipliers], [hessian])

    rows, cols = hessian.sparsity().get_triplet()
    density = hessian.nnz() / (variables.size1() * (variables.size1() + 1) / 2)
    graph = hessian_function.n_nodes()
    structures[label] = (np.array(rows), np.array(cols), variables.size1())
    print(f"{label:<22}{variables.size1():>10}{hessian.nnz():>10}"
          f"{density:>8.0%}{graph:>8}")

print(
    "\nThe local models are the same in both rows. Sharing the design block removes\n"
    "sixteen variables, keeps the nonzeros at the same level, and runs its columns\n"
    "through every condition."
)

# ---------------------------------------------------------------------- figure
figure, axes = plt.subplots(1, 2, figsize=(10, 5))
for axis, label in zip(axes, structures):
    rows, cols, nvars = structures[label]
    axis.plot(cols, rows, ".", color="#1f4e79" if "shared" in label else "#2e8b57",
              markersize=2.2)
    axis.set_xlim(-1, nvars)
    axis.set_ylim(nvars, -1)
    axis.set_aspect("equal")
    axis.set_title(f"{label}\n{nvars} variables, {len(rows)} nonzeros", fontsize=10.5)
    axis.set_xlabel("variables", fontsize=9.5)
    edge = 4 if "shared" in label else 0
    if edge:
        axis.axvline(edge - 0.5, color="0.5", linewidth=0.7, linestyle=":")
axes[0].set_ylabel("variables", fontsize=9.5)
figure.suptitle(
    "One design block shared by every condition (left), one block per condition (right)",
    fontsize=11.5,
    y=0.98,
)
figure.tight_layout(rect=(0, 0.02, 1, 0.93))
figure.savefig("hessian_structure.png", dpi=170)
print("wrote hessian_structure.png")
