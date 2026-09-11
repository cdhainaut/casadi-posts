"""A dense linear solve can be cheap to evaluate and expensive to differentiate.

The model, in one page:

    design parameters p, controls u, state v
    A(p, v) y = b(p, u, v)          dense, and A depends on the variables
    outputs   = f(y, u, p)          some non-linear reading of the response

Evaluating the outputs is cheap. Computing their exact Hessian is not, once the
coefficient matrix depends on the variables: that dependence is what generates
the expensive terms. This script measures the four quantities side by side:

    gradient                         first order
    Hessian-vector product           directional second order
    exact Hessian                    the full matrix
    exact Hessian, A constant        same model, coefficients independent of x

Run:  python example.py
"""

import time

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------ parameters
N = 192  # size of the dense system
OMEGA = 2.0 * np.pi
WEIGHT = 0.4
COUPLING = 0.3
EVALUATIONS = 10
design_value = np.array([0.5, 0.3, 1.0, 0.4])
state_value = 12.0
locator = np.cos(np.pi * (np.arange(N) + 0.5) / N)

# --------------------------------------------------------------------- symbols
design = cas.MX.sym("design", 4)
controls = cas.MX.sym("controls", N)
state = cas.MX.sym("state")
variables = cas.vertcat(design, controls, state)

# ------------------------------------------ dense matrix, from the coordinates
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

# ---------------------------------------------------------------- the responses
response = cas.solve(matrix, rhs)
argument = controls + COUPLING * (kernel @ response) / state
warped = OMEGA * argument - 5.0 * argument**3
outputs = cas.vertcat(
    state * cas.sum1(warped * weights), cas.sum1(argument * weights)
)
objective = cas.sum1(outputs)

# ------------------------------- the same model, coefficients free of variables
coordinates_ref = (
    locator
    + 0.12 * design_value[0] * (1.0 - locator**2)
    + 0.08 * design_value[1] * locator * (1.0 - locator**2)
)
difference_ref = coordinates_ref[:, None] - coordinates_ref[None, :]
np.fill_diagonal(difference_ref, 1.0)
kernel_ref = ((coordinates_ref[1] - coordinates_ref[0]) / (4.0 * np.pi)) / difference_ref
np.fill_diagonal(kernel_ref, 0.0)
weights_ref = WEIGHT * design_value[2] * (1.0 - 0.4 * design_value[3] * locator)
matrix_ref = cas.DM(
    np.eye(N)
    + np.diag(np.asarray(weights_ref).ravel() * OMEGA / state_value)
    @ np.asarray(kernel_ref)
)
response_ref = cas.solve(matrix_ref, rhs)
argument_ref = controls + COUPLING * (kernel @ response_ref) / state
warped_ref = OMEGA * argument_ref - 5.0 * argument_ref**3
outputs_ref = cas.vertcat(
    state * cas.sum1(warped_ref * weights), cas.sum1(argument_ref * weights)
)
objective_ref = cas.sum1(outputs_ref)

# ------------------------------------------------------------------ derivatives
direction = cas.MX.sym("direction", variables.size1())
gradient = cas.Function(
    "gradient_of", [variables], [cas.gradient(objective, variables)]
)
hessian_vector = cas.Function(
    "hessian_vector_of",
    [variables, direction],
    [cas.jtimes(cas.gradient(objective, variables), variables, direction)],
)
hessian = cas.Function(
    "hessian_of",
    [variables],
    [cas.tril(cas.hessian(objective, variables)[0], True)],
)
hessian_ref = cas.Function(
    "hessian_ref_of",
    [variables],
    [cas.tril(cas.hessian(objective_ref, variables)[0], True)],
)

# ------------------------------------------------------------------- the timing
point = np.concatenate([design_value, np.full(N, 0.1), [state_value]])
direction_value = np.random.default_rng(0).standard_normal(variables.size1())
calls = {
    "gradient": (gradient, (point,)),
    "Hessian-vector product": (hessian_vector, (point, direction_value)),
    "exact Hessian": (hessian, (point,)),
    "exact Hessian, A constant": (hessian_ref, (point,)),
}
for function, arguments in calls.values():
    for _ in range(2):
        function(*arguments)

print(f"{'quantity':<28}{'graph nodes':>13}{'one evaluation':>17}")
results = []
for name, (function, arguments) in calls.items():
    runs = []
    for _ in range(EVALUATIONS):
        start = time.perf_counter()
        function(*arguments)
        runs.append(time.perf_counter() - start)
    milliseconds = float(np.median(runs)) * 1e3
    results.append((name, function.n_nodes(), milliseconds))
    print(f"{name:<28}{function.n_nodes():>13}{milliseconds:>14.2f} ms")

print(
    f"\nAt N = {N}, the exact Hessian costs "
    f"{results[2][2] / results[3][2]:.1f}x the same model with constant coefficients, "
    f"and {results[2][2] / results[1][2]:.1f}x its own Hessian-vector product."
)

# ---------------------------------------------------------------------- figure
names = [row[0] for row in results]
times = [row[2] for row in results]
colors = ["#2e8b57", "#7fb3d5", "#1f4e79", "#8e44ad"]
figure, axis = plt.subplots(figsize=(8.5, 3.6))
axis.barh(np.arange(len(names))[::-1], times, color=colors, height=0.55)
axis.set_xscale("log")
axis.set_xlim(right=max(times) * 6)
axis.set_yticks(np.arange(len(names))[::-1])
axis.set_yticklabels(names, fontsize=10)
axis.set_xlabel("milliseconds for one evaluation (log scale)", fontsize=9.5)
axis.grid(axis="x", alpha=0.25, linewidth=0.6)
axis.set_axisbelow(True)
for position, value in zip(np.arange(len(names))[::-1], times):
    axis.text(value * 1.25, position, f"{value:.1f} ms", va="center", fontsize=9)
axis.set_title(
    f"Same model at N = {N}: the cost is the dependence of A on the variables",
    fontsize=11,
)
figure.tight_layout()
figure.savefig("hessian_cost.png", dpi=170)
print("wrote hessian_cost.png")
