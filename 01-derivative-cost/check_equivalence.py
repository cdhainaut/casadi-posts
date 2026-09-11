"""Check that the two ways of writing the dense kernel are the same object.

The kernel can be assembled entry by entry in a Python loop, or written as one
matrix expression. With MX the graph is very different — that was the point of
the first lesson — but the numbers must be identical, derivatives included.

This script builds both forms and compares, at several design points:

    the kernel itself
    the outputs
    the gradient of the summed outputs
    the Lagrangian Hessian of the summed outputs

Run:  python check_equivalence.py
"""

import casadi as cas
import numpy as np

N = 24
OMEGA = 2.0 * np.pi
WEIGHT = 0.4
COUPLING = 0.3
TOLERANCE = 1e-12

locator = np.cos(np.pi * (np.arange(N) + 0.5) / N)
design_ref = np.array([0.5, 0.3, 1.0, 0.4])
controls_ref = np.full(N, 0.1)
state_ref = 12.0
test_points = [
    np.array([0.0, 0.0, 1.0, 0.0]),
    np.array([0.5, 0.3, 1.0, 0.4]),
    np.array([1.0, -0.5, 0.7, 0.9]),
    np.array([-0.4, 0.8, 1.3, -0.2]),
]

design = cas.MX.sym("design", 4)
controls = cas.MX.sym("controls", N)
state = cas.MX.sym("state")
variables = cas.vertcat(design, controls, state)

coordinates = (
    locator
    + 0.12 * design[0] * (1.0 - locator**2)
    + 0.08 * design[1] * locator * (1.0 - locator**2)
)


def kernel_scalar():
    """Entry-by-entry assembly: one node per element of the matrix."""
    out = cas.MX.zeros(N, N)
    for i in range(N):
        for j in range(N):
            if i != j:
                out[i, j] = (coordinates[1] - coordinates[0]) / (
                    4.0 * np.pi * (coordinates[i] - coordinates[j])
                )
    return out


def kernel_matrix():
    """One matrix expression: repmat, elementwise division, diagonal cleared."""
    span = cas.reshape(coordinates, N, 1)
    difference = cas.repmat(span, 1, N) - cas.repmat(span.T, N, 1) + cas.MX.eye(N)
    out = ((coordinates[1] - coordinates[0]) / (4.0 * np.pi)) / difference
    return out - cas.diag(cas.diag(out))


def model(kernel):
    weights = WEIGHT * design[2] * (1.0 - 0.4 * design[3] * locator)
    matrix = cas.MX.eye(N) + cas.diag(OMEGA * weights / state) @ kernel
    rhs = OMEGA * weights * (controls + 0.1 * locator)
    response = cas.solve(matrix, rhs)
    argument = controls + COUPLING * (kernel @ response) / state
    transferred = OMEGA * argument - 5.0 * argument**3
    return cas.vertcat(
        state * cas.sum1(transferred * weights), cas.sum1(argument * weights)
    )


functions = {}
for name, kernel in (("scalar", kernel_scalar()), ("matrix", kernel_matrix())):
    outputs = model(kernel)
    objective = cas.sum1(outputs)
    functions[name] = {
        "kernel": cas.Function(f"kernel_{name}", [design], [kernel]),
        "outputs": cas.Function(f"outputs_{name}", [variables], [outputs]),
        "gradient": cas.Function(
            f"gradient_{name}", [variables], [cas.gradient(objective, variables)]
        ),
        "hessian": cas.Function(
            f"hessian_{name}",
            [variables],
            [cas.tril(cas.hessian(objective, variables)[0], True)],
        ),
    }

print(f"{'quantity':<12}{'max abs difference over 4 design points':>42}")
worst = {}
for quantity in ("kernel", "outputs", "gradient", "hessian"):
    largest = 0.0
    for point in test_points:
        arguments = (
            [point]
            if quantity == "kernel"
            else [np.concatenate([point, controls_ref, [state_ref]])]
        )
        left = np.asarray(functions["scalar"][quantity](*arguments)).ravel()
        right = np.asarray(functions["matrix"][quantity](*arguments)).ravel()
        largest = max(largest, float(np.max(np.abs(left - right))))
    worst[quantity] = largest
    verdict = "ok" if largest < TOLERANCE else "FAILED"
    print(f"{quantity:<12}{largest:>42.3e}   {verdict}")

import time

print(f"\nnodes: kernel {functions['scalar']['kernel'].n_nodes()} (loop) against "
      f"{functions['matrix']['kernel'].n_nodes()} (matrix), "
      f"Hessian {functions['scalar']['hessian'].n_nodes()} against "
      f"{functions['matrix']['hessian'].n_nodes()}")

point = np.concatenate([design_ref, controls_ref, [state_ref]])
print(f"\n{'writing':<10}{'Hessian nodes':>15}{'one evaluation':>16}")
for name in ("scalar", "matrix"):
    function = functions[name]["hessian"]
    for _ in range(3):
        function(point)
    runs = []
    for _ in range(20):
        start = time.perf_counter()
        function(point)
        runs.append(time.perf_counter() - start)
    print(f"{name:<10}{function.n_nodes():>15}{np.median(runs) * 1e3:>13.2f} ms")
if max(worst.values()) < TOLERANCE:
    print("the two writings are the same object")
else:
    raise SystemExit("the two writings disagree")
