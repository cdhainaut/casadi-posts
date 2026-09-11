"""How the derivative cost grows with the size of the dense system.

Same model as example.py, five system sizes, with the kernel written once as a
matrix expression. For each size:

    primal     the outputs themselves
    gradient   first-order derivatives of the summed outputs
    hessian    exact Lagrangian Hessian of the summed outputs
    frozen     same Hessian with the design parameters frozen inside the matrix

Reports graph sizes (machine independent) and evaluation times. The interesting
question is which curve leaves the others behind, and at which size.

Writes scaling.json next to this file; figures.py draws it.

Run:  python scaling.py
"""

import json
import time
from pathlib import Path

import casadi as cas
import numpy as np

SIZES = (12, 24, 48, 96, 192)
OMEGA = 2.0 * np.pi
WEIGHT = 0.4
COUPLING = 0.3
EVALUATIONS = 20

design_ref = np.array([0.5, 0.3, 1.0, 0.4])
state_value = 12.0

print(f"{'N':>5}{'vars':>6}{'nnz H':>9}{'nodes primal':>14}{'nodes grad':>12}"
      f"{'nodes H':>10}{'nodes H frozen':>16}{'build H (s)':>13}"
      f"{'eval H (ms)':>13}{'eval H frozen (ms)':>20}")
rows = []
for N in SIZES:
    locator = np.cos(np.pi * (np.arange(N) + 0.5) / N)
    design = cas.MX.sym("design", 4)
    controls = cas.MX.sym("controls", N)
    state = cas.MX.sym("state")

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

    variables = cas.vertcat(design, controls, state)
    response = cas.solve(matrix, rhs)
    argument = controls + COUPLING * (kernel @ response) / state
    transferred = OMEGA * argument - 5.0 * argument**3
    outputs = cas.vertcat(
        state * cas.sum1(transferred * weights), cas.sum1(argument * weights)
    )
    response_ref = cas.solve(matrix_ref, rhs_ref)
    argument_ref = controls + COUPLING * (kernel_ref @ response_ref) / state
    transferred_ref = OMEGA * argument_ref - 5.0 * argument_ref**3
    outputs_ref = cas.vertcat(
        state * cas.sum1(transferred_ref * weights_ref),
        cas.sum1(argument_ref * weights_ref),
    )

    timer = time.perf_counter()
    prime = cas.Function("outputs_of", [variables], [outputs])
    gradient = cas.Function("gradient_of", [variables], [cas.gradient(cas.sum1(outputs), variables)])
    hessian = cas.Function(
        "hessian_of", [variables], [cas.tril(cas.hessian(cas.sum1(outputs), variables)[0], True)]
    )
    build = time.perf_counter() - timer
    hessian_frozen = cas.Function(
        "hessian_frozen_of",
        [variables],
        [cas.tril(cas.hessian(cas.sum1(outputs_ref), variables)[0], True)],
    )

    point = np.concatenate([design_ref, np.full(N, 0.1), [state_value]])
    for _ in range(2):
        prime(point)
        gradient(point)
        hessian(point)
        hessian_frozen(point)
    timings, timings_gradient, timings_frozen = [], [], []
    for _ in range(EVALUATIONS):
        start = time.perf_counter()
        gradient(point)
        timings_gradient.append(time.perf_counter() - start)
        start = time.perf_counter()
        hessian(point)
        timings.append(time.perf_counter() - start)
        start = time.perf_counter()
        hessian_frozen(point)
        timings_frozen.append(time.perf_counter() - start)

    rows.append({
        "N": N,
        "variables": int(variables.size1()),
        "nnz_hessian": int(hessian.nnz_out(0)),
        "nodes_primal": int(prime.n_nodes()),
        "nodes_gradient": int(gradient.n_nodes()),
        "nodes_hessian": int(hessian.n_nodes()),
        "nodes_hessian_frozen": int(hessian_frozen.n_nodes()),
        "build_s": build,
        "gradient_ms": float(np.median(timings_gradient)) * 1e3,
        "hessian_ms": float(np.median(timings)) * 1e3,
        "hessian_frozen_ms": float(np.median(timings_frozen)) * 1e3,
    })
    print(f"{N:>5}{variables.size1():>6}{hessian.nnz_out(0):>9}{prime.n_nodes():>14}"
          f"{gradient.n_nodes():>12}{hessian.n_nodes():>10}"
          f"{hessian_frozen.n_nodes():>16}{build:>13.2f}"
          f"{np.median(timings_gradient) * 1e3:>15.3f}"
          f"{np.median(timings) * 1e3:>13.2f}{np.median(timings_frozen) * 1e3:>20.2f}")

Path(__file__).with_name("scaling.json").write_text(
    json.dumps(rows, indent=2) + "\n", encoding="utf-8"
)
print("\nwrote scaling.json")
