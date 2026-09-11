"""How the derivative cost grows with the size of the dense system.

Same model as example.py, five system sizes, plus two control objects of the same
dimension so that a dense algebra cost is not mistaken for a differentiation cost:

    model          primal, gradient, Hessian-vector product, exact Hessian,
                   and the same Hessian with the design parameters frozen
    constant       a dense constant matrix of the same shape: the floor cost of
                   materialising the output
    cubic          sum((C x)^3) with C dense and constant: a genuinely dense,
                   genuinely variable Hessian, with no model structure at all

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
rng = np.random.default_rng(0)

print(f"{'N':>4}{'vars':>6}{'nnz':>7} | {'grad ms':>9}{'H.v ms':>9}{'H ms':>9}"
      f"{'H frozen ms':>13}{'constant ms':>13}{'cubic ms':>10} |"
      f"{'nodes grad':>11}{'nodes H':>9}{'nodes Hf':>9}{'nodes cubic':>13}")
rows = []
for N in SIZES:
    locator = np.cos(np.pi * (np.arange(N) + 0.5) / N)
    design = cas.MX.sym("design", 4)
    controls = cas.MX.sym("controls", N)
    state = cas.MX.sym("state")
    variables = cas.vertcat(design, controls, state)
    nvars = int(variables.size1())

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

    response = cas.solve(matrix, rhs)
    argument = controls + COUPLING * (kernel @ response) / state
    transferred = OMEGA * argument - 5.0 * argument**3
    outputs = cas.vertcat(
        state * cas.sum1(transferred * weights), cas.sum1(argument * weights)
    )
    response_ref = cas.solve(
        cas.MX.eye(N) + cas.diag(OMEGA * weights_ref / state) @ kernel_ref,
        OMEGA * weights_ref * (controls + 0.1 * locator),
    )
    argument_ref = controls + COUPLING * (kernel_ref @ response_ref) / state
    transferred_ref = OMEGA * argument_ref - 5.0 * argument_ref**3
    outputs_ref = cas.vertcat(
        state * cas.sum1(transferred_ref * weights_ref),
        cas.sum1(argument_ref * weights_ref),
    )

    objective = cas.sum1(outputs)
    gradient_expression = cas.gradient(objective, variables)
    direction = cas.MX.sym("direction", nvars)
    dense_constant = cas.DM(np.ones((nvars, nvars)))
    dense_cubic = cas.DM(rng.standard_normal((nvars, nvars)) / np.sqrt(nvars))

    prime = cas.Function("outputs_of", [variables], [outputs])
    gradient = cas.Function("gradient_of", [variables], [gradient_expression])
    hessian_vector = cas.Function(
        "hv_of", [variables, direction],
        [cas.jtimes(gradient_expression, variables, direction)],
    )
    hessian = cas.Function(
        "hessian_of", [variables],
        [cas.tril(cas.hessian(objective, variables)[0], True)],
    )
    hessian_frozen = cas.Function(
        "hessian_frozen_of", [variables],
        [cas.tril(cas.hessian(cas.sum1(outputs_ref), variables)[0], True)],
    )
    constant = cas.Function(
        "constant_of", [variables], [cas.tril(dense_constant, True)]
    )
    cubic = cas.Function(
        "cubic_of", [variables],
        [cas.tril(cas.hessian(cas.sum1((dense_cubic @ variables) ** 3), variables)[0], True)],
    )

    point = np.concatenate([design_ref, np.full(N, 0.1), [state_value]])
    direction_value = rng.standard_normal(nvars)
    calls = {
        "gradient": (gradient, (point,)),
        "hessian_vector": (hessian_vector, (point, direction_value)),
        "hessian": (hessian, (point,)),
        "hessian_frozen": (hessian_frozen, (point,)),
        "constant": (constant, (point,)),
        "cubic": (cubic, (point,)),
    }
    for function, arguments in calls.values():
        for _ in range(2):
            function(*arguments)
    timings = {}
    for name, (function, arguments) in calls.items():
        runs = []
        for _ in range(EVALUATIONS):
            start = time.perf_counter()
            function(*arguments)
            runs.append(time.perf_counter() - start)
        timings[name] = float(np.median(runs)) * 1e3

    row = {
        "N": N,
        "variables": nvars,
        "nnz_hessian": int(hessian.nnz_out(0)),
        "nodes_gradient": int(gradient.n_nodes()),
        "nodes_hessian_vector": int(hessian_vector.n_nodes()),
        "nodes_hessian": int(hessian.n_nodes()),
        "nodes_hessian_frozen": int(hessian_frozen.n_nodes()),
        "nodes_cubic": int(cubic.n_nodes()),
        "gradient_ms": timings["gradient"],
        "hessian_vector_ms": timings["hessian_vector"],
        "hessian_ms": timings["hessian"],
        "hessian_frozen_ms": timings["hessian_frozen"],
        "constant_ms": timings["constant"],
        "cubic_ms": timings["cubic"],
    }
    row["hessian_us_per_nonzero"] = row["hessian_ms"] * 1e3 / row["nnz_hessian"]
    rows.append(row)
    print(f"{N:>4}{nvars:>6}{row['nnz_hessian']:>7} | "
          f"{row['gradient_ms']:>9.3f}{row['hessian_vector_ms']:>9.3f}{row['hessian_ms']:>9.2f}"
          f"{row['hessian_frozen_ms']:>13.2f}{row['constant_ms']:>13.4f}"
          f"{row['cubic_ms']:>10.2f} |"
          f"{row['nodes_gradient']:>11}{row['nodes_hessian']:>9}"
          f"{row['nodes_hessian_frozen']:>9}{row['nodes_cubic']:>13}")

Path(__file__).with_name("scaling.json").write_text(
    json.dumps(rows, indent=2) + "\n", encoding="utf-8"
)
print("\nwrote scaling.json")
