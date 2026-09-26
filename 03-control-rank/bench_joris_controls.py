"""One isolated case of Joris Gillis's sparse-operator example, with control reduction.

Adapted from sparse_operator.py supplied by Joris (see reference/README.md).
The full-control case preserves his equations, bounds and initialisation.
This standalone measurement harness needs only NumPy, CasADi and IPOPT.
"""

import argparse
import json
import platform
import resource
import time
import traceback
from pathlib import Path

import casadi as cas
import numpy as np

OMEGA = 2.0 * np.pi
WEIGHT = 0.4
COUPLING = 0.3
LENGTH = 0.15
DESIGN_GUESS = np.array([0.5, 0.3, 1.0, 0.4])
CONTROL_GUESS = 0.1
STATE_GUESS = 12.0
HESSIAN_REPEATS = 3
SOLVER_OPTIONS = {
    "hessian_approximation": "exact",
    "linear_solver": "mumps",
    "tol": 1e-8,
    "max_iter": 3000,
    "print_level": 5,
    "print_user_options": "yes",
}


def interaction(design: cas.MX, n: int) -> tuple:
    """Original lattice constants, including the absorbing-end correction."""
    length = LENGTH * cas.exp(0.6 * design[0] - 0.4 * design[1])
    stiffness = length**2 / (2.0 / n) ** 2
    mu = 1.0 + 1.0 / (2.0 * stiffness)
    ratio = mu - cas.sqrt(mu**2 - 1.0)
    amplitude = 1.0 / (1.0 + 2.0 * stiffness * (1.0 - ratio))
    return stiffness, ratio, amplitude


def dense_kernel(design: cas.MX, n: int) -> cas.MX:
    _, ratio, amplitude = interaction(design, n)
    offsets = cas.DM(np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]))
    return amplitude * ratio**offsets


def sparse_operator(design: cas.MX, n: int) -> cas.MX:
    stiffness, ratio, _ = interaction(design, n)
    laplacian = cas.sparsify(cas.DM(
        np.diag(2.0 * np.ones(n)) - np.diag(np.ones(n - 1), 1)
        - np.diag(np.ones(n - 1), -1)
    ))
    ends = cas.sparsify(cas.DM(np.diag(np.eye(n)[0] + np.eye(n)[-1])))
    return cas.DM.eye(n) + stiffness * (laplacian - ratio * ends)


def control_basis(n: int, rank: int) -> np.ndarray:
    """Full controls, a constant, or the first four Legendre modes on the stations."""
    if rank == n:
        return np.eye(n)
    if rank not in (1, 4) or rank > n:
        raise ValueError("Control rank must be 1, 4, or the station count")
    locator = -1.0 + (np.arange(n) + 0.5) * 2.0 / n
    return np.polynomial.legendre.legvander(locator, rank - 1)


def outputs_of(smoothed, controls, state, weights) -> cas.MX:
    """Original nonlinear nodewise reading, unchanged from Joris's example."""
    argument = controls + COUPLING * smoothed / state
    warped = OMEGA * argument - 5.0 * argument**3
    return cas.vertcat(state * cas.sum1(warped * weights), cas.sum1(argument * weights))


def build(n: int, rank: int, kernel: str, lifted: bool) -> tuple:
    """Return the original Opti tuple; only the spatial control basis is changed."""
    basis = control_basis(n, rank)
    opti = cas.Opti()
    design = opti.variable(4)
    coefficients = opti.variable(rank)
    controls = coefficients if rank == n else cas.DM(basis) @ coefficients
    state = opti.variable()
    locator = -1.0 + (np.arange(n) + 0.5) * 2.0 / n
    weights = WEIGHT * design[2] * (1.0 - 0.4 * design[3] * locator)
    rhs = OMEGA * weights * (controls + 0.1 * locator)
    if kernel == "dense":
        influence = dense_kernel(design, n)
        matrix = cas.MX.eye(n) + cas.diag(OMEGA * weights / state) @ influence
        eliminated = cas.solve(matrix, rhs)
        if lifted:
            variable = response = opti.variable(n)
            opti.subject_to(matrix @ response == rhs)
        else:
            response = eliminated
        smoothed = influence @ response
    elif kernel == "sparse":
        matrix = sparse_operator(design, n) + cas.diag(OMEGA * weights / state)
        eliminated = cas.solve(matrix, rhs)
        if lifted:
            variable = smoothed = opti.variable(n)
            opti.subject_to(matrix @ smoothed == rhs)
        else:
            smoothed = eliminated
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    outputs = outputs_of(smoothed, controls, state, weights)
    opti.minimize(cas.sum1(outputs))
    opti.subject_to(opti.bounded(0.25, design, 1.0))
    opti.subject_to(opti.bounded(-1.0, controls, 1.0))
    opti.subject_to(opti.bounded(5.0, state, 20.0))
    coefficient_guess = np.full(rank, CONTROL_GUESS) if rank == n else np.zeros(rank)
    if rank != n:
        coefficient_guess[0] = CONTROL_GUESS
    opti.set_initial(design, DESIGN_GUESS)
    opti.set_initial(coefficients, coefficient_guess)
    opti.set_initial(state, STATE_GUESS)
    if lifted:
        guess = cas.Function("guess", [design, coefficients, state], [eliminated])
        opti.set_initial(variable, guess(DESIGN_GUESS, coefficient_guess, STATE_GUESS))
    return opti, design, controls, state, outputs


def kernel_check(n: int) -> float:
    """Maximum entrywise inverse residual at three fixed admissible designs."""
    design = cas.MX.sym("design", 4)
    residual = sparse_operator(design, n) @ dense_kernel(design, n) - cas.DM.eye(n)
    evaluate = cas.Function("pair_check", [design], [residual])
    points = (DESIGN_GUESS, np.full(4, 0.25), np.ones(4))
    return max(float(cas.norm_inf(evaluate(point))) for point in points)


def constraint_violation(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(max(0.0, np.max(lower - values), np.max(values - upper)))


def initial_metrics(opti: cas.Opti, lifted: bool, n: int) -> dict:
    values = np.asarray(opti.debug.value(opti.g, opti.initial())).ravel()
    lower = np.asarray(opti.value(opti.lbg)).ravel()
    upper = np.asarray(opti.value(opti.ubg)).ravel()
    return {
        "primal_violation_inf": constraint_violation(values, lower, upper),
        "response_residual_inf": float(np.max(np.abs(values[:n]))) if lifted else None,
        "variable_scaling": "identity (original CasADi Opti)",
        "constraint_scaling": "identity before IPOPT internal scaling",
    }


def solution_metrics(opti, solution, design, controls, state) -> dict:
    """Unscaled KKT residuals using the actual solver gradient and Jacobian."""
    solver = opti.debug.casadi_solver
    x = solution.value(opti.x)
    multipliers = np.asarray(solution.value(opti.lam_g)).ravel()
    gradient = solver.get_function("nlp_grad_f")(x, [])[-1]
    jacobian = solver.get_function("nlp_jac_g")(x, [])[-1]
    stationarity = gradient + jacobian.T @ multipliers
    values = np.asarray(solution.value(opti.g)).ravel()
    lower = np.asarray(solution.value(opti.lbg)).ravel()
    upper = np.asarray(solution.value(opti.ubg)).ravel()
    inequality = lower != upper
    lower_active = inequality & (multipliers < 0) & np.isfinite(lower)
    upper_active = inequality & (multipliers > 0) & np.isfinite(upper)
    products = np.concatenate((
        multipliers[lower_active] * (values[lower_active] - lower[lower_active]),
        multipliers[upper_active] * (upper[upper_active] - values[upper_active]),
        [0.0],
    ))
    return {
        "objective": float(solution.value(opti.f)),
        "design": np.asarray(solution.value(design)).ravel().tolist(),
        "controls": np.asarray(solution.value(controls)).ravel().tolist(),
        "state": float(solution.value(state)),
        "primal_violation_inf": constraint_violation(values, lower, upper),
        "stationarity_inf": float(cas.norm_inf(stationarity)),
        "complementarity_inf": float(np.max(np.abs(products))),
    }


def derivative_metrics(opti: cas.Opti, stats: dict, x_initial: np.ndarray) -> dict:
    """IPOPT triangle and warmed evaluation at the initial primal point, lambda=1."""
    solver = opti.debug.casadi_solver
    hessian = solver.get_function("nlp_hess_l")
    jacobian = solver.get_function("nlp_jac_g")
    arguments = (x_initial, [], 1.0, np.ones(opti.ng))
    hessian(*arguments)
    timings = []
    for _ in range(HESSIAN_REPEATS):
        start = time.perf_counter()
        hessian(*arguments)
        timings.append(time.perf_counter() - start)
    calls = stats.get("n_call_nlp_hess_l", 0)
    return {
        "jacobian_nnz": jacobian.sparsity_out(1).nnz(),
        "hessian_triangle_nnz": hessian.sparsity_out(0).nnz(),
        "hessian_graph_nodes": hessian.n_nodes(),
        "hessian_calls": calls,
        "hessian_total_seconds": stats.get("t_wall_nlp_hess_l", 0.0),
        "hessian_mean_seconds": stats.get("t_wall_nlp_hess_l", 0.0) / calls if calls else None,
        "hessian_initial_seconds": timings,
        "hessian_initial_median_seconds": float(np.median(timings)),
        "initial_multiplier_convention": "objective factor 1, all constraint multipliers 1",
        "jacobian_total_seconds": stats.get("t_wall_nlp_jac_g", 0.0),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", type=int, required=True)
    parser.add_argument("--controls", type=int, required=True)
    parser.add_argument("--kernel", choices=("dense", "sparse"), required=True)
    parser.add_argument("--lifted", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "stations": args.stations, "control_rank": args.controls,
        "kernel": args.kernel, "lifted": args.lifted,
        "versions": {"python": platform.python_version(), "casadi": cas.__version__,
                     "numpy": np.__version__},
        "ipopt_options": SOLVER_OPTIONS, "status": "preflight",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    try:
        result["kernel_inverse_error_inf"] = kernel_check(args.stations)
        if result["kernel_inverse_error_inf"] > 1e-10:
            raise ValueError("Kernel inverse check exceeds 1e-10")
        start = time.perf_counter()
        opti, design, controls, state, _ = build(
            args.stations, args.controls, args.kernel, args.lifted
        )
        result["model_build_seconds"] = time.perf_counter() - start
        result.update(variables=opti.nx, constraints=opti.ng)
        x_initial = np.asarray(opti.debug.value(opti.x, opti.initial())).ravel()
        result["initial"] = initial_metrics(opti, args.lifted, args.stations)
        if result["initial"]["primal_violation_inf"] > 1e-8:
            raise ValueError("Initial primal violation exceeds 1e-8")
        opti.solver("ipopt", {"print_time": True}, SOLVER_OPTIONS)
        result["status"] = "solving"
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        start = time.perf_counter()
        solution = opti.solve()
        result["solve_seconds"] = time.perf_counter() - start
        stats = solution.stats()
        result.update(status=stats["return_status"], iterations=stats["iter_count"])
        result["solution"] = solution_metrics(opti, solution, design, controls, state)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        result["derivatives"] = derivative_metrics(opti, stats, x_initial)
    except Exception:
        result["status"] = "failed"
        result["exception"] = traceback.format_exc()
        raise
    finally:
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
