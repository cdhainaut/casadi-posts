"""One isolated case of the field/response-ratio homotopy.

Adapted from Joris Gillis's sparse-operator example (see ../03-control-rank/reference/).
The response model at N points is his, unchanged. The field is a chain of M = z*N
nodes whose Green function compresses exactly to his response kernel, K = R L^-1 S,
so all four writings stay algebraically equivalent at every field ratio z. The dense
writings are literally his; only the sparse ones see the growing field.

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


def interaction(design: cas.MX, nodes: int) -> tuple:
    """Joris's lattice constants for a chain of ``nodes`` stations."""
    length = LENGTH * cas.exp(0.6 * design[0] - 0.4 * design[1])
    stiffness = length**2 / (2.0 / nodes) ** 2
    mu = 1.0 + 1.0 / (2.0 * stiffness)
    ratio = mu - cas.sqrt(mu**2 - 1.0)
    amplitude = 1.0 / (1.0 + 2.0 * stiffness * (1.0 - ratio))
    return stiffness, ratio, amplitude


def dense_kernel(design: cas.MX, n: int) -> cas.MX:
    """Joris's response kernel, unchanged: amplitude * ratio ** |j - k|."""
    _, ratio, amplitude = interaction(design, n)
    offsets = cas.DM(np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]))
    return amplitude * ratio**offsets


def field_constants(design: cas.MX, n: int, z: int) -> tuple:
    """Field-chain constants whose Green function compresses to the response kernel.

    The field chain carries z nodes per response step, so its per-node decay is the
    z-th root of the response decay; stiffness then follows from Joris's lattice
    relation and the amplitude ratio becomes the operator scale. At z = 1 every
    constant reduces to Joris's.
    """
    _, ratio_n, amplitude_n = interaction(design, n)
    ratio_m = ratio_n ** (1.0 / z)
    mu_m = (1.0 + ratio_m**2) / (2.0 * ratio_m)
    stiffness_m = 1.0 / (2.0 * (mu_m - 1.0))
    amplitude_m = 1.0 / (1.0 + 2.0 * stiffness_m * (1.0 - ratio_m))
    return stiffness_m, ratio_m, amplitude_m, amplitude_m / amplitude_n


def field_operator(design: cas.MX, n: int, z: int) -> cas.MX:
    """Sparse field operator L with R L^-1 S equal to the response kernel exactly."""
    m = z * n
    stiffness, ratio, _, scale = field_constants(design, n, z)
    lap = cas.sparsify(cas.DM(
        np.diag(2.0 * np.ones(m)) - np.diag(np.ones(m - 1), 1) - np.diag(np.ones(m - 1), -1)
    ))
    ends = cas.sparsify(cas.DM(np.diag(np.eye(m)[0] + np.eye(m)[-1])))
    return scale * (cas.DM.eye(m) + stiffness * (lap - ratio * ends))


def selection_matrix(n: int, z: int) -> cas.DM:
    """S selects the observed nodes, R = S^T observes: response j at node j*z."""
    return cas.DM.triplet([j * z for j in range(n)], list(range(n)), [1.0] * n, z * n, n)


def outputs_of(smoothed, controls, state, weights) -> cas.MX:
    """Original nonlinear nodewise reading, unchanged from Joris's example."""
    argument = controls + COUPLING * smoothed / state
    warped = OMEGA * argument - 5.0 * argument**3
    return cas.vertcat(state * cas.sum1(warped * weights), cas.sum1(argument * weights))


def build(n: int, z: int, conditions: int, kernel: str, lifted: bool) -> tuple:
    """Four writings of one model; only the field size z*n changes between them."""
    opti = cas.Opti()
    design = opti.variable(4)
    locator = -1.0 + (np.arange(n) + 0.5) * 2.0 / n
    weights = WEIGHT * design[2] * (1.0 - 0.4 * design[3] * locator)
    influence = dense_kernel(design, n) if kernel == "dense" else None
    operator = field_operator(design, n, z) if kernel == "sparse" else None
    selection = selection_matrix(n, z) if kernel == "sparse" else None
    objective = 0
    guesses = []
    lifted_pairs = []
    controls_list = []
    states_list = []
    for _ in range(conditions):
        controls = opti.variable(n)
        state = opti.variable()
        controls_list.append(controls)
        states_list.append(state)
        rhs = OMEGA * weights * (controls + 0.1 * locator)
        if kernel == "dense":
            matrix = cas.MX.eye(n) + cas.diag(OMEGA * weights / state) @ influence
            eliminated = cas.solve(matrix, rhs)
            if lifted:
                variable = response = opti.variable(n)
                opti.subject_to(matrix @ response == rhs)
            else:
                response = eliminated
            smoothed = influence @ response
        elif kernel == "sparse":
            system = operator + selection @ cas.diag(OMEGA * weights / state) @ selection.T
            eliminated = cas.solve(system, selection @ rhs)
            if lifted:
                variable = field = opti.variable(z * n)
                opti.subject_to(system @ field == selection @ rhs)
            else:
                field = eliminated
            smoothed = selection.T @ field
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
        objective = objective + cas.sum1(outputs_of(smoothed, controls, state, weights))
        guesses.append(cas.Function(f"guess{len(guesses)}", [design, controls, state],
                                    [eliminated]))
        if lifted:
            lifted_pairs.append((variable, controls, state))
    opti.minimize(objective)
    opti.subject_to(opti.bounded(0.25, design, 1.0))
    for controls in controls_list:
        opti.subject_to(opti.bounded(-1.0, controls, 1.0))
    for state in states_list:
        opti.subject_to(opti.bounded(5.0, state, 20.0))
    opti.set_initial(design, DESIGN_GUESS)
    for controls in controls_list:
        opti.set_initial(controls, np.full(n, CONTROL_GUESS))
    for state in states_list:
        opti.set_initial(state, STATE_GUESS)
    if lifted:
        for index, (variable, controls, state) in enumerate(lifted_pairs):
            opti.set_initial(variable,
                             guesses[index](DESIGN_GUESS, np.full(n, CONTROL_GUESS),
                                            STATE_GUESS))
    return opti, design, controls_list, states_list, objective


def compression_check(n: int, z: int) -> float:
    """Max residual of K = R L^-1 S at three fixed admissible designs."""
    design = cas.MX.sym("design", 4)
    operator = cas.Function("operator", [design], [field_operator(design, n, z)])
    kernel = cas.Function("kernel", [design], [dense_kernel(design, n)])
    selection = selection_matrix(n, z)
    points = (DESIGN_GUESS, np.full(4, 0.25), np.ones(4))
    worst = 0.0
    for point in points:
        matrix = operator(point)
        solver = cas.Linsol("check", "csparse", matrix.sparsity())
        observed = cas.DM(selection.T) @ solver.solve(matrix, selection)
        worst = max(worst, float(cas.norm_inf(observed - kernel(point))))
    return worst


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


def solution_metrics(opti, solution, design, controls_list, states_list) -> dict:
    """Unscaled KKT residuals using the actual solver gradient and Jacobian."""
    solver = opti.debug.casadi_solver
    x = solution.value(opti.x)
    multipliers = np.asarray(solution.value(opti.lam_g)).ravel()
    gradient = solver.get_function("nlp_grad_f")(x, [])[-1]
    jacobian = solver.get_function("nlp_jac_g")(x, [])[-1]
    stationarity = gradient + jacobian.T @ multipliers
    values = np.asarray(solution.value(opti.g)).ravel()
    lower = np.asarray(opti.value(opti.lbg)).ravel()
    upper = np.asarray(opti.value(opti.ubg)).ravel()
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
        "controls": [np.asarray(solution.value(c)).ravel().tolist() for c in controls_list],
        "state": [float(solution.value(s)) for s in states_list],
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
    parser.add_argument("--field-ratio", type=int, required=True)
    parser.add_argument("--conditions", type=int, required=True)
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
        "stations": args.stations, "field_ratio": args.field_ratio,
        "field_unknowns": args.field_ratio * args.stations * args.conditions,
        "conditions": args.conditions,
        "kernel": args.kernel, "lifted": args.lifted, "controls": "full (rank = N)",
        "versions": {"python": platform.python_version(), "casadi": cas.__version__,
                     "numpy": np.__version__},
        "ipopt_options": SOLVER_OPTIONS, "status": "preflight",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    try:
        result["compression_error_inf"] = compression_check(args.stations, args.field_ratio)
        if result["compression_error_inf"] > 1e-10:
            raise ValueError("Kernel compression check exceeds 1e-10")
        start = time.perf_counter()
        opti, design, controls, states, _ = build(
            args.stations, args.field_ratio, args.conditions, args.kernel, args.lifted
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
        result["solution"] = solution_metrics(opti, solution, design, controls, states)
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
