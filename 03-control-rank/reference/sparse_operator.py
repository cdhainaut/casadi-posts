"""The same dense influence matrix, written as the inverse of a sparse operator.

The model of example.py has a dense influence matrix K(p) that comes from an
interaction law. Here the interaction is an exponential kernel,

    K_ij(p) = a(p) r(p)^|i-j|,                       dense, every entry depends on p

which is exactly the discrete Green's function of a tridiagonal operator,

    L(p) = I + c(p) T,    T = tridiag(-1, 2, -1) with absorbing ends,    K = L^-1.

The model is otherwise that of example.py:

    A(p, v) y = b(p, u, v),     A = I + (omega / v) diag(w) K
    z = K y,   a = u + kappa z / v,   outputs = f(a, w, v)

and the example problem is min sum(outputs) under a box; the lower bound on the
design is 0.25 rather than 0, which keeps the weights away from zero and removes
a degenerate local minimum with w = 0 and objective 0.
Four writings of the same problem are solved with IPOPT:

    dense, eliminated     y = solve(A, b) with the dense K       197 variables
    dense, lifted         y a variable, A y = b a constraint     389 variables
    sparse, eliminated    z = solve(L + (omega/v) diag(w), b)    197 variables
    sparse, lifted        z a variable, (L + ...) z = b          389 variables

Run:  python sparse_operator.py            (about two minutes)
"""

import json
import time
from pathlib import Path

import casadi as cas
import numpy as np

# ------------------------------------------------------------------ parameters
N = 192
OMEGA = 2.0 * np.pi
WEIGHT = 0.4
COUPLING = 0.3
LENGTH = 0.15  # reference range of the interaction
design_guess = np.array([0.5, 0.3, 1.0, 0.4])
controls_guess = 0.1
state_guess = 12.0
spacing = 2.0 / N
locator = -1.0 + (np.arange(N) + 0.5) * spacing
offsets = cas.DM(np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]))


def interaction(design):
    """Range of the interaction and the derived lattice constants."""
    length = LENGTH * cas.exp(0.6 * design[0] - 0.4 * design[1])
    stiffness = length**2 / spacing**2
    mu = 1.0 + 1.0 / (2.0 * stiffness)
    ratio = mu - cas.sqrt(mu**2 - 1.0)
    amplitude = 1.0 / (1.0 + 2.0 * stiffness * (1.0 - ratio))
    return stiffness, ratio, amplitude


def dense_kernel(design):
    """K(p) as a formula: N x N, every entry depends on p."""
    _, ratio, amplitude = interaction(design)
    return amplitude * ratio**offsets


LAPLACIAN = cas.sparsify(cas.DM(
    np.diag(2.0 * np.ones(N)) - np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1)
))
ENDS = cas.sparsify(cas.DM(np.diag(np.eye(N)[0] + np.eye(N)[-1])))


def sparse_operator(design):
    """L(p) = I + c T, tridiagonal, with the ends absorbing so that L^-1 = K."""
    stiffness, ratio, _ = interaction(design)
    return cas.DM.eye(N) + stiffness * (LAPLACIAN - ratio * ENDS)


def weights_of(design):
    return WEIGHT * design[2] * (1.0 - 0.4 * design[3] * locator)


def rhs_of(design, controls):
    return OMEGA * weights_of(design) * (controls + 0.1 * locator)


def outputs_of(smoothed, controls, state, weights):
    """The non-linear reading, on z = K y."""
    argument = controls + COUPLING * smoothed / state
    warped = OMEGA * argument - 5.0 * argument**3
    return cas.vertcat(
        state * cas.sum1(warped * weights), cas.sum1(argument * weights)
    )


def build(kernel, lifted):
    """Return the Opti problem and its variables for one writing."""
    opti = cas.Opti()
    design = opti.variable(4)
    controls = opti.variable(N)
    state = opti.variable()
    weights = weights_of(design)
    rhs = rhs_of(design, controls)

    if kernel == "dense":
        K = dense_kernel(design)
        A = cas.MX.eye(N) + cas.diag(OMEGA * weights / state) @ K
        if lifted:
            variable = response = opti.variable(N)
            eliminated = cas.solve(A, rhs)
            opti.subject_to(A @ response == rhs)
        else:
            response = cas.solve(A, rhs)
        smoothed = K @ response
    else:
        # z = K y and A y = b  <=>  (L + (omega / v) diag(w)) z = b
        M = sparse_operator(design) + cas.diag(OMEGA * weights / state)
        if lifted:
            variable = smoothed = opti.variable(N)
            eliminated = cas.solve(M, rhs)
            opti.subject_to(M @ smoothed == rhs)
        else:
            smoothed = cas.solve(M, rhs)

    outputs = outputs_of(smoothed, controls, state, weights)
    opti.minimize(cas.sum1(outputs))
    opti.subject_to(opti.bounded(0.25, design, 1.0))
    opti.subject_to(opti.bounded(-1.0, controls, 1.0))
    opti.subject_to(opti.bounded(5.0, state, 20.0))

    opti.set_initial(design, design_guess)
    opti.set_initial(controls, controls_guess)
    opti.set_initial(state, state_guess)
    if lifted:
        # start from the value consistent with the other guesses
        guess = cas.Function("guess", [design, controls, state], [eliminated])
        opti.set_initial(variable, guess(design_guess, controls_guess, state_guess))
    return opti, design, controls, state, outputs


def solve(kernel, lifted, hessian_approximation="exact"):
    opti, design, controls, state, outputs = build(kernel, lifted)
    opti.solver(
        "ipopt",
        {"print_time": False},
        {"hessian_approximation": hessian_approximation, "print_level": 0},
    )
    start = time.perf_counter()
    solution = opti.solve()
    elapsed = time.perf_counter() - start
    stats = solution.stats()
    solver = opti.debug.casadi_solver
    jac_g = solver.get_function("nlp_jac_g")
    result = {
        "kernel": kernel,
        "lifted": lifted,
        "hessian": hessian_approximation,
        "status": stats["return_status"],
        "variables": opti.nx,
        "constraints": opti.ng,
        "jac_nnz": jac_g.sparsity_out(1).nnz(),
        "iterations": stats["iter_count"],
        "objective": float(solution.value(opti.f)),
        "design": np.asarray(solution.value(design)).ravel().tolist(),
        "state": float(solution.value(state)),
        "wall_s": elapsed,
        "t_hess_s": stats.get("t_wall_nlp_hess_l", 0.0),
        "hessian_calls": stats.get("n_call_nlp_hess_l", 0),
        "t_jac_g_s": stats["t_wall_nlp_jac_g"],
        "t_grad_s": stats["t_wall_nlp_grad_f"],
        "hessian_nnz": None,
        "hessian_nodes": None,
    }
    if hessian_approximation == "exact":
        hess_l = solver.get_function("nlp_hess_l")
        result["hessian_nnz"] = hess_l.sparsity_out(0).nnz()
        result["hessian_nodes"] = hess_l.n_nodes()
    return result


def check_kernel():
    """K(p) is the inverse of L(p), to machine precision."""
    design = cas.MX.sym("design", 4)
    K = cas.Function("K", [design], [dense_kernel(design)])(design_guess)
    L = cas.Function("L", [design], [sparse_operator(design)])(design_guess)
    return float(cas.norm_inf(cas.mtimes(L, K) - cas.DM.eye(N)))


if __name__ == "__main__":
    print(f"|L K - I| = {check_kernel():.1e}")

    cases = [
        ("dense, eliminated", "dense", False),
        ("dense, lifted", "dense", True),
        ("sparse, eliminated", "sparse", False),
        ("sparse, lifted", "sparse", True),
    ]
    results = {name: solve(kernel, lifted) for name, kernel, lifted in cases}

    names = list(results)
    print(f"\n{'':<16}" + "".join(f"{name:>20}" for name in names))
    for key, fmt in (
        ("status", "s"),
        ("variables", "d"),
        ("constraints", "d"),
        ("jac_nnz", "d"),
        ("hessian_nnz", "d"),
        ("hessian_nodes", "d"),
        ("iterations", "d"),
        ("objective", ".6f"),
        ("wall_s", ".2f"),
        ("t_hess_s", ".3f"),
        ("t_jac_g_s", ".3f"),
        ("t_grad_s", ".3f"),
    ):
        row = [format(results[name][key], fmt) for name in names]
        print(f"{key:<16}" + "".join(f"{cell:>20}" for cell in row))
    print(
        "\nHessian per call (ms): "
        + ", ".join(
            f"{name} {1e3 * r['t_hess_s'] / r['hessian_calls']:.1f}"
            for name, r in results.items()
        )
    )
    Path("results.json").write_text(json.dumps(results, indent=2))
    print("wrote results.json")
