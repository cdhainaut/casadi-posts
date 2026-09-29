"""CasADi versus JAX on the same dense-kernel model: does jaxifying pay?

Follows Joris Gillis's suggestion that ML tooling may help when the bottleneck is
large dense matrix/tensor operations. The model is the dense writing of
bench_field_ratio.py: T conditions sharing one design, each with its own controls
and state, one dense solve per condition. The same objective is built twice, once
in CasADi (graph per condition) and once in JAX (batched solve), and objective,
gradient and Hessian are compared at one point before their eval times.

Optional dependency: jax (CPU). Everything else is NumPy and CasADi.
"""

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import casadi as cas
import numpy as np

from bench_field_ratio import (
    CONTROL_GUESS,
    COUPLING,
    DESIGN_GUESS,
    LENGTH,
    OMEGA,
    STATE_GUESS,
    WEIGHT,
    interaction,
)

EVAL_REPEATS = 5


def initial_point(n: int, conditions: int) -> np.ndarray:
    controls = np.tile(np.full(n, CONTROL_GUESS), conditions)
    return np.concatenate([DESIGN_GUESS, controls, np.full(conditions, STATE_GUESS)])


def casadi_model(n: int, conditions: int):
    """Objective, gradient and Hessian of the batched dense-kernel model in CasADi."""
    locator = -1.0 + (np.arange(n) + 0.5) * 2.0 / n
    x = cas.MX.sym("x", 4 + conditions * n + conditions)
    design = x[:4]
    weights = WEIGHT * design[2] * (1.0 - 0.4 * design[3] * locator)
    _, ratio, amplitude = interaction(design, n)
    offsets = cas.DM(np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]))
    kernel = amplitude * ratio**offsets
    objective = 0
    for condition in range(conditions):
        start = 4 + condition * n
        controls = x[start:start + n]
        state = x[4 + conditions * n + condition]
        rhs = OMEGA * weights * (controls + 0.1 * locator)
        matrix = cas.MX.eye(n) + cas.diag(OMEGA * weights / state) @ kernel
        smoothed = kernel @ cas.solve(matrix, rhs)
        argument = controls + COUPLING * smoothed / state
        warped = OMEGA * argument - 5.0 * argument**3
        objective = objective + state * cas.sum1(warped * weights) \
            + cas.sum1(argument * weights)
    gradient = cas.gradient(objective, x)
    hessian, _ = cas.hessian(objective, x)
    return (cas.Function("objective", [x], [objective]),
            cas.Function("gradient", [x], [gradient]),
            cas.Function("hessian", [x], [hessian]))


def jax_model(n: int, conditions: int):
    """The same objective, gradient and Hessian in JAX with a batched solve."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    locator = -1.0 + (np.arange(n) + 0.5) * 2.0 / n
    offsets = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    identity = np.eye(n)

    def objective(x):
        design = x[:4]
        length = LENGTH * jnp.exp(0.6 * design[0] - 0.4 * design[1])
        stiffness = length**2 / (2.0 / n) ** 2
        mu = 1.0 + 1.0 / (2.0 * stiffness)
        ratio = mu - jnp.sqrt(mu**2 - 1.0)
        amplitude = 1.0 / (1.0 + 2.0 * stiffness * (1.0 - ratio))
        weights = WEIGHT * design[2] * (1.0 - 0.4 * design[3] * locator)
        kernel = amplitude * ratio**offsets
        controls = x[4:4 + conditions * n].reshape(conditions, n)
        states = x[4 + conditions * n:]
        rhs = OMEGA * weights * (controls + 0.1 * locator)
        matrix = identity + (OMEGA * weights / states[:, None])[:, :, None] * kernel
        smoothed = jnp.linalg.solve(matrix, rhs[:, :, None])[:, :, 0] @ kernel.T
        argument = controls + COUPLING * smoothed / states[:, None]
        warped = OMEGA * argument - 5.0 * argument**3
        return jnp.sum(states * jnp.sum(warped * weights, axis=1)
                       + jnp.sum(argument * weights, axis=1))

    # Exact Hessian, reverse-over-reverse: jacfwd(jacrev) batches N tangents and
    # exceeds the 2 GiB envelope from t=8 onward; jacrev(jacrev) is the same
    # matrix computed row by row (harness fix, traced in the study README).
    return (jax.jit(objective), jax.jit(jax.grad(objective)),
            jax.jit(jax.jacrev(jax.grad(objective))))


def timed(function, point, repeats: int = EVAL_REPEATS) -> tuple:
    function(point)
    if hasattr(function, "lower_compile"):  # jax jit: force completion
        function(point).block_until_ready()
    timings = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = function(point)
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()
        timings.append(time.perf_counter() - start)
    return value, float(np.median(timings))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", type=int, required=True)
    parser.add_argument("--conditions", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n, conditions = args.stations, args.conditions
    point = initial_point(n, conditions)
    result = {
        "stations": n, "conditions": conditions,
        "variables": int(point.size),
        "versions": {"python": platform.python_version(), "casadi": cas.__version__,
                     "numpy": np.__version__},
        "status": "preflight",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    try:
        import jax
        result["versions"]["jax"] = jax.__version__
        start = time.perf_counter()
        cas_functions = casadi_model(n, conditions)
        result["casadi_build_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        jax_functions = jax_model(n, conditions)
        for function in jax_functions:
            function(point)
        result["jax_compile_seconds"] = time.perf_counter() - start
        result["eval_repeats"] = EVAL_REPEATS
        casadi_values = {}
        jax_values = {}
        for name, cas_function, jax_function in zip(
            ("objective", "gradient", "hessian"), cas_functions, jax_functions
        ):
            casadi_values[name], casadi_seconds = timed(cas_function, point)
            jax_values[name], jax_seconds = timed(jax_function, point)
            reference = np.asarray(casadi_values[name], dtype=float).ravel()
            candidate = np.asarray(jax_values[name], dtype=float).ravel()
            scale = max(1.0, float(np.max(np.abs(reference))))
            result.setdefault("comparison", {})[name] = {
                "casadi_seconds": casadi_seconds,
                "jax_seconds": jax_seconds,
                "relative_error": float(np.max(np.abs(reference - candidate))) / scale,
            }
            if result["comparison"][name]["relative_error"] > 1e-9:
                raise ValueError(f"CasADi/JAX {name} mismatch exceeds 1e-9")
        result["speedup_gradient"] = (
            result["comparison"]["gradient"]["casadi_seconds"]
            / result["comparison"]["gradient"]["jax_seconds"])
        result["speedup_hessian"] = (
            result["comparison"]["hessian"]["casadi_seconds"]
            / result["comparison"]["hessian"]["jax_seconds"])
        result["status"] = "complete"
    except Exception as exc:
        result["status"] = "failed"
        result["exception"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
