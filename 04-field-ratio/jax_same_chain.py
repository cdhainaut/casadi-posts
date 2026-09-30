"""Compare exact derivatives of an active synthetic chain in CasADi and JAX.

Geometry, a regularised vortex kernel, a dense solve and a Gaussian RBF map
feed six outputs. Inputs to the RBF are dimensionless. Validate values and
both derivative orders at three points before synchronized CPU timings.
The chain is a computational example, not a validated aerodynamic model.
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

N_PANELS = 42
N_UNITS = 3
N_PER_UNIT = N_PANELS // N_UNITS
N_CONTROLS = 9  # trim(3), twist_root(3), twist_tip(3)
N_RBF = 250
CORE_AREA = 0.05  # m^2, regularises the squared-distance denominator
CONST = 1.0 / (4.0 * np.pi)
REYNOLDS_REFERENCE = 1e6
Q = 0.5 * 1.225 * 8.0**2
N_REPEATS = 15
PARITY_GATE = 1e-9
SEED = 7


def synthetic_rig() -> dict:
    """Seeded stand-in rig: panels in three units, plausible geometry ranges."""
    rng = np.random.default_rng(SEED)
    unit = np.repeat(np.arange(N_UNITS), N_PER_UNIT)
    span = np.tile(np.linspace(-1.0, 1.0, N_PER_UNIT), N_UNITS)
    pos = np.stack([0.5 + 0.1 * unit + 0.05 * span,
                    span + 0.2 * unit,
                    0.1 * np.sin(2.0 * span)], axis=1)
    normal = np.stack([0.3 + 0.05 * span, 0.1 * unit, 1.0 + 0.1 * span], axis=1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    chord = 1.5 + 0.3 * rng.random(N_PANELS)
    left = pos + np.stack([-0.1 * np.ones(N_PANELS), -0.5 * chord,
                           np.zeros(N_PANELS)], axis=1)
    right = pos + np.stack([-0.1 * np.ones(N_PANELS), 0.5 * chord,
                            np.zeros(N_PANELS)], axis=1)
    return {
        "pos": pos, "normal": normal, "chord": chord, "left": left, "right": right,
        "centres": rng.standard_normal((N_RBF, 3)),
        "cl": rng.standard_normal(N_RBF),
        "cd": rng.standard_normal(N_RBF),
        "sigma": 0.5,
        "point": np.array([4., 4., 4., -6., -6., -6., -9., -9., -9.]),
    }


def build_casadi(data: dict):
    n = N_PANELS
    p = ca.MX.sym("p", N_CONTROLS)
    eta = np.tile(np.linspace(0.0, 1.0, N_PER_UNIT), N_UNITS)
    per_panel_twist = ca.vertcat(*[
        ca.repmat(p[3 + s], N_PER_UNIT, 1)
        + ca.DM(eta[:N_PER_UNIT]).reshape((N_PER_UNIT, 1))
        * ca.repmat(p[6 + s] - p[3 + s], N_PER_UNIT, 1) for s in range(N_UNITS)])
    trim = ca.vertcat(*[ca.repmat(p[s], N_PER_UNIT, 1) for s in range(N_UNITS)])
    angle = np.pi / 180.0 * (per_panel_twist + trim)
    normal = ca.DM(data["normal"])
    nrm = ca.horzcat(ca.cos(angle) * normal[:, 0] - ca.sin(angle) * normal[:, 2],
                     normal[:, 1],
                     ca.sin(angle) * normal[:, 0] + ca.cos(angle) * normal[:, 2])
    shift = ca.horzcat(*[ca.repmat(p[s] * 0.1, n, 1) for s in range(3)])
    left = ca.DM(data["left"]) + shift
    right = ca.DM(data["right"]) + shift
    pos = ca.DM(data["pos"])

    def pair(column, other):
        return ca.repmat(column, 1, n) - ca.repmat(other.T, n, 1)

    ax, ay, az = pair(pos[:, 0], left[:, 0]), pair(pos[:, 1], left[:, 1]), \
        pair(pos[:, 2], left[:, 2])
    bx, by, bz = pair(pos[:, 0], right[:, 0]), pair(pos[:, 1], right[:, 1]), \
        pair(pos[:, 2], right[:, 2])
    abx = ay * bz - az * by
    adotb = ax * bx + ay * by + az * bz
    na = ca.sqrt(ax * ax + ay * ay + az * az)
    nb = ca.sqrt(bx * bx + by * by + bz * bz)
    s = na * nb + adotb
    influence = CONST * abx * (1 / na + 1 / nb) * s / (s * s + CORE_AREA**2)
    rhs = 0.1 + 0.01 * trim + 0.02 * ca.DM(eta)
    gamma = ca.solve(influence + ca.diag(0.02 + 0 * ca.DM(eta)), rhs)
    alpha = rhs + influence @ gamma
    # RBF section polar: squared distances expanded, no 3-D tensor
    reynolds = REYNOLDS_REFERENCE * (1.0 + 0.1 * p[0])
    stations = ca.horzcat(alpha, (reynolds / REYNOLDS_REFERENCE - 1.0) + 0 * alpha,
                          0.04 + 0.01 * per_panel_twist)
    centres = ca.DM(data["centres"])
    r2 = ca.repmat(ca.sum2(stations**2), 1, N_RBF) \
        + ca.repmat(ca.sum2(centres**2).T, n, 1) - 2.0 * stations @ centres.T
    kernel = ca.exp(-r2 / (2.0 * data["sigma"] ** 2))
    cl_ = ca.reshape(kernel @ ca.DM(data["cl"]), (n, 1))
    cd_ = ca.reshape(kernel @ ca.DM(data["cd"]), (n, 1))
    chord = ca.DM(data["chord"])
    lift = cl_ * Q * chord * 0.5
    drag = cd_ * Q * chord * 0.5
    fx = lift * nrm[:, 0] + drag
    fy = lift * nrm[:, 1]
    fz = lift * nrm[:, 2]
    posx, posy, posz = pos[:, 0], pos[:, 1], pos[:, 2]
    wrench = ca.vertcat(ca.sum1(fx), ca.sum1(fy), ca.sum1(fz),
                        ca.sum1(posy * fz - posz * fy),
                        ca.sum1(posz * fx - posx * fz),
                        ca.sum1(posx * fy - posy * fx))
    scalar = ca.cse(ca.sum1(wrench))[0]
    return (ca.Function("forward", [p], [wrench]),
            ca.Function("jacobian", [p], [ca.jacobian(wrench, p)]),
            ca.Function("hessian", [p], [ca.hessian(scalar, p)[0]]))


def build_jax(data: dict):
    pos = jnp.asarray(data["pos"])
    normal = jnp.asarray(data["normal"])
    chord = jnp.asarray(data["chord"])
    left0 = jnp.asarray(data["left"])
    right0 = jnp.asarray(data["right"])
    centres = jnp.asarray(data["centres"])
    cl_data = jnp.asarray(data["cl"])
    cd_data = jnp.asarray(data["cd"])
    eta = jnp.tile(jnp.linspace(0.0, 1.0, N_PER_UNIT), N_UNITS)
    sigma2 = 2.0 * float(data["sigma"]) ** 2

    def forward(p):
        per_panel_twist = jnp.concatenate([
            p[3 + s] + eta[:N_PER_UNIT] * (p[6 + s] - p[3 + s])
            for s in range(N_UNITS)])
        trim = jnp.concatenate([jnp.full(N_PER_UNIT, p[s]) for s in range(N_UNITS)])
        angle = jnp.radians(per_panel_twist + trim)
        nx = jnp.cos(angle) * normal[:, 0] - jnp.sin(angle) * normal[:, 2]
        nz = jnp.sin(angle) * normal[:, 0] + jnp.cos(angle) * normal[:, 2]
        nrm = jnp.stack([nx, normal[:, 1], nz], axis=-1)
        shift = p[0:3] * 0.1
        left = left0 + shift
        right = right0 + shift

        def pair(column, other):
            return column[:, None] - other[None, :]

        ax = pair(pos[:, 0], left[:, 0])
        ay = pair(pos[:, 1], left[:, 1])
        az = pair(pos[:, 2], left[:, 2])
        bx = pair(pos[:, 0], right[:, 0])
        by = pair(pos[:, 1], right[:, 1])
        bz = pair(pos[:, 2], right[:, 2])
        abx = ay * bz - az * by
        adotb = ax * bx + ay * by + az * bz
        na = jnp.sqrt(ax * ax + ay * ay + az * az)
        nb = jnp.sqrt(bx * bx + by * by + bz * bz)
        s = na * nb + adotb
        influence = CONST * abx * (1 / na + 1 / nb) * s / (s * s + CORE_AREA**2)
        rhs = 0.1 + 0.01 * trim + 0.02 * eta
        gamma = jnp.linalg.solve(influence + jnp.diag(0.02 + 0 * eta), rhs)
        alpha = rhs + influence @ gamma
        reynolds = REYNOLDS_REFERENCE * (1.0 + 0.1 * p[0])
        stations = jnp.stack([alpha, (reynolds / REYNOLDS_REFERENCE - 1.0) + 0 * alpha,
                              0.04 + 0.01 * per_panel_twist], axis=-1)
        r2 = jnp.sum(stations**2, axis=1)[:, None] \
            + jnp.sum(centres**2, axis=1)[None, :] - 2.0 * stations @ centres.T
        kernel = jnp.exp(-r2 / sigma2)
        cl_ = kernel @ cl_data
        cd_ = kernel @ cd_data
        lift = cl_ * Q * chord * 0.5
        drag = cd_ * Q * chord * 0.5
        fx = lift * nrm[:, 0] + drag
        fy = lift * nrm[:, 1]
        fz = lift * nrm[:, 2]
        force = jnp.stack([jnp.sum(fx), jnp.sum(fy), jnp.sum(fz)])
        moment = jnp.stack([
            jnp.sum(pos[:, 1] * fz - pos[:, 2] * fy),
            jnp.sum(pos[:, 2] * fx - pos[:, 0] * fz),
            jnp.sum(pos[:, 0] * fy - pos[:, 1] * fx),
        ])
        return jnp.concatenate([force, moment])

    return (jax.jit(forward), jax.jit(jax.jacobian(forward)),
            jax.jit(jax.hessian(lambda p: jnp.sum(forward(p)))))


def relative_error(reference, candidate) -> float:
    """Reject non-finite or mismatched outputs before computing a scaled error."""
    reference = np.asarray(reference, dtype=float).ravel()
    candidate = np.asarray(candidate, dtype=float).ravel()
    if reference.shape != candidate.shape:
        raise ValueError("Output sizes differ")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Non-finite output")
    scale = max(1.0, float(np.max(np.abs(reference))))
    error = float(np.max(np.abs(reference - candidate))) / scale
    if error > PARITY_GATE:
        raise ValueError(f"Parity error {error} exceeds {PARITY_GATE}")
    return error


def validate(cas_functions, jax_functions, points: np.ndarray) -> list[dict]:
    """Check active outputs, both derivative orders and every control at each point."""
    checks = []
    for point in points:
        row = {"point": point.tolist()}
        for name, cas_fn, jax_fn in zip(
            ("forward", "jacobian", "hessian"), cas_functions, jax_functions
        ):
            reference = np.asarray(cas_fn(ca.DM(point)), dtype=float)
            candidate = np.asarray(jax_fn(jnp.asarray(point)), dtype=float)
            error = relative_error(reference, candidate)
            norm = float(np.max(np.abs(reference)))
            if norm == 0.0:
                raise ValueError(f"Degenerate {name}: all entries are zero")
            row[name] = {"relative_error": error, "max_abs": norm}
            if name == "jacobian":
                column_norms = np.max(np.abs(reference), axis=0)
                if np.any(column_norms == 0.0):
                    raise ValueError("An inactive control has a zero Jacobian column")
                row["control_column_max_abs"] = column_norms.tolist()
        checks.append(row)
    return checks


def timed(function, value) -> dict:
    """Warm once, then include completion of every call in its measured interval."""
    out = function(value)
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    samples = []
    for _ in range(N_REPEATS):
        started = perf_counter()
        out = function(value)
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        samples.append((perf_counter() - started) * 1e3)
    return {"median_ms": float(np.median(samples)), "samples_ms": samples}


def compiled_hessian(cas_hessian, points: np.ndarray, point) -> dict:
    """Compile the same graph; gate parity at all points before synchronized timing."""
    with tempfile.TemporaryDirectory() as folder:
        previous = os.getcwd()
        os.chdir(folder)
        try:
            cas_hessian.generate("same_chain", {})
        finally:
            os.chdir(previous)
        sources = sorted(Path(folder).glob("same_chain*.c"))
        include = Path(ca.__file__).resolve().parent / "include"
        shared = Path(folder) / "libsame_chain.so"
        started = perf_counter()
        subprocess.run(
            ["gcc", "-O2", "-fPIC", "-shared", "-o", str(shared),
             *[str(s) for s in sources], f"-I{include}", f"-I{include / 'casadi'}",
             "-lm"], check=True, capture_output=True, timeout=300)
        compile_seconds = perf_counter() - started
        compiled = ca.external(cas_hessian.name(), str(shared))
        errors = [relative_error(cas_hessian(p), compiled(p)) for p in points]
        timing = timed(compiled, point)
    return {"compiler": "gcc -O2", "compile_seconds": compile_seconds,
            "parity_errors": errors, **timing}


def provenance() -> dict:
    """Record the measured source and the actual Linux execution envelope."""
    cpu_model = next((line.split(":", 1)[1].strip()
                      for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), platform.processor())
    return {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "jax": jax.__version__, "casadi": ca.__version__},
        "cpu_model": cpu_model, "architecture": platform.machine(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "address_space_bytes": resource.getrlimit(resource.RLIMIT_AS),
        "stack_bytes": resource.getrlimit(resource.RLIMIT_STACK),
        "core_bytes": resource.getrlimit(resource.RLIMIT_CORE),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "PYTHONHASHSEED", "XLA_FLAGS")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-c", action="store_true", help="compare the same C graph")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = synthetic_rig()
    points = np.stack([data["point"], data["point"] + 0.25, data["point"] - 0.5])
    report = {"n_panels": N_PANELS, "n_controls": N_CONTROLS, "n_rbf": N_RBF,
              "seed": SEED, "parity_gate": PARITY_GATE, "repeats": N_REPEATS,
              "rbf_features": ["alpha", "Re/Re_reference - 1", "relative_thickness"],
              "hessian_scalar": "sum of the six synthetic outputs",
              "provenance": provenance(), "status": "preflight"}
    try:
        cas_functions = build_casadi(data)
        jax_functions = build_jax(data)
        started = perf_counter()
        report["validation"] = validate(cas_functions, jax_functions, points)
        report["validation_seconds_including_jax_compilation"] = perf_counter() - started
        for name, cas_fn, jax_fn in zip(
            ("forward", "jacobian", "hessian"), cas_functions, jax_functions
        ):
            cas_timing = timed(cas_fn, ca.DM(data["point"]))
            jax_timing = timed(jax_fn, jnp.asarray(data["point"]))
            report[name] = {"casadi": cas_timing, "jax": jax_timing,
                            "speedup": cas_timing["median_ms"] / jax_timing["median_ms"]}
        if args.compile_c:
            report["compiled_hessian"] = compiled_hessian(
                cas_functions[2], points, ca.DM(data["point"]))
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", exception=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
