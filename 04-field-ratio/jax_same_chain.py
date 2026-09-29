"""CasADi versus JAX on the same lifting-line chain, parity gated before timing.

The chain is the generic shape of a sail-aero model: rig geometry, Biot-Savart
influence matrix, one dense solve, an RBF section polar, force and moment
recovery. The same formulas and the same numbers are implemented twice, once in
CasADi and once in JAX, with the same dependence on the shape controls. Parity
of the wrench and of the Hessian is gated at 1e-9 before any timing is reported.

What is compared is the evaluation time of the forward pass, the Jacobian and
the Hessian: what a solver pays every iteration.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 nice -n 10 python jax_same_chain.py

NumPy, CasADi and JAX only. No model dependency.
"""

import argparse
import json
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
R_CORE = 0.05
CONST = 1.0 / (4.0 * np.pi)
REYNOLDS = 1e6
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
    eta = np.linspace(0.0, 1.0, n)
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
    influence = CONST * abx * (1 / na + 1 / nb) * s / (s * s + R_CORE**2)
    rhs = 0.1 + 0.01 * trim + 0.02 * ca.DM(eta)
    gamma = ca.solve(influence + ca.diag(0.02 + 0 * ca.DM(eta)), rhs)
    alpha = rhs + influence @ gamma
    # RBF section polar: squared distances expanded, no 3-D tensor
    stations = ca.horzcat(alpha, REYNOLDS * (1.0 + 0.1 * p[0]) + 0 * alpha,
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
    eta = jnp.linspace(0.0, 1.0, N_PANELS)
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
        influence = CONST * abx * (1 / na + 1 / nb) * s / (s * s + R_CORE**2)
        rhs = 0.1 + 0.01 * trim + 0.02 * eta
        gamma = jnp.linalg.solve(influence + jnp.diag(0.02 + 0 * eta), rhs)
        alpha = rhs + influence @ gamma
        stations = jnp.stack([alpha, REYNOLDS * (1.0 + 0.1 * p[0]) + 0 * alpha,
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


def timed(function, value) -> float:
    out = function(value)
    if hasattr(out, "block_until_ready"):
        jax.block_until_ready(out)
    samples = []
    for _ in range(N_REPEATS):
        started = perf_counter()
        function(value)
        samples.append(perf_counter() - started)
    return float(np.median(np.array(samples)) * 1e3)


def mechanism(cas_hessian, point) -> dict:
    """Where the gain comes from: interpretation floor, per-node cost, JIT.

    The interpretation floor is a chain of trivially cheap nodes: if the real
    chain ran at the same cost per node, its cost would be interpretation and a
    compiler would remove it. The JIT row compiles the very same CasADi graph to
    C and times it again: that is what compilation alone is worth.
    """
    import os
    import subprocess
    import tempfile

    x = ca.MX.sym("x")
    y = x
    for _ in range(20000):
        y = y + 1.0
    trivial = ca.Function("trivial", [x], [y])
    value = ca.DM(1.0)
    trivial(value)
    samples = []
    for _ in range(N_REPEATS):
        started = perf_counter()
        trivial(value)
        samples.append((perf_counter() - started) * 1e3)
    trivial_us = 1e3 * float(np.median(np.array(samples))) / trivial.n_nodes()

    plain = timed(cas_hessian, point)
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
        subprocess.run(
            ["gcc", "-O2", "-fPIC", "-shared", "-o", str(shared),
             *[str(s) for s in sources], f"-I{include}", f"-I{include / 'casadi'}",
             "-lm"], check=True, capture_output=True)
        compiled = timed(ca.external(cas_hessian.name(), str(shared)), point)
    return {
        "trivial_node_us": trivial_us,
        "chain_node_us": 1e3 * plain / cas_hessian.n_nodes(),
        "chain_nodes": cas_hessian.n_nodes(),
        "hessian_interpreted_ms": plain,
        "hessian_compiled_ms": compiled,
        "compile_alone_speedup": plain / compiled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mechanism", action="store_true",
                        help="also measure the interpretation floor and the "
                             "same graph compiled to C (needs gcc)")
    args = parser.parse_args()
    data = synthetic_rig()
    point_c = ca.DM(data["point"].tolist())
    point_j = jnp.asarray(data["point"])
    cas_forward, cas_jacobian, cas_hessian = build_casadi(data)
    jax_forward, jax_jacobian, jax_hessian = build_jax(data)

    report = {"n_panels": N_PANELS, "n_controls": N_CONTROLS, "n_rbf": N_RBF,
              "seed": SEED, "parity_gate": PARITY_GATE,
              "versions": {"jax": jax.__version__, "casadi": ca.__version__}}
    for name, cas_fn, jax_fn in (("forward", cas_forward, jax_forward),
                                 ("jacobian", cas_jacobian, jax_jacobian),
                                 ("hessian", cas_hessian, jax_hessian)):
        reference = np.asarray(cas_fn(point_c), dtype=float).ravel()
        candidate = np.asarray(jax_fn(point_j), dtype=float).ravel()
        scale = max(1.0, float(np.max(np.abs(reference))))
        report[f"{name}_parity_rel"] = float(np.max(np.abs(reference - candidate))) / scale
        if report[f"{name}_parity_rel"] > PARITY_GATE:
            report["status"] = "parity_fail"
            print(json.dumps(report, indent=2))
            raise SystemExit(1)
    for name, cas_fn, jax_fn in (("forward", cas_forward, jax_forward),
                                 ("jacobian", cas_jacobian, jax_jacobian),
                                 ("hessian", cas_hessian, jax_hessian)):
        report[name] = {"casadi_ms": timed(cas_fn, point_c),
                        "jax_ms": timed(jax_fn, point_j)}
        report[f"{name}_speedup"] = report[name]["casadi_ms"] / report[name]["jax_ms"]
    if args.mechanism:
        report["mechanism"] = mechanism(cas_hessian, point_c)
    report["status"] = "complete"
    print(json.dumps(report, indent=2))
    if args.output:
        with open(args.output, "w") as stream:
            json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
