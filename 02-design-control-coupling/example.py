"""Exact derivative structure with shared, independent or frozen design.

Build the same local equations in three layouts. The frozen layout removes
all design variables. Save numerical records and sparsity patterns in a fresh
output directory; no calculation runs when this module is imported.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

N = 16
K = 5
OMEGA = 2.0 * np.pi
WEIGHT = 0.4
COUPLING = 0.3
TARGET = 0.5
DESIGN_VALUE = np.array([0.5, 0.3, 1.0, 0.4])
EVALUATIONS = 20
LAYOUTS = ("design shared", "design per condition", "design frozen")


def local_outputs(design: cas.MX | cas.DM, controls: cas.MX, state: cas.MX,
                  locator: np.ndarray) -> tuple[cas.MX, cas.MX]:
    """One dense influence model, unchanged between design layouts."""
    n = len(locator)
    coordinates = (locator + 0.12 * design[0] * (1.0 - locator**2)
                   + 0.08 * design[1] * locator * (1.0 - locator**2))
    weights = WEIGHT * design[2] * (1.0 - 0.4 * design[3] * locator)
    span = cas.reshape(coordinates, n, 1)
    difference = cas.repmat(span, 1, n) - cas.repmat(span.T, n, 1) + cas.MX.eye(n)
    kernel = ((coordinates[1] - coordinates[0]) / (4.0 * np.pi)) / difference
    kernel -= cas.diag(cas.diag(kernel))
    matrix = cas.MX.eye(n) + cas.diag(OMEGA * weights / state) @ kernel
    rhs = OMEGA * weights * (controls + 0.1 * locator)
    response = cas.solve(matrix, rhs)
    argument = controls + COUPLING * (kernel @ response) / state
    warped = OMEGA * argument - 5.0 * argument**3
    return cas.sum1(warped * weights), cas.sum1(argument * weights) - TARGET


def build_layout(n: int, conditions: int, layout: str) -> tuple:
    """Return H, closure Jacobian, common primal point and block sizes."""
    if layout not in LAYOUTS:
        raise ValueError(layout)
    locator = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    controls = [cas.MX.sym(f"u_{k}", n) for k in range(conditions)]
    states = [cas.MX.sym(f"v_{k}") for k in range(conditions)]
    if layout == "design shared":
        design = cas.MX.sym("p", 4)
        designs = [design] * conditions
        blocks = [design] + [cas.vertcat(u, v) for u, v in zip(controls, states)]
        initial_blocks = [DESIGN_VALUE] + [np.r_[np.full(n, 0.1), 12.0]] * conditions
        design_block, condition_block = 4, n + 1
    elif layout == "design per condition":
        designs = [cas.MX.sym(f"p_{k}", 4) for k in range(conditions)]
        blocks = [cas.vertcat(p, u, v) for p, u, v in zip(designs, controls, states)]
        initial_blocks = [np.r_[DESIGN_VALUE, np.full(n, 0.1), 12.0]] * conditions
        design_block, condition_block = 0, n + 5
    else:
        designs = [cas.DM(DESIGN_VALUE)] * conditions
        blocks = [cas.vertcat(u, v) for u, v in zip(controls, states)]
        initial_blocks = [np.r_[np.full(n, 0.1), 12.0]] * conditions
        design_block, condition_block = 0, n + 1
    variables = cas.vertcat(*blocks)
    terms = [local_outputs(p, u, v, locator) for p, u, v in zip(designs, controls, states)]
    objective = sum(term[0] for term in terms)
    closures = cas.vertcat(*[term[1] for term in terms])
    multipliers = cas.MX.sym("lambda", conditions)
    lagrangian = objective + cas.dot(multipliers, closures)
    hessian = cas.tril(cas.hessian(lagrangian, variables)[0], True)
    jacobian = cas.jacobian(closures, variables)
    return (cas.Function("hessian", [variables, multipliers], [hessian]),
            cas.Function("jacobian", [variables], [jacobian]),
            np.concatenate(initial_blocks), design_block, condition_block)


def measure(n: int, conditions: int) -> tuple[list[dict], dict]:
    """Measure the lower Hessian triangle at one common point with lambda=1."""
    records, patterns = [], {}
    for layout in LAYOUTS:
        started = time.perf_counter()
        hessian, jacobian, point, design_block, condition_block = build_layout(
            n, conditions, layout)
        build_seconds = time.perf_counter() - started
        multipliers = np.ones(conditions)
        hessian(point, multipliers)
        samples = []
        for _ in range(EVALUATIONS):
            started = time.perf_counter()
            hessian(point, multipliers)
            samples.append((time.perf_counter() - started) * 1e3)
        rows, cols = hessian.sparsity_out(0).get_triplet()
        records.append({
            "label": layout, "stations": n, "conditions": conditions,
            "variables": len(point), "constraints": conditions,
            "nnz_jacobian": jacobian.nnz_out(0), "nnz_hessian": len(rows),
            "nodes_jacobian": jacobian.n_nodes(), "nodes_hessian": hessian.n_nodes(),
            "build_s": build_seconds, "hessian_ms": float(np.median(samples)),
            "samples_ms": samples, "point": point.tolist(),
            "multipliers": multipliers.tolist(),
        })
        for key, value in {"hessian_rows": rows, "hessian_cols": cols,
                           "hessian_shape": (len(point), len(point)),
                           "design_block": design_block,
                           "condition_block": condition_block}.items():
            patterns[f"{layout}|{key}"] = np.asarray(value)
    return records, patterns


def plot_patterns(records: list[dict], patterns: dict, output: Path) -> None:
    """Draw the shared border and the independent condition blocks."""
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, record in zip(axes, records):
        label = record["label"]
        rows, cols = (patterns[f"{label}|{key}"] for key in ("hessian_rows", "hessian_cols"))
        axis.plot(cols, rows, ".", markersize=2)
        axis.set_xlim(-1, record["variables"])
        axis.set_ylim(record["variables"], -1)
        axis.set_aspect("equal")
        axis.set_title(f"{label}\n{record['variables']} variables, {len(rows)} nonzeros")
        axis.set_xlabel("variables")
    axes[0].set_ylabel("variables")
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/reproduction_NEW"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records, patterns = measure(N, K)
    (args.output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    np.savez(args.output / "patterns.npz", **patterns)
    provenance = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "casadi": cas.__version__,
                     "numpy": np.__version__},
        "stations": N, "conditions": K, "triangle": "lower", "repeats": EVALUATIONS,
        "multiplier_convention": "all closure multipliers 1; objective factor 1",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plot_patterns(records, patterns, args.output / "hessian_structure.png")
    print(f"{'layout':<22}{'variables':>10}{'nonzeros':>10}{'density':>9}{'graph':>8}")
    for record in records:
        nvars = record["variables"]
        density = record["nnz_hessian"] / (nvars * (nvars + 1) / 2)
        print(f"{record['label']:<22}{nvars:>10}{record['nnz_hessian']:>10}"
              f"{density:>8.0%}{record['nodes_hessian']:>8}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
