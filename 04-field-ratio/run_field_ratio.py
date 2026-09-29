"""Run the field/response-ratio homotopy and the JAX bench, one process per cell.

Group H sweeps the field ratio z at one condition; group T raises the number of
conditions at fixed z; group J is the CasADi-versus-JAX bench. Each cell runs in a
fresh Python process. Stop on the first process or validation failure; never retry.
The caller must apply the documented resource envelope before launching.
"""

import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
from pathlib import Path

WRITINGS = (("dense", False), ("dense", True), ("sparse", False), ("sparse", True))
H_GROUPS = tuple(("H", 42, z, 1) for z in (1, 2, 4, 8, 16, 32, 64, 128))
T_GROUPS = tuple(("T", 42, 8, t) for t in (2, 8, 32))
J_GROUPS = tuple(("J", 42, 1, t) for t in (1, 8, 32))
TIMEOUT_SECONDS = 300
JAX_TIMEOUT_SECONDS = 900
PRIMAL_LIMIT = 1e-6
DUAL_LIMIT = 1e-4
COMPLEMENTARITY_LIMIT = 1e-4
OBJECTIVE_RTOL = 1e-8
OBJECTIVE_ATOL = 1e-7
COMPRESSION_LIMIT = 1e-10
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


def validate_case(row: dict, first: dict) -> None:
    """Frozen gates; no timing claim is accepted from an unsuccessful solve.

    Objective agreement is reported, not enforced: a formulation may converge to
    another stationary point of the same model on large fields (measured at z=64).
    """
    if row["status"] != "Solve_Succeeded":
        raise ValueError(f"Unsuccessful solve: {row['status']}")
    if row["compression_error_inf"] > COMPRESSION_LIMIT:
        raise ValueError(f"Compression error {row['compression_error_inf']}")
    solution = row["solution"]
    for key, limit in (("primal_violation_inf", PRIMAL_LIMIT),
                       ("stationarity_inf", DUAL_LIMIT),
                       ("complementarity_inf", COMPLEMENTARITY_LIMIT)):
        if not math.isfinite(solution[key]) or solution[key] > limit:
            raise ValueError(f"{key}={solution[key]} exceeds {limit}")
    if first is not None:
        reference = first["solution"]["objective"]
        row["objective_reference"] = reference
        row["objective_gap_rel"] = abs(solution["objective"] - reference) \
            / max(abs(reference), 1.0)
        if row["objective_gap_rel"] > OBJECTIVE_RTOL:
            row["finding"] = "different_stationary_point"


def flag_memory_wall(row: dict) -> None:
    """Record the measured JAX memory wall as a finding, not a harness failure."""
    if "RESOURCE_EXHAUSTED" in row.get("exception", ""):
        row["finding"] = "jax_hessian_memory_wall"


def validate_jax(row: dict) -> None:
    if row.get("finding") == "jax_hessian_memory_wall":
        return
    if row["status"] != "complete":
        raise ValueError(f"Unsuccessful jax bench: {row['status']}")
    for name, comparison in row["comparison"].items():
        if comparison["relative_error"] > 1e-9:
            raise ValueError(f"{name} mismatch {comparison['relative_error']}")


def resource_envelope() -> dict:
    limits = {
        "address_space_bytes": resource.getrlimit(resource.RLIMIT_AS),
        "stack_bytes": resource.getrlimit(resource.RLIMIT_STACK),
        "core_bytes": resource.getrlimit(resource.RLIMIT_CORE),
    }
    if limits["address_space_bytes"][0] != 2 * 1024**3:
        raise ValueError("Expected a 2 GiB virtual address-space limit")
    if limits["stack_bytes"][0] != 64 * 1024**2 or limits["core_bytes"][0] != 0:
        raise ValueError("Expected a 64 MiB stack and disabled core dumps")
    if any(os.environ.get(key) != "1" for key in THREAD_VARIABLES):
        raise ValueError("All numerical thread counts must be set to one")
    if os.environ.get("PYTHONHASHSEED") != "0" or os.getpriority(os.PRIO_PROCESS, 0) != 10:
        raise ValueError("Expected PYTHONHASHSEED=0 and nice=10")
    return {**limits, "threads": 1, "hash_seed": 0, "nice": 10,
            "timeout_seconds": TIMEOUT_SECONDS, "jax_timeout_seconds": JAX_TIMEOUT_SECONDS}


def run_command(command: list, label: str, output: Path, campaign: dict, timeout: int,
                validate) -> dict:
    result_path = output / f"{label}.json"
    case = {"case": label, "command": command, "status": "running"}
    campaign["cases"].append(case)
    (output / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    print(f"[campaign] {label}", flush=True)
    try:
        with (output / f"{label}.log").open("w") as log:
            completed = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
            )
        case["returncode"] = completed.returncode
        row = json.loads(result_path.read_text())
        flag_memory_wall(row)
        if completed.returncode != 0 and not row.get("finding"):
            raise RuntimeError(f"Child exit code {completed.returncode}")
        validate(row)
        case.update(status="validated", result=row)
        print(f"[validated] {label}", flush=True)
        return row
    except Exception as exc:
        case.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    envelope = resource_envelope()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("bench_field_ratio.py", "jax_dense_bench.py"):
        shutil.copyfile(Path(__file__).with_name(name), output / name)
    campaign = {
        "scope": ("H: field ratio z at T=1; T: conditions at z=8; "
                  "J: CasADi versus JAX on the dense writing"),
        "envelope": envelope,
        "groups": [list(group) for group in H_GROUPS + J_GROUPS + T_GROUPS],
        "writings": [list(writing) for writing in WRITINGS],
        "gates": {"primal": PRIMAL_LIMIT, "stationarity": DUAL_LIMIT,
                  "complementarity": COMPLEMENTARITY_LIMIT,
                  "objective": "reported (finding: different_stationary_point)",
                  "compression": COMPRESSION_LIMIT, "jax_relative": 1e-9},
        "status": "running", "cases": [],
    }
    try:
        for phase, n, z, t in H_GROUPS:
            first = None
            for kernel, lifted in WRITINGS:
                label = f"{phase}_n{n}_z{z}_t{t}_{kernel}_{'lifted' if lifted else 'eliminated'}"
                command = [sys.executable, "-u", str(output / "bench_field_ratio.py"),
                           "--stations", str(n), "--field-ratio", str(z),
                           "--conditions", str(t), "--kernel", kernel,
                           "--output", str(output / f"{label}.json")]
                if lifted:
                    command.append("--lifted")

                def validate(row, first=first):
                    validate_case(row, first)

                row = run_command(command, label, output, campaign, TIMEOUT_SECONDS,
                                  validate)
                if first is None:
                    first = row
        for phase, n, z, t in J_GROUPS:
            label = f"{phase}_n{n}_z{z}_t{t}"
            command = [sys.executable, "-u", str(output / "jax_dense_bench.py"),
                       "--stations", str(n), "--conditions", str(t),
                       "--output", str(output / f"{label}.json")]
            run_command(command, label, output, campaign, JAX_TIMEOUT_SECONDS,
                        lambda row: validate_jax(row))
        for phase, n, z, t in T_GROUPS:
            first = None
            for kernel, lifted in WRITINGS:
                label = f"{phase}_n{n}_z{z}_t{t}_{kernel}_{'lifted' if lifted else 'eliminated'}"
                command = [sys.executable, "-u", str(output / "bench_field_ratio.py"),
                           "--stations", str(n), "--field-ratio", str(z),
                           "--conditions", str(t), "--kernel", kernel,
                           "--output", str(output / f"{label}.json")]
                if lifted:
                    command.append("--lifted")

                def validate(row, first=first):
                    validate_case(row, first)

                row = run_command(command, label, output, campaign, TIMEOUT_SECONDS,
                                  validate)
                if first is None:
                    first = row
        campaign["status"] = "complete"
    except Exception:
        campaign["status"] = "stopped_at_first_failure"
        raise
    finally:
        (output / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
        hashes = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
                  for path in sorted(output.iterdir()) if path.is_file()]
        (output / "sha256sums.txt").write_text("\n".join(hashes) + "\n")


if __name__ == "__main__":
    main()
