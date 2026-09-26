"""Run J0 then J1 sequentially, one fresh Python process per writing.

Save sources, commands, raw logs and incremental results in a new directory.
Stop on the first process or validation failure; never retry a case.
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
GROUPS = (("J0", 192, 192), ("J1", 42, 42), ("J1", 42, 4), ("J1", 42, 1))
TIMEOUT_SECONDS = 300
PRIMAL_LIMIT = 1e-6
DUAL_LIMIT = 1e-4
COMPLEMENTARITY_LIMIT = 1e-4
OBJECTIVE_RTOL = 1e-8
OBJECTIVE_ATOL = 1e-7
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


def validate_case(row: dict, reference: dict | None, first: dict | None) -> None:
    """Frozen gates; no timing claim is accepted from an unsuccessful solve."""
    if row["status"] != "Solve_Succeeded":
        raise ValueError(f"Unsuccessful solve: {row['status']}")
    solution = row["solution"]
    for key, limit in (("primal_violation_inf", PRIMAL_LIMIT),
                       ("stationarity_inf", DUAL_LIMIT),
                       ("complementarity_inf", COMPLEMENTARITY_LIMIT)):
        if not math.isfinite(solution[key]) or solution[key] > limit:
            raise ValueError(f"{key}={solution[key]} exceeds {limit}")
    target = first["solution"]["objective"] if first else None
    if reference:
        target = reference["objective"]
    if target is not None and not math.isclose(
        solution["objective"], target, rel_tol=OBJECTIVE_RTOL, abs_tol=OBJECTIVE_ATOL
    ):
        raise ValueError(f"Objective mismatch: {solution['objective']} versus {target}")
    if reference:
        for key in ("variables", "constraints"):
            if row[key] != reference[key]:
                raise ValueError(f"Original {key} mismatch")
        for key, old_key in (("jacobian_nnz", "jac_nnz"), ("hessian_triangle_nnz", "hessian_nnz")):
            if row["derivatives"][key] != reference[old_key]:
                raise ValueError(f"Original {key} mismatch")


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
    return {**limits, "threads": 1, "hash_seed": 0, "nice": 10, "timeout_seconds": TIMEOUT_SECONDS}


def run_cases(output: Path, reference: dict, campaign: dict) -> None:
    for phase, n, rank in GROUPS:
        first = None
        for kernel, lifted in WRITINGS:
            label = f"{phase}_n{n}_r{rank}_{kernel}_{'lifted' if lifted else 'eliminated'}"
            result_path = output / f"{label}.json"
            command = [sys.executable, "-u", str(output / "bench_joris_controls.py"),
                       "--stations", str(n), "--controls", str(rank), "--kernel", kernel,
                       "--output", str(result_path)]
            if lifted:
                command.append("--lifted")
            case = {"case": label, "command": command, "status": "running"}
            campaign["cases"].append(case)
            (output / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
            print(f"[campaign] {label}", flush=True)
            try:
                with (output / f"{label}.log").open("w") as log:
                    completed = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT_SECONDS
                    )
                case["returncode"] = completed.returncode
                if completed.returncode != 0:
                    raise RuntimeError(f"Child exit code {completed.returncode}")
                row = json.loads(result_path.read_text())
                original = reference[f"{kernel}, {'lifted' if lifted else 'eliminated'}"]
                validate_case(row, original if phase == "J0" else None, first)
                if first is None:
                    first = row
                case.update(status="validated", result=row)
                print(f"[validated] {label}: {row['solve_seconds']:.3f} s", flush=True)
            except Exception as exc:
                case.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                (output / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    envelope = resource_envelope()
    reference = json.loads(args.reference_results.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("bench_joris_controls.py")
    for origin, name in ((source, source.name), (Path(__file__), "run_joris_controls.py"),
                         (args.reference_source, "joris_original.py"),
                         (args.reference_results, "joris_original_results.json")):
        shutil.copyfile(origin, output / name)
    campaign = {
        "scope": "J0 original model; J1 control ranks 42, 4, 1; T=1 throughout",
        "envelope": envelope, "groups": GROUPS, "writings": WRITINGS,
        "gates": {"primal": PRIMAL_LIMIT, "stationarity": DUAL_LIMIT,
                  "complementarity": COMPLEMENTARITY_LIMIT, "objective_rtol": OBJECTIVE_RTOL,
                  "objective_atol": OBJECTIVE_ATOL},
        "status": "running", "cases": [],
    }
    try:
        run_cases(output, reference, campaign)
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
