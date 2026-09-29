"""Algebra, equivalence and CLI smoke for the field/response-ratio homotopy."""

import json
import subprocess
import sys
from pathlib import Path

import casadi as cas
import numpy as np
import pytest

import bench_field_ratio as bench
import run_field_ratio as campaign

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "03-control-rank/reference/sparse_operator.py"


def original_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("joris_original", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("z", [1, 2, 5, 8])
def test_compression_identity(z):
    assert bench.compression_check(42, z) <= 1e-10


def test_z1_reduces_to_joris():
    original = original_module()
    p = cas.MX.sym("p", 4)
    for ours, theirs in ((bench.dense_kernel(p, 192), original.dense_kernel(p)),
                         (bench.field_operator(p, 192, 1), original.sparse_operator(p))):
        evaluate = cas.Function("difference", [p], [ours - theirs])
        np.testing.assert_allclose(evaluate(bench.DESIGN_GUESS), 0.0, atol=1e-12)


def test_field_is_sparse_and_grows():
    p = cas.MX.sym("p", 4)
    for z in (1, 4, 16):
        operator = bench.field_operator(p, 42, z)
        assert operator.shape == (42 * z, 42 * z)
        assert operator.sparsity().nnz() <= 4 * 42 * z


def test_selection_observes_response_nodes():
    for z in (1, 3):
        selection = bench.selection_matrix(6, z)
        np.testing.assert_array_equal(np.asarray(selection), np.eye(6 * z)[:, ::z])


@pytest.mark.parametrize("z", [1, 3])
def test_four_writings_same_objective(z):
    objectives = []
    for kernel, lifted in (("dense", False), ("dense", True),
                           ("sparse", False), ("sparse", True)):
        opti, *_ = bench.build(6, z, 2, kernel, lifted)
        opti.solver("ipopt", {"print_time": False},
                    {**bench.SOLVER_OPTIONS, "print_level": 0, "max_iter": 500})
        objectives.append(float(opti.solve().value(opti.f)))
    np.testing.assert_allclose(objectives, objectives[0], rtol=1e-8, atol=1e-7)


def test_jax_parity():
    pytest.importorskip("jax")
    from jax_dense_bench import casadi_model, initial_point, jax_model

    n, conditions = 8, 2
    point = initial_point(n, conditions)
    cas_functions = casadi_model(n, conditions)
    jax_functions = jax_model(n, conditions)
    for name, cas_function, jax_function in zip(
        ("objective", "gradient", "hessian"), cas_functions, jax_functions
    ):
        reference = np.asarray(cas_function(point), dtype=float).ravel()
        candidate = np.asarray(jax_function(point), dtype=float).ravel()
        scale = max(1.0, float(np.max(np.abs(reference))))
        assert np.max(np.abs(reference - candidate)) / scale <= 1e-9, name


def test_jax_same_chain_parity():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    import jax_same_chain as same

    data = same.synthetic_rig()
    point_c = cas.DM(data["point"].tolist())
    point_j = jnp.asarray(data["point"])
    cas_forward, _, cas_hessian = same.build_casadi(data)
    jax_forward, _, jax_hessian = same.build_jax(data)
    for name, cas_fn, jax_fn in (("forward", cas_forward, jax_forward),
                                 ("hessian", cas_hessian, jax_hessian)):
        reference = np.asarray(cas_fn(point_c), dtype=float).ravel()
        candidate = np.asarray(jax_fn(point_j), dtype=float).ravel()
        scale = max(1.0, float(np.max(np.abs(reference))))
        assert np.max(np.abs(reference - candidate)) / scale <= 1e-9, name


def test_campaign_gates_report_different_stationary_point():
    first = {"solution": {"objective": 1.0, "primal_violation_inf": 0.0,
                          "stationarity_inf": 0.0, "complementarity_inf": 0.0},
             "status": "Solve_Succeeded", "compression_error_inf": 0.0}
    campaign.validate_case(json.loads(json.dumps(first)), first)
    bad = json.loads(json.dumps(first))
    bad["solution"]["objective"] = 1.5
    campaign.validate_case(bad, first)
    assert bad["finding"] == "different_stationary_point"
    assert bad["objective_gap_rel"] > 0.4


def test_campaign_gates_accept_memory_wall_finding():
    row = {"status": "failed", "exception": "RESOURCE_EXHAUSTED: Out of memory",
           "comparison": {}}
    campaign.flag_memory_wall(row)
    campaign.validate_jax(row)
    assert row["finding"] == "jax_hessian_memory_wall"


def test_campaign_gates_reject_unsuccessful_solve():
    first = {"solution": {"objective": 1.0, "primal_violation_inf": 0.0,
                          "stationarity_inf": 0.0, "complementarity_inf": 0.0},
             "status": "Solve_Succeeded", "compression_error_inf": 0.0}
    bad = json.loads(json.dumps(first))
    bad["status"] = "Maximum_Iterations_Exceeded"
    with pytest.raises(ValueError, match="Unsuccessful solve"):
        campaign.validate_case(bad, first)


@pytest.mark.slow
def test_cli_smoke(tmp_path):
    output = tmp_path / "case.json"
    command = [sys.executable, "-u", str(ROOT / "bench_field_ratio.py"),
               "--stations", "6", "--field-ratio", "2", "--conditions", "1",
               "--kernel", "sparse", "--lifted", "--output", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr
    row = json.loads(output.read_text())
    assert row["status"] == "Solve_Succeeded"
    assert row["compression_error_inf"] <= 1e-10
    assert row["field_unknowns"] == 12
