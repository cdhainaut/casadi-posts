"""Algebra, derivatives and isolated CLI smoke for the non-domain Joris reproducer."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import casadi as cas
import numpy as np
import pytest

import bench_joris_controls as bench
import run_joris_controls as campaign

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference/sparse_operator.py"


@pytest.mark.parametrize("rank", [1, 4, 6])
def test_control_basis(rank):
    basis = bench.control_basis(6, rank)
    assert np.linalg.matrix_rank(basis) == rank
    if rank == 6:
        np.testing.assert_array_equal(basis, np.eye(6))
    else:
        np.testing.assert_array_equal(basis[:, 0], np.ones(6))


def test_original_kernel_and_objective_parity():
    spec = importlib.util.spec_from_file_location("joris_original", REFERENCE)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    p = cas.MX.sym("p", 4)
    for ours, theirs in ((bench.dense_kernel(p, 192), original.dense_kernel(p)),
                         (bench.sparse_operator(p, 192), original.sparse_operator(p))):
        evaluate = cas.Function("difference", [p], [ours - theirs])
        np.testing.assert_allclose(evaluate(bench.DESIGN_GUESS), 0.0, atol=1e-12)
    old_opti, *_ = original.build("sparse", False)
    new_opti, *_ = bench.build(192, 192, "sparse", False)
    assert new_opti.nx == old_opti.nx == 197
    np.testing.assert_allclose(
        old_opti.debug.value(old_opti.f, old_opti.initial()),
        new_opti.debug.value(new_opti.f, new_opti.initial()), rtol=1e-13,
    )


@pytest.mark.parametrize("rank", [1, 4, 6])
def test_reduced_derivatives_and_finite_differences(rank):
    rng = np.random.default_rng(0)
    functions = []
    for kernel in ("dense", "sparse"):
        opti, *_ = bench.build(6, rank, kernel, False)
        expression = cas.cse(opti.f)
        gradient = cas.gradient(expression, opti.x)
        hessian = cas.hessian(expression, opti.x)[0]
        functions.append(cas.Function(kernel, [opti.x], [expression, gradient, hessian]))
    point = np.asarray(opti.debug.value(opti.x, opti.initial())).ravel()
    dense, sparse = (function(point) for function in functions)
    for left, right in zip(dense, sparse):
        np.testing.assert_allclose(left, right, rtol=1e-9, atol=1e-10)
    direction = rng.normal(size=point.size)
    direction /= np.linalg.norm(direction)
    step = 1e-5
    plus = functions[0](point + step * direction)
    minus = functions[0](point - step * direction)
    np.testing.assert_allclose(
        float((plus[0] - minus[0]) / (2 * step)),
        float(dense[1].T @ direction), rtol=1e-6, atol=1e-8,
    )
    np.testing.assert_allclose(
        (plus[1] - minus[1]) / (2 * step), dense[2] @ direction, rtol=1e-6, atol=1e-8,
    )
    assert bench.kernel_check(6) < 1e-12


@pytest.mark.slow
@pytest.mark.parametrize("kernel,lifted", campaign.WRITINGS)
def test_isolated_cli_and_metrics(tmp_path, kernel, lifted):
    output = tmp_path / "result.json"
    command = [sys.executable, str(ROOT / "bench_joris_controls.py"),
               "--stations", "6", "--controls", "4", "--kernel", kernel,
               "--output", str(output)]
    if lifted:
        command.append("--lifted")
    completed = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    (tmp_path / "raw.log").write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(output.read_text())
    campaign.validate_case(result, reference=None, first=None)
    assert result["variables"] == 9 + (6 if lifted else 0)
    assert result["initial"]["primal_violation_inf"] < 1e-10
    assert result["derivatives"]["hessian_triangle_nnz"] > 0
    assert result["derivatives"]["hessian_initial_median_seconds"] > 0


def test_campaign_stops_on_first_child_failure(tmp_path, monkeypatch):
    calls = []

    def failed_child(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(campaign.subprocess, "run", failed_child)
    record = {"cases": []}
    with pytest.raises(RuntimeError, match="Child exit code 2"):
        campaign.run_cases(tmp_path, {}, record)
    assert len(calls) == 1
    assert json.loads((tmp_path / "campaign.json").read_text())["cases"][0]["status"] == "failed"
