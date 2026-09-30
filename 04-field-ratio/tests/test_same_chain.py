"""Active derivatives, independent differences and synchronized measurement."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

same = pytest.importorskip("jax_same_chain")


@pytest.fixture(scope="module")
def chain():
    data = same.synthetic_rig()
    return data, same.build_casadi(data), same.build_jax(data)


@pytest.mark.parametrize("offset", [0.0, 0.25, -0.5])
def test_active_parity_at_multiple_points(chain, offset):
    data, cas_functions, jax_functions = chain
    checks = same.validate(cas_functions, jax_functions, [data["point"] + offset])
    for name in ("forward", "jacobian", "hessian"):
        assert checks[0][name]["relative_error"] <= same.PARITY_GATE
        assert checks[0][name]["max_abs"] > 0.0
    assert np.all(np.asarray(checks[0]["control_column_max_abs"]) > 0.0)


@pytest.mark.parametrize("index", [0, 3, 6])
def test_derivatives_against_finite_differences(chain, index):
    data, functions, _ = chain
    forward, jacobian, hessian = functions
    point = data["point"].copy()
    step = 1e-4
    shift = np.zeros(same.N_CONTROLS)
    shift[index] = step
    forward_difference = (np.asarray(forward(point + shift)).ravel()
                          - np.asarray(forward(point - shift)).ravel()) / (2.0 * step)
    np.testing.assert_allclose(np.asarray(jacobian(point))[:, index], forward_difference,
                               rtol=2e-4, atol=1e-6)
    gradient_difference = (np.asarray(jacobian(point + shift)).sum(axis=0)
                           - np.asarray(jacobian(point - shift)).sum(axis=0)) / (2.0 * step)
    np.testing.assert_allclose(np.asarray(hessian(point))[:, index], gradient_difference,
                               rtol=2e-4, atol=1e-6)
    np.testing.assert_allclose(hessian(point), np.asarray(hessian(point)).T,
                               rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_parity_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="Non-finite"):
        same.relative_error([value], [value])


def test_parity_rejects_mismatched_sizes():
    with pytest.raises(ValueError, match="sizes"):
        same.relative_error(np.ones(3), np.ones(4))


def test_underflow_is_rejected_before_timing():
    data = same.synthetic_rig()
    data["centres"][:, 1] += same.REYNOLDS_REFERENCE
    functions = same.build_casadi(data)
    np.testing.assert_array_equal(functions[0](data["point"]), np.zeros((6, 1)))
    zeros = (lambda p: np.zeros(6), lambda p: np.zeros((6, 9)),
             lambda p: np.zeros((9, 9)))
    with pytest.raises(ValueError, match="Degenerate forward"):
        same.validate(functions, zeros, [data["point"]])


def test_timing_waits_for_each_call():
    events = []

    class Pending:
        def block_until_ready(self):
            events.append("wait")

    def evaluate(_):
        events.append("call")
        return Pending()

    result = same.timed(evaluate, None)
    assert events == ["call", "wait"] * (same.N_REPEATS + 1)
    assert len(result["samples_ms"]) == same.N_REPEATS
    assert result["median_ms"] == float(np.median(result["samples_ms"]))


def test_cli_does_not_overwrite_existing_output(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("preserve\n")
    script = Path(same.__file__)
    result = subprocess.run([sys.executable, str(script), "--output", str(output)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "FileExistsError" in result.stderr
    assert output.read_text() == "preserve\n"
