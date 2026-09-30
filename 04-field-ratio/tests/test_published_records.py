"""Recheck published evidence without launching another timing campaign."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import run_field_ratio as campaign

ROOT = Path(__file__).resolve().parents[1]
RECORDS = json.loads((ROOT / "validation/results.json").read_text())
PROVENANCE = json.loads((ROOT / "validation/provenance.json").read_text())


@pytest.mark.parametrize("name", sorted(RECORDS["homotopy"]))
def test_historical_homotopy_passes_gates(name):
    row = RECORDS["homotopy"][name]
    first = RECORDS["homotopy"][f"H_n42_z{row['field_ratio']}_t1_dense_eliminated"]
    campaign.validate_case(row, first)
    assert row["campaign_case_status"] == "validated"
    assert row["derivatives"]["hessian_calls"] > 0
    assert row["derivatives"]["hessian_mean_seconds"] > 0.0


def test_complete_h_sweep_and_explicit_nonconverged_cases():
    assert len(RECORDS["homotopy"]) == 32
    for z in (64, 128):
        assert RECORDS["homotopy"][f"H_n42_z{z}_t1_sparse_lifted"]["finding"] \
            == "different_stationary_point"
    assert PROVENANCE["historical_campaign"]["source_status"] == "stopped_at_first_failure"
    assert RECORDS["batched_dense"]["J_n42_z1_t32"]["status"] == "failed"
    last = RECORDS["additional_conditions"]["T_n42_z8_t32_dense_eliminated"]
    assert last["campaign_case_status"] == "failed"
    assert last["solution"]["complementarity_inf"] > campaign.COMPLEMENTARITY_LIMIT


def test_measured_source_hashes():
    assert hashlib.sha256((ROOT / "jax_same_chain.py").read_bytes()).hexdigest() \
        == RECORDS["same_chain"]["provenance"]["source_sha256"]
    for name, expected in PROVENANCE["historical_campaign"]["measured_sources_sha256"].items():
        assert hashlib.sha256((ROOT / "validation/sources" / name).read_bytes()).hexdigest() \
            == expected
    invalidated = PROVENANCE["invalidated"]
    assert hashlib.sha256((ROOT / "validation" / invalidated["file"]).read_bytes()).hexdigest() \
        == invalidated["sha256"]


def test_current_same_chain_is_active_and_timings_are_traceable():
    row = RECORDS["same_chain"]
    assert row["status"] == "complete"
    assert len(row["validation"]) == 3
    for check in row["validation"]:
        assert len(check["control_column_max_abs"]) == row["n_controls"]
        assert min(check["control_column_max_abs"]) > 0.0
        for name in ("forward", "jacobian", "hessian"):
            assert check[name]["max_abs"] > 0.0
            assert 0.0 <= check[name]["relative_error"] <= row["parity_gate"]
    for name in ("forward", "jacobian", "hessian"):
        for backend in ("casadi", "jax"):
            timing = row[name][backend]
            assert len(timing["samples_ms"]) == row["repeats"]
            assert np.isfinite(timing["samples_ms"]).all()
            assert min(timing["samples_ms"]) > 0.0
            assert timing["median_ms"] == float(np.median(timing["samples_ms"]))
        assert row[name]["speedup"] == row[name]["casadi"]["median_ms"] \
            / row[name]["jax"]["median_ms"]
    assert max(row["compiled_hessian"]["parity_errors"]) <= row["parity_gate"]
