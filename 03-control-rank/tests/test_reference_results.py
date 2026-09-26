"""Validate the published measurements and attribution without running a benchmark."""

import hashlib
import json
from pathlib import Path

import pytest

import run_joris_controls as campaign

ROOT = Path(__file__).resolve().parents[1]
MEASURED = json.loads((ROOT / "validation/results.json").read_text())
REFERENCE = json.loads((ROOT / "reference/results.json").read_text())


@pytest.mark.parametrize("case", MEASURED["cases"], ids=lambda case: case["case"])
def test_published_case_passes_original_gates(case):
    row = case["result"]
    label = f"{row['kernel']}, {'lifted' if row['lifted'] else 'eliminated'}"
    first = next(
        item["result"] for item in MEASURED["cases"]
        if (item["result"]["stations"], item["result"]["control_rank"])
        == (row["stations"], row["control_rank"])
    )
    campaign.validate_case(row, REFERENCE[label] if row["stations"] == 192 else None, first)
    assert case["status"] == "validated"


def test_published_campaign_is_complete_and_portable():
    assert MEASURED["status"] == "complete"
    assert len(MEASURED["cases"]) == 16
    actual = {
        (case["result"]["stations"], case["result"]["control_rank"],
         case["result"]["kernel"], case["result"]["lifted"])
        for case in MEASURED["cases"]
    }
    expected = {(n, rank, kernel, lifted) for _, n, rank in campaign.GROUPS
                for kernel, lifted in campaign.WRITINGS}
    assert actual == expected
    assert all("command" not in case for case in MEASURED["cases"])


def test_reference_files_match_the_measured_sources():
    provenance = json.loads((ROOT / "validation/provenance.json").read_text())
    sources = provenance["source_files_sha256"]
    for name, original in (("reference/sparse_operator.py", "joris_original.py"),
                           ("reference/results.json", "joris_original_results.json"),
                           ("run_joris_controls.py", "run_joris_controls.py")):
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sources[original]
    source = (ROOT / "bench_joris_controls.py").read_bytes().replace(
        b"see reference/README.md", b"see docs/protocole_pont_joris.md"
    )
    assert hashlib.sha256(source).hexdigest() == sources["bench_joris_controls.py"]
