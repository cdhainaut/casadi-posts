"""Condition blocks, the shared border and removal of frozen design variables."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import example


@pytest.mark.parametrize("layout,design_block,condition_block", [
    ("design shared", 4, 5), ("design per condition", 0, 9), ("design frozen", 0, 5)
])
def test_layout_dimensions_and_blocks(layout, design_block, condition_block):
    hessian, jacobian, point, border, block = example.build_layout(4, 3, layout)
    assert (border, block) == (design_block, condition_block)
    assert len(point) == design_block + 3 * condition_block
    lower = np.asarray(hessian(point, np.ones(3)))
    full = lower + lower.T - np.diag(np.diag(lower))
    assert np.isfinite(full).all()
    assert np.max(np.abs(full)) > 0.0
    for first in range(3):
        for second in range(first):
            a = design_block + first * condition_block
            b = design_block + second * condition_block
            np.testing.assert_array_equal(full[a:a + block, b:b + block], 0.0)
    columns = np.max(np.abs(np.asarray(jacobian(point))), axis=0)
    assert np.all(columns > 0.0), "inactive variables must not survive freezing"


@pytest.mark.parametrize("layout,nvars,nnz", [
    ("design shared", 89, 1115), ("design per condition", 105, 1155),
    ("design frozen", 85, 765)
])
def test_published_structure(layout, nvars, nnz):
    hessian, _, point, *_ = example.build_layout(example.N, example.K, layout)
    assert len(point) == nvars
    assert hessian.nnz_out(0) == nnz


def test_archived_records_match_current_generator():
    root = Path(example.__file__).parent
    records = json.loads((root / "validation/results.json").read_text())
    provenance = json.loads((root / "validation/provenance.json").read_text())
    assert hashlib.sha256(Path(example.__file__).read_bytes()).hexdigest() \
        == provenance["source_sha256"]
    assert (provenance["stations"], provenance["conditions"]) == (example.N, example.K)
    with np.load(root / "validation/patterns.npz") as patterns:
        for row in records:
            hessian, jacobian, point, border, block = example.build_layout(
                example.N, example.K, row["label"])
            assert len(point) == row["variables"]
            assert hessian.n_nodes() == row["nodes_hessian"]
            assert hessian.nnz_out(0) == row["nnz_hessian"]
            assert jacobian.nnz_out(0) == row["nnz_jacobian"]
            assert len(row["samples_ms"]) == example.EVALUATIONS
            assert row["hessian_ms"] == float(np.median(row["samples_ms"]))
            for key, values in zip(("hessian_rows", "hessian_cols"),
                                   hessian.sparsity_out(0).get_triplet()):
                np.testing.assert_array_equal(patterns[f"{row['label']}|{key}"], values)
            assert int(patterns[f"{row['label']}|design_block"]) == border
            assert int(patterns[f"{row['label']}|condition_block"]) == block


def test_invalid_layout():
    with pytest.raises(ValueError):
        example.build_layout(4, 2, "unknown")


def test_import_does_not_execute(tmp_path):
    root = Path(example.__file__).parent
    command = [sys.executable, "-c", "import example"]
    import os

    environment = {**os.environ, "PYTHONPATH": str(root)}
    result = subprocess.run(command, cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert not list(tmp_path.iterdir())


def test_existing_output_is_preserved(tmp_path):
    output = tmp_path / "published"
    output.mkdir()
    record = output / "results.json"
    record.write_text("preserve\n")
    result = subprocess.run([sys.executable, example.__file__, "--output", str(output)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "FileExistsError" in result.stderr
    assert record.read_text() == "preserve\n"
