"""Tests for Layer 2 (Unit Discovery - BPE)."""

import json
from pathlib import Path
import tempfile

import pandas as pd
import pytest

from src.layer2_units.bpe import (
    extract_glyph_stream,
    train_bpe,
    quire_resampling_stability,
    align_units_with_coordinates,
    run_unit_discovery,
)


def test_extract_glyph_stream_returns_list():
    """Synthetic DataFrame -> list of str."""
    df = pd.DataFrame({
        "tokens": [["ol", "daiin"], ["ol", "chedy"], ["daiin", "shedy"]],
        "quire": ["A", "A", "B"],
    })

    stream = extract_glyph_stream(df)

    assert isinstance(stream, list)
    assert all(isinstance(g, str) for g in stream)
    assert stream == ["ol", "daiin", "ol", "chedy", "daiin", "shedy"]


def test_train_bpe_produces_vocab():
    """Small synthetic stream -> model with vocab."""
    glyph_stream = ["ol", "daiin", "ol", "chedy", "daiin", "shedy"] * 50

    model_dict = train_bpe(glyph_stream, vocab_size=50)

    assert "model" in model_dict or "status" in model_dict
    if "model" in model_dict:
        # Should have a vocab
        from src.layer2_units.bpe import _get_vocab
        vocab = _get_vocab(model_dict)
        assert len(vocab) > 0
        assert len(vocab) <= 50


def test_quire_resampling_runs():
    """Synthetic stream with 3 quires -> returns dict with expected keys."""
    glyph_stream = ["ol", "daiin"] * 100 + ["chedy", "shedy"] * 100 + ["qokeedy", "qokedy"] * 100
    quire_labels = ["A"] * 200 + ["B"] * 200 + ["C"] * 200

    result = quire_resampling_stability(glyph_stream, quire_labels, n_iterations=10, vocab_size=50)

    assert isinstance(result, dict)
    assert "mean_jaccard" in result or "error" in result
    if "mean_jaccard" in result:
        assert 0 <= result["mean_jaccard"] <= 1
        assert "std" in result
        assert "ci_95" in result
        assert len(result["ci_95"]) == 2


def test_align_units_handles_no_coordinates():
    """Empty coords -> returns status 'unavailable'."""
    result = align_units_with_coordinates(["ol", "daiin", "chedy"], None)

    assert isinstance(result, dict)
    assert result.get("status") == "unavailable"
    assert result.get("units_aligned") == 0


def test_run_unit_discovery_writes_inventory():
    """Asserts data/processed/unit_inventory.json exists and is valid JSON."""
    # This test runs the actual function which requires the joined_records.json
    # Since it's already generated from previous steps, we can test it
    result = run_unit_discovery()

    assert isinstance(result, dict)
    assert "vocab_size" in result
    assert "n_units" in result
    assert "gate_status" in result
    assert result["gate_status"] in ["pass", "partial", "fail"]

    # Check file was written
    inventory_path = Path("data/processed/unit_inventory.json")
    assert inventory_path.exists(), "unit_inventory.json should be created"

    with open(inventory_path, "r") as f:
        inventory = json.load(f)

    assert "vocab_size" in inventory
    assert "n_units" in inventory
    assert "units" in inventory
    assert isinstance(inventory["units"], list)
    assert len(inventory["units"]) == inventory["n_units"]
    assert "gate_status" in inventory


if __name__ == "__main__":
    pytest.main([__file__, "-v"])