"""Tests for Layer 3 evaluation engine."""

import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pytest

from src.layer3_mechanisms.evaluate import (
    JointSignature,
    compute_conditional_glyph_entropy,
    compute_token_length_stats,
    compute_mi_plateau,
    compute_d1_agreement,
    compute_edge_mutual_information,
    compute_zipf_slope,
    compute_hapax_ratio,
    compute_lz77_compression_ratio,
    compute_quire_stability,
    compute_cross_transcription_agreement,
    compute_positional_vocab_divergence,
    compute_page_template_transition_ll,
    compute_fractal_properties,
    compute_joint_signature,
    save_signature,
    load_observed_signature,
)
from src.layer3_mechanisms.data_loader import DataBundle


@dataclass
class MockBundle:
    """Mock DataBundle for testing."""
    tokenized: list[list[str]]
    line_boundaries: list[int]
    quire_labels: list[str]
    quire_token_lists: dict[str, list[list[str]]]
    mismatch_df: Optional[object]
    page_records: Optional[list]
    token_count: int


def test_conditional_glyph_entropy_on_degenerate_sequence():
    """Single-glyph sequence returns H = 0."""
    tokenized = [["a"], ["a"], ["a"]]
    entropy = compute_conditional_glyph_entropy(tokenized)
    assert entropy == 0.0


def test_token_length_stats_on_known_input():
    """[['a'], ['a','b'], ['a','b','c']] gives mean 2.0, variance 2/3 (population)."""
    tokenized = [["a"], ["a", "b"], ["a", "b", "c"]]
    stats = compute_token_length_stats(tokenized)
    assert abs(stats["mean"] - 2.0) < 1e-10
    # Population variance of [1,2,3] is 2/3
    assert abs(stats["variance"] - 2.0/3.0) < 1e-10
    # Skew for [1,2,3] should be 0 (symmetric)
    assert abs(stats["skew"]) < 1e-10


def test_zipf_slope_on_uniform_frequencies():
    """Uniform distribution gives slope ≈ 0."""
    # Create tokens with exactly equal frequencies
    tokenized = [["a"], ["b"], ["c"], ["d"]]
    slope = compute_zipf_slope(tokenized)
    # For uniform frequencies, log-log regression should give slope near 0
    assert slope is not None
    assert abs(slope) < 1.0


def test_hapax_ratio_all_unique():
    """All tokens unique gives hapax ratio = 1.0."""
    tokenized = [["a"], ["b"], ["c"]]
    ratio = compute_hapax_ratio(tokenized)
    assert ratio == 1.0


def test_hapax_ratio_none_unique():
    """All tokens same gives hapax ratio = 0.0."""
    tokenized = [["a"], ["a"], ["a"]]
    ratio = compute_hapax_ratio(tokenized)
    assert ratio == 0.0


def test_joint_signature_returns_all_fields():
    """Every S1–S15 field is populated (or explicitly None for S12) when given a synthetic DataBundle."""
    # Create a synthetic bundle
    tokenized = [
        ["a", "b"], ["b", "c"], ["c", "a"], ["a", "b", "c"],
        ["b", "a"], ["c", "b"], ["a", "c"], ["b", "c", "a"],
    ]
    line_boundaries = [0]
    for t in tokenized:
        line_boundaries.append(line_boundaries[-1] + len(t))
    
    quire_labels = ["A", "A", "B", "B", "A", "A", "B", "B"]
    quire_token_lists = {}
    for i, q in enumerate(quire_labels):
        quire_token_lists.setdefault(q, []).append(tokenized[i])
    
    bundle = MockBundle(
        tokenized=tokenized,
        line_boundaries=line_boundaries,
        quire_labels=quire_labels,
        quire_token_lists=quire_token_lists,
        mismatch_df=None,
        page_records=None,
        token_count=sum(len(t) for t in tokenized),
    )
    
    sig = compute_joint_signature(bundle)
    
    # Check all fields exist
    fields = [
        "s1_conditional_glyph_entropy", "s2_token_length_mean", "s2_token_length_variance",
        "s2_token_length_skew", "s3_mi_plateau", "s4_d1_agreement_onset",
        "s4_d1_agreement_coda", "s4_d1_agreement_ratio", "s5_edge_mutual_information",
        "s6_zipf_slope", "s7_hapax_ratio", "s8_lz77_compression_ratio",
        "s9_quire_stability_mean", "s10_cross_transcription_agreement",
        "s11_positional_vocab_divergence", "s12_page_template_transition_ll",
        "s13_hurst_dfa", "s14_multifractal_delta_h", "s15_box_counting_dimension"
    ]
    
    for field in fields:
        assert hasattr(sig, field), f"Missing field: {field}"
    
    # S12 should be None (placeholder)
    assert sig.s12_page_template_transition_ll is None


def test_save_and_load_signature_roundtrip():
    """Write then read gives identical values."""
    tokenized = [["a", "b"], ["b", "c"], ["c", "a"]]
    line_boundaries = [0, 2, 4, 6]
    quire_labels = ["A", "A", "B"]
    quire_token_lists = {"A": [["a", "b"], ["b", "c"]], "B": [["c", "a"]]}
    
    bundle = MockBundle(
        tokenized=tokenized,
        line_boundaries=line_boundaries,
        quire_labels=quire_labels,
        quire_token_lists=quire_token_lists,
        mismatch_df=None,
        page_records=None,
        token_count=6,
    )
    
    sig = compute_joint_signature(bundle)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_signature.json"
        save_signature(sig, str(path))
        
        # Load and compare
        with open(path, "r") as f:
            loaded = json.load(f)
        
        original = {k: round(v, 6) if isinstance(v, float) and v is not None else v 
                    for k, v in asdict(sig).items()}
        
        for k, v in original.items():
            assert k in loaded, f"Key {k} missing in loaded"
            if v is None:
                assert loaded[k] is None, f"Expected None for {k}, got {loaded[k]}"
            else:
                assert abs(loaded[k] - v) < 1e-6, f"Mismatch for {k}: {loaded[k]} vs {v}"


def test_fractal_properties_loaded_from_baseline():
    """Asserts S13–S15 match fractal_baseline.json (if it exists)."""
    path = Path("data/processed/fractal_baseline.json")
    if not path.exists():
        pytest.skip("fractal_baseline.json not found")
    
    with open(path, "r") as f:
        baseline = json.load(f)
    
    fractal = compute_fractal_properties()
    
    if "hurst" in baseline and "dfa" in baseline["hurst"]:
        expected_hurst = baseline["hurst"]["dfa"]
        assert fractal["hurst_dfa"] is not None
        assert abs(fractal["hurst_dfa"] - expected_hurst) < 1e-6
    
    if "multifractal" in baseline and "delta_h" in baseline["multifractal"]:
        expected_delta_h = baseline["multifractal"]["delta_h"]
        assert fractal["multifractal_delta_h"] is not None
        assert abs(fractal["multifractal_delta_h"] - expected_delta_h) < 1e-6
    
    if "box_counting" in baseline and "dimension" in baseline["box_counting"]:
        expected_bc = baseline["box_counting"]["dimension"]
        assert fractal["box_counting_dimension"] is not None
        assert abs(fractal["box_counting_dimension"] - expected_bc) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])