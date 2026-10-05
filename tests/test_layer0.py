"""Tests for Layer 0 (Foundational Audit)."""

import json
import tempfile
from pathlib import Path

import pytest

from src.layer0_audit.audit import compute_sha256, generate_manifest


def test_compute_sha256_stability():
    """Same input → same hash; different input → different hash."""
    data1 = {"a": 1, "b": [2, 3], "c": {"d": 4}}
    data2 = {"a": 1, "b": [2, 3], "c": {"d": 4}}
    data3 = {"a": 1, "b": [2, 3], "c": {"d": 5}}

    hash1 = compute_sha256(data1)
    hash2 = compute_sha256(data2)
    hash3 = compute_sha256(data3)

    assert hash1 == hash2, "Same input should produce same hash"
    assert hash1 != hash3, "Different input should produce different hash"
    assert len(hash1) == 64, "SHA-256 should be 64 hex characters"


def test_generate_manifest_writes_file(tmp_path):
    """Passes a synthetic sources list, asserts manifest exists and has generated_at."""
    sources = [
        {
            "name": "test_source",
            "source_type": "test",
            "repo_id": "test/repo",
            "status": "loaded",
            "row_count": 100,
            "schema": {"col1": "int", "col2": "string"},
            "sha256": "abc123",
            "error": None,
        }
    ]

    # Change to temp directory to avoid polluting actual data
    old_cwd = Path.cwd()
    try:
        import os
        os.chdir(tmp_path)
        # Create the data/processed directory structure
        (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)

        manifest = generate_manifest(sources)

        manifest_path = tmp_path / "data" / "processed" / "data_manifest.json"
        assert manifest_path.exists(), "Manifest file should be created"

        with open(manifest_path, "r") as f:
            loaded = json.load(f)

        assert "generated_at" in loaded, "Manifest should have generated_at field"
        assert loaded["sources"] == sources, "Sources should match input"
        assert "cross_transcription_stats" in loaded, "Should have cross_transcription_stats"
    finally:
        os.chdir(old_cwd)


def test_manifest_handles_failed_sources(tmp_path):
    """Passes a list with a status: 'failed' entry, asserts manifest is produced."""
    sources = [
        {
            "name": "failed_source",
            "source_type": "test",
            "repo_id": "test/repo",
            "status": "failed",
            "row_count": 0,
            "schema": {},
            "sha256": None,
            "error": "Connection timeout",
        },
        {
            "name": "loaded_source",
            "source_type": "test",
            "repo_id": "test/repo2",
            "status": "loaded",
            "row_count": 50,
            "schema": {"col1": "int"},
            "sha256": "def456",
            "error": None,
        },
    ]

    old_cwd = Path.cwd()
    try:
        import os
        os.chdir(tmp_path)
        (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)

        manifest = generate_manifest(sources)

        manifest_path = tmp_path / "data" / "processed" / "data_manifest.json"
        assert manifest_path.exists(), "Manifest file should be created even with failed sources"

        with open(manifest_path, "r") as f:
            loaded = json.load(f)

        assert len(loaded["sources"]) == 2, "Both sources should be in manifest"
        assert loaded["sources"][0]["status"] == "failed", "Failed status should be preserved"
        assert loaded["sources"][1]["status"] == "loaded", "Loaded status should be preserved"
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])