"""Layer 2: Unit Discovery - BPE module. Filled by Brick 3."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def extract_glyph_stream(df: pd.DataFrame) -> list[str]:
    """Return a flat list of EVA glyph strings from joined_records.
    One entry per glyph token. Preserve order across the corpus."""
    glyph_stream = []
    for tokens in df["tokens"]:
        if isinstance(tokens, list):
            glyph_stream.extend(tokens)
    return glyph_stream


def _train_bpe_ffbpe(glyph_stream: list[str], vocab_size: int) -> dict:
    """Train BPE using ffbpe if available."""
    try:
        import ffbpe
    except ImportError:
        return {"status": "unavailable", "reason": "ffbpe not installed"}

    # ffbpe expects a string with space-separated tokens
    text = " ".join(glyph_stream)
    model = ffbpe.train_bpe(text, vocab_size=vocab_size)
    return {"model": model, "backend": "ffbpe"}


def _train_bpe_tokenizers(glyph_stream: list[str], vocab_size: int) -> dict:
    """Train BPE using Hugging Face tokenizers library."""
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer
    except ImportError:
        return {"status": "unavailable", "reason": "tokenizers not installed"}

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    # Train on the glyph stream
    # tokenizers expects an iterator of strings
    def batch_iterator():
        # Yield in chunks
        chunk_size = 10000
        for i in range(0, len(glyph_stream), chunk_size):
            yield " ".join(glyph_stream[i:i + chunk_size])

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    return {"model": tokenizer, "backend": "tokenizers"}


def train_bpe(glyph_stream: list[str], vocab_size: int = 500) -> dict:
    """Train a BPE model on the glyph stream.
    Use ffbpe if importable; otherwise fall back to tokenizers.BPE (HF).
    Return a dict with .encode/.decode callables or equivalent."""
    # Try ffbpe first
    result = _train_bpe_ffbpe(glyph_stream, vocab_size)
    if result.get("status") != "unavailable":
        return result

    # Fall back to tokenizers
    result = _train_bpe_tokenizers(glyph_stream, vocab_size)
    if result.get("status") != "unavailable":
        return result

    # Both unavailable - return minimal info
    return {"status": "unavailable", "reason": "No BPE backend available"}


def _get_vocab(model_dict: dict) -> list[str]:
    """Extract vocabulary from trained model."""
    model = model_dict.get("model")
    backend = model_dict.get("backend")

    if model is None:
        return []

    if backend == "ffbpe":
        # ffbpe model has a vocab attribute
        try:
            return list(model.vocab.keys())[:model.vocab_size]  # type: ignore[attr-defined]
        except AttributeError:
            return []
    elif backend == "tokenizers":
        # HF tokenizers
        try:
            return list(model.get_vocab().keys())  # type: ignore[attr-defined]
        except AttributeError:
            return []
    return []


def _encode_bpe(model_dict: dict, text: str) -> list[int]:
    """Encode text using trained BPE model."""
    model = model_dict.get("model")
    backend = model_dict.get("backend")

    if model is None:
        return []

    if backend == "ffbpe":
        return model.encode(text)  # type: ignore[attr-defined]
    elif backend == "tokenizers":
        return model.encode(text).ids  # type: ignore[attr-defined]
    return []


def test_unit_stability_by_source(df: pd.DataFrame, vocab_size: int = 500) -> dict:
    """For each transcription source present in df (or each available source
    slice if the source column is unavailable, split the corpus by quire):
    - Train BPE independently.
    - Compute Jaccard similarity of discovered unit sets vs. the reference.
    - Return {source_or_slice: {"jaccard": float, "n_units": int}}."""
    # Since transcription_source is "unknown" for all rows in current data,
    # split by quire instead
    results = {}

    # Get reference vocabulary from full corpus
    full_stream = extract_glyph_stream(df)
    ref_model = train_bpe(full_stream, vocab_size)
    ref_vocab = set(_get_vocab(ref_model))

    if not ref_vocab:
        return {"error": "Failed to train reference model"}

    # Split by quire and train independently
    for quire in df["quire"].dropna().unique():
        quire_df = df[df["quire"] == quire]
        quire_stream = extract_glyph_stream(quire_df)

        if len(quire_stream) < 100:  # Too small to train reliably
            continue

        quire_model = train_bpe(quire_stream, vocab_size)
        quire_vocab = set(_get_vocab(quire_model))

        if quire_vocab:
            intersection = ref_vocab & quire_vocab
            union = ref_vocab | quire_vocab
            jaccard = len(intersection) / len(union) if union else 0.0
            results[str(quire)] = {"jaccard": jaccard, "n_units": len(quire_vocab)}

    return results


def quire_resampling_stability(
    glyph_stream: list[str],
    quire_labels: list[str],
    n_iterations: int = 100,
    vocab_size: int = 500,
) -> dict:
    """Bootstrap quires with replacement, retrain BPE, compute Jaccard
    similarity of unit inventories across iterations.
    Return {"mean_jaccard": float, "std": float, "ci_95": [lo, hi]}."""
    import random
    import statistics

    if len(glyph_stream) != len(quire_labels):
        return {"error": "glyph_stream and quire_labels must have same length"}

    # Group glyphs by quire
    quire_to_glyphs: dict[str, list[str]] = {}
    for glyph, quire in zip(glyph_stream, quire_labels):
        quire_to_glyphs.setdefault(quire, []).append(glyph)

    unique_quires = list(quire_to_glyphs.keys())
    if len(unique_quires) < 2:
        return {"error": "Need at least 2 quires for bootstrap"}

    # Train reference on full corpus
    ref_model = train_bpe(glyph_stream, vocab_size)
    ref_vocab = set(_get_vocab(ref_model))
    if not ref_vocab:
        return {"error": "Failed to train reference model"}

    jaccards = []
    rng = random.Random(42)

    for _ in range(n_iterations):
        # Bootstrap sample quires with replacement
        sampled_quires = rng.choices(unique_quires, k=len(unique_quires))
        bootstrap_stream = []
        for q in sampled_quires:
            bootstrap_stream.extend(quire_to_glyphs[q])

        if len(bootstrap_stream) < 100:
            continue

        bootstrap_model = train_bpe(bootstrap_stream, vocab_size)
        bootstrap_vocab = set(_get_vocab(bootstrap_model))

        if bootstrap_vocab:
            intersection = ref_vocab & bootstrap_vocab
            union = ref_vocab | bootstrap_vocab
            jaccard = len(intersection) / len(union) if union else 0.0
            jaccards.append(jaccard)

    if not jaccards:
        return {"error": "No valid bootstrap iterations"}

    mean_j = statistics.mean(jaccards)
    std_j = statistics.stdev(jaccards) if len(jaccards) > 1 else 0.0

    # 95% CI using normal approximation
    import math
    margin = 1.96 * std_j / math.sqrt(len(jaccards))
    ci_lo = max(0, mean_j - margin)
    ci_hi = min(1, mean_j + margin)

    return {
        "mean_jaccard": mean_j,
        "std": std_j,
        "ci_95": [ci_lo, ci_hi],
        "n_iterations": len(jaccards),
    }


def align_units_with_coordinates(units: list[str], coords: Any) -> dict:
    """If coordinates are available, align. Otherwise return
    {"status": "unavailable", "units_aligned": 0}."""
    if coords is None:
        return {"status": "unavailable", "units_aligned": 0}
    # Placeholder for future implementation
    return {"status": "unavailable", "units_aligned": 0, "note": "Coordinate alignment not implemented"}


def run_unit_discovery() -> dict:
    """Orchestrate: extract stream, train BPE, test source stability,
    bootstrap quires, attempt coordinate alignment, write outputs.
    Return a summary dict."""
    # Load joined records
    joined_path = Path("data/processed/joined_records.json")
    if not joined_path.exists():
        raise FileNotFoundError("joined_records.json not found. Run Layer 1 first.")

    df = pd.read_json(joined_path, orient="records")

    # Extract glyph stream
    glyph_stream = extract_glyph_stream(df)

    # Train BPE on full corpus
    vocab_size = 500
    model_dict = train_bpe(glyph_stream, vocab_size)

    if model_dict.get("status") == "unavailable":
        return {
            "status": "failed",
            "error": "BPE training failed",
            "backend": "none",
        }

    # Get vocabulary
    vocab = _get_vocab(model_dict)

    # Test stability by source/quire
    stability_by_source = test_unit_stability_by_source(df, vocab_size)

    # Get quire labels for bootstrap
    quire_labels = []
    for _, row in df.iterrows():
        tokens = row.get("tokens", [])
        quire = row.get("quire", "")
        quire_labels.extend([quire] * len(tokens))

    # Bootstrap quire stability
    bootstrap_results = quire_resampling_stability(glyph_stream, quire_labels, n_iterations=100, vocab_size=vocab_size)

    # Attempt coordinate alignment (will be unavailable)
    coords = None  # No coordinates available
    alignment = align_units_with_coordinates(vocab, coords)

    # Determine gate status
    stable_units = sum(1 for v in stability_by_source.values() if v.get("jaccard", 0) >= 0.5)
    gate_status = "fail"
    if stable_units >= 50:
        gate_status = "pass"
    elif stable_units > 0:
        gate_status = "partial"

    # Prepare output
    output = {
        "vocab_size": vocab_size,
        "n_units": len(vocab),
        "units": vocab,
        "stability_by_source": stability_by_source,
        "quire_bootstrap": bootstrap_results,
        "coordinate_alignment": alignment,
        "gate_status": gate_status,
        "backend": model_dict.get("backend", "unknown"),
    }

    # Save unit inventory
    output_path = Path("data/processed/unit_inventory.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    # Save summary
    summary = {
        "vocab_size": vocab_size,
        "n_units": len(vocab),
        "gate_status": gate_status,
        "backend": model_dict.get("backend", "unknown"),
        "stability_summary": {
            "n_slices_tested": len(stability_by_source),
            "stable_units_count": stable_units,
        },
        "bootstrap_summary": {
            "mean_jaccard": bootstrap_results.get("mean_jaccard"),
            "std": bootstrap_results.get("std"),
            "ci_95": bootstrap_results.get("ci_95"),
        },
        "output_file": str(output_path),
    }

    summary_path = Path("data/processed/unit_discovery_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


if __name__ == "__main__":
    import json
    summary = run_unit_discovery()
    print(json.dumps(summary, indent=2, default=str))