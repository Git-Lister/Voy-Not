"""Fractal analysis package. Filled by Brick 4."""

from .box_counting import (
    box_counting_dimension,
    box_counting_for_observed,
    glyph_distribution_to_points,
)
from .hurst import (
    compute_hurst_dfa,
    compute_hurst_ensemble,
    compute_hurst_rs,
    hurst_by_quire,
)
from .multifractal import compute_multifractal, multifractal_by_quire

__all__ = [
    "box_counting_dimension",
    "box_counting_for_observed",
    "compute_hurst_dfa",
    "compute_hurst_ensemble",
    "compute_hurst_rs",
    "compute_multifractal",
    "glyph_distribution_to_points",
    "hurst_by_quire",
    "multifractal_by_quire",
    "run_fractal_analysis",
]


def _extract_glyph_stream(df) -> list[int]:
    """Extract a numeric sequence from the glyph stream.
    Maps each unique EVA glyph to an integer ID."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if "tokens" not in df.columns:
        raise ValueError("DataFrame must have 'tokens' column")

    # Parse EVA tokens into individual glyphs
    # Tokens are like "fachys.ykal.ar.ataiin" - split on dots and commas
    all_glyphs = []
    for tokens in df["tokens"]:
        if isinstance(tokens, list):
            for token in tokens:
                if isinstance(token, str):
                    # Split on dots and commas, keep non-empty parts
                    parts = token.replace(',', '.').split('.')
                    for part in parts:
                        if part:
                            all_glyphs.append(part)

    # Build vocabulary
    vocab = sorted(set(all_glyphs))
    glyph_to_id = {g: i for i, g in enumerate(vocab)}

    # Convert to numeric sequence
    sequence = [glyph_to_id[g] for g in all_glyphs]

    return sequence


def run_fractal_analysis(joined_records_path: str = "data/processed/joined_records.json") -> dict:
    """Orchestrate: load joined records, compute Hurst, multifractal, box-counting.
    Write data/processed/fractal_baseline.json. Return summary dict."""
    import json
    from pathlib import Path

    import pandas as pd

    # Load joined records
    path = Path(joined_records_path)
    if not path.exists():
        raise FileNotFoundError(f"joined_records.json not found at {joined_records_path}. Run Layer 1 first.")

    df = pd.read_json(path, orient="records")

    # Extract glyph stream as numeric sequence
    sequence_int = _extract_glyph_stream(df)
    sequence = [float(x) for x in sequence_int]

    if len(sequence) < 1000:
        raise ValueError(f"Insufficient sequence length ({len(sequence)}), need at least 1000 for fractal analysis")

    results = {}

    # 1. Hurst analysis
    try:
        hurst_result = compute_hurst_ensemble(sequence)
        results["hurst"] = hurst_result
        print(f"Hurst: RS={hurst_result.get('rs'):.4f}, DFA={hurst_result.get('dfa'):.4f}, mean={hurst_result.get('mean'):.4f}")
    except Exception as e:  # noqa: BLE001
        results["hurst"] = {"error": str(e)}
        print(f"Hurst analysis failed: {e}")

    # 2. Multifractal analysis
    try:
        mf_result = compute_multifractal(sequence)
        results["multifractal"] = mf_result
        print(f"Multifractal: delta_h={mf_result.get('delta_h'):.4f}, delta_alpha={mf_result.get('delta_alpha'):.4f}, backend={mf_result.get('backend')}")
    except Exception as e:  # noqa: BLE001
        results["multifractal"] = {"error": str(e)}
        print(f"Multifractal analysis failed: {e}")

    # 3. Box-counting analysis
    try:
        # Convert glyph distribution to points
        points = glyph_distribution_to_points(df, scale="line")
        if len(points) >= 100:
            bc_result = box_counting_dimension(points)
            results["box_counting"] = bc_result
            print(f"Box-counting: dimension={bc_result.get('dimension'):.4f}, R^2={bc_result.get('r_squared'):.4f}")
        else:
            results["box_counting"] = {"error": f"Insufficient points ({len(points)})"}
            print(f"Box-counting: insufficient points ({len(points)})")
    except Exception as e:  # noqa: BLE001
        results["box_counting"] = {"error": str(e)}
        print(f"Box-counting analysis failed: {e}")

    # Add metadata
    results["metadata"] = {
        "sequence_length": len(sequence),
        "n_unique_glyphs": len(set(sequence)),
        "n_points_box_counting": len(glyph_distribution_to_points(df, scale="line")) if "box_counting" in results and "error" not in results.get("box_counting", {}) else 0,
        "source": "joined_records",
        "value_column": "glyph_stream",
    }

    # Save to file
    output_path = Path("data/processed/fractal_baseline.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Fractal baseline saved to {output_path}")

    return results


if __name__ == "__main__":
    import json
    result = run_fractal_analysis()
    print(json.dumps(result, indent=2, default=str))