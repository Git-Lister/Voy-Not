"""Layer 1: Representation - Splits module. Filled by Brick 2."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def create_splits(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> dict:
    """Create quire-level folds. No quire appears in more than one fold.
    Save each fold's index to data/splits/fold_{i}.json.
    Write data/splits/split_manifest.json containing:
    {
      "n_folds": 5,
      "seed": 42,
      "created_at": "<ISO 8601>",
      "folds": [{"fold": 0, "quires": [...], "n_rows": N}, ...]
    }
    Return the manifest dict."""
    if "quire" not in df.columns:
        raise ValueError("DataFrame must have 'quire' column for quire-level splits")

    # Get unique quires
    quires = df["quire"].dropna().unique().tolist()
    if not quires:
        raise ValueError("No quires found in DataFrame")

    # Shuffle quires deterministically
    import random
    rng = random.Random(seed)
    quires_shuffled = quires.copy()
    rng.shuffle(quires_shuffled)

    # Assign quires to folds
    folds = []
    for i in range(n_folds):
        fold_quires = quires_shuffled[i::n_folds]
        # Get row indices for this fold
        fold_mask = df["quire"].isin(fold_quires)
        fold_indices = df[fold_mask].index.tolist()

        folds.append({
            "fold": i,
            "quires": fold_quires,
            "n_rows": len(fold_indices),
            "row_indices": fold_indices,
        })

    # Save each fold's indices
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    for fold_info in folds:
        fold_file = splits_dir / f"fold_{fold_info['fold']}.json"
        with open(fold_file, "w", encoding="utf-8") as f:
            json.dump({
                "fold": fold_info["fold"],
                "quires": fold_info["quires"],
                "row_indices": fold_info["row_indices"],
            }, f, indent=2)

    # Create manifest
    manifest = {
        "n_folds": n_folds,
        "seed": seed,
        "created_at": datetime.now(UTC).isoformat(),
        "folds": [
            {"fold": f["fold"], "quires": f["quires"], "n_rows": f["n_rows"]}
            for f in folds
        ],
    }

    manifest_path = splits_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest


def verify_no_leakage(splits: dict) -> bool:
    """Assert no quire appears in multiple folds. Return True if clean."""
    seen_quires = set()
    for fold in splits.get("folds", []):
        for quire in fold.get("quires", []):
            if quire in seen_quires:
                return False
            seen_quires.add(quire)
    return True


def run_splits() -> dict:
    """Load joined records and create splits."""
    joined_path = Path("data/processed/joined_records.json")
    if not joined_path.exists():
        raise FileNotFoundError("joined_records.json not found. Run parse.py first.")

    df = pd.read_json(joined_path, orient="records")
    manifest = create_splits(df)

    # Verify
    no_leakage = verify_no_leakage(manifest)

    return {
        "manifest": manifest,
        "no_leakage": no_leakage,
    }


if __name__ == "__main__":
    import json
    result = run_splits()
    print(json.dumps(result, indent=2, default=str))