"""Layer 1: Representation - Splits module. Filled by Brick 2, updated in Brick 3."""

import json
import random
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.layer1_representation.parse import normalize_quire


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

    # Get unique quires (normalized)
    df["quire_norm"] = df["quire"].apply(normalize_quire)
    quires = df["quire_norm"].dropna().unique().tolist()
    # Filter out empty strings
    quires = [q for q in quires if q]
    if not quires:
        raise ValueError("No quires found in DataFrame after normalization")

    # Shuffle quires deterministically
    rng = random.Random(seed)
    quires_shuffled = quires.copy()
    rng.shuffle(quires_shuffled)

    # Assign quires to folds
    folds = []
    for i in range(n_folds):
        fold_quires = quires_shuffled[i::n_folds]
        # Get row indices for this fold (using normalized quire)
        fold_mask = df["quire_norm"].isin(fold_quires)
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


def verify_no_leakage(splits: dict, all_quires: list[str] | None = None, total_rows: int | None = None) -> bool:
    """Assert no quire appears in multiple folds. Return True if clean.
    If all_quires provided, also assert union of fold quires covers all_quires.
    If total_rows provided, also assert every row index 0..total_rows-1 appears exactly once."""
    seen_quires = set()
    for fold in splits.get("folds", []):
        for quire in fold.get("quires", []):
            if quire in seen_quires:
                return False
            seen_quires.add(quire)

    if all_quires is not None:
        # Normalize all_quires for comparison
        norm_all: set[str] = {normalize_quire(q) for q in all_quires if q}
        norm_folds: set[str] = {q for q in seen_quires if q}
        if norm_all != norm_folds:
            missing_quires = norm_all - norm_folds
            extra_quires = norm_folds - norm_all
            if missing_quires:
                print(f"verify_no_leakage: MISSING quires in folds: {missing_quires}")
            if extra_quires:
                print(f"verify_no_leakage: EXTRA quires in folds: {extra_quires}")
            return False

    if total_rows is not None:
        # Check full coverage: every row index appears exactly once
        # Load row_indices from individual fold files
        from pathlib import Path

        splits_dir = Path("data/splits")
        all_indices: list[int] = []
        for fold in splits.get("folds", []):
            fold_file = splits_dir / f"fold_{fold['fold']}.json"
            if fold_file.exists():
                with open(fold_file) as f:
                    fold_data = json.load(f)
                all_indices.extend(fold_data.get("row_indices", []))
            else:
                print("verify_no_leakage: fold file not found: {fold_file}")
                return False

        if len(all_indices) != total_rows:
            print(f"verify_no_leakage: row count mismatch - got {len(all_indices)}, expected {total_rows}")
            return False
        if len(set(all_indices)) != total_rows:
            print("verify_no_leakage: duplicate row indices found")
            return False
        if set(all_indices) != set(range(total_rows)):
            missing_rows = set(range(total_rows)) - set(all_indices)
            print(f"verify_no_leakage: missing row indices: {sorted(missing_rows)[:10]}...")
            return False

    return True


def run_splits() -> dict:
    """Load joined records and create splits."""
    joined_path = Path("data/processed/joined_records.json")
    if not joined_path.exists():
        raise FileNotFoundError("joined_records.json not found. Run parse.py first.")

    df = pd.read_json(joined_path, orient="records")
    manifest = create_splits(df)

    # Get all quires from data (normalized)
    df["quire_norm"] = df["quire"].apply(normalize_quire)
    all_quires = df["quire_norm"].dropna().unique().tolist()
    all_quires = [q for q in all_quires if q]

    # Verify
    no_leakage = verify_no_leakage(manifest, all_quires, total_rows=len(df))

    return {
        "manifest": manifest,
        "no_leakage": no_leakage,
    }


if __name__ == "__main__":
    import json
    result = run_splits()
    print(json.dumps(result, indent=2, default=str))