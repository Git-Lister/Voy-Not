#!/usr/bin/env python
"""Diagnose the Q quire normalization issue."""

from pathlib import Path

import pandas as pd

from src.layer1_representation.parse import normalize_quire


def main():
    joined_path = Path("data/processed/joined_records.json")
    if not joined_path.exists():
        print("joined_records.json not found. Run Layer 1 first.")
        return

    df = pd.read_json(joined_path, orient="records")

    print("=== Raw quire values (before normalization) ===")
    raw_counts = df["quire"].value_counts()
    print(raw_counts)
    print(f"\nTotal unique: {len(raw_counts)}")
    print(f"Total rows: {len(df)}")

    # Check for "Q" specifically
    q_rows = df[df["quire"] == "Q"]
    print(f"\nRows with raw quire == 'Q': {len(q_rows)}")
    if len(q_rows) > 0:
        print(f"Sample: {q_rows[['page', 'line_number', 'quire']].head(3).to_string()}")

    # Apply current normalization
    print("\n=== Current normalize_quire behavior ===")
    test_values = ["qA", "Q", "A", "", "qB", "qQ", "q"]
    for v in test_values:
        result = normalize_quire(v)
        print(f"  normalize_quire({v!r}) = {result!r}")

    # Apply to all data
    df["quire_norm"] = df["quire"].apply(normalize_quire)
    norm_counts = df["quire_norm"].value_counts()
    print("\n=== Normalized quire values (after current normalize_quire) ===")
    print(norm_counts)
    print(f"\nUnique after normalization: {len(norm_counts)}")
    print(f"Rows with empty string: {(df['quire_norm'] == '').sum()}")

    # Check what "Q" becomes
    q_norm = normalize_quire("Q")
    print(f"\nnormalize_quire('Q') = {q_norm!r}")

    # Check fold coverage
    print("\n=== Current fold coverage ===")
    splits_dir = Path("data/splits")
    if splits_dir.exists():
        all_indices = set()
        for fold_file in sorted(splits_dir.glob("fold_*.json")):
            with open(fold_file) as f:
                fold_data = json.load(f)
            indices = fold_data.get("row_indices", [])
            all_indices.update(indices)
            print(f"  Fold {fold_data.get('fold')}: {len(indices)} rows")

        print(f"\nTotal unique row indices in folds: {len(all_indices)}")
        print(f"Total rows in joined_records: {len(df)}")
        print(f"Missing rows: {len(df) - len(all_indices)}")


if __name__ == "__main__":
    import json
    main()