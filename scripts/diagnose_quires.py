#!/usr/bin/env python
"""Diagnose quire label inconsistency in joined_records."""

import json
from pathlib import Path

import pandas as pd


def main():
    joined_path = Path("data/processed/joined_records.json")
    if not joined_path.exists():
        print("joined_records.json not found. Run Layer 1 first.")
        return

    df = pd.read_json(joined_path, orient="records")

    print("=== Unique quire values with counts ===")
    if "quire" in df.columns:
        quire_counts = df["quire"].value_counts()
        print(quire_counts)
        print(f"\nTotal unique: {len(quire_counts)}")
        print(f"Total rows: {len(df)}")

        print("\n=== Folio -> Quire mapping ===")
        if "folio" in df.columns:
            folio_quire = df[["folio", "quire"]].drop_duplicates().sort_values("folio")
            print(folio_quire.to_string(index=False))

        print("\n=== Page -> Quire mapping ===")
        if "page" in df.columns:
            page_quire = df[["page", "quire"]].drop_duplicates().sort_values("page")
            print(page_quire.to_string(index=False))

        print("\n=== Normalization check: strip leading q/Q ===")
        def normalize(q):
            if pd.isna(q):
                return q
            q_str = str(q).strip()
            if q_str.lower().startswith('q'):
                return q_str[1:]
            return q_str

        df["quire_norm"] = df["quire"].apply(normalize)
        norm_counts = df["quire_norm"].value_counts()
        print(norm_counts)
        print(f"\nUnique after normalization: {len(norm_counts)}")

        # Check if normalization collapses labels
        original_to_norm = df[["quire", "quire_norm"]].drop_duplicates().sort_values("quire")
        print("\n=== Mapping original -> normalized ===")
        for _, row in original_to_norm.iterrows():
            if row["quire"] != row["quire_norm"]:
                print(f"  {row['quire']} -> {row['quire_norm']}")

        # Proposed normalization dict
        mapping = {}
        for _, row in original_to_norm.iterrows():
            if row["quire"] != row["quire_norm"]:
                mapping[str(row["quire"])] = str(row["quire_norm"])
        print("\n=== Proposed normalization dict ===")
        print(json.dumps(mapping, indent=2))

        # Check leakage after normalization
        print("\n=== Leakage check on normalized quires ===")
        # Load current splits
        splits_dir = Path("data/splits")
        if splits_dir.exists():
            all_fold_quires = []
            for fold_file in sorted(splits_dir.glob("fold_*.json")):
                with open(fold_file) as f:
                    fold_data = json.load(f)
                fold_quires = fold_data.get("quires", [])
                # Normalize fold quires
                norm_fold_quires = [normalize(q) for q in fold_quires]
                all_fold_quires.extend(norm_fold_quires)
                print(f"  Fold {fold_data.get('fold')}: {norm_fold_quires}")

            seen = set()
            leaks = []
            for q in all_fold_quires:
                if q in seen:
                    leaks.append(q)
                seen.add(q)
            if leaks:
                print(f"  LEAKS DETECTED: {leaks}")
            else:
                print("  No leaks after normalization")

    else:
        print("No 'quire' column found")


if __name__ == "__main__":
    main()