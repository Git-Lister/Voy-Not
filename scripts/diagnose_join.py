#!/usr/bin/env python
"""Diagnose join row count discrepancy."""

from pathlib import Path

import pandas as pd
from datasets import load_dataset


def main():
    print("=== Raw EVA dataset ===")
    eva_ds = load_dataset("Ched-ai/voynich-eva", split="train")
    print(f"Rows: {len(eva_ds)}")
    print(f"Columns: {eva_ds.column_names}")
    print(f"Unique pages: {eva_ds.to_pandas()['page_id'].nunique()}")
    print(f"Unique quires: {eva_ds.to_pandas()['quire'].nunique()}")
    print()

    print("=== Metadata configs ===")
    for config in ["pages", "folios", "quires"]:
        ds = load_dataset("Ched-ai/voynich-manuscript-metadata", config, split="train")
        df = ds.to_pandas()
        print(f"{config}: {len(df)} rows, columns: {df.columns.tolist()}")
        if "page_id" in df.columns:
            print(f"  Unique page_ids: {df['page_id'].nunique()}")
        if "folio_id" in df.columns:
            print(f"  Unique folio_ids: {df['folio_id'].nunique()}")
        if "quire_id" in df.columns:
            print(f"  Unique quire_ids: {df['quire_id'].nunique()}")
        # Check for duplicates
        if "page_id" in df.columns:
            dupes = df[df.duplicated(subset=["page_id"], keep=False)]
            if len(dupes) > 0:
                print(f"  DUPLICATE page_ids: {len(dupes)} rows")
                print(dupes["page_id"].value_counts().head(10))
        print()

    print("=== Joined records ===")
    joined_path = Path("data/processed/joined_records.json")
    if joined_path.exists():
        joined_df = pd.read_json(joined_path, orient="records")
        print(f"Rows: {len(joined_df)}")
        print(f"Columns: {joined_df.columns.tolist()}")

        # Check for duplicate page-line combinations
        if "page" in joined_df.columns and "line_number" in joined_df.columns:
            dupes = joined_df[joined_df.duplicated(subset=["page", "line_number"], keep=False)]
            if len(dupes) > 0:
                print(f"  DUPLICATE page+line: {len(dupes)} rows")
                print(dupes[["page", "line_number"]].value_counts().head(10))

        # Pages in EVA but not in metadata
        eva_pages = set(eva_ds.to_pandas()["page_id"].unique())
        joined_pages = set(joined_df["page"].dropna().unique()) if "page" in joined_df.columns else set()
        print(f"\nPages in EVA but not in joined: {len(eva_pages - joined_pages)}")
        print(f"  Sample: {list(eva_pages - joined_pages)[:10]}")

        # Pages in joined but not in EVA (should be 0 or few)
        print(f"Pages in joined but not in EVA: {len(joined_pages - eva_pages)}")

        # Check metadata pages
        meta_pages = set()
        for config in ["pages", "folios", "quires"]:
            ds = load_dataset("Ched-ai/voynich-manuscript-metadata", config, split="train")
            if "page_id" in ds.column_names:
                meta_pages.update(ds.to_pandas()["page_id"].unique())
            elif "folio_id" in ds.column_names:
                meta_pages.update(ds.to_pandas()["folio_id"].unique())

        print(f"Pages in metadata: {len(meta_pages)}")
        print(f"EVA pages not in metadata: {len(eva_pages - meta_pages)}")
        print(f"  Sample: {list(eva_pages - meta_pages)[:10]}")
    else:
        print("joined_records.json not found")


if __name__ == "__main__":
    main()