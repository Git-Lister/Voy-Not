#!/usr/bin/env python
"""Diagnostic script to list available configs for voynich-manuscript-metadata."""

from datasets import get_dataset_config_names, load_dataset

configs = get_dataset_config_names("Ched-ai/voynich-manuscript-metadata")
print("Available configs:", configs)
for cfg in configs:
    try:
        ds = load_dataset("Ched-ai/voynich-manuscript-metadata", cfg, split="train")
        print(f"--- config={cfg} | rows={len(ds)} | columns={ds.column_names}")
        print(f"    first row: {dict(ds[0])}")
    except Exception as e:  # noqa: BLE001
        print(f"--- config={cfg} | FAILED: {e}")