#!/usr/bin/env python
"""Diagnose mismatch dataset metrics."""

from datasets import load_dataset


def main():
    ds = load_dataset("Ched-ai/voynich-transcription-mismatch", split="train")
    df = ds.to_pandas()

    print("=== similarity_score distribution ===")
    print(f"Type: {df['similarity_score'].dtype}")
    print(f"Min: {df['similarity_score'].min()}")
    print(f"Max: {df['similarity_score'].max()}")
    print(f"Mean: {df['similarity_score'].mean():.4f}")
    print(f"Median: {df['similarity_score'].median():.4f}")
    print(f"Unique values: {df['similarity_score'].nunique()}")
    print("\nValue counts (top 20):")
    print(df['similarity_score'].value_counts().head(20))

    print("\n=== eva_agreement distribution ===")
    print(df['eva_agreement'].value_counts())

    print("\n=== Sample rows ===")
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        print(f"\nRow {i}:")
        print(f"  page_id: {row['page_id']}")
        print(f"  line_number: {row['line_number']}")
        print(f"  zl_text: {row['zl_text'][:50]}...")
        print(f"  it_text: {row['it_text'][:50]}...")
        print(f"  cd_text: {row['cd_text'][:50]}...")
        print(f"  fg_text: {row['fg_text'][:50]}...")
        print(f"  gc_text: {row['gc_text'][:50]}...")
        print(f"  eva_agreement: {row['eva_agreement']}")
        print(f"  similarity_score: {row['similarity_score']}")
        print(f"  sources_present: {row['sources_present']}")
        print(f"  sources_missing: {row['sources_missing']}")

    print("\n=== Check if similarity_score is binary ===")
    unique_scores = sorted(df['similarity_score'].unique())
    print(f"All unique scores: {unique_scores}")

    # Check if eva_agreement == (similarity_score == 1.0)
    agreement_from_score = (df['similarity_score'] == 1.0)
    print(f"\neva_agreement == (score==1.0): {(df['eva_agreement'] == agreement_from_score).all()}")


if __name__ == "__main__":
    main()