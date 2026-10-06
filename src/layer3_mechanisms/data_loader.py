"""Data loader for Layer 3 evaluation engine. Filled by Brick 3a."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pandas import DataFrame


@dataclass
class DataBundle:
    tokenized: list[list[str]]
    line_boundaries: list[int]
    quire_labels: list[str]
    quire_token_lists: dict[str, list[list[str]]]
    mismatch_df: "DataFrame | None"
    page_records: list | None
    token_count: int


def _load_joined_records() -> pd.DataFrame:
    """Load joined records from parquet or JSON."""
    parquet_path = Path("data/processed/joined_records.parquet")
    json_path = Path("data/processed/joined_records.json")
    
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except ImportError:
            pass
    
    if json_path.exists():
        return pd.read_json(json_path, orient="records")
    
    raise FileNotFoundError("joined_records.parquet or .json not found. Run Layer 1 first.")


def _load_mismatch_df() -> pd.DataFrame | None:
    """Load mismatch data from local JSON or HF dataset."""
    local_path = Path("data/raw/vcat_mismatch/data.json")
    
    if local_path.exists():
        try:
            return pd.read_json(local_path, orient="records")
        except Exception:  # noqa: BLE001
            pass
    
    try:
        from datasets import load_dataset
        ds = load_dataset("Ched-ai/voynich-transcription-mismatch", split="train")
        return ds.to_pandas()
    except Exception:  # noqa: BLE001
        return None


def _load_page_records() -> list | None:
    """Load page records from joined records."""
    try:
        df = _load_joined_records()
        if "page_id" in df.columns:
            return df[["page_id", "folio", "quire", "section", "illustration_type"]].drop_duplicates().to_dict("records")
    except Exception:  # noqa: BLE001
        pass
    return None


def _parse_tokens_column(df: pd.DataFrame) -> list[list[str]]:
    """Parse the tokens column from joined records.
    
    The tokens column contains a list with a single string representing
    the EVA transcription of the line. Tokens are separated by dots (.)
    and commas (,).
    """
    tokenized = []
    
    for tokens in df["tokens"]:
        if isinstance(tokens, list) and tokens:
            # Take the first element and split on dots and commas
            text = str(tokens[0])
            # Split on dots and commas, keep non-empty parts
            parts = text.replace(',', '.').split('.')
            glyphs = [p for p in parts if p]
            tokenized.append(glyphs)
        elif isinstance(tokens, str):
            text = tokens.strip()
            parts = text.replace(',', '.').split('.')
            glyphs = [p for p in parts if p]
            tokenized.append(glyphs)
        else:
            tokenized.append([])
    
    return tokenized


def load_data_bundle() -> DataBundle:
    """Load all Layer 1 outputs and assemble a DataBundle.
    
    Sources:
    - data/processed/joined_records.parquet (or .json fallback)
    - data/processed/mismatch_report.json
    - data/raw/vcat_mismatch/data.json (for mismatch_df)
    
    If joined_records is missing, raises FileNotFoundError.
    
    Returns:
        DataBundle with all required fields for evaluation engine and null models.
    """
    # Load joined records
    df = _load_joined_records()
    
    # Parse tokens
    tokenized = _parse_tokens_column(df)
    
    # Line boundaries: cumulative token counts
    line_lengths = [len(t) for t in tokenized]
    line_boundaries = [0]
    for length in line_lengths:
        line_boundaries.append(line_boundaries[-1] + length)
    
    # Quire labels (one per line)
    quire_labels = df["quire"].tolist()
    
    # Group by quire
    quire_token_lists: dict[str, list[list[str]]] = {}
    for i, quire in enumerate(quire_labels):
        if quire not in quire_token_lists:
            quire_token_lists[quire] = []
        quire_token_lists[quire].append(tokenized[i])
    
    # Mismatch DataFrame
    mismatch_df = _load_mismatch_df()
    
    # Page records
    page_records = _load_page_records()
    
    # Total token count
    token_count = sum(line_lengths)
    
    return DataBundle(
        tokenized=tokenized,
        line_boundaries=line_boundaries,
        quire_labels=quire_labels,
        quire_token_lists=quire_token_lists,
        mismatch_df=mismatch_df,
        page_records=page_records,
        token_count=token_count,
    )