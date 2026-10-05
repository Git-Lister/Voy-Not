"""Layer 1: Representation - Parse module. Filled by Brick 2, updated in Brick 3."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def normalize_quire(q: str) -> str:
    """Normalize quire labels by stripping leading 'q' or 'Q' if present.
    Single-letter quires (A-Z) are preserved as-is."""
    if not q or pd.isna(q):
        return ""
    q_str = str(q).strip()
    # Only strip leading q/Q if the result would be a non-empty single letter
    if len(q_str) > 1 and q_str[0].lower() == 'q':
        return q_str[1:]
    return q_str


@dataclass
class LineRecord:
    page: str
    folio: str
    quire: str
    line_number: int
    tokens: list[str]
    raw_text: str
    transcription_source: str  # "ZL" | "IT" | "CD" | "FG" | "GC" | "unknown"


@dataclass
class PageRecord:
    page_id: str
    folio: str
    quire: str
    section: str | None  # e.g. "botanical", "astronomical", if metadata exposes it
    illustration_type: str | None


@dataclass
class TokenCoordinate:
    token_id: str
    page: str
    line: int
    x: float
    y: float
    width: float
    height: float


def _tokenize_eva(text: str) -> list[str]:
    """Simple EVA tokenizer: split on spaces, keep punctuation as separate tokens."""
    if not text or pd.isna(text):
        return []
    # EVA uses spaces between tokens, sometimes with special chars
    return text.strip().split()


def parse_eva_lines(dataset) -> list[LineRecord]:
    """Parse the loaded vcat_eva dataset into LineRecords.
    Detect transcription source from dataset schema/metadata where possible."""
    records = []
    for row in dataset:
        page_id = row.get("page_id", "")
        # Extract folio from page_id (e.g., "f1r" -> "f1")
        folio = page_id[:-1] if page_id and page_id[-1] in ("r", "v") else page_id
        quire = normalize_quire(row.get("quire", ""))
        line_number = row.get("line_number", 0)
        text = row.get("text", "")
        text_clean = row.get("text_clean", text)
        tokens = _tokenize_eva(text_clean)

        # The EVA dataset doesn't specify which transcription source was used
        # Default to "unknown" - the mismatch dataset has the 5-source comparison
        transcription_source = "unknown"

        records.append(LineRecord(
            page=page_id,
            folio=folio,
            quire=quire,
            line_number=line_number,
            tokens=tokens,
            raw_text=text,
            transcription_source=transcription_source,
        ))
    return records


def parse_metadata(loaded_configs: dict) -> list[PageRecord]:
    """Parse whichever metadata configs loaded into PageRecords.
    Merge pages/folios/quires if all three available."""
    records = []
    pages_ds = loaded_configs.get("pages")
    folios_ds = loaded_configs.get("folios")

    # Build lookup maps from folios
    folio_map = {}
    if folios_ds:
        for row in folios_ds:
            folio_id = row.get("folio_id", "")
            folio_map[folio_id] = {
                "quire_id": normalize_quire(row.get("quire_id", "")),
                "section_value": row.get("section_value"),
                "illustration_type": None,  # folios config doesn't have this directly
            }

    # Use pages config as primary since it has the most detail including illustration_type
    if pages_ds:
        for row in pages_ds:
            page_id = row.get("page_id", "")
            folio = row.get("folio_id", "")
            quire = normalize_quire(row.get("quire_id", ""))
            section = row.get("section_value")
            illustration_type = row.get("illustration_type")

            records.append(PageRecord(
                page_id=page_id,
                folio=folio,
                quire=quire,
                section=section,
                illustration_type=illustration_type,
            ))
    else:
        # Fallback: create PageRecords from folios if pages not available
        for folio_id, info in folio_map.items():
            records.append(PageRecord(
                page_id=folio_id,
                folio=folio_id,
                quire=info["quire_id"],
                section=info["section_value"],
                illustration_type=None,
            ))

    return records


def parse_yale_coordinates(payload: dict) -> list[TokenCoordinate]:
    """Parse YaleDHLab or Beinecke payload into TokenCoordinates.
    Return empty list if payload is None or unavailable."""
    coords: list[TokenCoordinate] = []
    if not payload:
        return coords

    # Expected format: list of dicts with token_id, page, line, x, y, width, height
    # or dict with these keys
    items = payload if isinstance(payload, list) else [payload]

    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            coords.append(TokenCoordinate(
                token_id=str(item.get("token_id", "")),
                page=str(item.get("page", "")),
                line=int(item.get("line", 0)),
                x=float(item.get("x", 0.0)),
                y=float(item.get("y", 0.0)),
                width=float(item.get("width", 0.0)),
                height=float(item.get("height", 0.0)),
            ))
        except (ValueError, TypeError):
            continue

    return coords


def join_records(
    lines: list[LineRecord],
    pages: list[PageRecord],
    coords: list[TokenCoordinate],
) -> pd.DataFrame:
    """Join into a unified DataFrame. Use a LEFT JOIN from lines (EVA) so row count
    equals the EVA line count exactly. Page-level fields attached where available,
    nulls elsewhere."""
    # Convert to DataFrames
    lines_df = pd.DataFrame([{
        "page": lr.page,
        "folio": lr.folio,
        "quire": lr.quire,
        "line_number": lr.line_number,
        "tokens": lr.tokens,
        "raw_text": lr.raw_text,
        "transcription_source": lr.transcription_source,
    } for lr in lines])

    pages_df = pd.DataFrame([{
        "page_id": pr.page_id,
        "folio": pr.folio,
        "quire": pr.quire,
        "section": pr.section,
        "illustration_type": pr.illustration_type,
    } for pr in pages])

    # LEFT JOIN from lines to pages on page_id
    merged = pd.merge(
        lines_df,
        pages_df,
        left_on="page",
        right_on="page_id",
        how="left",
        suffixes=("", "_page"),
    )

    # Coalesce folio and quire columns (prefer line's version, fall back to page's)
    if "folio_page" in merged.columns:
        merged["folio"] = merged["folio"].fillna(merged["folio_page"])
        merged.drop(columns=["folio_page"], inplace=True)
    if "quire_page" in merged.columns:
        merged["quire"] = merged["quire"].fillna(merged["quire_page"])
        merged.drop(columns=["quire_page"], inplace=True)

    # Coordinates would need token-level join, but we only have line-level data
    # For now, skip coordinate merge since we don't have token coordinates
    # In future, we could expand lines to token-level and join with coords

    return merged


def compare_transcriptions(mismatch_dataset) -> dict:
    """Report the dataset's actual fields without inventing thresholds.
    
    Returns:
    {
        "n_lines": int,
        "eva_agreement_pct": float,
        "similarity_score_stats": {
            "mean": float, "median": float, "std": float,
            "min": float, "max": float,
            "p10": float, "p50": float, "p90": float, "p99": float
        },
        "similarity_score_histogram": {
            "bins": [0.0, 0.1, ..., 1.0],
            "counts": [n0, n1, ..., n10]
        },
        "per_source_present_pct": {"ZL": float, "IT": float, "CD": float, "FG": float, "GC": float},
        "notes": "No arbitrary thresholds. Report the distribution; downstream analysis may define thresholds with pre-registration."
    }"""
    import numpy as np
    
    total = len(mismatch_dataset)
    if total == 0:
        return {}

    eva_agreement_count = 0
    similarity_scores = []
    source_present_counts = {"ZL": 0, "IT": 0, "CD": 0, "FG": 0, "GC": 0}

    for row in mismatch_dataset:
        # eva_agreement: dataset's flag
        if row.get("eva_agreement", False):
            eva_agreement_count += 1

        # similarity_score: collect all values for statistics
        sim_score = row.get("similarity_score")
        if sim_score is not None:
            similarity_scores.append(sim_score)

        # Per-source presence: count lines where source has text
        sources_present = row.get("sources_present", [])
        if isinstance(sources_present, list):
            for src in sources_present:
                if src in source_present_counts:
                    source_present_counts[src] += 1

    # Compute similarity_score statistics
    if similarity_scores:
        scores_arr = np.array(similarity_scores)
        stats = {
            "mean": float(np.mean(scores_arr)),
            "median": float(np.median(scores_arr)),
            "std": float(np.std(scores_arr)),
            "min": float(np.min(scores_arr)),
            "max": float(np.max(scores_arr)),
            "p10": float(np.percentile(scores_arr, 10)),
            "p50": float(np.percentile(scores_arr, 50)),
            "p90": float(np.percentile(scores_arr, 90)),
            "p99": float(np.percentile(scores_arr, 99)),
        }
        
        # Histogram with 11 bins (0.0, 0.1, ..., 1.0)
        bins = np.linspace(0.0, 1.0, 11)
        counts, _ = np.histogram(scores_arr, bins=bins)
        histogram = {
            "bins": bins.tolist(),
            "counts": counts.tolist(),
        }
    else:
        stats = {
            "mean": 0.0, "median": 0.0, "std": 0.0,
            "min": 0.0, "max": 0.0,
            "p10": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0,
        }
        histogram = {
            "bins": np.linspace(0.0, 1.0, 11).tolist(),
            "counts": [0] * 10,
        }

    # Per-source presence rate
    per_source_rates = {}
    for src in ["ZL", "IT", "CD", "FG", "GC"]:
        present = source_present_counts[src]
        if present > 0:
            per_source_rates[src] = present / total * 100
        else:
            per_source_rates[src] = 0.0

    report = {
        "n_lines": total,
        "eva_agreement_pct": (eva_agreement_count / total * 100) if total > 0 else 0.0,
        "similarity_score_stats": stats,
        "similarity_score_histogram": histogram,
        "per_source_present_pct": per_source_rates,
        "notes": "No arbitrary thresholds. Report the distribution; downstream analysis may define thresholds with pre-registration.",
    }

    # Save to file
    output_path = Path("data/processed/mismatch_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def run_representation() -> dict:
    """Orchestrate the full Layer 1 pipeline. Returns a summary dict."""
    from datasets import load_dataset

    # Load datasets
    eva_ds = load_dataset("Ched-ai/voynich-eva", split="train")
    mismatch_ds = load_dataset("Ched-ai/voynich-transcription-mismatch", split="train")

    # Load metadata configs
    metadata_configs = {}
    for config in ["pages", "folios", "quires"]:
        metadata_configs[config] = load_dataset("Ched-ai/voynich-manuscript-metadata", config, split="train")

    # Parse
    lines = parse_eva_lines(eva_ds)
    pages = parse_metadata(metadata_configs)
    coords: list[TokenCoordinate] = []  # No coordinates available yet

    # Join
    joined_df = join_records(lines, pages, coords)

    # Save joined records
    output_path = Path("data/processed/joined_records.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joined_df.to_json(output_path, orient="records", indent=2)

    # Compare transcriptions
    mismatch_report = compare_transcriptions(mismatch_ds)

    # Save representation summary
    summary = {
        "n_lines": len(lines),
        "n_pages": len(pages),
        "n_coords": len(coords),
        "joined_shape": list(joined_df.shape),
        "mismatch_report": mismatch_report,
        "output_files": {
            "joined_records": str(output_path),
            "mismatch_report": "data/processed/mismatch_report.json",
        }
    }

    summary_path = Path("data/processed/representation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


if __name__ == "__main__":
    import json
    summary = run_representation()
    print(json.dumps(summary, indent=2, default=str))