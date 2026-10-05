"""Box counting module. Filled by Brick 4."""

from typing import Any

import numpy as np


def box_counting_dimension(points: list[tuple[float, float]], n_scales: int = 30) -> dict:
    """Box-counting dimension for 2D point cloud.

    Args:
        points: List of (x, y) coordinates
        n_scales: Number of box scales to test

    Return:
        {"dimension": float, "scales": [...], "counts": [...], "r_squared": float}

    Raise ValueError if fewer than 100 points."""
    if len(points) < 100:
        raise ValueError("Fewer than 100 points - box-counting unreliable")

    points_arr = np.array(points, dtype=float)
    if points_arr.shape[1] != 2:
        raise ValueError("Points must be 2D (x, y)")

    # Normalize points to unit square [0, 1] x [0, 1]
    min_vals = points_arr.min(axis=0)
    max_vals = points_arr.max(axis=0)
    range_vals = max_vals - min_vals

    # Avoid division by zero - if range is 0, add small epsilon
    range_vals = np.where(range_vals == 0, 1.0, range_vals)
    normalized = (points_arr - min_vals) / range_vals

    # Estimate minimum point spacing to avoid saturation regime
    n_pts = len(normalized)
    if n_pts > 500:
        idx = np.random.choice(n_pts, 500, replace=False)
        sample_pts = normalized[idx]
    else:
        sample_pts = normalized

    from scipy.spatial.distance import pdist
    try:
        dists = pdist(sample_pts)
        min_spacing = np.percentile(dists, 1)  # 1st percentile as proxy for min spacing
    except Exception:  # noqa: BLE001
        min_spacing = 1.0 / np.sqrt(len(normalized))  # rough estimate

    # Define scales: from ~1/2 to slightly above min spacing
    max_scale = 0.5
    min_scale = max(min_spacing * 2, 1e-3)
    if min_scale >= max_scale:
        min_scale = max_scale / 10

    scales = np.logspace(np.log10(max_scale), np.log10(min_scale), n_scales)
    counts = []
    valid_scales = []

    for scale in scales:
        # Grid size
        n_boxes = int(np.ceil(1.0 / scale))
        if n_boxes < 2 or n_boxes > 1024:
            continue

        # Assign points to boxes
        box_indices = np.floor(normalized * n_boxes).astype(int)
        box_indices = np.clip(box_indices, 0, n_boxes - 1)

        # Count unique boxes
        unique_boxes = np.unique(box_indices[:, 0] * n_boxes + box_indices[:, 1])
        n_occupied = len(unique_boxes)

        if n_occupied > 1:
            counts.append(n_occupied)
            valid_scales.append(scale)

    if len(counts) < 6:
        raise ValueError("Insufficient valid scales for box-counting")

    # Remove saturated regime: only use scales where count < 0.9 * total_points
    # and count > 4
    total_pts = len(points_arr)
    valid_mask = (np.array(counts) < 0.9 * total_pts) & (np.array(counts) > 4)
    if np.sum(valid_mask) < 4:
        # Fallback: use all non-saturated scales
        valid_mask = np.array(counts) < 0.95 * total_pts
    counts = np.array(counts)[valid_mask].tolist()
    valid_scales = np.array(valid_scales)[valid_mask].tolist()

    if len(counts) < 4:
        raise ValueError("Insufficient valid scales in scaling regime for box-counting")

    # Fit log(N) vs log(1/scale)
    log_scales = np.log(1.0 / np.array(valid_scales))
    log_counts = np.log(np.array(counts))

    # Linear regression
    A = np.vstack([log_scales, np.ones(len(log_scales))]).T
    slope, intercept = np.linalg.lstsq(A, log_counts, rcond=None)[0]

    # R-squared
    predicted = slope * log_scales + intercept
    ss_res = np.sum((log_counts - predicted) ** 2)
    ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "dimension": float(slope),
        "scales": valid_scales,
        "counts": counts,
        "r_squared": float(r_squared),
    }


def glyph_distribution_to_points(df: Any, scale: str = "line") -> list[tuple[float, float]]:
    """Convert the glyph stream into a 2D point cloud for box-counting.

    One option: (position_in_line, glyph_id) as (x, y).
    - x: normalized position within line (0 to 1)
    - y: normalized glyph index in vocabulary (0 to 1)

    Args:
        df: DataFrame with 'tokens' column (list of glyph strings per line)
        scale: "line" (per line) or "corpus" (global position)

    Returns:
        List of (x, y) tuples."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if "tokens" not in df.columns:
        raise ValueError("DataFrame must have 'tokens' column")

    # Parse tokens into individual glyphs (split on dots and commas)
    all_glyphs = []
    for tokens in df["tokens"]:
        if isinstance(tokens, list):
            for token in tokens:
                if isinstance(token, str):
                    parts = token.replace(',', '.').split('.')
                    for part in parts:
                        if part:
                            all_glyphs.append(part)

    if not all_glyphs:
        return []

    vocab = sorted(set(all_glyphs))
    glyph_to_id = {g: i for i, g in enumerate(vocab)}
    n_glyphs = len(vocab)

    points = []

    if scale == "line":
        # For each line, normalize position within line
        for tokens in df["tokens"]:
            if not isinstance(tokens, list) or len(tokens) == 0:
                continue
            # Parse each token into glyphs
            line_glyphs = []
            for token in tokens:
                if isinstance(token, str):
                    parts = token.replace(',', '.').split('.')
                    for part in parts:
                        if part:
                            line_glyphs.append(part)

            n_line_glyphs = len(line_glyphs)
            for i, glyph in enumerate(line_glyphs):
                x = i / max(1, n_line_glyphs - 1)  # normalized position in line [0, 1]
                y = glyph_to_id.get(glyph, 0) / max(1, n_glyphs - 1)  # normalized glyph id [0, 1]
                points.append((x, y))

    elif scale == "corpus":
        # Global position across entire corpus
        position = 0
        for tokens in df["tokens"]:
            if not isinstance(tokens, list) or len(tokens) == 0:
                continue
            for token in tokens:
                if isinstance(token, str):
                    parts = token.replace(',', '.').split('.')
                    for part in parts:
                        if part:
                            x = position / max(1, len(all_glyphs) - 1)
                            y = glyph_to_id.get(part, 0) / max(1, n_glyphs - 1)
                            points.append((x, y))
                            position += 1

    else:
        raise ValueError(f"Unknown scale: {scale}. Use 'line' or 'corpus'.")

    return points


def box_counting_for_observed(df: Any) -> dict:
    """Apply to the observed dataset. Return summary."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    # Convert to points
    points = glyph_distribution_to_points(df, scale="line")

    if len(points) < 100:
        return {"error": f"Insufficient points ({len(points)})"}

    # Compute box-counting dimension
    result = box_counting_dimension(points)
    result["n_points"] = len(points)
    result["method"] = "glyph_distribution_to_points(scale='line')"

    return result