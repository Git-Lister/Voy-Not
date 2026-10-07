"""Evaluation engine for joint signature S1–S15. Filled by Brick 3a."""

import collections
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pandas import DataFrame


@dataclass
class JointSignature:
    s1_conditional_glyph_entropy: float | None
    s2_token_length_mean: float | None
    s2_token_length_variance: float | None
    s2_token_length_skew: float | None
    s3_mi_plateau: float | None
    s4_d1_agreement_onset: float | None
    s4_d1_agreement_coda: float | None
    s4_d1_agreement_ratio: float | None
    s5_edge_mutual_information: float | None
    s6_zipf_slope: float | None
    s7_hapax_ratio: float | None
    s8_lz77_compression_ratio: float | None
    s9_quire_stability_mean: float | None
    s10_cross_transcription_agreement: float | None
    s11_positional_vocab_divergence: float | None
    s12_page_template_transition_ll: float | None
    s13_hurst_dfa: float | None
    s14_multifractal_delta_h: float | None
    s15_box_counting_dimension: float | None


def compute_conditional_glyph_entropy(tokenized: list[list[str]]) -> float | None:
    """H(next_glyph | current_glyph) in bits. Pooled across all tokens."""
    if not tokenized:
        return None

    # Flatten all tokens into a single sequence of glyphs
    all_glyphs = []
    for token in tokenized:
        all_glyphs.extend(token)

    if len(all_glyphs) < 2:
        return None

    # Count bigrams and unigrams
    bigram_counts = collections.Counter(itertools.pairwise(all_glyphs))
    unigram_counts = collections.Counter(all_glyphs)

    # Compute conditional entropy
    total_bigrams = sum(bigram_counts.values())
    entropy = 0.0

    for (prev, curr), count in bigram_counts.items():
        p_joint = count / total_bigrams
        p_prev = unigram_counts[prev] / len(all_glyphs)
        p_cond = p_joint / p_prev if p_prev > 0 else 0
        if p_cond > 0:
            entropy -= p_joint * math.log2(p_cond)

    return entropy


def compute_token_length_stats(tokenized: list[list[str]]) -> dict:
    """Returns {'mean': ..., 'variance': ..., 'skew': ...}."""
    lengths = [len(t) for t in tokenized if t]
    if not lengths:
        return {"mean": None, "variance": None, "skew": None}

    arr = np.array(lengths, dtype=float)
    mean = float(np.mean(arr))
    variance = float(np.var(arr))

    # Skewness: third standardized moment
    if np.std(arr) > 0:
        skew = float(np.mean(((arr - mean) / np.std(arr)) ** 3))
    else:
        skew = 0.0

    return {"mean": mean, "variance": variance, "skew": skew}


def compute_mi_plateau(tokenized: list[list[str]], max_distance: int = 50) -> float | None:
    """Mutual information between tokens at distance d, averaged over d in [10, 50].
    Tokens treated as opaque units."""
    if not tokenized:
        return None

    # Flatten tokens into a single sequence
    all_tokens = [str(t) for token in tokenized for t in token]
    if len(all_tokens) < max_distance + 1:
        return None

    # Compute MI for each distance
    mi_values = []
    for d in range(10, min(max_distance + 1, len(all_tokens))):
        pairs = list(zip(all_tokens[:-d], all_tokens[d:]))
        if not pairs:
            continue

        # Count joint and marginal frequencies
        joint_counts = collections.Counter(pairs)
        marginal1 = collections.Counter(p[0] for p in pairs)
        marginal2 = collections.Counter(p[1] for p in pairs)
        total = len(pairs)

        mi = 0.0
        for (x, y), count in joint_counts.items():
            p_xy = count / total
            p_x = marginal1[x] / total
            p_y = marginal2[y] / total
            if p_x > 0 and p_y > 0:
                mi += p_xy * math.log2(p_xy / (p_x * p_y))

        if mi > 0:
            mi_values.append(mi)

    return float(np.mean(mi_values)) if mi_values else None


def compute_d1_agreement(tokenized: list[list[str]]) -> dict:
    """For each position i in each token, compare the glyph at position i
    to the glyph at position i+1. Compute onset agreement (first two glyphs),
    coda agreement (last two glyphs), and their ratio."""
    if not tokenized:
        return {"onset": None, "coda": None, "ratio": None}

    onset_agreements = 0
    onset_total = 0
    coda_agreements = 0
    coda_total = 0

    for token in tokenized:
        if len(token) < 2:
            continue

        # Onset: first two glyphs
        if token[0] == token[1]:
            onset_agreements += 1
        onset_total += 1

        # Coda: last two glyphs
        if token[-1] == token[-2]:
            coda_agreements += 1
        coda_total += 1

    onset = onset_agreements / onset_total if onset_total > 0 else None
    coda = coda_agreements / coda_total if coda_total > 0 else None
    ratio = (onset / coda) if (onset is not None and coda is not None and coda > 0) else None

    return {"onset": onset, "coda": coda, "ratio": ratio}


def compute_edge_mutual_information(tokenized: list[list[str]]) -> float | None:
    """MI between the last glyph of token t and the first glyph of token t+1."""
    if not tokenized:
        return None

    edges = []
    for i in range(len(tokenized) - 1):
        if tokenized[i] and tokenized[i + 1]:
            edges.append((tokenized[i][-1], tokenized[i + 1][0]))

    if not edges:
        return None

    joint_counts = collections.Counter(edges)
    left_counts = collections.Counter(e[0] for e in edges)
    right_counts = collections.Counter(e[1] for e in edges)
    total = len(edges)

    mi = 0.0
    for (x, y), count in joint_counts.items():
        p_xy = count / total
        p_x = left_counts[x] / total
        p_y = right_counts[y] / total
        if p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))

    return mi if mi > 0 else None


def compute_zipf_slope(tokenized: list[list[str]]) -> float | None:
    """Log-log regression of frequency rank vs. frequency. Return slope."""
    if not tokenized:
        return None

    all_tokens = [str(t) for token in tokenized for t in token]
    if not all_tokens:
        return None

    freq = collections.Counter(all_tokens)
    counts = sorted(freq.values(), reverse=True)
    ranks = np.arange(1, len(counts) + 1)

    log_ranks = np.log(ranks)
    log_counts = np.log(counts)

    # Linear regression: log(f) = a + b * log(r) => slope = b
    A = np.vstack([log_ranks, np.ones(len(log_ranks))]).T
    slope, _ = np.linalg.lstsq(A, log_counts, rcond=None)[0]

    return float(slope)


def compute_hapax_ratio(tokenized: list[list[str]]) -> float | None:
    """Proportion of distinct tokens that occur exactly once."""
    if not tokenized:
        return None

    all_tokens = [str(t) for token in tokenized for t in token]
    if not all_tokens:
        return None

    freq = collections.Counter(all_tokens)
    hapax = sum(1 for count in freq.values() if count == 1)
    distinct = len(freq)

    return hapax / distinct if distinct > 0 else None


def compute_lz77_compression_ratio(tokenized: list[list[str]]) -> float | None:
    """Compressed length / raw length using a simple LZ77 implementation.
    Report the ratio, not the compressed size."""
    if not tokenized:
        return None

    # Flatten to single sequence
    sequence = [str(t) for token in tokenized for t in token]
    if not sequence:
        return None

    # Simple LZ77: sliding window of size 4096
    window_size = 4096
    raw_length = len(sequence)
    compressed_tokens = 0
    i = 0

    while i < raw_length:
        best_len = 0

        # Search in sliding window
        start = max(0, i - window_size)
        for j in range(start, i):
            length = 0
            while i + length < raw_length and sequence[j + length] == sequence[i + length]:
                length += 1
                best_len = max(best_len, length)

        if best_len >= 3:  # Minimum match length
            compressed_tokens += 2  # distance + length
            i += best_len
        else:
            compressed_tokens += 1
            i += 1

    return compressed_tokens / raw_length if raw_length > 0 else None


def compute_quire_stability(quire_token_lists: dict[str, list[list[str]]], top_n: int = 100) -> float | None:
    """For the top N most frequent tokens overall, compute the average
    fraction of quires in which each appears. Returns mean fraction."""
    if not quire_token_lists:
        return None

    # Get top N tokens overall
    all_tokens = []
    for tokens in quire_token_lists.values():
        for token in tokens:
            all_tokens.extend(token)

    if not all_tokens:
        return None

    freq = collections.Counter(all_tokens)
    top_tokens = {t for t, _ in freq.most_common(top_n)}

    if not top_tokens:
        return None

    n_quires = len(quire_token_lists)
    fractions = []

    for token in top_tokens:  # type: ignore[assignment]
        quire_count = sum(1 for q_tokens in quire_token_lists.values() if any(token in t for t in q_tokens))
        fractions.append(quire_count / n_quires)

    return float(np.mean(fractions)) if fractions else None


def compute_cross_transcription_agreement(mismatch_df: "DataFrame | None") -> float | None:
    """Mean similarity_score from the mismatch dataset. Return None if mismatch_df is None."""
    if mismatch_df is None or "similarity_score" not in mismatch_df.columns:
        return None

    scores = mismatch_df["similarity_score"].dropna()
    if len(scores) == 0:
        return None

    return float(scores.mean())


def compute_positional_vocab_divergence(tokenized: list[list[str]]) -> float | None:
    """For each token, compute the Shannon entropy (bits) of its line-position
    distribution. Positions are normalized to [0,1] and binned into 10 equal-width
    bins. Returns the mean entropy across tokens that appear in >=2 lines.
    Shannon entropy is always >= 0."""
    if not tokenized:
        return None

    n_lines = len(tokenized)
    token_positions: collections.defaultdict[str, list[float]] = collections.defaultdict(list)

    for line_idx, token in enumerate(tokenized):
        for t in token:
            token_positions[str(t)].append(line_idx / max(1, n_lines - 1))

    entropies: list[float] = []
    for token, positions in token_positions.items():  # type: ignore[assignment]
        if len(positions) < 2:
            continue
        hist, _ = np.histogram(positions, bins=10, range=(0, 1), density=False)
        total = hist.sum()
        if total == 0:
            continue
        probs = hist / total
        probs = probs[probs > 0]
        if len(probs) > 1:
            entropy = -np.sum(probs * np.log2(probs))
            entropies.append(entropy)

    return float(np.mean(entropies)) if entropies else None


def compute_page_template_transition_ll(page_records, template_labels) -> float | None:
    """PLACEHOLDER in this Brick. Return None. Will be implemented in Brick 5.
    Do NOT attempt to compute it here."""
    return None


def compute_fractal_properties(tokenized: list[list[str]] | None = None) -> dict:
    """Compute S13, S14, S15 on the given corpus.

    If tokenized is None, fall back to loading the observed values from
    data/processed/fractal_baseline.json (used only when scoring the
    observed corpus itself). If tokenized is provided, compute fresh
    values on that corpus using the functions in src/shared/fractal/.
    """
    if tokenized is None:
        import json
        from pathlib import Path
        path = Path("data/processed/fractal_baseline.json")
        if path.exists():
            data = json.loads(path.read_text())
            return {
                "hurst_dfa": data.get("hurst", {}).get("dfa"),
                "multifractal_delta_h": data.get("multifractal", {}).get("delta_h"),
                "box_counting_dimension": data.get("box_counting", {}).get("dimension"),
            }
        return {"hurst_dfa": None, "multifractal_delta_h": None,
                "box_counting_dimension": None}

# Compute fresh on the provided corpus
    from src.shared.fractal.box_counting import box_counting_dimension
    from src.shared.fractal.hurst import compute_hurst_ensemble
    from src.shared.fractal.multifractal import compute_multifractal

    # Parse glyph stream into numeric IDs using order of first appearance
    # This is deterministic and independent of alphabetical or frequency order
    all_glyphs = [g for token in tokenized for g in token]
    
    # Parse glyph stream into numeric IDs using order of first appearance
    # This is deterministic and independent of alphabetical or frequency order
    seen: dict[str, int] = {}
    numeric_stream: list[float] = []
    for g in all_glyphs:
        if g not in seen:
            seen[g] = len(seen)
        numeric_stream.append(float(seen[g]))
    
    if len(numeric_stream) < 1000:
        return {"hurst_dfa": None, "multifractal_delta_h": None,
                "box_counting_dimension": None}

    try:
        h_result = compute_hurst_ensemble(numeric_stream)
        h = h_result.get("dfa")
    except Exception:  # noqa: BLE001
        h = None
    try:
        mf_result = compute_multifractal(numeric_stream)
        dh = mf_result.get("delta_h")
    except Exception:  # noqa: BLE001
        dh = None

    # Box counting: use per-line normalization (matching original Brick 4b)
    # Create points with per-line normalization: x = position in line (0 to 1),
    # y = glyph_id / n_glyphs. This matches the original glyph_distribution_to_points
    # with scale="line" that produced BC=0.995.
    try:
        # Build vocabulary for glyph IDs
        all_glyphs = []
        for token in tokenized:
            for g in token:
                all_glyphs.append(g)
        unique_glyphs = sorted(set(all_glyphs))
        glyph_to_id = {g: i for i, g in enumerate(unique_glyphs)}
        n_glyphs = len(unique_glyphs)

        # Create points with per-line normalization (x = position in line, y = glyph_id / n_glyphs)
        points: list[tuple[float, float]] = []
        for line_tokens in tokenized:
            if len(line_tokens) < 2:
                continue
            n_tokens = len(line_tokens)
            for i, glyph in enumerate(line_tokens):
                x = i / (n_tokens - 1) if n_tokens > 1 else 0.5
                y = glyph_to_id[glyph] / (n_glyphs - 1) if n_glyphs > 1 else 0.5
                points.append((x, y))

        if len(points) >= 100:
            bc_result = box_counting_dimension(points, normalize=True)
            bc = bc_result.get("dimension") if isinstance(bc_result, dict) else bc_result
        else:
            bc = None
    except Exception:  # noqa: BLE001
        bc = None

    return {
        "hurst_dfa": h,
        "multifractal_delta_h": dh,
        "box_counting_dimension": bc,
    }


def compute_joint_signature(bundle, compute_fractal_fresh: bool = False) -> JointSignature:
    """Orchestrate all property computations using a DataBundle.
    S12 is None (placeholder). S13–S15 loaded from fractal_baseline.json
    unless compute_fractal_fresh=True, in which case they are computed fresh.
    Returns a JointSignature."""

    # S1: Conditional glyph entropy
    s1 = compute_conditional_glyph_entropy(bundle.tokenized)

    # S2: Token length statistics
    s2 = compute_token_length_stats(bundle.tokenized)
    s2_mean = s2.get("mean")
    s2_var = s2.get("variance")
    s2_skew = s2.get("skew")

    # S3: MI plateau
    s3 = compute_mi_plateau(bundle.tokenized)

    # S4: D1 agreement
    s4 = compute_d1_agreement(bundle.tokenized)
    s4_onset = s4.get("onset")
    s4_coda = s4.get("coda")
    s4_ratio = s4.get("ratio")

    # S5: Edge mutual information
    s5 = compute_edge_mutual_information(bundle.tokenized)

    # S6: Zipf slope
    s6 = compute_zipf_slope(bundle.tokenized)

    # S7: Hapax ratio
    s7 = compute_hapax_ratio(bundle.tokenized)

    # S8: LZ77 compression ratio
    s8 = compute_lz77_compression_ratio(bundle.tokenized)

    # S9: Quire stability
    s9 = compute_quire_stability(bundle.quire_token_lists)

    # S10: Cross-transcription agreement
    s10 = compute_cross_transcription_agreement(bundle.mismatch_df)

    # S11: Positional vocabulary divergence
    s11 = compute_positional_vocab_divergence(bundle.tokenized)

    # S12: Page template transition LL (placeholder)
    s12 = compute_page_template_transition_ll(bundle.page_records, None)

    # S13-S15: Fractal properties from baseline or fresh
    fractal = compute_fractal_properties(bundle.tokenized if compute_fractal_fresh else None)
    s13 = fractal.get("hurst_dfa")
    s14 = fractal.get("multifractal_delta_h")
    s15 = fractal.get("box_counting_dimension")

    return JointSignature(
        s1_conditional_glyph_entropy=s1,
        s2_token_length_mean=s2_mean,
        s2_token_length_variance=s2_var,
        s2_token_length_skew=s2_skew,
        s3_mi_plateau=s3,
        s4_d1_agreement_onset=s4_onset,
        s4_d1_agreement_coda=s4_coda,
        s4_d1_agreement_ratio=s4_ratio,
        s5_edge_mutual_information=s5,
        s6_zipf_slope=s6,
        s7_hapax_ratio=s7,
        s8_lz77_compression_ratio=s8,
        s9_quire_stability_mean=s9,
        s10_cross_transcription_agreement=s10,
        s11_positional_vocab_divergence=s11,
        s12_page_template_transition_ll=s12,
        s13_hurst_dfa=s13,
        s14_multifractal_delta_h=s14,
        s15_box_counting_dimension=s15,
    )


def save_signature(sig: JointSignature, path: str) -> None:
    """Write the signature to JSON with rounding to 6 decimal places.
    Handles None values (serialize as JSON null)."""
    output: dict[str, float | None] = {}
    for key, value in asdict(sig).items():
        if value is not None and isinstance(value, float):
            output[key] = round(value, 6)
        else:
            output[key] = value

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def load_observed_signature(path: str = "data/processed/observed_signature.json") -> JointSignature:
    """Load the observed Voynich signature from disk.
    If missing, compute it from the loaded data bundle and save it."""
    p = Path(path)

    if p.exists():
        with open(p, "r") as f:
            data = json.load(f)
        # Convert back to JointSignature
        return JointSignature(**data)

    # Compute if missing
    from src.layer3_mechanisms.data_loader import load_data_bundle
    bundle = load_data_bundle()
    sig = compute_joint_signature(bundle)
    save_signature(sig, path)
    return sig


if __name__ == "__main__":
    import json

    from src.layer3_mechanisms.data_loader import load_data_bundle

    bundle = load_data_bundle()
    sig = compute_joint_signature(bundle)
    save_signature(sig, "data/processed/observed_signature.json")
    print(json.dumps(asdict(sig), indent=2, default=str))