"""Sweep the three alternative vocabulary mechanisms, score against observed."""
from __future__ import annotations
import json
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.layer3_mechanisms.data_loader import load_data_bundle
from src.layer3_mechanisms.alt_vocab_dynamics import (
    PureNoveltyParams, generate_pure_novelty,
    FixedVocabParams, generate_fixed_vocab,
    TwoRegimeParams, generate_two_regime,
)
from src.layer3_mechanisms.score import load_tolerance_intervals, score_corpus

logger = logging.getLogger(__name__)
OUTPUT = Path("data/processed/diag_alt_vocab.json")


def _hapax_ratio(tokenized):
    counts = Counter(tuple(t) for t in tokenized)
    if not counts:
        return 0.0
    return sum(1 for v in counts.values() if v == 1) / len(counts)


def _zipf_slope(tokenized):
    counts = Counter(tuple(t) for t in tokenized)
    freqs = sorted(counts.values(), reverse=True)
    if len(freqs) < 10:
        return None
    ranks = np.arange(1, len(freqs) + 1)
    return float(np.polyfit(np.log(ranks), np.log(freqs), 1)[0])


def main():
    bundle = load_data_bundle()
    intervals = load_tolerance_intervals()
    n = bundle.token_count

    results = []

    # Pure novelty
    for mut in [0.1, 0.3, 0.5, 0.7]:
        params = PureNoveltyParams(mutation_strength=mut)
        tokens = generate_pure_novelty(bundle.tokenized, n, params)
        score = score_corpus(tokens, intervals)
        results.append({
            "mechanism": "pure_novelty",
            "params": asdict(params),
            "hapax": _hapax_ratio(tokens),
            "zipf": _zipf_slope(tokens),
            "matches": score["matches"],
            "total": score["total"],
        })

    # Fixed vocab
    for vs in [100, 500, 2000, 5000]:
        params = FixedVocabParams(vocab_size=vs)
        tokens = generate_fixed_vocab(bundle.tokenized, n, params)
        score = score_corpus(tokens, intervals)
        results.append({
            "mechanism": "fixed_vocab",
            "params": asdict(params),
            "hapax": _hapax_ratio(tokens),
            "zipf": _zipf_slope(tokens),
            "matches": score["matches"],
            "total": score["total"],
        })

    # Two regime
    for core_frac in [0.3, 0.5, 0.7, 0.9]:
        params = TwoRegimeParams(core_fraction=core_frac,
                                 novelty_fraction=1 - core_frac)
        tokens = generate_two_regime(bundle.tokenized, n, params)
        score = score_corpus(tokens, intervals)
        results.append({
            "mechanism": "two_regime",
            "params": asdict(params),
            "hapax": _hapax_ratio(tokens),
            "zipf": _zipf_slope(tokens),
            "matches": score["matches"],
            "total": score["total"],
        })

    output = {
        "observed_hapax": 0.701,
        "observed_zipf": -0.901,
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, default=str))
    logger.info("wrote %s", OUTPUT)

    # Which mechanism gets closest to (0.70, -0.90)?
    def distance(r):
        return ((r["hapax"] - 0.701) ** 2 + (r["zipf"] - (-0.901)) ** 2) ** 0.5
    best = min(results, key=distance)
    print(json.dumps({"best_by_hapax_zipf_distance": best}, indent=2))
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()