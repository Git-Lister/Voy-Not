"""Exhaustive PA parameter sweep with frequency-spectrum instrumentation.

For each config:
  - Sweep novelty_prob on a log scale from 0.001 to 0.99
  - Sweep mutation_strength over {0.1, 0.3, 0.5, 0.7}
  - Run 5 seeds per config, report mean and std of hapax ratio,
    Zipf slope, and score against observed signature.

Also instrument the generator:
  - How many novel types are introduced?
  - Of those, what fraction remain hapax at the end?
  - What is the mean frequency of a novel type after 100, 1000, 10000 steps?
"""
from __future__ import annotations
import json
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.layer3_mechanisms.data_loader import load_data_bundle
from src.layer3_mechanisms.preferential_attachment import (
    PAParams, PreferentialAttachmentGenerator,
)
from src.layer3_mechanisms.evaluate import compute_joint_signature
from src.layer3_mechanisms.score import load_tolerance_intervals, score_corpus

logger = logging.getLogger(__name__)
OUTPUT = Path("data/processed/diag_pa_exhaustive.json")


NOVELTY_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]
MUTATION_GRID = [0.1, 0.3, 0.5, 0.7]
SEEDS = [42, 101, 202, 303, 404]
N_TOKENS = 33728


def _zipf_slope(tokenized):
    from collections import Counter
    import math
    counts = Counter(tuple(t) for t in tokenized)
    freqs = sorted(counts.values(), reverse=True)
    if len(freqs) < 10:
        return None
    ranks = list(range(1, len(freqs) + 1))
    log_r = np.log(ranks)
    log_f = np.log(freqs)
    slope = np.polyfit(log_r, log_f, 1)[0]
    return float(slope)


def _hapax_ratio(tokenized):
    counts = Counter(tuple(t) for t in tokenized)
    if not counts:
        return 0.0
    hapax = sum(1 for v in counts.values() if v == 1)
    return hapax / len(counts)


def _instrument_generator(gen, n_tokens):
    """Track novel type survival. Returns dict with survival statistics."""
    # Track when each type was first introduced
    first_seen: dict[tuple, int] = {}
    for i, t in enumerate(gen.tokens[:n_tokens]):
        key = tuple(t)
        if key not in first_seen:
            first_seen[key] = i

    final_counts = Counter(tuple(t) for t in gen.tokens[:n_tokens])

    # Novel types = those first seen after index 0 (i.e., not in the initial seed)
    # We approximate: types first seen after index 1000 are "novel"
    novel_keys = [k for k, idx in first_seen.items() if idx > 1000]

    if not novel_keys:
        return {"n_novel": 0, "novel_hapax_ratio": None,
                "novel_mean_freq": None}

    novel_hapax = sum(1 for k in novel_keys if final_counts[k] == 1)
    novel_freqs = [final_counts[k] for k in novel_keys]

    return {
        "n_novel": len(novel_keys),
        "novel_hapax_ratio": novel_hapax / len(novel_keys),
        "novel_mean_freq": float(np.mean(novel_freqs)),
        "novel_median_freq": float(np.median(novel_freqs)),
    }


def main():
    bundle = load_data_bundle()
    intervals = load_tolerance_intervals()
    seed_corpus = bundle.tokenized

    results = []
    for novelty in NOVELTY_GRID:
        for mutation in MUTATION_GRID:
            per_seed = []
            for seed in SEEDS:
                params = PAParams(novelty_prob=novelty,
                                  mutation_strength=mutation, seed=seed)
                gen = PreferentialAttachmentGenerator(
                    seed_corpus, params, rng_seed=seed,
                )
                tokens = gen.generate(N_TOKENS)
                hapax = _hapax_ratio(tokens)
                zipf = _zipf_slope(tokens)
                instr = _instrument_generator(gen, N_TOKENS)
                score = score_corpus(tokens, intervals)
                per_seed.append({
                    "seed": seed,
                    "hapax": hapax,
                    "zipf": zipf,
                    "matches": score["matches"],
                    "total": score["total"],
                    **instr,
                })

            def _mean(key):
                vals = [p[key] for p in per_seed if p.get(key) is not None]
                return float(np.mean(vals)) if vals else None

            def _std(key):
                vals = [p[key] for p in per_seed if p.get(key) is not None]
                return float(np.std(vals)) if vals else None

            results.append({
                "novelty": novelty,
                "mutation": mutation,
                "hapax_mean": _mean("hapax"),
                "hapax_std": _std("hapax"),
                "zipf_mean": _mean("zipf"),
                "zipf_std": _std("zipf"),
                "matches_mean": _mean("matches"),
                "matches_std": _std("matches"),
                "n_novel_mean": _mean("n_novel"),
                "novel_hapax_mean": _mean("novel_hapax_ratio"),
                "novel_mean_freq": _mean("novel_mean_freq"),
                "per_seed": per_seed,
            })

    output = {
        "n_configs": len(results),
        "seeds": SEEDS,
        "observed_hapax": 0.701,
        "observed_zipf": -0.901,
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, default=str))
    logger.info("wrote %s", OUTPUT)

    # Print the best config by matches and by hapax proximity
    best_matches = max(results, key=lambda r: r["matches_mean"] or 0)
    best_hapax = max(results, key=lambda r: -(abs((r["hapax_mean"] or 0) - 0.701)))
    print(json.dumps({
        "best_by_matches": {k: best_matches[k] for k in
                            ["novelty", "mutation", "matches_mean", "hapax_mean"]},
        "best_by_hapax": {k: best_hapax[k] for k in
                          ["novelty", "mutation", "matches_mean", "hapax_mean"]},
    }, indent=2))
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()