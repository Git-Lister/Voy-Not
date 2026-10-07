"""Sweep the dictionary-based compression generator, score against observed."""
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

from src.layer3_mechanisms.compression import DictParams, generate_compression_corpus
from src.layer3_mechanisms.data_loader import load_data_bundle
from src.layer3_mechanisms.score import load_tolerance_intervals, score_corpus
from src.shared.null_models import within_token_shuffle

logger = logging.getLogger(__name__)
OUTPUT = Path("data/processed/mechanism_results_compression.json")


def main() -> dict:
    bundle = load_data_bundle()
    intervals = load_tolerance_intervals()
    n_tokens = bundle.token_count
    seed = bundle.tokenized

    grid = list(itertools.product(
        [0.4, 0.6, 0.8],      # reference_prob
        [0.1, 0.3],           # mutate_prob
        [250, 1000],          # max_reference_distance
    ))

    results = []
    for ref_p, mut_p, max_ref in grid:
        literal_p = round(1.0 - ref_p - mut_p, 6)
        if literal_p < 0:
            continue
        params = DictParams(
            reference_prob=ref_p, mutate_prob=mut_p,
            literal_prob=literal_p, max_reference_distance=max_ref,
        )
        logger.info("config: ref=%.2f mut=%.2f", ref_p, mut_p)
        generated = generate_compression_corpus(seed, n_tokens, params, rng_seed=42)
        score = score_corpus(generated, intervals)
        results.append({
            "params": params.__dict__,
            "matches": score["matches"],
            "total": score["total"],
            "match_fraction": score["match_fraction"],
            "properties": score["properties"],
        })

    # Null baseline
    null_corpus = within_token_shuffle(bundle.tokenized, seed=42)
    null_score = score_corpus(null_corpus, intervals)

    output = {
        "n_configs": len(results),
        "null_baseline": {
            "matches": null_score["matches"],
            "total": null_score["total"],
            "match_fraction": null_score["match_fraction"],
        },
        "configs": sorted(results, key=lambda r: -r["matches"]),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, default=str))
    logger.info("wrote %s", OUTPUT)
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = main()
    print(json.dumps({
        "null_baseline": result["null_baseline"],
        "top_3_configs": [
            {"params": r["params"], "matches": r["matches"], "total": r["total"]}
            for r in result["configs"][:3]
        ],
    }, indent=2))