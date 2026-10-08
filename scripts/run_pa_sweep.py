"""Sweep preferential-attachment parameters, score against observed signature."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.layer3_mechanisms.data_loader import load_data_bundle
from src.layer3_mechanisms.preferential_attachment import (
    PAParams,
    generate_preferential_corpus,
)
from src.layer3_mechanisms.score import load_tolerance_intervals, score_corpus
from src.shared.null_models import within_token_shuffle

logger = logging.getLogger(__name__)
OUTPUT = Path("data/processed/mechanism_results_pa.json")


def main() -> dict:
    bundle = load_data_bundle()
    intervals = load_tolerance_intervals()

    grid = [
        {"novelty_prob": 0.05, "mutation_strength": 0.3},
        {"novelty_prob": 0.10, "mutation_strength": 0.3},
        {"novelty_prob": 0.10, "mutation_strength": 0.5},
        {"novelty_prob": 0.20, "mutation_strength": 0.3},
        {"novelty_prob": 0.20, "mutation_strength": 0.5},
        {"novelty_prob": 0.30, "mutation_strength": 0.5},
    ]

    results = []
    for cfg in grid:
        params = PAParams(**cfg)
        logger.info("config: %s", cfg)
        generated = generate_preferential_corpus(bundle.tokenized, bundle.token_count, params, rng_seed=42)
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
        "n_configs": result["n_configs"],
        "null_baseline": result["null_baseline"],
        "top_3_configs": [
            {"params": r["params"], "matches": r["matches"], "total": r["total"]}
            for r in result["configs"][:3]
        ],
    }, indent=2))