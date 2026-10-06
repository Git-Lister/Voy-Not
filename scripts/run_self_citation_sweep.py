"""Run the self-citation generator across a parameter grid, score each
configuration against the observed signature, and compare against a null
baseline (within-token shuffle)."""
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

from src.layer3_mechanisms.data_loader import load_data_bundle
from src.layer3_mechanisms.score import load_tolerance_intervals, score_corpus
from src.layer3_mechanisms.self_citation import (
    SelfCitationParams,
    generate_corpus,
)

logger = logging.getLogger(__name__)
OUTPUT = Path("data/processed/mechanism_results_self_citation.json")


def main() -> dict:
    bundle = load_data_bundle()
    intervals = load_tolerance_intervals()
    n_tokens = bundle.token_count
    seed = bundle.tokenized  # Use full observed corpus as seed

    grid = list(itertools.product(
        [0.7, 0.85, 0.95],   # copy_prob — higher values are closer to Timm-Schinner
        [0.0, 1.0],          # grammar_strength
    ))

    results = []
    for copy_prob, grammar_strength in grid:
        # Balance remaining probability across mutation/insert/delete
        remainder = 1.0 - copy_prob
        params = SelfCitationParams(
            copy_prob=copy_prob,
            mutation_prob=remainder * 0.6,
            insert_prob=remainder * 0.3,
            delete_prob=remainder * 0.1,
            source_weighting="frequency",
            grammar_strength=grammar_strength,
        )
        logger.info("config: copy=%.2f grammar=%.2f", copy_prob, grammar_strength)
        generated = generate_corpus(seed, n_tokens, params, rng_seed=42)
        score = score_corpus(generated, intervals)
        results.append({
            "params": params.__dict__,
            "matches": score["matches"],
            "total": score["total"],
            "match_fraction": score["match_fraction"],
            "properties": score["properties"],
        })

    # Null baseline
    from src.shared.null_models import within_token_shuffle
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
            {"params": r["params"], "matches": r["matches"]}
            for r in result["configs"][:3]
        ],
    }, indent=2, default=str))