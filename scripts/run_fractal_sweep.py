"""Sweep fractal generator parameters, score against observed signature."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.layer3_mechanisms.data_loader import load_data_bundle
from src.layer3_mechanisms.fractal_generator import (
    FractalParams,
    generate_fractal_corpus,
)
from src.layer3_mechanisms.score import load_tolerance_intervals, score_corpus
from src.shared.null_models import within_token_shuffle

logger = logging.getLogger(__name__)
OUTPUT = Path("data/processed/mechanism_results_fractal.json")


def main() -> dict:
    bundle = load_data_bundle()
    intervals = load_tolerance_intervals()
    vocab = sorted({g for t in bundle.tokenized for g in t})

    grid = [
        {"hurst": 0.55, "local_mutation_rate": 0.03},
        {"hurst": 0.61, "local_mutation_rate": 0.03},
        {"hurst": 0.65, "local_mutation_rate": 0.03},
        {"hurst": 0.61, "local_mutation_rate": 0.05},
        {"hurst": 0.61, "local_mutation_rate": 0.10},
        {"hurst": 0.70, "local_mutation_rate": 0.05},
    ]

    results = []
    for cfg in grid:
        params = FractalParams(**cfg)
        logger.info("config: %s", cfg)
        generated = generate_fractal_corpus(vocab, params)
        score = score_corpus(generated, intervals)
        results.append({
            "params": params.__dict__,
            "matches": score["matches"],
            "total": score["total"],
            "match_fraction": score["match_fraction"],
            "properties": score["properties"],
        })

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