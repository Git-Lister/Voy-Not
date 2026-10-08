"""Test whether hapax tokens have different length statistics than non-hapax.
If they do, this is a signature of a particular generative mechanism.
"""
from __future__ import annotations
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from src.layer3_mechanisms.data_loader import load_data_bundle

logger = logging.getLogger(__name__)
OUTPUT = Path("data/processed/diag_hapax_length.json")


def _analyse(tokenized, name):
    counts = Counter(tuple(t) for t in tokenized)
    hapax_keys = {k for k, v in counts.items() if v == 1}
    non_hapax_keys = {k for k, v in counts.items() if v > 1}

    hapax_lengths = [len(k) for k in hapax_keys]
    non_hapax_lengths = [len(k) for k in non_hapax_keys]

    return {
        "name": name,
        "n_hapax_types": len(hapax_keys),
        "n_non_hapax_types": len(non_hapax_keys),
        "hapax_length_mean": float(np.mean(hapax_lengths)) if hapax_lengths else None,
        "hapax_length_std": float(np.std(hapax_lengths)) if hapax_lengths else None,
        "non_hapax_length_mean": float(np.mean(non_hapax_lengths)) if non_hapax_lengths else None,
        "non_hapax_length_std": float(np.std(non_hapax_lengths)) if non_hapax_lengths else None,
        "length_difference": (float(np.mean(hapax_lengths)) - float(np.mean(non_hapax_lengths)))
                              if hapax_lengths and non_hapax_lengths else None,
    }


def main():
    bundle = load_data_bundle()
    observed = _analyse(bundle.tokenized, "observed")

    # Also compute for generated corpora from the best configs
    from src.layer3_mechanisms.self_citation import (
        SelfCitationParams, generate_corpus,
    )
    pa_best = SelfCitationParams(copy_prob=0.85, mutation_prob=0.09,
                                 insert_prob=0.045, delete_prob=0.015,
                                 grammar_strength=0.0)
    generated_selfcite = generate_corpus(bundle.tokenized, bundle.token_count, pa_best)
    selfcite = _analyse(generated_selfcite, "self_citation_best")

    output = {
        "observed": observed,
        "self_citation_best": selfcite,
        "interpretation": (
            "If length_difference is significantly non-zero for observed but "
            "zero for generators, the mechanism must be producing hapax with "
            "the wrong length profile."
        ),
    }
    OUTPUT.write_text(json.dumps(output, indent=2, default=str))
    logger.info("wrote %s", OUTPUT)
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()