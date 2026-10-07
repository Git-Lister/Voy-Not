"""Quire-level bootstrap tolerance intervals for the joint signature.

For each bootstrap iteration, resample quires with replacement, rebuild the
tokenized corpus, and recompute S1-S15. Record the standard deviation of each
property across iterations. Tolerance = observed_value +/- 2 * SD.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict
from pathlib import Path

from src.layer3_mechanisms.data_loader import load_data_bundle
from src.layer3_mechanisms.evaluate import compute_joint_signature

logger = logging.getLogger(__name__)

N_BOOTSTRAP = 15
RNG_SEED = 42
OUTPUT_PATH = Path("data/processed/tolerance_intervals.json")


def _resample_quires(bundle, rng: random.Random):
    """Return a new DataBundle with quires resampled with replacement."""
    quire_labels = list(bundle.quire_token_lists.keys())
    sampled = [rng.choice(quire_labels) for _ in quire_labels]

    new_tokenized: list[list[str]] = []
    new_quire_labels: list[str] = []
    new_quire_lists: dict[str, list[list[str]]] = {}

    for q in sampled:
        lines = bundle.quire_token_lists[q]
        for line in lines:
            new_tokenized.append(line)
            new_quire_labels.append(q)
        new_quire_lists.setdefault(q, []).extend(lines)

    new_boundaries = [0]
    for t in new_tokenized:
        new_boundaries.append(new_boundaries[-1] + len(t))

    from src.layer3_mechanisms.data_loader import DataBundle
    return DataBundle(
        tokenized=new_tokenized,
        line_boundaries=new_boundaries,
        quire_labels=new_quire_labels,
        quire_token_lists=new_quire_lists,
        mismatch_df=bundle.mismatch_df,
        page_records=bundle.page_records,
        token_count=sum(len(t) for t in new_tokenized),
    )


def compute_tolerance_intervals() -> dict:
    bundle = load_data_bundle()
    observed = compute_joint_signature(bundle, compute_fractal_fresh=True)

    rng = random.Random(RNG_SEED)
    sigs: list = []

    for i in range(N_BOOTSTRAP):
        resampled = _resample_quires(bundle, rng)
        try:
            sig = compute_joint_signature(resampled, compute_fractal_fresh=True)
            sigs.append(sig)
        except Exception as e:  # noqa: BLE001
            logger.warning("bootstrap %d failed: %s", i, e)

    fields = list(asdict(observed).keys())
    intervals: dict[str, object] = {
        "n_bootstrap": len(sigs),
        "seed": RNG_SEED,
        "properties": {},
    }
    for field in fields:
        values = [getattr(s, field) for s in sigs if getattr(s, field) is not None]
        obs = getattr(observed, field)
        if not values or obs is None:
            intervals["properties"][field] = {  # type: ignore[index]
                "observed": obs,
                "sd": None,
                "lower": None,
                "upper": None,
            }
            continue
        mean = sum(values) / len(values)
        sd = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        intervals["properties"][field] = {  # type: ignore[index]
            "observed": obs,
            "bootstrap_mean": mean,
            "sd": sd,
            "lower": obs - 2 * sd,
            "upper": obs + 2 * sd,
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(intervals, indent=2, default=str))
    logger.info("wrote %s", OUTPUT_PATH)
    return intervals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = compute_tolerance_intervals()
    print(json.dumps(result["properties"], indent=2, default=str))