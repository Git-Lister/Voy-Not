"""Score a generated corpus against the observed signature using tolerance intervals."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.layer3_mechanisms.data_loader import DataBundle
from src.layer3_mechanisms.evaluate import (
    compute_joint_signature,
)

logger = logging.getLogger(__name__)

# Fields that are not scoreable for generated corpora
EXCLUDED_FIELDS = {
    "s9_quire_stability_mean",
    "s10_cross_transcription_agreement",
    "s12_page_template_transition_ll",
}


def load_tolerance_intervals(path: str = "data/processed/tolerance_intervals.json") -> dict:
    return json.loads(Path(path).read_text())


def _make_bundle_from_tokens(tokenized: list[list[str]]) -> DataBundle:
    boundaries = [0]
    for t in tokenized:
        boundaries.append(boundaries[-1] + len(t))
    return DataBundle(
        tokenized=tokenized,
        line_boundaries=boundaries,
        quire_labels=["generated"] * len(tokenized),
        quire_token_lists={"generated": tokenized},
        mismatch_df=None,
        page_records=None,
        token_count=sum(len(t) for t in tokenized),
    )


def score_corpus(tokenized: list[list[str]], intervals: dict) -> dict:
    """Compute S1-S15 on the generated corpus, compare against tolerance intervals.
    Returns {'matches': N, 'total': M, 'properties': {name: {generated, observed, within}}}.
    Properties where the generated value is None are excluded from both numerator and denominator."""
    bundle = _make_bundle_from_tokens(tokenized)
    sig = compute_joint_signature(bundle, compute_fractal_fresh=True)

    props = intervals["properties"]
    matches = 0
    total = 0
    detail = {}
    for field, spec in props.items():
        if field in EXCLUDED_FIELDS:
            detail[field] = {"excluded": True, "reason": "not scoreable for generated corpus"}
            continue
        gen_val = getattr(sig, field)
        if gen_val is None:
            # Not scoreable — exclude from denominator
            detail[field] = {
                "generated": None,
                "observed": spec.get("observed"),
                "excluded": True,
                "reason": "generated value is None",
            }
            continue
        lower, upper = spec.get("lower"), spec.get("upper")
        within = False
        if lower is not None and upper is not None:
            within = lower <= gen_val <= upper
        detail[field] = {
            "generated": gen_val,
            "observed": spec.get("observed"),
            "lower": lower,
            "upper": upper,
            "within": within,
        }
        total += 1
        if within:
            matches += 1

    return {
        "matches": matches,
        "total": total,
        "match_fraction": matches / total if total else 0.0,
        "properties": detail,
    }