"""Leave-one-quire-out diagnostic.

For each quire in the corpus, remove it, recompute the joint signature
on the remaining corpus, and compare to the full-corpus signature. If a
removed quire causes a large change in a property, that quire carries
distinct structure for that property.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.layer3_mechanisms.data_loader import DataBundle, load_data_bundle
from src.layer3_mechanisms.evaluate import compute_joint_signature

logger = logging.getLogger(__name__)
OUTPUT = Path("data/processed/homogeneity_diagnostic.json")

# Properties to ignore in stability analysis (not scoreable or constant)
IGNORE = {"s9_quire_stability_mean", "s10_cross_transcription_agreement",
          "s12_page_template_transition_ll"}


def _bundle_without_quire(bundle, drop_quire: str):
    """Return a new DataBundle with the specified quire removed."""
    kept_tokenized: list[list[str]] = []
    kept_labels: list[str] = []
    kept_lists: dict[str, list[list[str]]] = {}
    for label, lines in bundle.quire_token_lists.items():
        if label == drop_quire:
            continue
        kept_lists[label] = lines
        for line in lines:
            kept_tokenized.append(line)
            kept_labels.append(label)
    boundaries = [0]
    for t in kept_tokenized:
        boundaries.append(boundaries[-1] + len(t))
    return DataBundle(
        tokenized=kept_tokenized,
        line_boundaries=boundaries,
        quire_labels=kept_labels,
        quire_token_lists=kept_lists,
        mismatch_df=bundle.mismatch_df,
        page_records=bundle.page_records,
        token_count=sum(len(t) for t in kept_tokenized),
    )


def main() -> dict:

    bundle = load_data_bundle()
    quires = sorted(bundle.quire_token_lists.keys())
    logger.info("quires: %s", quires)

    full = asdict(compute_joint_signature(bundle))

    leave_one_out: dict[str, dict] = {}
    for q in quires:
        logger.info("dropping quire %s", q)
        reduced = _bundle_without_quire(bundle, q)
        sig = asdict(compute_joint_signature(reduced))
        leave_one_out[q] = sig

    # Load tolerance SDs
    tol_path = Path("data/processed/tolerance_intervals.json")
    tol = json.loads(tol_path.read_text())["properties"] if tol_path.exists() else {}

    # For each property, compute how much leave-one-out shifts
    summary: dict[str, dict] = {}
    for field, full_val in full.items():
        if field in IGNORE:
            continue
        loo_vals = [leave_one_out[q][field] for q in quires
                    if leave_one_out[q].get(field) is not None]
        if not loo_vals or full_val is None:
            summary[field] = {"full": full_val, "note": "not scoreable"}
            continue
        loo_min = min(loo_vals)
        loo_max = max(loo_vals)
        max_abs_delta = max(abs(v - full_val) for v in loo_vals)
        sd = (tol.get(field, {}) or {}).get("sd")
        delta_in_sd = (max_abs_delta / sd) if sd and sd > 0 else None

        # Which quire caused the biggest shift?
        biggest = max(quires, key=lambda q: abs(
            (leave_one_out[q].get(field) or full_val) - full_val))

        summary[field] = {
            "full": full_val,
            "loo_min": loo_min,
            "loo_max": loo_max,
            "max_abs_delta": max_abs_delta,
            "sd": sd,
            "delta_in_sd": delta_in_sd,
            "biggest_effect_quire": biggest,
        }

    # Sort by homogeneity (lower delta_in_sd = more homogeneous)
    ranked = sorted(
        [(f, s) for f, s in summary.items() if s.get("delta_in_sd") is not None],
        key=lambda x: x[1]["delta_in_sd"],
    )

    output = {
        "n_quires": len(quires),
        "quires": quires,
        "full_signature": full,
        "leave_one_out": leave_one_out,
        "per_property": summary,
        "ranked_by_homogeneity": [
            {"property": f, "delta_in_sd": s["delta_in_sd"],
             "biggest_effect_quire": s["biggest_effect_quire"]}
            for f, s in ranked
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, default=str))
    logger.info("wrote %s", OUTPUT)
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = main()
    print(json.dumps({
        "n_quires": result["n_quires"],
        "ranked_by_homogeneity": result["ranked_by_homogeneity"],
    }, indent=2))