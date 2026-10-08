"""Collect (hapax, Zipf, score) triples from all previous model outputs
plus the two new diagnostics. Produce a single scatter JSON for analysis."""
from __future__ import annotations
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
OUTPUT = Path("data/processed/diag_hapax_zipf_scatter.json")

SOURCES = [
    ("self_citation", "data/processed/mechanism_results_self_citation.json"),
    ("compression", "data/processed/mechanism_results_compression.json"),
    ("fractal", "data/processed/mechanism_results_fractal.json"),
    ("pa", "data/processed/mechanism_results_pa.json"),
    ("pa_exhaustive", "data/processed/diag_pa_exhaustive.json"),
    ("alt_vocab", "data/processed/diag_alt_vocab.json"),
]


def main():
    points = []
    for name, path in SOURCES:
        p = Path(path)
        if not p.exists():
            logger.warning("missing %s", p)
            continue
        data = json.loads(p.read_text())

        if name in ("self_citation", "compression", "fractal", "pa"):
            for cfg in data.get("configs", []):
                props = cfg.get("properties", {})
                hapax = props.get("s7_hapax_ratio", {}).get("generated")
                zipf = props.get("s6_zipf_slope", {}).get("generated")
                points.append({
                    "source": name,
                    "params": cfg.get("params"),
                    "hapax": hapax,
                    "zipf": zipf,
                    "matches": cfg.get("matches"),
                })
        elif name == "pa_exhaustive":
            for r in data.get("results", []):
                points.append({
                    "source": "pa_exhaustive",
                    "params": {"novelty": r["novelty"], "mutation": r["mutation"]},
                    "hapax": r["hapax_mean"],
                    "zipf": r["zipf_mean"],
                    "matches": r["matches_mean"],
                })
        elif name == "alt_vocab":
            for r in data.get("results", []):
                points.append({
                    "source": "alt_vocab_" + r["mechanism"],
                    "params": r["params"],
                    "hapax": r["hapax"],
                    "zipf": r["zipf"],
                    "matches": r["matches"],
                })

    output = {
        "observed": {"hapax": 0.701, "zipf": -0.901},
        "n_points": len(points),
        "points": points,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, default=str))
    logger.info("wrote %s with %d points", OUTPUT, len(points))
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()