"""Compare Voynich compression profile against natural-language controls.

The controls are short samples repeated to match length. This is a coarse
test; a rigorous version would use matched-length corpora. Documented here
so the limitation is visible.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.layer3_mechanisms.compression import profile_corpus
from src.layer3_mechanisms.data_loader import load_data_bundle

logger = logging.getLogger(__name__)
OUTPUT = Path("data/processed/compression_profile.json")

CONTROLS = {
    "latin_sample": (
        "Gallia est omnis divisa in partes tres quarum unam incolunt Belgae "
        "aliam Aquitani tertiam qui ipsorum lingua Celtae nostra Galli appellantur "
        "Hi omnes lingua institutis legibus inter se differunt"
    ),
    "english_sample": (
        "It was the best of times it was the worst of times it was the age "
        "of wisdom it was the age of foolishness it was the epoch of belief "
        "it was the epoch of incredulity"
    ),
    "germanic_sample": (
        "der die das ein eine einer eines dem den die der des "
        "und oder aber wenn weil dass obwohl trotzdem dennoch"
    ),
}


def _control_tokenized(text: str, target_glyphs: int) -> list[list[str]]:
    chars = list(text.replace(" ", ""))
    if not chars:
        return []
    repeated: list[str] = []
    while len(repeated) < target_glyphs:
        repeated.extend(chars)
    repeated = repeated[:target_glyphs]
    tokens = []
    for i in range(0, len(repeated), 8):
        chunk = repeated[i:i + 8]
        if chunk:
            tokens.append(chunk)
    return tokens


def main() -> dict:
    bundle = load_data_bundle()
    target = sum(len(t) for t in bundle.tokenized)
    logger.info("target glyphs: %d", target)

    profiles = {"voynich": profile_corpus("voynich", bundle.tokenized).__dict__}
    for name, text in CONTROLS.items():
        tokens = _control_tokenized(text, target)
        profiles[name] = profile_corpus(name, tokens).__dict__

    output = {
        "note": (
            "Controls are short samples repeated to match length. This is a "
            "coarse comparison. A rigorous version would use matched-length "
            "natural language corpora."
        ),
        "profiles": profiles,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2))
    logger.info("wrote %s", OUTPUT)
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = main()
    print(json.dumps(result["profiles"], indent=2))