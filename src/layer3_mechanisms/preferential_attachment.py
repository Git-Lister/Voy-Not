"""Preferential-attachment generator for the Voynich Manuscript.

Implements a Yule-Simon-style process: each new token either copies a
previously-used token with probability proportional to its current
frequency (preferential attachment), or introduces a novel token with
probability p_novel. Novel tokens are synthesized by glyph-level mutation
of an existing token, so they remain within the observed glyph vocabulary.
"""
from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PAParams:
    novelty_prob: float = 0.10         # per-token probability of introducing a new token
    mutation_strength: float = 0.5     # mutation magnitude when creating novel tokens
    seed: int = 42

    def validate(self) -> None:
        if not 0.0 < self.novelty_prob < 1.0:
            raise ValueError("novelty_prob must be in (0, 1)")
        if not 0.0 < self.mutation_strength < 1.0:
            raise ValueError("mutation_strength must be in (0, 1)")


class PreferentialAttachmentGenerator:
    """Yule-Simon process over tokens.

    State:
      self.tokens: list of emitted tokens
      self.freq: Counter mapping token_id (as tuple) to frequency
      self.vocab: list of unique token tuples seen so far

    Emission:
      With probability novelty_prob: emit a NEW token (mutant of a
      random existing token).
      Otherwise: emit an existing token chosen with probability
      proportional to its current frequency (preferential attachment).
    """

    def __init__(
        self,
        seed_corpus: list[list[str]],
        params: PAParams,
        rng_seed: int = 42,
    ):
        params.validate()
        self.params = params
        self.rng = random.Random(rng_seed)
        self.glyph_vocab = sorted({g for t in seed_corpus for g in t})
        # Seed with unique tokens from the seed corpus
        self.tokens: list[list[str]] = []
        self.freq: dict[tuple[str, ...], int] = {}
        for t in seed_corpus:
            key = tuple(t)
            self.freq[key] = self.freq.get(key, 0) + 1
            self.tokens.append(list(t))

    def _preferential_sample(self) -> list[str]:
        """Sample an existing token with probability ∝ its frequency."""
        keys = list(self.freq.keys())
        weights = [self.freq[k] for k in keys]
        chosen = self.rng.choices(keys, weights=weights, k=1)[0]
        return list(chosen)

    def _mutate_token(self, token: list[str]) -> list[str]:
        """Create a novel token by glyph mutation of an existing one."""
        if not token:
            return [self.rng.choice(self.glyph_vocab)]
        result = list(token)
        n_mutations = max(1, round(self.params.mutation_strength * len(token)))
        n_mutations = min(n_mutations, len(result))
        positions = self.rng.sample(range(len(result)), n_mutations)
        for pos in positions:
            choices = [g for g in self.glyph_vocab if g != result[pos]]
            if choices:
                result[pos] = self.rng.choice(choices)
        return result

    def _novel_token(self) -> list[str]:
        parent = self._preferential_sample()
        return self._mutate_token(parent)

    def generate(self, n_tokens: int) -> list[list[str]]:
        while len(self.tokens) < n_tokens:
            if self.rng.random() < self.params.novelty_prob:
                new_tok = self._novel_token()
            else:
                new_tok = self._preferential_sample()
            key = tuple(new_tok)
            self.freq[key] = self.freq.get(key, 0) + 1
            self.tokens.append(new_tok)
        return self.tokens[:n_tokens]


def generate_preferential_corpus(
    seed_corpus: list[list[str]],
    n_tokens: int,
    params: PAParams,
    rng_seed: int = 42,
) -> list[list[str]]:
    gen = PreferentialAttachmentGenerator(seed_corpus, params, rng_seed=rng_seed)
    return gen.generate(n_tokens)


if __name__ == "__main__":
    import json
    from collections import Counter

    from src.layer3_mechanisms.data_loader import load_data_bundle
    bundle = load_data_bundle()
    params = PAParams()
    out = generate_preferential_corpus(bundle.tokenized, bundle.token_count, params)
    types = Counter(tuple(t) for t in out)
    hapax = sum(1 for v in types.values() if v == 1) / len(types)
    print(json.dumps({
        "n_tokens": len(out),
        "n_types": len(types),
        "hapax_ratio": hapax,
    }, indent=2))