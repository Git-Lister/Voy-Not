"""Three alternative vocabulary-dynamics mechanisms.

PureNovelty: never reuse a token. Hapax = 1.0 by construction. Zipf undefined.

FixedVocab: sample tokens from a fixed pool with fixed frequencies.
Hapax ratio is determined by the pool and sample size.

TwoRegime: a "core" vocabulary sampled with high probability, plus a
"novelty" regime where each token is used exactly once. The core fraction
f_core controls the mix.
"""
from __future__ import annotations
import collections
import random
from dataclasses import dataclass


@dataclass
class PureNoveltyParams:
    mutation_strength: float = 0.5
    seed: int = 42


def generate_pure_novelty(seed_corpus, n_tokens, params):
    rng = random.Random(params.seed)
    glyph_vocab = sorted({g for t in seed_corpus for g in t})
    base_tokens = [list(t) for t in seed_corpus]

    def mutate(token):
        result = list(token)
        n_mut = max(1, int(round(params.mutation_strength * len(result))))
        n_mut = min(n_mut, len(result))
        for pos in rng.sample(range(len(result)), n_mut):
            choices = [g for g in glyph_vocab if g != result[pos]]
            if choices:
                result[pos] = rng.choice(choices)
        return result

    out = []
    for _ in range(n_tokens):
        parent = rng.choice(base_tokens)
        out.append(mutate(parent))
    return out


@dataclass
class FixedVocabParams:
    vocab_size: int = 500
    seed: int = 42


def generate_fixed_vocab(seed_corpus, n_tokens, params):
    rng = random.Random(params.seed)
    # Build a pool of vocab_size types by sampling from the seed corpus
    unique = list({tuple(t) for t in seed_corpus})
    if len(unique) < params.vocab_size:
        # Pad by mutating existing types
        glyph_vocab = sorted({g for t in seed_corpus for g in t})
        while len(unique) < params.vocab_size:
            parent = list(rng.choice(unique))
            pos = rng.randrange(len(parent))
            choices = [g for g in glyph_vocab if g != parent[pos]]
            if choices:
                parent[pos] = rng.choice(choices)
                unique.append(tuple(parent))
    pool = [list(t) for t in unique[:params.vocab_size]]
    # Zipf-like frequencies
    weights = [1.0 / (i + 1) for i in range(len(pool))]
    return [list(rng.choices(pool, weights=weights, k=1)[0])
            for _ in range(n_tokens)]


@dataclass
class TwoRegimeParams:
    core_fraction: float = 0.5   # fraction of tokens from core
    novelty_fraction: float = 0.5
    mutation_strength: float = 0.5
    seed: int = 42


def generate_two_regime(seed_corpus, n_tokens, params):
    rng = random.Random(params.seed)
    glyph_vocab = sorted({g for t in seed_corpus for g in t})
    unique = list({tuple(t) for t in seed_corpus})
    pool = [list(t) for t in unique]
    weights = [1.0 / (i + 1) for i in range(len(pool))]

    def mutate(token):
        result = list(token)
        n_mut = max(1, int(round(params.mutation_strength * len(result))))
        n_mut = min(n_mut, len(result))
        for pos in rng.sample(range(len(result)), n_mut):
            choices = [g for g in glyph_vocab if g != result[pos]]
            if choices:
                result[pos] = rng.choice(choices)
        return result

    out = []
    for _ in range(n_tokens):
        if rng.random() < params.core_fraction:
            out.append(list(rng.choices(pool, weights=weights, k=1)[0]))
        else:
            out.append(mutate(rng.choice(pool)))
    return out