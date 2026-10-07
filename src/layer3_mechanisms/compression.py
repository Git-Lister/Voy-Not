"""Compression-based mechanisms for the Voynich Manuscript.

Implements LZ77 (token and glyph level) and Huffman encoding, a
compression-profile test comparing the Voynich to natural-language
controls, and a dictionary-based generative model.
"""
from __future__ import annotations

import collections
import heapq
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------- LZ77 ----------

@dataclass
class LZ77Triple:
    offset: int
    length: int
    next_symbol: str | None


def lz77_encode(
    sequence: Sequence[str],
    window_size: int = 1024,
    min_match: int = 3,
) -> list[LZ77Triple]:
    """Standard LZ77 encoding. Returns list of (offset, length, next_symbol)
    triples. offset=0, length=0 => literal (next_symbol carries the value)."""
    triples: list[LZ77Triple] = []
    i = 0
    n = len(sequence)
    while i < n:
        window_start = max(0, i - window_size)
        best_offset = 0
        best_length = 0
        for j in range(window_start, i):
            length = 0
            while (
                i + length < n
                and j + length < i
                and sequence[j + length] == sequence[i + length]
                and length < window_size
            ):
                length += 1
            if length > best_length and length >= min_match:
                best_length = length
                best_offset = i - j
        if best_length >= min_match:
            triples.append(LZ77Triple(offset=best_offset, length=best_length,
                                       next_symbol=None))
            i += best_length
        else:
            triples.append(LZ77Triple(offset=0, length=0,
                                       next_symbol=sequence[i]))
            i += 1
    return triples


def lz77_decode(triples: list[LZ77Triple]) -> list[str]:
    output: list[str] = []
    for t in triples:
        if t.length == 0:
            if t.next_symbol is not None:
                output.append(t.next_symbol)
        else:
            start = len(output) - t.offset
            for k in range(t.length):
                output.append(output[start + k])
    return output


def lz77_ratio(sequence: Sequence[str], window_size: int = 1024) -> float:
    """Encoded size / raw size in symbol-count units. Encoded size counts
    each triple as 3 symbols (offset, length, next)."""
    triples = lz77_encode(sequence, window_size=window_size)
    encoded_size = 3 * len(triples)
    raw_size = len(sequence)
    return encoded_size / raw_size if raw_size else 0.0


# ---------- Huffman ----------

def _huffman_tree(frequencies: dict[str, int]) -> list:
    heap: list[list] = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return heap[0]


def huffman_codes(sequence: Sequence[str]) -> dict[str, str]:
    freq = collections.Counter(sequence)
    tree = _huffman_tree(dict(freq))
    return {symbol: code for symbol, code in tree[1:]}


def huffman_ratio(sequence: Sequence[str]) -> float:
    if not sequence:
        return 0.0
    codes = huffman_codes(sequence)
    total_bits = sum(len(codes[g]) for g in sequence)
    raw_bits = len(sequence) * 8
    return total_bits / raw_bits


# ---------- Compression profile ----------

@dataclass
class CompressionProfile:
    name: str
    glyph_lz77_w64: float
    glyph_lz77_w256: float
    glyph_lz77_w1024: float
    glyph_huffman: float
    token_lz77_w256: float


def profile_corpus(name: str, tokenized: list[list[str]]) -> CompressionProfile:
    """Flatten to glyph sequence for glyph-level ratios; use tokens for
    token-level LZ77."""
    glyph_stream = [g for token in tokenized for g in token]
    token_stream = ["".join(t) for t in tokenized if t]
    if len(glyph_stream) < 100:
        return CompressionProfile(name, 0, 0, 0, 0, 0)
    return CompressionProfile(
        name=name,
        glyph_lz77_w64=lz77_ratio(glyph_stream, window_size=64),
        glyph_lz77_w256=lz77_ratio(glyph_stream, window_size=256),
        glyph_lz77_w1024=lz77_ratio(glyph_stream, window_size=1024),
        glyph_huffman=huffman_ratio(glyph_stream),
        token_lz77_w256=lz77_ratio(token_stream, window_size=256),
    )


# ---------- Dictionary-based generative model ----------

@dataclass
class DictParams:
    reference_prob: float = 0.6
    mutate_prob: float = 0.2
    literal_prob: float = 0.2
    max_reference_distance: int = 500

    def validate(self) -> None:
        total = self.reference_prob + self.mutate_prob + self.literal_prob
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"probabilities must sum to 1.0, got {total}")


class DictionaryGenerator:
    """LZ77-inspired: at each step, either reference+copy a previous token,
    mutate a referenced token, or emit a literal (fresh) token."""

    def __init__(
        self,
        seed_corpus: list[list[str]],
        params: DictParams,
        rng_seed: int = 42,
    ):
        params.validate()
        self.params = params
        self.rng = random.Random(rng_seed)
        self.corpus: list[list[str]] = [list(t) for t in seed_corpus]
        self._glyph_vocab = sorted({g for t in seed_corpus for g in t})

    def _reference(self) -> list[str]:
        upper = max(0, len(self.corpus) - self.params.max_reference_distance)
        idx = self.rng.randrange(upper, len(self.corpus))
        return list(self.corpus[idx])

    def _mutate(self, token: list[str]) -> list[str]:
        if not token:
            return [self.rng.choice(self._glyph_vocab)]
        result = list(token)
        idx = self.rng.randrange(len(result))
        choices = [g for g in self._glyph_vocab if g != result[idx]]
        if not choices:
            return result
        result[idx] = self.rng.choice(choices)
        return result

    def _literal(self) -> list[str]:
        length = self.rng.randint(1, 8)
        return [self.rng.choice(self._glyph_vocab) for _ in range(length)]

    def _next(self) -> list[str]:
        r = self.rng.random()
        p = self.params
        if r < p.reference_prob:
            return self._reference()
        r -= p.reference_prob
        if r < p.mutate_prob:
            return self._mutate(self._reference())
        return self._literal()

    def generate(self, n_tokens: int) -> list[list[str]]:
        while len(self.corpus) < n_tokens:
            self.corpus.append(self._next())
        return self.corpus[:n_tokens]


def generate_compression_corpus(
    seed_corpus: list[list[str]],
    n_tokens: int,
    params: DictParams,
    rng_seed: int = 42,
) -> list[list[str]]:
    gen = DictionaryGenerator(seed_corpus, params, rng_seed=rng_seed)
    return gen.generate(n_tokens)


if __name__ == "__main__":
    import json

    from src.layer3_mechanisms.data_loader import load_data_bundle
    bundle = load_data_bundle()
    profile = profile_corpus("voynich", bundle.tokenized)
    print(json.dumps(profile.__dict__, indent=2))