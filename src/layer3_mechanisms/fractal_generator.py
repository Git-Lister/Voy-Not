"""Fractal generative model for the Voynich Manuscript.

Approach: synthesize a 1-D signal with a target Hurst exponent using
spectral synthesis (fractional Gaussian noise). Map the signal to glyphs
via rank-quantile encoding, then group into tokens matching the observed
length distribution. Local mutations break exact repetition.

This is a "generative fractal" in the sense that each scale of the
resulting token stream mirrors structure at coarser scales.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FractalParams:
    hurst: float = 0.614             # target H
    n_glyphs: int = 33728            # total glyph count
    token_length_mean: float = 8.28  # from observed
    token_length_std: float = 3.20   # from observed (sqrt(10.22))
    local_mutation_rate: float = 0.05
    seed: int = 42

    def validate(self) -> None:
        if not 0.05 < self.hurst < 0.95:
            raise ValueError("hurst must be in (0.05, 0.95)")
        if self.n_glyphs < 1000:
            raise ValueError("n_glyphs must be >= 1000")
        if not 0.0 <= self.local_mutation_rate <= 1.0:
            raise ValueError("local_mutation_rate must be in [0,1]")


def synthesize_fgn(N: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """Fractional Gaussian noise via spectral synthesis.

    Returns a 1-D array of length N with approximately the given Hurst
    exponent. Uses the fBm spectral density S(f) ~ |f|^(-(2H-1)) and
    inverse FFT."""
    # Frequency grid
    freqs = np.fft.fftfreq(N)
    freqs[0] = 1e-10  # avoid division by zero at DC

    # Power spectrum for fBm increments: |f|^(-(2H-1))
    power = np.abs(freqs) ** (-(2 * H - 1))
    power[0] = 0.0  # kill DC to avoid drift

    # Random phases
    phases = rng.uniform(0, 2 * np.pi, N)
    spectrum = np.sqrt(power) * np.exp(1j * phases)

    # Inverse FFT to get real signal
    signal = np.real(np.fft.ifft(spectrum))
    # Normalize to zero mean, unit std
    signal = signal - signal.mean()
    std = signal.std()
    if std > 0:
        signal = signal / std
    return signal


def rank_quantile_encode(signal: np.ndarray, vocab: list[str]) -> list[str]:
    """Map a real-valued signal to glyphs by rank: the lowest values get the
    least-frequent glyph, the highest get the most-frequent (or vice versa).
    Preserves marginal glyph frequencies approximately if we use quantile bins
    matching observed frequencies.
    """
    n = len(signal)
    # Argsort positions by signal value
    order = np.argsort(signal)
    # Divide sorted order into n_symbols equal quantile bins
    bins = np.linspace(0, n, len(vocab) + 1, dtype=int)
    output = [""] * n
    for i, symbol in enumerate(vocab):
        start, end = bins[i], bins[i + 1]
        for pos in order[start:end]:
            output[pos] = symbol
    return output


def group_into_tokens(
    glyph_stream: list[str],
    length_mean: float,
    length_std: float,
    rng: random.Random,
) -> list[list[str]]:
    """Group a glyph stream into tokens with lengths ~ Normal(mean, std)."""
    tokens: list[list[str]] = []
    i = 0
    while i < len(glyph_stream):
        length = max(1, round(rng.gauss(length_mean, length_std)))
        if i + length > len(glyph_stream):
            length = len(glyph_stream) - i
        token = glyph_stream[i : i + length]
        if token:
            tokens.append(token)
        i += length
    return tokens


def mutate_locally(
    tokens: list[list[str]],
    vocab: list[str],
    rate: float,
    rng: random.Random,
) -> list[list[str]]:
    """Apply local mutations at the given per-glyph rate."""
    out: list[list[str]] = []
    for token in tokens:
        new_token = list(token)
        for j in range(len(new_token)):
            if rng.random() < rate:
                new_token[j] = rng.choice(vocab)
        if new_token:
            out.append(new_token)
    return out


def generate_fractal_corpus(
    observed_vocab: list[str],
    params: FractalParams,
) -> list[list[str]]:
    """Main entry point. Synthesize fGn, encode to glyphs, group to tokens,
    apply local mutations. Returns list of token glyph-lists."""
    params.validate()
    rng_np = np.random.default_rng(params.seed)
    rng_py = random.Random(params.seed)

    signal = synthesize_fgn(params.n_glyphs, params.hurst, rng_np)
    glyphs = rank_quantile_encode(signal, observed_vocab)
    tokens = group_into_tokens(glyphs, params.token_length_mean,
                               params.token_length_std, rng_py)
    tokens = mutate_locally(tokens, observed_vocab, params.local_mutation_rate,
                            rng_py)
    return tokens


if __name__ == "__main__":
    import json

    from src.layer3_mechanisms.data_loader import load_data_bundle
    from src.shared.fractal.hurst import compute_hurst_dfa

    bundle = load_data_bundle()
    vocab = sorted({g for t in bundle.tokenized for g in t})
    params = FractalParams()
    generated = generate_fractal_corpus(vocab, params)
    glyph_stream = [g for t in generated for g in t]
    
    # Convert glyphs to numeric IDs for Hurst computation
    def glyphs_to_numeric(glyphs):
        vocab = sorted(set(glyphs))
        glyph_to_id = {g: i for i, g in enumerate(vocab)}
        return [float(glyph_to_id[g]) for g in glyphs]
    
    obs_stream = [g for t in bundle.tokenized for g in t]
    print(json.dumps({
        "n_tokens": len(generated),
        "n_glyphs": len(glyph_stream),
        "hurst_observed": compute_hurst_dfa(glyphs_to_numeric(obs_stream)),
        "hurst_generated": compute_hurst_dfa(glyphs_to_numeric(glyph_stream)),
    }, indent=2))