import numpy as np
from src.layer3_mechanisms.fractal_generator import (
    FractalParams, synthesize_fgn, rank_quantile_encode,
    group_into_tokens, generate_fractal_corpus,
)
from src.shared.fractal.hurst import compute_hurst_dfa
import random
from collections import Counter


def test_fgn_has_target_hurst():
    rng = np.random.default_rng(42)
    sig = synthesize_fgn(4096, H=0.7, rng=rng)
    # Hurst via DFA on numeric series
    h = compute_hurst_dfa(sig.tolist())
    assert 0.55 < h < 0.85  # loose band, DFA has variance


def test_rank_quantile_preserves_marginals():
    rng = np.random.default_rng(42)
    sig = rng.normal(size=1000)
    vocab = ["a", "b", "c", "d"]
    out = rank_quantile_encode(sig, vocab)
    counts = Counter(out)
    # Each glyph should get ~250 positions
    for g in vocab:
        assert 200 <= counts[g] <= 300


def test_group_into_tokens_respects_length():
    glyphs = ["a"] * 100
    rng = random.Random(42)
    tokens = group_into_tokens(glyphs, 8.0, 1.0, rng)
    assert sum(len(t) for t in tokens) == 100


def test_generator_produces_requested_count():
    vocab = ["a", "b", "c", "d", "e", "f"]
    params = FractalParams(hurst=0.6, n_glyphs=2000,
                           token_length_mean=8.0, token_length_std=1.0)
    out = generate_fractal_corpus(vocab, params)
    total = sum(len(t) for t in out)
    assert total >= 1900  # allow for truncation


def test_generator_deterministic():
    vocab = ["a", "b", "c", "d"]
    params = FractalParams(hurst=0.6, n_glyphs=2000, seed=42)
    out1 = generate_fractal_corpus(vocab, params)
    out2 = generate_fractal_corpus(vocab, params)
    assert out1 == out2


def test_params_validate():
    import pytest
    with pytest.raises(ValueError):
        FractalParams(hurst=1.5).validate()
    with pytest.raises(ValueError):
        FractalParams(hurst=0.0).validate()