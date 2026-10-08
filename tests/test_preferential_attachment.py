import pytest
from src.layer3_mechanisms.preferential_attachment import (
    PAParams, PreferentialAttachmentGenerator,
    generate_preferential_corpus,
)


def _seed_corpus():
    return [["a", "b", "c"], ["b", "c", "d"], ["c", "d", "e"],
            ["a", "b", "c"], ["b", "c", "d"]]


def test_generator_produces_requested_count():
    out = generate_preferential_corpus(_seed_corpus(), 200, PAParams())
    assert len(out) == 200


def test_generator_deterministic():
    out1 = generate_preferential_corpus(_seed_corpus(), 200, PAParams(), rng_seed=42)
    out2 = generate_preferential_corpus(_seed_corpus(), 200, PAParams(), rng_seed=42)
    assert out1 == out2


def test_generator_produces_hapax():
    """With novelty_prob > 0, some tokens should appear only once."""
    from collections import Counter
    out = generate_preferential_corpus(_seed_corpus(), 500, PAParams(novelty_prob=0.2))
    types = Counter(tuple(t) for t in out)
    hapax = sum(1 for v in types.values() if v == 1)
    assert hapax > 0


def test_generator_uses_only_seed_vocabulary():
    out = generate_preferential_corpus(_seed_corpus(), 200, PAParams())
    seed_vocab = {"a", "b", "c", "d", "e"}
    for t in out:
        assert all(g in seed_vocab for g in t)


def test_params_validate():
    with pytest.raises(ValueError):
        PAParams(novelty_prob=0.0).validate()
    with pytest.raises(ValueError):
        PAParams(novelty_prob=1.0).validate()
    with pytest.raises(ValueError):
        PAParams(mutation_strength=0.0).validate()