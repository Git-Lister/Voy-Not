import pytest
from src.layer3_mechanisms.self_citation import (
    SelfCitationParams,
    SelfCitationGenerator,
    generate_corpus,
)


def test_params_validate_rejects_bad_probabilities():
    with pytest.raises(ValueError):
        SelfCitationParams(copy_prob=0.5, mutation_prob=0.5,
                           insert_prob=0.5, delete_prob=0.5).validate()


def test_generator_produces_requested_count():
    seed = [["a", "b"], ["b", "c"], ["c", "a"]]
    out = generate_corpus(seed, n_tokens=100, params=SelfCitationParams())
    assert len(out) == 100


def test_generator_is_deterministic():
    seed = [["a", "b"], ["b", "c"], ["c", "a"]]
    out1 = generate_corpus(seed, 100, SelfCitationParams(), rng_seed=42)
    out2 = generate_corpus(seed, 100, SelfCitationParams(), rng_seed=42)
    assert out1 == out2


def test_generator_uses_only_seed_vocabulary():
    seed = [["a", "b"], ["b", "c"], ["c", "a"]]
    out = generate_corpus(seed, 100, SelfCitationParams())
    vocab = {"a", "b", "c"}
    for token in out:
        assert all(g in vocab for g in token)


def test_positional_grammar_reduces_vocabulary_drift():
    """With grammar_strength=1.0, generated tokens must start/end with glyphs
    seen in the seed's onset/coda positions."""
    seed = [["a", "b"], ["b", "c"], ["c", "a"]]
    onset = {t[0] for t in seed}
    coda = {t[-1] for t in seed}
    params = SelfCitationParams(grammar_strength=1.0)
    out = generate_corpus(seed, 200, params)
    for token in out:
        assert token[0] in onset
        assert token[-1] in coda