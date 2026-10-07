import pytest
from src.layer3_mechanisms.compression import (
    DictParams, DictionaryGenerator, generate_compression_corpus,
    lz77_encode, lz77_decode, lz77_ratio, huffman_codes, huffman_ratio,
)


def test_lz77_roundtrip_simple():
    seq = list("abcabcabcabc")
    assert lz77_decode(lz77_encode(seq, 64, 2)) == seq


def test_lz77_roundtrip_repetitive():
    seq = list("ab" * 100)
    assert lz77_decode(lz77_encode(seq, 64, 3)) == seq


def test_lz77_ratio_lower_for_repetitive():
    rep = list("ab" * 500)
    rnd = list("abcdefghij" * 100)
    assert lz77_ratio(rep, 64) < lz77_ratio(rnd, 64)


def test_huffman_codes_complete():
    codes = huffman_codes(list("aaabbc"))
    assert set(codes.keys()) == {"a", "b", "c"}


def test_huffman_ratio_in_range():
    seq = list("the quick brown fox " * 50)
    r = huffman_ratio(seq)
    assert 0 < r < 1


def test_dict_generator_produces_requested_count():
    seed = [["a", "b", "c"], ["b", "c", "d"], ["c", "d", "e"]]
    assert len(generate_compression_corpus(seed, n_tokens=100, params=DictParams())) == 100


def test_dict_generator_deterministic():
    seed = [["a", "b"], ["b", "c"], ["c", "a"]]
    o1 = generate_compression_corpus(seed, 100, DictParams(), rng_seed=42)
    o2 = generate_compression_corpus(seed, 100, DictParams(), rng_seed=42)
    assert o1 == o2


def test_dict_params_validate():
    with pytest.raises(ValueError):
        DictParams(reference_prob=0.5, mutate_prob=0.5, literal_prob=0.5).validate()