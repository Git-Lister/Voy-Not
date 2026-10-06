"""Null model suite for generative model testing. Filled by Brick 3a."""

import random
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.layer3_mechanisms.data_loader import DataBundle


def within_token_shuffle(tokenized: list[list[str]], seed: int = 42) -> list[list[str]]:
    """Shuffle glyphs within each token. Preserves token lengths and glyph marginals."""
    rng = random.Random(seed)
    shuffled = []
    for token in tokenized:
        if len(token) <= 1:
            shuffled.append(token[:])
        else:
            glyphs = token[:]
            rng.shuffle(glyphs)
            shuffled.append(glyphs)
    return shuffled


def within_line_shuffle(tokenized: list[list[str]], line_boundaries: list[int], seed: int = 42) -> list[list[str]]:
    """Shuffle tokens within each line. Preserves line lengths and token marginals.
    line_boundaries is the cumulative-token-count array from DataBundle."""
    rng = random.Random(seed)
    shuffled = []

    for i in range(len(line_boundaries) - 1):
        start = line_boundaries[i]
        end = line_boundaries[i + 1]
        line_tokens = tokenized[start:end]

        if len(line_tokens) <= 1:
            shuffled.extend(line_tokens)
        else:
            line_copy = line_tokens[:]
            rng.shuffle(line_copy)
            shuffled.extend(line_copy)

    return shuffled


def within_quire_shuffle(tokenized: list[list[str]], quire_labels: list[str], seed: int = 42) -> list[list[str]]:
    """Shuffle whole lines within each quire. Preserves quire lengths and line
    marginals. quire_labels is the per-line array from DataBundle."""
    rng = random.Random(seed)

    # Group line indices by quire
    quire_to_indices: dict[str, list[int]] = {}
    for idx, quire in enumerate(quire_labels):
        if quire not in quire_to_indices:
            quire_to_indices[quire] = []
        quire_to_indices[quire].append(idx)

    # Shuffle within each quire
    new_order = list(range(len(tokenized)))
    for quire, indices in quire_to_indices.items():
        if len(indices) > 1:
            shuffled_indices = indices[:]
            rng.shuffle(shuffled_indices)
            for old, new in zip(indices, shuffled_indices):
                new_order[old] = new

    # Reorder tokenized
    return [tokenized[i] for i in new_order]


def frequency_matched_synthetic(tokenized: list[list[str]], seed: int = 42) -> list[list[str]]:
    """Generate synthetic corpus where token lengths and glyph frequencies are
    matched to the original, but token identities are drawn i.i.d. from the
    observed token-frequency distribution."""
    rng = random.Random(seed)

    # Flatten all tokens and get token frequency distribution
    all_tokens = [tuple(token) for token in tokenized]
    if not all_tokens:
        return [[] for _ in tokenized]

    token_freq = Counter(all_tokens)
    token_list = list(token_freq.keys())
    token_probs = np.array([token_freq[t] for t in token_list], dtype=float)
    token_probs /= token_probs.sum()

    # Also get token length distribution
    lengths = [len(t) for t in tokenized]

    synthetic: list[list[str]] = []
    for length in lengths:
        # Generate a new token of the same length by sampling from observed tokens of that length
        same_length_tokens = [t for t in token_list if len(t) == length]
        if not same_length_tokens:
            # Fallback: sample from all tokens of the correct length
            all_tokens_of_length = [t for t in token_list if len(t) == length]
            if not all_tokens_of_length:
                # Ultimate fallback: just use the first token from token_list
                new_tok = list(token_list[0]) if token_list else []
            else:
                chosen = rng.choices(all_tokens_of_length, weights=[token_freq[t] for t in all_tokens_of_length], k=1)[0]
                new_tok = list(chosen)
        else:
            same_length_probs = np.array([token_freq[t] for t in same_length_tokens], dtype=float)
            same_length_probs /= same_length_probs.sum()
            chosen_list = rng.choices(same_length_tokens, weights=same_length_probs.tolist(), k=1)
            chosen_tuple = chosen_list[0]
            new_tok = list(chosen_tuple)
        synthetic.append(new_tok)

    return synthetic


def block_bootstrap(tokenized: list[list[str]], block_size: int = 20, seed: int = 42) -> list[list[str]]:
    """Resample contiguous blocks of tokens with replacement."""
    rng = random.Random(seed)
    n_tokens = len(tokenized)
    if n_tokens == 0:
        return []

    # Create blocks
    blocks = [tokenized[i:i + block_size] for i in range(0, n_tokens, block_size)]
    if not blocks:
        return tokenized[:]

    # Resample blocks
    n_blocks = len(blocks)
    resampled_indices = [rng.randint(0, n_blocks - 1) for _ in range(n_blocks)]

    synthetic = []
    for idx in resampled_indices:
        synthetic.extend(blocks[idx])

    # Trim or pad to match original length
    if len(synthetic) > n_tokens:
        synthetic = synthetic[:n_tokens]
    elif len(synthetic) < n_tokens:
        # Pad with random tokens from original
        while len(synthetic) < n_tokens:
            synthetic.append(rng.choice(tokenized))

    return synthetic


def markov_chain_null(tokenized: list[list[str]], order: int = 2, seed: int = 42) -> list[list[str]]:
    """Fit a Markov chain of the given order to the token sequence. Generate
    a new sequence of the same length."""
    rng = random.Random(seed)

    # Flatten tokens
    all_tokens = [str(t) for token in tokenized for t in token]
    if not all_tokens:
        return [[] for _ in tokenized]

    # Build transition counts
    transitions: Counter[tuple[tuple[str, ...], str]] = Counter()
    states: Counter[tuple[str, ...]] = Counter()

    for i in range(len(all_tokens) - order):
        state = tuple(all_tokens[i:i + order])
        next_token = all_tokens[i + order]
        transitions[(state, next_token)] += 1
        states[state] += 1

    # Build transition probabilities
    trans_probs: dict[tuple[str, ...], dict[str, float]] = {}
    for (state, next_t), count in transitions.items():
        if state not in trans_probs:
            trans_probs[state] = {}
        trans_probs[state][next_t] = count / states[state]

    # Generate new sequence
    # Start with a random state
    if not trans_probs:
        return tokenized[:]

    start_state = rng.choice(list(trans_probs.keys()))
    generated = list(start_state)

    target_length = len(all_tokens)
    while len(generated) < target_length:
        state = tuple(generated[-order:])
        if state in trans_probs:
            probs = trans_probs[state]
            next_t = rng.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]
            generated.append(next_t)
        else:
            # Fallback: random token
            generated.append(rng.choice(all_tokens))

    # Reshape to match original tokenized structure
    synthetic_tokenized: list[list[str]] = []
    idx = 0
    for token in tokenized:
        length = len(token)
        if idx + length <= len(generated):
            synthetic_tokenized.append(generated[idx:idx + length])
            idx += length
        else:
            synthetic_tokenized.append(generated[idx:])
            break

    return synthetic_tokenized


def apply_all_nulls(bundle: "DataBundle", seed: int = 42) -> dict[str, list[list[str]]]:
    """Return a dict mapping null-model name to shuffled corpus.
    Takes a DataBundle, not raw lists — this avoids ambiguity about
    line_boundaries and quire_labels."""

    return {
        "within_token_shuffle": within_token_shuffle(bundle.tokenized, seed),
        "within_line_shuffle": within_line_shuffle(bundle.tokenized, bundle.line_boundaries, seed),
        "within_quire_shuffle": within_quire_shuffle(bundle.tokenized, bundle.quire_labels, seed),
        "frequency_matched_synthetic": frequency_matched_synthetic(bundle.tokenized, seed),
        "block_bootstrap": block_bootstrap(bundle.tokenized, block_size=20, seed=seed),
        "markov_chain_null": markov_chain_null(bundle.tokenized, order=2, seed=seed),
    }