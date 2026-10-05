"""Hurst exponent module. Filled by Brick 4."""

from typing import Any

import numpy as np


def _hurst_rs(sequence: np.ndarray) -> float:
    """Rescaled range (R/S) estimator.
    Based on Hurst (1951) and Mandelbrot & Wallis (1969).
    Uses overlapping windows for better estimation.
    Returns H ∈ [0, 1]."""
    n = len(sequence)
    if n < 20:
        raise ValueError("Sequence too short for R/S estimation (minimum 20 points)")

    # Mean-center the sequence
    mean_seq = np.mean(sequence)
    y = sequence - mean_seq

    # Cumulative sum
    z = np.cumsum(y)

    # Use a range of window sizes
    min_k = 8
    max_k = min(n // 4, 1000)
    if max_k < min_k:
        raise ValueError("Sequence too short")

    # Generate window sizes - use more points for better fit
    ks = np.unique(np.logspace(np.log10(min_k), np.log10(max_k), 30).astype(int))
    ks = ks[ks <= max_k]
    ks = ks[ks >= min_k]

    rs_vals = []

    for k in ks:
        # Use overlapping windows (step = k//2)
        step = max(1, k // 2)
        n_windows = (n - k) // step + 1
        if n_windows < 4:
            continue

        rs_k = []
        for i in range(n_windows):
            start = i * step
            end = start + k
            if end > n:
                break
            window = z[start:end]

            # Range
            r = np.max(window) - np.min(window)
            # Standard deviation of original (not cumulative) sequence
            s = np.std(sequence[start:end])

            if s > 0:
                rs_k.append(r / s)

        if len(rs_k) >= 4:
            rs_vals.append((k, np.mean(rs_k)))

    if len(rs_vals) < 5:
        raise ValueError("Insufficient valid windows for R/S estimation")

    # Fit log(R/S) vs log(k)
    ks_arr = np.array([v[0] for v in rs_vals])
    rs_arr = np.array([v[1] for v in rs_vals])

    log_k = np.log(ks_arr)
    log_rs = np.log(rs_arr)

    # Linear regression
    A = np.vstack([log_k, np.ones(len(log_k))]).T
    slope, _ = np.linalg.lstsq(A, log_rs, rcond=None)[0]

    # Hurst exponent is the slope
    return float(np.clip(slope, 0.0, 1.0))


def _hurst_dfa(sequence: np.ndarray) -> float:
    """Detrended Fluctuation Analysis (DFA) estimator.
    Based on Peng et al. (1994).
    Returns H ∈ [0, 1]."""
    n = len(sequence)
    if n < 20:
        raise ValueError("Sequence too short for DFA estimation (minimum 20 points)")

    # Mean-center and cumulative sum
    y = sequence - np.mean(sequence)
    profile = np.cumsum(y)

    # Window sizes
    min_k = 4
    max_k = min(n // 4, 1000)
    if max_k < min_k:
        raise ValueError("Sequence too short")

    ks = np.unique(np.logspace(np.log10(min_k), np.log10(max_k), 30).astype(int))
    ks = ks[ks <= max_k]
    ks = ks[ks >= min_k]

    fluct = []

    for k in ks:
        # Non-overlapping windows
        n_windows = n // k
        if n_windows < 4:
            continue

        rms_k = []
        for i in range(n_windows):
            start = i * k
            end = start + k
            segment = profile[start:end]

            # Detrend with linear fit
            x = np.arange(k)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            detrended = segment - trend

            rms = np.sqrt(np.mean(detrended**2))
            if rms > 0:
                rms_k.append(rms)

        if len(rms_k) >= 4:
            fluct.append((k, np.mean(rms_k)))

    if len(fluct) < 5:
        raise ValueError("Insufficient valid windows for DFA")

    # Fit log(F) vs log(k)
    ks_arr = np.array([v[0] for v in fluct])
    f_arr = np.array([v[1] for v in fluct])

    log_k = np.log(ks_arr)
    log_f = np.log(f_arr)

    A = np.vstack([log_k, np.ones(len(log_k))]).T
    slope, _ = np.linalg.lstsq(A, log_f, rcond=None)[0]

    return float(np.clip(slope, 0.0, 1.0))


def compute_hurst_rs(sequence: list[float]) -> float:
    """Rescaled range (R/S) estimator.
    Returns H ∈ [0, 1]. Raises ValueError if sequence too short."""
    arr = np.array(sequence, dtype=float)
    if np.all(arr == arr[0]):
        raise ValueError("Constant sequence - Hurst exponent undefined")
    return _hurst_rs(arr)


def compute_hurst_dfa(sequence: list[float]) -> float:
    """Detrended fluctuation analysis estimator.
    Returns H ∈ [0, 1]."""
    arr = np.array(sequence, dtype=float)
    if np.all(arr == arr[0]):
        raise ValueError("Constant sequence - Hurst exponent undefined")
    return _hurst_dfa(arr)


def compute_hurst_ensemble(sequence: list[float]) -> dict:
    """Compute both RS and DFA. Return {"rs": float, "dfa": float, "mean": float, "std": float}."""
    arr = np.array(sequence, dtype=float)
    if np.all(arr == arr[0]):
        raise ValueError("Constant sequence - Hurst exponent undefined")

    h_rs = None
    h_dfa = None

    try:
        h_rs = _hurst_rs(arr)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: R/S estimation failed: {e}")

    try:
        h_dfa = _hurst_dfa(arr)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: DFA estimation failed: {e}")

    valid = [h for h in [h_rs, h_dfa] if h is not None]

    if not valid:
        raise ValueError("Both R/S and DFA estimation failed")

    return {
        "rs": h_rs,
        "dfa": h_dfa,
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)) if len(valid) > 1 else 0.0,
    }


def hurst_by_quire(df: Any, value_column: str) -> dict:
    """Apply compute_hurst_ensemble per quire. Return {quire: {...}, "aggregate": {...}}."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if "quire" not in df.columns:
        raise ValueError("DataFrame must have 'quire' column")

    if value_column not in df.columns:
        raise ValueError(f"DataFrame must have '{value_column}' column")

    results = {}
    all_values = []

    for quire in df["quire"].dropna().unique():
        quire_data = df[df["quire"] == quire]
        values = quire_data[value_column].dropna().tolist()

        if len(values) < 20:
            results[str(quire)] = {"error": f"Insufficient data points ({len(values)})"}
            continue

        try:
            results[str(quire)] = compute_hurst_ensemble(values)
            all_values.extend(values)
        except Exception as e:  # noqa: BLE001
            results[str(quire)] = {"error": str(e)}

    # Aggregate across all quires
    if len(all_values) >= 20:
        try:
            results["aggregate"] = compute_hurst_ensemble(all_values)
        except Exception as e:  # noqa: BLE001
            results["aggregate"] = {"error": str(e)}
    else:
        results["aggregate"] = {"error": "Insufficient aggregate data"}

    return results