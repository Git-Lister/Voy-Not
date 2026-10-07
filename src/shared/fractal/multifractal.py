"""Multifractal module. Filled by Brick 4."""

import importlib.util
from typing import Any

import numpy as np


def _compute_multifractal_fathon(sequence: list[float], q_range: tuple = (-5, 5), q_step: float = 0.5) -> dict:
    """Compute MF-DFA using fathon."""
    if importlib.util.find_spec("fathon") is None:
        raise RuntimeError("fathon not installed")
    
    try:
        from fathon import fathonUtils as fu
    except ImportError:
        raise RuntimeError("fathon not installed")
    
    if len(sequence) < 1000:
        raise ValueError("Sequence too short for multifractal analysis (minimum 1000 points recommended)")
    
    arr = np.array(sequence, dtype=float)
    if np.all(arr == arr[0]):
        raise ValueError("Constant sequence - multifractal spectrum undefined")
    
    # Compute MF-DFA
    # q values
    q_vals = np.arange(q_range[0], q_range[1] + q_step/2, q_step)
    
    # Use fathon's MFDFA
    p = fu.MFDFA(arr, q=q_vals)
    
    # Extract results
    hq = p.hq.tolist()  # generalized Hurst exponent
    tau_q = p.tau.tolist()  # mass exponent
    alpha = p.alpha.tolist()  # singularity strength
    f_alpha = p.f_alpha.tolist()  # singularity spectrum
    
    delta_h = float(np.max(hq) - np.min(hq)) if hq else 0.0
    delta_alpha = float(np.max(alpha) - np.min(alpha)) if alpha else 0.0
    
    return {
        "q": q_vals.tolist(),
        "hq": hq,
        "tau_q": tau_q,
        "alpha": alpha,
        "f_alpha": f_alpha,
        "delta_h": delta_h,
        "delta_alpha": delta_alpha,
        "backend": "fathon",
    }


def _compute_multifractal_numpy(sequence: list[float], q_range: tuple = (-5, 5), q_step: float = 0.5) -> dict:
    """Compute MF-DFA using numpy (manual implementation following Kantelhardt et al. 2002)."""
    if len(sequence) < 1000:
        raise ValueError("Sequence too short for multifractal analysis (minimum 1000 points recommended)")
    
    arr = np.array(sequence, dtype=float)
    if np.all(arr == arr[0]):
        raise ValueError("Constant sequence - multifractal spectrum undefined")
    
    # Step 1: Compute profile (cumulative sum of detrended signal)
    profile = np.cumsum(arr - np.mean(arr))
    N = len(profile)
    
    # Step 2: Define scales
    # Use scales from 16 to N/4, logarithmically spaced
    min_scale = 16
    max_scale = N // 4
    if max_scale <= min_scale:
        raise ValueError("Sequence too short for valid scale range")
    
    scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), 20).astype(int))
    
    # Step 3: q values
    q_vals = np.arange(q_range[0], q_range[1] + q_step/2, q_step)
    
    # Step 4: For each scale, compute fluctuation function
    # This is a simplified MF-DFA implementation
    Fq: dict[float, list[float]] = {q: [] for q in q_vals}
    
    for s in scales:
        # Divide profile into N_s non-overlapping segments
        N_s = N // s
        if N_s < 2:
            continue
            
        # RMS fluctuation for each segment
        rms_segments = []
        for i in range(N_s):
            segment = profile[i*s:(i+1)*s]
            # Detrend with linear fit
            x = np.arange(s)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            detrended = segment - trend
            rms = np.sqrt(np.mean(detrended**2))
            if rms > 0:
                rms_segments.append(rms)
        
        if len(rms_segments) < 2:
            continue
            
        # Compute F(q, s) for each q
        for q in q_vals:
            if q == 0:
                # q=0 case: logarithmic averaging
                fq = np.exp(np.mean(np.log(rms_segments)))
            else:
                fq = np.mean(np.array(rms_segments)**q)**(1/q)
            Fq[q].append(fq)
    
    # Step 5: Fit F(q, s) ~ s^h(q) for each q
    hq = []
    tau_q = []
    valid_q = []
    valid_scales = []
    r_squared_per_q = []
    
    for q in q_vals:
        if len(Fq[q]) < 4:  # Need at least 4 points for reliable fit
            continue
        
        # Use only scales where we have data
        valid_s = scales[:len(Fq[q])]
        log_s = np.log(valid_s)
        log_Fq = np.log(Fq[q])
        
        # Linear fit
        try:
            coeffs = np.polyfit(log_s, log_Fq, 1)
            h = coeffs[0]
            hq.append(float(h))
            tau_q.append(float(q * h - 1))
            valid_q.append(float(q))
            valid_scales.append(valid_s)
            
            # Compute R² for this q
            log_Fq_pred = coeffs[0] * log_s + coeffs[1]
            ss_res = np.sum((log_Fq - log_Fq_pred) ** 2)
            ss_tot = np.sum((log_Fq - np.mean(log_Fq)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            r_squared_per_q.append(float(r2))
        except np.linalg.LinAlgError:
            continue
    
    if len(hq) < 3:
        raise ValueError("Insufficient q values for multifractal spectrum")
    
    # Step 6: Compute singularity spectrum via Legendre transform
    hq_arr = np.array(hq)
    q_arr = np.array(valid_q)
    
    # alpha = d(tau)/dq = h + q*dh/dq
    # Approximate derivative
    if len(hq) > 2:
        dh_dq = np.gradient(hq_arr, q_arr)
        alpha = hq_arr + q_arr * dh_dq
        # f(alpha) = q*alpha - tau(q)
        f_alpha = q_arr * alpha - np.array(tau_q)
    else:
        alpha = hq_arr.copy()
        f_alpha = np.zeros_like(hq_arr)
    
    delta_h = float(np.max(hq_arr) - np.min(hq_arr))
    delta_alpha = float(np.max(alpha) - np.min(alpha))
    
    mean_r2 = float(np.mean(r_squared_per_q)) if r_squared_per_q else None
    min_r2 = float(np.min(r_squared_per_q)) if r_squared_per_q else None
    
    return {
        "q": valid_q,
        "hq": hq,
        "tau_q": tau_q,
        "alpha": alpha.tolist(),
        "f_alpha": f_alpha.tolist(),
        "delta_h": delta_h,
        "delta_alpha": delta_alpha,
        "backend": "numpy",
        "scales_used": valid_scales[0] if valid_scales else [],
        "r_squared_per_q": r_squared_per_q,
        "mean_r_squared": mean_r2,
        "min_r_squared": min_r2,
    }


def compute_multifractal(sequence: list[float], q_range: tuple = (-5, 5), q_step: float = 0.5) -> dict:
    """MF-DFA via fathon (or fallback to numpy).
    
    Return:
    {
      "q": [list of q values],
      "hq": [generalized Hurst exponent],
      "tau_q": [mass exponent],
      "f_alpha": [singularity spectrum f(α)],
      "alpha": [singularity strength α],
      "delta_h": float,   # width of h(q) spectrum
      "delta_alpha": float  # width of f(α) spectrum
    }
    
    If fathon unavailable, implement a numpy-based MF-DFA following Kantelhardt et al. (2002).
    Raise ValueError if sequence too short (< 1000 points recommended)."""
    # Try fathon first
    if importlib.util.find_spec("fathon") is not None:
        try:
            return _compute_multifractal_fathon(sequence, q_range, q_step)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: fathon MF-DFA failed: {e}, falling back to numpy")
    
    # Fallback to numpy
    try:
        return _compute_multifractal_numpy(sequence, q_range, q_step)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Both fathon and numpy MF-DFA failed: {e}")


def compute_multifractal_delta_h(sequence: list[float], q_range: tuple = (-5, 5), q_step: float = 0.5, return_diagnostics: bool = False) -> float | dict:
    """Compute the multifractal delta_h (width of h(q) spectrum) for a sequence.
    
    Args:
        sequence: Input numeric sequence
        q_range: Tuple of (q_min, q_max)
        q_step: Step size for q values
        return_diagnostics: If True, return full diagnostic info instead of just delta_h
    
    Returns:
        float (delta_h) or dict with diagnostics if return_diagnostics=True
    """
    result = compute_multifractal(sequence, q_range, q_step)
    dh = result.get("delta_h", 0.0)
    
    if return_diagnostics:
        def _to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, list):
                return [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in obj]
            return obj
        
        return {
            "delta_h": result.get("delta_h", 0.0),
            "q_values": _to_list(result.get("q", [])),
            "h_q": _to_list(result.get("hq", [])),
            "tau_q": _to_list(result.get("tau_q", [])),
            "alpha": _to_list(result.get("alpha", [])),
            "f_alpha": _to_list(result.get("f_alpha", [])),
            "scales_used": _to_list(result.get("scales_used", [])),
            "r_squared_per_q": _to_list(result.get("r_squared_per_q", [])),
            "mean_r_squared": result.get("mean_r_squared"),
            "min_r_squared": result.get("min_r_squared"),
        }
    return dh


def multifractal_by_quire(df: Any, value_column: str) -> dict:
    """Per-quire multifractal analysis. Return {quire: {...}, "aggregate": {...}}."""
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
        
        if len(values) < 1000:
            results[str(quire)] = {"error": f"Insufficient data points ({len(values)}, need 1000+)"}
            continue
        
        try:
            results[str(quire)] = compute_multifractal(values)
            all_values.extend(values)
        except Exception as e:  # noqa: BLE001
            results[str(quire)] = {"error": str(e)}
    
    # Aggregate across all quires
    if len(all_values) >= 1000:
        try:
            results["aggregate"] = compute_multifractal(all_values)
        except Exception as e:  # noqa: BLE001
            results["aggregate"] = {"error": str(e)}
    else:
        results["aggregate"] = {"error": "Insufficient aggregate data"}
    
    return results