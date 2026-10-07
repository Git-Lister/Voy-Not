# Preregistration

## The Essential Question
(to be populated in v0.4.1)

## Target Statistical Signature (S1–S15)
(to be populated in v0.4.1)

## Datasets and Infrastructure
(to be populated in v0.4.1)

## Null Models and Controls
(to be populated in v0.4.1)

## Model Classes (A–F)
(to be populated in v0.4.1)

## Falsification Criteria
(to be populated in v0.4.1)

## Amendments Log
(to be populated in v0.4.1)

### Amendment 2026-10-07 — Fractal Properties: Operational Definitions
S13, S14, S15 are now defined operationally as follows:

**S13 (Hurst DFA):** computed on the full glyph stream, converted to
first-appearance-order integer IDs. DFA over log-spaced window sizes
from 8 to N//4 (capped at 1000). Slope of log(F(s)) vs log(s).

**S14 (MF-DFA Δh):** computed on the same integer-ID series. q in
[-5, 5] step 0.5. Scales log-spaced from 16 to N//4. Δh = h(q_min) - h(q_max). 
Validity: mean R² ≥ 0.95 and min R² ≥ 0.85.

**S15 (box-counting):** computed on point set [(i, glyph_id(i))] for
i in [0, N). Raw coordinates, no normalization. Box sizes log-spaced
from 0.5 to min_spacing*2. Slope of log(N(ε)) vs log(1/ε).

Original Brick 4b values (H=0.653, Δh=0.041, BC=0.995) are RETRACTED.
They were computed on a line-level series (4072 points), not the glyph
stream (33728 points). The current values (H≈0.614, Δh≈0.577, BC≈1.728)
replace them and are frozen as of 2026-10-07.