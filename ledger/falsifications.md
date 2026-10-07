# Falsification Record

Append-only. Every hypothesis run records its status here.

---

## 2026-10-06 — H-A-001 Self-Citation

**Status:** FALSIFIED
**Best config:** copy_prob=0.85, grammar_strength=0.0, source_weighting=frequency
**Result:** 7/14 scoreable properties matched (threshold: >=12/15)
**Matched:** S1 (conditional glyph entropy), S4_onset, S4_coda, S4_ratio, S8 (LZ77 compression), S11 (positional vocab divergence)
**Failed:** S2_mean (2.95 vs 8.28), S2_skew, S3 (MI plateau), S5 (edge MI), S6 (Zipf), S7 (hapax)
**Artefacts corrected in this entry:** S9 excluded (single-quire generation makes it meaningless); S10, S12 excluded (no cross-transcription data for generated text); S13-S15 pending box-counting robustness fix.
**Interpretation:** Self-citation as implemented produces locally plausible text but lacks a mechanism for maintaining long-range structure. The manuscript has structure at multiple scales; this model has structure at one scale.
**Next:** Brick 3c — test compression hypothesis.