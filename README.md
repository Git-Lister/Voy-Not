# The Voynich Generative Apparatus

A reproducible, falsifiable computational apparatus to investigate the Voynich Manuscript (Beinecke MS 408) as a generative artifact. The central question is: **"What are the necessary and sufficient constraints on a generative process that can produce the Voynich manuscript's observable structure?"** We do NOT seek a key or a translation. We build, test, and falsify generative models against a pre-registered joint statistical signature (S1–S15), including fractal parameters (Hurst, multifractal width, box-counting dimension). All hypotheses are pre-registered. Every run records its falsification status.

This is a pre-registered research programme, not a decipherment attempt.

## Documentation

- [BLUEPRINT.md](BLUEPRINT.md) — System architecture and brick execution order
- [PREREGISTRATION.md](PREREGISTRATION.md) — Pre-registered hypotheses and statistical signatures

## Getting Started

```bash
uv sync
pytest tests/ -v
```