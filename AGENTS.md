# Agents Configuration

## Project Purpose
Build a reproducible, falsifiable computational apparatus to investigate the Voynich Manuscript (Beinecke MS 408) as a generative artifact. The central question is: "What are the necessary and sufficient constraints on a generative process that can produce the Voynich manuscript's observable structure?" We do NOT seek a key or a translation. We build, test, and falsify generative models against a pre-registered joint statistical signature (S1–S15), including fractal parameters (Hurst, multifractal width, box-counting dimension). All hypotheses are pre-registered. Every run records its falsification status.

## Three-Party Workflow
- **USER** produces ideas, approves gates, tests UI/interface.
- **ARCHITECT** (external, separate from you) produces code logic, Brick specifications, and recalibrations.
- **YOU (OpenCode)** apply code, scaffold, run tests, and return a STAGE REPORT.

## Seven Phase Gates
1. Intake & Audit
2. Representation
3. Descriptive Structure
4. Null Tests
5. Multimodal Association
6. Explanatory Hypotheses
7. Semantic Claims

## Hard Rules (Never Violate)
1. Never claim decipherment. The output is falsifiable modelling, not translation.
2. Cross-transcription validation is mandatory. Report results by source (ZL, IT, CD, FG, GC).
3. Every hypothesis must be pre-registered in ledger/hypotheses.json before any run.
4. Every run must record its falsification status.
5. Fractal parameters (Hurst, multifractal width, box-counting dimension) must be computable for all generated output.
6. Do not add scope. If a Brick is ambiguous, stop and record it under "Blockers".
7. Use `uv` for Python package management. Use `pytest` for tests. Use `ruff` and `mypy` for linting (report-only; lint warnings do not fail a Brick unless stated).
8. Python version 3.11 (pinned via `.python-version`). Project root is the current working directory.

## Stage Report Template
===== STAGE REPORT =====
Brick ID:            Brick 0
Status:              <COMPLETE | PARTIAL | BLOCKED | FAILED>
Timestamp:           <ISO 8601>

--- Files created ---
<path> | <bytes> | <purpose one-line>

--- Files modified ---
<path> | <change summary>

--- Commands run ---
<command> | <exit code> | <stdout summary (max 5 lines)>

--- Tests ---
<test name> | <PASS | FAIL | SKIP> | <duration ms>

--- Errors / warnings ---
<error> | <file:line> | <attempted fix | unresolved>

--- Assumptions made ---
<assumption> | <reason> | <risk if wrong>

--- Blockers for next stage ---
<blocker> | <required input>

--- Noticed during build (optional, only if genuinely stood out) ---
<observation> | <why it stood out>

--- Ready for next Brick? ---
<YES | NO> | <if NO, what is needed>
=========================

## Notes
- Bricks are executed one at a time. No scope is to be added.
- The model/LLM used by OpenCode is user-configured and must not be hard-coded.