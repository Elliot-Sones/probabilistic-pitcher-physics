# V2.3 Model Tournament Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare alternate feature breakdowns and model families against the current V2.1/V2.2 baselines using the same temporal C2ST validation.

**Architecture:** Add a focused tournament module that can fit and sample candidate models without changing the default generator. The first tournament includes PCA latent residual sampling, context nearest-neighbor sampling, and a derived-feature Gaussian/copula representation. A runner evaluates all variants on the same Skubal split over repeated seeds and writes a clean report.

**Tech Stack:** Python, pandas, numpy, existing feature groups, existing factorized model, existing C2ST validator, pytest, ruff.

---

## Scope

This does not replace the product model automatically. A tournament model only becomes a candidate default if it improves `physics_core` without breaking component layers.

## Tasks

### Task 1: Derived Feature Representation

**Files:**
- Create: `src/pitcher_twin/tournament.py`
- Create: `tests/test_tournament.py`

- [x] Add tests for derived feature columns from a pitch-physics frame.
- [x] Implement derived features for spin angle, movement magnitude/angle, release geometry, and velocity-adjusted movement.
- [x] Verify: `pytest tests/test_tournament.py -q`.
- [x] Commit.

### Task 2: Tournament Model Primitives

**Files:**
- Modify: `src/pitcher_twin/tournament.py`
- Modify: `tests/test_tournament.py`

- [x] Add tests for PCA latent residual fit/sample.
- [x] Add tests for context nearest-neighbor fit/sample.
- [x] Implement `TournamentModel`, `fit_pca_latent_model`, `sample_pca_latent_model`.
- [x] Implement `fit_context_neighbor_model`, `sample_context_neighbor_model`.
- [x] Verify finite full-physics output rows.
- [x] Commit.

### Task 3: Tournament Evaluation

**Files:**
- Modify: `src/pitcher_twin/tournament.py`
- Modify: `tests/test_tournament.py`

- [x] Add tests that tournament evaluation returns layer AUC rows for all candidates.
- [x] Include current factorized baseline.
- [x] Include PCA latent and context-neighbor variants.
- [x] Include derived-feature Gaussian/covariance variant.
- [x] Aggregate repeated-seed results.
- [x] Commit.

### Task 4: CLI Runner

**Files:**
- Create: `scripts/run_model_tournament.py`
- Modify: `tests/test_tournament.py`

- [x] Add CLI import smoke test.
- [x] Implement runner with `--data`, `--output-dir`, `--pitcher-id`, `--pitch-type`, `--repeats`.
- [x] Write JSON and Markdown report.
- [x] Commit.

### Task 5: Real Skubal Tournament

**Files:**
- Generated: `outputs/model_tournament_skubal_2025_ff/model_tournament_report.json`
- Generated: `outputs/model_tournament_skubal_2025_ff/model_tournament_summary.md`
- Modify: `README.md`
- Modify: `docs/presentation.md`

- [x] Run tournament on Skubal 2025 FF.
- [x] Compare winner against V2.1/V2.2 baselines.
- [x] Document whether any variant earns candidate-default status.
- [x] Commit artifacts and docs.

### Task 6: Full Verification

- [x] Run `python3 -m compileall src app scripts tests`.
- [x] Run `ruff check src tests app scripts`.
- [x] Run `pytest -q`.
- [x] Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`.
- [x] Commit final fixes if needed.
