# V2.4 Conditional State Mixture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a V2.4 tournament candidate that models pitcher state and state-specific residual distributions instead of relying on one widened covariance matrix.

**Architecture:** Keep the existing tournament runner and C2ST validation. Add recent pitcher-state feature engineering, a conditional state-mixture model in `src/pitcher_twin/tournament.py`, and tests proving it fits finite samples and appears in tournament reports.

**Tech Stack:** Python, pandas, numpy, scikit-learn mixture models, existing C2ST validator, pytest.

---

### Task 1: Recent Pitcher State Features

**Files:**
- Modify: `src/pitcher_twin/features.py`
- Test: `tests/test_features_real.py`

- [ ] **Step 1: Write the failing test**

Add a test that builds a small pitcher/game sequence and asserts previous-pitch plus rolling state columns are derived from earlier rows only.

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/test_features_real.py::test_recent_pitcher_state_features_use_prior_pitches_only -q`

Expected: import or assertion failure because the helper does not exist yet.

- [ ] **Step 3: Implement minimal feature engineering**

Add `RECENT_STATE_FEATURES` and `add_recent_pitcher_state_features()`. Include previous release speed, previous plate location, previous movement, and rolling means over prior 5/10/20 pitches.

- [ ] **Step 4: Run the focused test**

Run: `pytest tests/test_features_real.py::test_recent_pitcher_state_features_use_prior_pitches_only -q`

Expected: pass.

### Task 2: Conditional State Mixture Model

**Files:**
- Modify: `src/pitcher_twin/tournament.py`
- Test: `tests/test_tournament.py`

- [ ] **Step 1: Write failing model tests**

Add tests for `fit_conditional_state_mixture_model()` and `sample_tournament_model()` that assert finite physics-core samples, state metadata, and spin-axis normalization.

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `pytest tests/test_tournament.py::test_conditional_state_mixture_model_samples_state_conditioned_physics -q`

Expected: import failure because the functions/model do not exist yet.

- [ ] **Step 3: Implement minimal model**

Use available context/recent-state features to define state buckets, fit regularized Gaussian or Bayesian Gaussian mixture residual distributions per state, and sample from the state nearest each requested context row.

- [ ] **Step 4: Run the focused test**

Run: `pytest tests/test_tournament.py::test_conditional_state_mixture_model_samples_state_conditioned_physics -q`

Expected: pass.

### Task 3: Tournament Integration and Reporting

**Files:**
- Modify: `src/pitcher_twin/tournament.py`
- Modify: `scripts/run_model_tournament.py`
- Test: `tests/test_tournament.py`

- [ ] **Step 1: Write failing integration assertion**

Update tournament tests to require `conditional_state_mixture_residual` in `report["model_names"]`, `candidate_notes`, and summary-ready results.

- [ ] **Step 2: Run tournament test and verify it fails**

Run: `pytest tests/test_tournament.py::test_model_tournament_reports_repeated_layer_results -q`

Expected: assertion failure because V2.4 is not included.

- [ ] **Step 3: Add V2.4 to `_fit_tournament_candidate_models()`**

Append the fitted conditional state mixture model to tournament candidates and add candidate notes describing the state/context/residual strategy.

- [ ] **Step 4: Run tournament tests**

Run: `pytest tests/test_tournament.py -q`

Expected: pass.

### Task 4: Real Skubal Evaluation

**Files:**
- Output: `outputs/model_tournament_skubal_2025_ff/model_tournament_report.json`
- Output: `outputs/model_tournament_skubal_2025_ff/model_tournament_summary.md`

- [ ] **Step 1: Run the model tournament**

Run: `python3 scripts/run_model_tournament.py --data data/processed/skubal_2025.csv --output-dir outputs/model_tournament_skubal_2025_ff --pitcher-id 669373 --pitch-type FF --repeats 30`

Expected: report and summary are regenerated with V2.4 included.

- [ ] **Step 2: Run full verification**

Run: `pytest -q` and `ruff check .`

Expected: all tests and lint pass.

