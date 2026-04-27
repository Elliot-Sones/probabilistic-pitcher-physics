# V4 Pitch-Family Release/Spin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pitch-family-aware release/spin tournament candidate and a model router that explains which model should be trusted by pitch type and layer.

**Architecture:** Extend the existing model tournament rather than replacing it. The new factorized variant starts from the short-memory factorized physics model, applies family-specific release recentering, blended release-geometry correction, and empirical circular spin-axis residual sampling; the router summarizes tournament reports into validated/candidate/diagnostic model choices.

**Tech Stack:** Python, pandas, numpy, existing factorized tournament code, pytest.

---

### Task 1: Pitch-Family Spin/Release Utilities

**Files:**
- Modify: `tests/test_tournament.py`
- Modify: `src/pitcher_twin/tournament.py`

- [x] **Step 1: Write failing tests**

Add tests for `pitch_family_for_pitch_type`, `pitch_family_release_spin_settings`, `fit_spin_axis_residual_model`, `apply_spin_axis_residual_model`, and blended release geometry.

- [x] **Step 2: Verify tests fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_tournament.py -q`

Expected: import failure for the new utility functions.

- [x] **Step 3: Implement utilities**

Implement family mapping, family-specific alphas, empirical circular spin residual fitting/sampling, and partial release-geometry blending.

- [x] **Step 4: Verify focused tests pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_tournament.py -q`

### Task 2: Tournament Candidate Integration

**Files:**
- Modify: `src/pitcher_twin/tournament.py`
- Modify: `tests/test_tournament.py`

- [x] **Step 1: Update expected model list**

Add `factorized_pitch_family_release_spin` to the tournament model-name test and assert candidate notes include pitch family, spin residual source rows, and release-geometry blend alpha.

- [x] **Step 2: Verify tests fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_tournament.py::test_model_tournament_reports_repeated_layer_results -q`

- [x] **Step 3: Wire the candidate**

Add the new variant to `_fit_factorized_variants`, apply the family-specific transforms inside `evaluate_model_tournament`, and add notes to the report.

- [x] **Step 4: Verify focused tests pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_tournament.py -q`

### Task 3: Model Router

**Files:**
- Create: `src/pitcher_twin/model_router.py`
- Create: `tests/test_model_router.py`
- Modify: `src/pitcher_twin/validation_board.py`

- [x] **Step 1: Write failing router tests**

Test that a fake tournament report is routed into validated, candidate, and diagnostic layer choices and that scorecards include the route recommendation.

- [x] **Step 2: Verify tests fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_model_router.py tests/test_validation_board.py -q`

- [x] **Step 3: Implement router and scorecard hook**

Implement `build_model_route(report)` and add the route summary to validation-board summaries and scorecards.

- [x] **Step 4: Verify focused tests pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_model_router.py tests/test_validation_board.py -q`

### Task 4: Real V4 Evaluation

**Files:**
- Generate: `outputs/validation_board_skubal_2025_top3_v4/`
- Create: `docs/research/2026-04-27-v4-pitch-family-release-spin-results.md`
- Modify: `README.md`
- Modify: `docs/presentation.md`

- [x] **Step 1: Run Skubal top-3 board**

Run: `/Users/elliot18/Desktop/Home/Projects/trajekt-pitcher-twin/.venv/bin/python scripts/run_validation_board.py --data data/processed/skubal_2025.csv --output-dir outputs/validation_board_skubal_2025_top3_v4 --top 3 --repeats 3 --samples 260`

- [x] **Step 2: Compare V3 vs V4**

Record whether `factorized_pitch_family_release_spin` improves FF, SI, or CH, and whether it becomes best physics-core model for any pitch.

- [x] **Step 3: Update docs**

Document the exact scores and whether V4 is promoted, candidate-only, or diagnostic.

### Task 5: Verification And Merge

**Files:**
- All changed files.

- [x] **Step 1: Run syntax checks**

Run: `python3 -m py_compile src/pitcher_twin/tournament.py src/pitcher_twin/model_router.py src/pitcher_twin/validation_board.py scripts/run_validation_board.py app/streamlit_app.py`

- [x] **Step 2: Run focused tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_tournament.py tests/test_model_router.py tests/test_validation_board.py -q`

- [x] **Step 3: Run full tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`

- [x] **Step 4: Commit and merge**

Commit the branch, merge into `main`, and rerun full tests on main.
