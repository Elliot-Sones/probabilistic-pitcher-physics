# Rolling Validation Board Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add rolling temporal validation and pitch-type failure explanations so model quality is measured across multiple future-game windows, not one lucky 70/30 split.

**Architecture:** Create `src/pitcher_twin/rolling_validation.py` for game-window splitting, tournament summarization, failure explanation, and report writing. Add `scripts/run_rolling_temporal_board.py` as the CLI entry point. Keep validation powered by the existing tournament/C2ST machinery.

**Tech Stack:** Python, pandas, numpy, existing tournament evaluator, pytest.

---

### Task 1: Rolling Game Splits

**Files:**
- Create: `src/pitcher_twin/rolling_validation.py`
- Test: `tests/test_rolling_validation.py`

- [ ] **Step 1: Write failing test**

Test that 14 ordered games with `initial_train_games=4`, `test_games=2`, and `step_games=2` produce folds `1-4 -> 5-6`, `1-6 -> 7-8`, etc.

- [ ] **Step 2: Verify red**

Run: `pytest tests/test_rolling_validation.py::test_rolling_game_splits_train_cumulative_and_test_future_games -q`

Expected: import failure because module does not exist.

- [ ] **Step 3: Implement split helper**

Add `rolling_game_splits()` and a `RollingGameSplit` dataclass.

- [ ] **Step 4: Verify green**

Run the focused test and confirm pass.

### Task 2: Failure Explainer

**Files:**
- Modify: `src/pitcher_twin/rolling_validation.py`
- Test: `tests/test_rolling_validation.py`

- [ ] **Step 1: Write failing test**

Test that failed C2ST rows with `spin_axis` and `release_extension` produce a release/spin explanation, while movement/trajectory features produce acceleration/movement consistency.

- [ ] **Step 2: Verify red**

Run: `pytest tests/test_rolling_validation.py::test_failure_explainer_labels_classifier_detection_features -q`

- [ ] **Step 3: Implement explainer helpers**

Add `explain_detection_features()` and `build_pitch_type_failure_explanations()`.

- [ ] **Step 4: Verify green**

Run focused test.

### Task 3: Board Evaluation and CLI

**Files:**
- Modify: `src/pitcher_twin/rolling_validation.py`
- Create: `scripts/run_rolling_temporal_board.py`
- Test: `tests/test_rolling_validation.py`

- [ ] **Step 1: Write failing tests**

Use an injected fake tournament evaluator to verify board summaries and output paths without running expensive real validation in unit tests.

- [ ] **Step 2: Verify red**

Run: `pytest tests/test_rolling_validation.py -q`

- [ ] **Step 3: Implement board and writer**

Add `evaluate_rolling_temporal_board()`, `write_rolling_board_outputs()`, and script `main()`.

- [ ] **Step 4: Verify green**

Run: `pytest tests/test_rolling_validation.py -q`

### Task 4: Real Run and Verification

**Files:**
- Output: `outputs/rolling_validation_skubal_2025_ff/rolling_validation_board.json`
- Output: `outputs/rolling_validation_skubal_2025_ff/rolling_validation_board.md`
- Output: `outputs/rolling_validation_skubal_2025_ff/pitch_type_failure_explainer.csv`

- [ ] **Step 1: Run Skubal board**

Run: `.venv/bin/python scripts/run_rolling_temporal_board.py --data data/processed/skubal_2025.csv --output-dir outputs/rolling_validation_skubal_2025_ff --pitcher-id 669373 --pitch-type FF --initial-train-games 10 --test-games 2 --step-games 2 --repeats 4`

- [ ] **Step 2: Full verification**

Run: `pytest -q`, `ruff check .`, and `git diff --check`.

