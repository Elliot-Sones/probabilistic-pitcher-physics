# V3 Generalized Validation Board Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable validation board that proves Pitcher Twin quality across pitcher/pitch candidates, not only one hand-picked Skubal fastball run.

**Architecture:** Add a focused validation-board module that selects real pitcher/pitch candidates, runs the existing model tournament, summarizes best models and layer status, and renders presentation-ready scorecards. Add a CLI wrapper that writes JSON, CSV, markdown, per-candidate reports, and optional rolling temporal windows.

**Tech Stack:** Python, pandas, existing Statcast loader/cleaner, existing `evaluate_model_tournament`, pytest.

---

### Task 1: Candidate Selection And Summary Contracts

**Files:**
- Create: `tests/test_validation_board.py`
- Create: `src/pitcher_twin/validation_board.py`

- [x] **Step 1: Write failing tests**

Add tests for `CandidateCriteria`, `candidate_pitcher_pitches`, `summarize_tournament_report`, `rolling_game_windows`, and `render_scorecard_markdown`.

- [x] **Step 2: Run focused tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_validation_board.py -q`

Expected: fail with missing module or missing functions.

- [x] **Step 3: Implement minimal validation-board module**

Implement candidate ranking, report summarization, rolling game split creation, and markdown rendering without running heavy model fits inside unit tests.

- [x] **Step 4: Run focused tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_validation_board.py -q`

Expected: all tests in the new file pass.

### Task 2: Validation Board Runner

**Files:**
- Create: `scripts/run_validation_board.py`
- Modify: `tests/test_validation_board.py`

- [x] **Step 1: Add failing CLI smoke test**

Import the script module, call the runner on a synthetic frame with a stub tournament function, and assert that leaderboard and scorecard artifacts are produced.

- [x] **Step 2: Run focused test**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_validation_board.py -q`

Expected: fail because the script does not exist.

- [x] **Step 3: Implement runner**

Load real Statcast data, clean features, select candidates, run tournament per candidate, write reports, write leaderboard CSV/JSON/markdown, write scorecards, and optionally run rolling windows.

- [x] **Step 4: Run focused test**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_validation_board.py -q`

Expected: all validation-board tests pass.

### Task 3: Real Artifact Generation

**Files:**
- Generate: `outputs/validation_board_*/`
- Modify: `README.md`
- Create: `docs/research/2026-04-27-v3-validation-board-results.md`

- [x] **Step 1: Run Skubal real-data board**

Run: `python3 scripts/run_validation_board.py --data /Users/elliot18/Desktop/Home/Projects/trajekt-pitcher-twin/data/processed/skubal_2025.csv --output-dir outputs/validation_board_skubal_2025 --top 1 --repeats 3 --samples 260 --rolling --rolling-repeats 1 --max-rolling-windows 3`

Expected: writes a one-candidate board proving the generalized pipeline works on the known real file.

- [x] **Step 2: Try broader real-data board when available**

Run: `python3 scripts/run_validation_board.py --data /Users/elliot18/assistant/projects/trajekt-scout/data/processed/latest_statcast.csv --output-dir outputs/validation_board_latest_statcast --top 3 --min-pitches 300 --min-games 8 --min-holdout 60 --repeats 2 --samples 220`

Expected: either writes a multi-candidate board or fails with a clear data/column availability error to record.

- [x] **Step 3: Document results**

Summarize the exact board outputs, whether V2.5 generalizes beyond Skubal, and what needs a larger season dataset.

### Task 4: Presentation Hook

**Files:**
- Modify: `app/streamlit_app.py`
- Modify: `README.md`

- [x] **Step 1: Inspect existing app tabs**

Read current UI structure and choose the smallest clean tab addition.

- [x] **Step 2: Add validation-board view**

Add a tab that loads a leaderboard CSV/JSON path, shows pitcher/pitch rows, best physics-core model, layer AUCs, pass rates, and links to generated scorecards.

- [x] **Step 3: Verify app syntax**

Run: `python3 -m py_compile app/streamlit_app.py`

Expected: compile succeeds. If Streamlit is unavailable locally, record that runtime launch was not possible.

### Task 5: Final Verification And Merge

**Files:**
- All changed files.

- [x] **Step 1: Run focused tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_validation_board.py -q`

- [x] **Step 2: Run full tests**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`

- [x] **Step 3: Review git diff**

Run: `git status --short && git diff --stat && git diff -- docs README.md src scripts tests app`

- [x] **Step 4: Commit and merge**

Commit the V3 worktree and merge it into `main` only after verification passes.
