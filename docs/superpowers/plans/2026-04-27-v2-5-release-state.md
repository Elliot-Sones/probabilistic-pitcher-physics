# V2.5 Release-State Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve full-physics temporal realism by making release mechanics more stable, especially spin axis, release extension, release position, velocity, and spin rate.

**Architecture:** Keep the V2.4 factorized tournament and add state postprocessors. One postprocessor models spin axis as a circular angle rather than independent `cos/sin`. Another learns a recency-weighted release-state center and applies it only to release fields. A third learns a bounded short-horizon game trend anchor, which addresses cases where the pitcher keeps drifting after the training window.

**Tech Stack:** Python, pandas, numpy, pytest, existing `pitcher_twin.tournament`, `pitcher_twin.factorized`, and C2ST validator.

---

### Task 1: Circular Spin-State Tests

**Files:**
- Modify: `tests/test_tournament.py`
- Modify: `src/pitcher_twin/tournament.py`

- [ ] **Step 1: Write the failing test**

Add imports:

```python
from pitcher_twin.tournament import apply_spin_axis_angle_anchor, fit_spin_axis_angle_anchor
```

Add a test that creates recent spin-axis drift, corrupts generated `spin_axis_cos/sin`, and verifies the postprocessor rotates samples toward the learned circular mean while preserving unit norm.

- [ ] **Step 2: Run red test**

Run:

```bash
pytest tests/test_tournament.py::test_spin_axis_angle_anchor_rotates_samples_and_preserves_unit_norm -q
```

Expected: FAIL because the spin-axis anchor functions do not exist.

- [ ] **Step 3: Implement minimal spin-axis anchor**

Add:

```python
def fit_spin_axis_angle_anchor(train, half_life_games=1.25) -> dict[str, object]:
    ...

def apply_spin_axis_angle_anchor(samples, anchor, alpha=0.70) -> pd.DataFrame:
    ...
```

The anchor computes a circular recent-game mean angle from `spin_axis_cos/sin`. The apply function rotates generated angles toward the target using wrapped angle differences and rewrites normalized `cos/sin`.

- [ ] **Step 4: Verify green**

Run:

```bash
pytest tests/test_tournament.py::test_spin_axis_angle_anchor_rotates_samples_and_preserves_unit_norm -q
```

Expected: PASS.

### Task 2: Release-State Candidate Integration

**Files:**
- Modify: `tests/test_tournament.py`
- Modify: `src/pitcher_twin/tournament.py`

- [ ] **Step 1: Write failing tournament expectation**

Update expected model names to include:

```python
"factorized_release_state_anchored"
```

Assert metadata:

```python
assert report["candidate_notes"]["factorized_release_state_anchored"]["release_anchor_alpha"] == 0.70
assert report["candidate_notes"]["factorized_release_state_anchored"]["spin_axis_anchor"] == "circular_recent_game_mean"
```

- [ ] **Step 2: Run red test**

Run:

```bash
pytest tests/test_tournament.py::test_model_tournament_reports_repeated_layer_results -q
```

Expected: FAIL because the candidate is not in the tournament yet.

- [ ] **Step 3: Implement tournament support**

Add `factorized_release_state_anchored` as a variant based on `factorized_recent_state_anchored`. During sampling:

1. Apply the full recent-state anchor as V2.4 does.
2. Apply a release-only recent-state anchor to `release_speed`, `release_spin_rate`, `release_pos_x`, `release_pos_y`, `release_pos_z`, `release_extension`.
3. Apply the circular spin-axis angle anchor.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/test_tournament.py -q
```

Expected: PASS.

### Task 2.5: Trend-State Candidate Integration

**Files:**
- Modify: `tests/test_tournament.py`
- Modify: `src/pitcher_twin/tournament.py`

- [ ] **Step 1: Write failing trend-anchor test**

Add import:

```python
fit_recent_trend_state_anchor
```

Add a synthetic test where per-game release speed and spin rate rise every game. Verify the trend anchor predicts a higher future state than the plain recent-state anchor while reporting the correct source row and game counts.

- [ ] **Step 2: Run red test**

Run:

```bash
pytest tests/test_tournament.py::test_recent_trend_state_anchor_extrapolates_bounded_game_drift -q
```

Expected: FAIL because `fit_recent_trend_state_anchor` does not exist.

- [ ] **Step 3: Implement trend anchor**

Add:

```python
def fit_recent_trend_state_anchor(
    train,
    feature_columns,
    half_life_games=5.0,
    horizon_games=1.0,
    trend_shrinkage=0.75,
    max_recent_std=1.5,
) -> dict[str, object]:
    ...
```

Then add `factorized_trend_state_anchored` to the tournament and apply `apply_recent_state_anchor(..., alpha=0.85)` using the trend anchor.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/test_tournament.py -q
```

Expected: PASS.

### Task 3: Real Skubal Evaluation

**Files:**
- Create: `docs/research/2026-04-27-v2-5-release-state-results.md`
- Generate: `outputs/model_tournament_skubal_2025_ff_v2_5*/`

- [ ] **Step 1: Run three seed-family tournaments**

Run `scripts/run_model_tournament.py` with seeds `42`, `101`, and `222`, outputting to V2.5-specific directories.

- [ ] **Step 2: Compare against V2.4**

Record:

- physics-core mean AUC and pass rate
- release-only mean AUC and pass rate
- whether the candidate becomes default
- what features remain as leakage

- [ ] **Step 3: Save result doc**

Write a concise interpretation in `docs/research/2026-04-27-v2-5-release-state-results.md`.

### Task 4: Verification and Merge

**Files:**
- All touched implementation, tests, docs, and generated results.

- [ ] **Step 1: Full verification**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Commit branch**

Commit the V2.5 branch.

- [ ] **Step 3: Merge to main**

Merge locally, keep the unrelated main `README.md` change untouched, then rerun full tests on `main`.
