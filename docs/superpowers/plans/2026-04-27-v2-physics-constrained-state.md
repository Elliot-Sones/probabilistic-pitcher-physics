# V2.4 Physics-Constrained State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add V2.4 tournament candidates that improve full-pitch realism by constraining release geometry, testing release copula structure, and anchoring generated clouds to the pitcher’s recent learned state.

**Architecture:** Keep the existing V2.3 tournament framework. Add small, testable postprocessors that learn the mechanical relationship between `release_pos_y` and `release_extension` and a recency-weighted game-state center from training data. Apply those postprocessors to factorized samples from the strongest short-memory model, then evaluate each candidate with the existing C2ST layer validator against the Skubal FF benchmark.

**Tech Stack:** Python, pandas, numpy, sklearn, pytest, existing `pitcher_twin.factorized` and `pitcher_twin.tournament` modules.

---

### Task 1: Release Geometry Constraint Tests

**Files:**
- Modify: `tests/test_tournament.py`
- Modify: `src/pitcher_twin/tournament.py`

- [ ] **Step 1: Write the failing test**

Add imports:

```python
from pitcher_twin.tournament import (
    apply_release_geometry_constraint,
    evaluate_model_tournament,
    fit_release_geometry_constraint,
    fit_tournament_models,
    sample_tournament_model,
)
```

Add a test that creates synthetic release data where `release_pos_y + release_extension` has a stable learned center, corrupts sample geometry, and verifies the postprocessor restores the learned sum:

```python
def test_release_geometry_constraint_restores_learned_extension_sum() -> None:
    frame = _synthetic_pitch_frame(n=80)
    drift = np.linspace(-0.04, 0.04, len(frame))
    frame["release_pos_y"] = 54.0 + drift
    frame["release_extension"] = 6.5 - drift

    constraint = fit_release_geometry_constraint(frame)

    samples = frame[["release_pos_y", "release_extension", "release_pos_x"]].head(24).copy()
    samples["release_pos_y"] = 53.25
    samples["release_extension"] = 6.10

    constrained = apply_release_geometry_constraint(samples, constraint, random_state=11)

    release_sum = constrained["release_pos_y"] + constrained["release_extension"]
    assert abs(float(release_sum.mean()) - constraint["sum_mean"]) < 0.08
    assert release_sum.std(ddof=0) > 0.0
    assert release_sum.std(ddof=0) < 0.15
    assert np.isfinite(constrained["release_pos_y"]).all()
    assert np.isfinite(constrained["release_extension"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_tournament.py::test_release_geometry_constraint_restores_learned_extension_sum -q
```

Expected: FAIL because `fit_release_geometry_constraint` is not defined/exported yet.

- [ ] **Step 3: Implement minimal code**

In `src/pitcher_twin/tournament.py`, add:

```python
def fit_release_geometry_constraint(train: pd.DataFrame) -> dict[str, float | int]:
    ...

def apply_release_geometry_constraint(
    samples: pd.DataFrame,
    constraint: Mapping[str, float | int],
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    ...
```

The learned target is `release_pos_y + release_extension`. The postprocessor samples plausible target sums and adjusts `release_pos_y` while preserving generated `release_extension`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_tournament.py::test_release_geometry_constraint_restores_learned_extension_sum -q
```

Expected: PASS.

### Task 2: Tournament Candidate Integration

**Files:**
- Modify: `tests/test_tournament.py`
- Modify: `src/pitcher_twin/tournament.py`

- [ ] **Step 1: Write the failing test**

Update the tournament report test to expect:

```python
"factorized_physics_constrained_state",
```

and assert that the report includes a candidate note:

```python
assert report["candidate_notes"]["factorized_physics_constrained_state"]["release_geometry_constraint"] == "release_pos_y_plus_extension"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_tournament.py::test_model_tournament_reports_repeated_layer_results -q
```

Expected: FAIL because the new model is not included in the tournament yet.

- [ ] **Step 3: Implement tournament support**

Modify `_fit_factorized_variants()` so the existing short-memory wide residual variant stays intact and a new `factorized_physics_constrained_state` variant is added. During sampling in `evaluate_model_tournament()`, if that variant has a release-geometry constraint, call `apply_release_geometry_constraint()` after `sample_factorized_physics()`.

Add metadata to `candidate_notes`:

```python
"factorized_physics_constrained_state": {
    "description": "Short-memory factorized model with release_pos_y + extension geometry constraint.",
    "release_geometry_constraint": "release_pos_y_plus_extension",
}
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/test_tournament.py -q
```

Expected: PASS.

### Task 2.5: Recent-State Anchor Candidate

**Files:**
- Modify: `tests/test_tournament.py`
- Modify: `src/pitcher_twin/tournament.py`

- [ ] **Step 1: Write the failing test**

Add imports for:

```python
fit_recent_state_anchor
apply_recent_state_anchor
```

Add a test that builds synthetic early/late game drift, creates generated samples far below the recent-state mean, and verifies `apply_recent_state_anchor(..., alpha=0.5)` moves the generated cloud halfway toward the training-only anchor.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_tournament.py::test_recent_state_anchor_moves_generated_cloud_toward_recent_games -q
```

Expected: FAIL because the recent-state anchor functions are not defined.

- [ ] **Step 3: Implement recent-state anchor**

Add:

```python
def fit_recent_state_anchor(train, feature_columns, half_life_games=1.25) -> dict[str, object]:
    ...

def apply_recent_state_anchor(samples, anchor, alpha=0.70) -> pd.DataFrame:
    ...
```

Then add `factorized_recent_state_anchored` to `_fit_factorized_variants()` and apply the anchor in `evaluate_model_tournament()`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/test_tournament.py -q
```

Expected: PASS.

### Task 3: Real-Data Benchmark

**Files:**
- No required code edits.
- Optional docs/results update after evaluation.

- [ ] **Step 1: Run the real Skubal tournament**

Run the project’s tournament diagnostic against `data/processed/skubal_2025.csv` for Skubal FF. If the existing script is unavailable, use `evaluate_model_tournament()` directly with the same temporal 70/30 setup used in V2.3.

- [ ] **Step 2: Compare against V2.3**

Record whether `factorized_physics_constrained_state` improves `physics_core` mean AUC versus the previous best `factorized_short_memory_wide_residual` at `0.603`.

- [ ] **Step 3: Save results**

Write a concise result artifact under:

```text
docs/research/2026-04-27-v2-physics-constrained-state-results.md
```

Include setup, model ranking, what improved, what did not, and next recommended angle.

### Task 4: Verification and Integration

**Files:**
- All touched implementation, tests, and docs.

- [ ] **Step 1: Run full tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Inspect git diff**

Run:

```bash
git diff --stat
git diff -- tests/test_tournament.py src/pitcher_twin/tournament.py
```

Expected: changes are scoped to V2.4.

- [ ] **Step 3: Commit the branch**

Run:

```bash
git add docs/superpowers/plans/2026-04-27-v2-physics-constrained-state.md tests/test_tournament.py src/pitcher_twin/tournament.py docs/research/2026-04-27-v2-physics-constrained-state-results.md
git commit -m "Add physics constrained state tournament candidate"
```

Expected: commit created on `v2-physics-state`.
