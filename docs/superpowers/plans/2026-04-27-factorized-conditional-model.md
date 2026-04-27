# Factorized Conditional Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a factorized conditional generator that samples release, movement, trajectory, and command in a physically staged order.

**Architecture:** Implement a new `src/pitcher_twin/factorized.py` module with a dataclass model, ridge helpers, fit, sample, and layer evaluation support. Integrate the new model into `validate_conditional_layers` so it appears in the same AUC comparison table as the existing game-drift models.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy-free linear algebra via numpy, existing C2ST validator, pytest.

---

### Task 1: Factorized Model Core

**Files:**
- Create: `src/pitcher_twin/factorized.py`
- Test: `tests/test_factorized.py`

- [ ] **Step 1: Write failing tests**

Add tests for training and finite full-physics sampling:

```python
def test_factorized_model_fits_named_stages() -> None:
    train, _ = _skubal_train_holdout()
    model = fit_factorized_pitch_model(train, pitcher_name="Skubal, Tarik", pitch_type="FF")
    assert model.model_name == "factorized_conditional"
    assert model.release_columns == PITCH_PHYSICS_FEATURES
    assert model.movement_columns == ["pfx_x", "pfx_z"]
    assert model.command_columns == ["plate_x", "plate_z"]
    assert model.trajectory_columns == TRAJECTORY_FEATURES
    assert set(model.stage_payloads) == {"release", "movement", "trajectory", "command"}


def test_factorized_model_samples_finite_physics_core_rows() -> None:
    train, holdout = _skubal_train_holdout()
    model = fit_factorized_pitch_model(train, pitcher_name="Skubal, Tarik", pitch_type="FF")
    samples = sample_factorized_pitch_model(model, n=25, context_df=holdout, random_state=7)
    assert samples.shape == (25, len(FEATURE_GROUPS["physics_core"]))
    assert samples.columns.tolist() == FEATURE_GROUPS["physics_core"]
    assert np.isfinite(samples.to_numpy(float)).all()
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
pytest tests/test_factorized.py -q
```

Expected: fail because `pitcher_twin.factorized` does not exist.

- [ ] **Step 3: Implement factorized model core**

Implement:

- `FactorizedPitchModel`
- `fit_factorized_pitch_model(...)`
- `sample_factorized_pitch_model(...)`

Stages:

- release uses recent weighted game mean plus fatigue-context ridge effects plus residual covariance;
- movement predicts `pfx_x`, `pfx_z` from sampled release;
- trajectory predicts trajectory columns from release plus movement;
- command predicts `plate_x`, `plate_z` from intent context plus sampled `release_speed`, `pfx_x`, `pfx_z`.

- [ ] **Step 4: Run factorized tests**

Run:

```bash
pytest tests/test_factorized.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/pitcher_twin/factorized.py tests/test_factorized.py
git commit -m "Add factorized conditional pitch model"
```

### Task 2: Validation Integration

**Files:**
- Modify: `src/pitcher_twin/conditional.py`
- Modify: `tests/test_conditional.py`

- [ ] **Step 1: Write failing validation assertion**

Update conditional validation tests to require `factorized_conditional` in each layer's `model_results`.

```python
assert "factorized_conditional" in row["model_results"]
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
pytest tests/test_conditional.py::test_conditional_layer_validation_reports_layers_and_models -q
```

Expected: fail because validation does not include `factorized_conditional`.

- [ ] **Step 3: Integrate factorized validation**

In `validate_conditional_layers`, fit one factorized model per train/holdout evaluation and add its generated samples to every requested feature group by selecting that group's columns before C2ST.

- [ ] **Step 4: Run conditional tests**

Run:

```bash
pytest tests/test_conditional.py tests/test_factorized.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/pitcher_twin/conditional.py tests/test_conditional.py
git commit -m "Compare factorized model in conditional validation"
```

### Task 3: Real Skubal Evaluation

**Files:**
- No source file changes unless tests expose a defect.

- [ ] **Step 1: Run full tests**

Run:

```bash
pytest -q
```

Expected: pass.

- [ ] **Step 2: Run real Skubal layer validation**

Run a one-off validation against `data/processed/skubal_2025.csv` for Skubal FF, reporting AUCs for:

- game-drift Gaussian
- game-drift copula
- conditional copula
- factorized conditional

Across:

- command
- movement
- release
- trajectory
- physics core

- [ ] **Step 3: Report results honestly**

Summarize whether factorization improves any layer and whether full physics remains diagnostic.
