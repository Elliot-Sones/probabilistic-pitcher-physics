# Side-By-Side Conditional Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a side-by-side conditional pitch distribution explorer using Statcast-supported game-state inputs.

**Architecture:** Engineer pitcher-perspective context features in `features.py`, keep model fitting/sampling primitives in `models.py`, add a focused `conditional.py` orchestration module, and expose the result through tests plus a Streamlit comparison tab. Validation remains C2ST-based and reports layer-by-layer confidence.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy, scikit-learn when available, pytest, Streamlit/Plotly for the dashboard.

---

### Task 1: Engineer Pitcher-Perspective Context Features

**Files:**
- Modify: `src/pitcher_twin/features.py`
- Modify: `src/pitcher_twin/models.py`
- Test: `tests/test_features_real.py`

- [ ] **Step 1: Write failing tests**

Add tests that prove raw Statcast `pitch_number` is not used as game pitch count and that score differential is pitcher-perspective.

```python
def test_pitcher_game_pitch_count_is_cumulative_within_pitcher_game() -> None:
    df = pd.DataFrame(
        {
            "game_date": ["2026-04-01"] * 5,
            "game_pk": [1] * 5,
            "pitcher": [10] * 5,
            "at_bat_number": [1, 1, 2, 2, 2],
            "pitch_number": [1, 2, 1, 2, 3],
        }
    )
    result = add_pitcher_game_pitch_count(df)
    assert result["pitcher_game_pitch_count"].tolist() == [1, 2, 3, 4, 5]


def test_pitcher_score_diff_is_positive_when_pitchers_team_leads() -> None:
    df = pd.DataFrame(
        {
            "pitcher_team": ["HOME", "AWAY"],
            "home_team": ["HOME", "HOME"],
            "away_team": ["AWAY", "AWAY"],
            "home_score": [5, 5],
            "away_score": [3, 8],
            "bat_score": [3, 5],
            "fld_score": [5, 8],
        }
    )
    result = add_pitcher_score_diff(df)
    assert result["pitcher_score_diff"].tolist() == [2.0, 3.0]
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
pytest tests/test_features_real.py::test_pitcher_game_pitch_count_is_cumulative_within_pitcher_game tests/test_features_real.py::test_pitcher_score_diff_is_positive_when_pitchers_team_leads -q
```

Expected: both fail because `add_pitcher_game_pitch_count` and `add_pitcher_score_diff` do not exist.

- [ ] **Step 3: Implement feature engineering**

Add `add_pitcher_game_pitch_count` and `add_pitcher_score_diff` to `features.py`. Update `add_real_context_features` to create both fields. Update context feature lists to prefer `pitcher_game_pitch_count` and `pitcher_score_diff`.

- [ ] **Step 4: Update model context ordering**

In `models.py`, replace `pitch_number` and `score_diff` in `CONTEXT_CONDITIONING_FEATURES` with `pitcher_game_pitch_count` and `pitcher_score_diff`. Update `_sort_pitch_rows` ordering to `game_date`, `game_pk`, `at_bat_number`, `pitch_number`.

- [ ] **Step 5: Run feature/model tests**

Run:

```bash
pytest tests/test_features_real.py tests/test_models_validation_real.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pitcher_twin/features.py src/pitcher_twin/models.py tests/test_features_real.py
git commit -m "Add pitcher-perspective context features"
```

### Task 2: Add Conditional Sampling Helpers

**Files:**
- Create: `src/pitcher_twin/conditional.py`
- Test: `tests/test_conditional.py`

- [ ] **Step 1: Write failing tests**

Add tests for context dataframe construction, model fallback selection, side-by-side summaries, and miss tendency.

```python
def test_make_context_dataframe_encodes_ui_inputs() -> None:
    context = make_context_dataframe(
        inning=7,
        pitcher_game_pitch_count=88,
        balls=2,
        strikes=2,
        batter_hand="L",
        pitcher_score_diff=1,
        repeat=3,
    )
    assert context.shape[0] == 3
    assert context["count_bucket_code"].iloc[0] == COUNT_BUCKET_CODES["even"]
    assert context["batter_stand_code"].iloc[0] == 1.0
    assert context["pitcher_score_diff"].iloc[0] == 1.0


def test_derive_miss_tendency_reports_zone_chase_and_spike() -> None:
    samples = pd.DataFrame({"plate_x": [-1.2, 0.0, 0.4, 1.4], "plate_z": [0.7, 2.4, 3.0, 4.1]})
    tendency = derive_miss_tendency(samples, pitcher_hand="R")
    assert tendency["sample_count"] == 4
    assert tendency["chase_rate"] > 0
    assert tendency["spike_risk_rate"] > 0
    assert tendency["primary_vertical"] in {"up", "down", "balanced"}
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
pytest tests/test_conditional.py -q
```

Expected: fails because `pitcher_twin.conditional` does not exist.

- [ ] **Step 3: Implement `conditional.py`**

Implement:

- `make_context_dataframe(...)`
- `select_conditional_model(...)`
- `sample_conditional_distribution(...)`
- `summarize_distribution(...)`
- `compare_context_distributions(...)`
- `derive_miss_tendency(...)`
- `layer_status_from_report(...)`

- [ ] **Step 4: Run conditional tests**

Run:

```bash
pytest tests/test_conditional.py -q
```

Expected: all conditional tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pitcher_twin/conditional.py tests/test_conditional.py
git commit -m "Add conditional pitch sampling helpers"
```

### Task 3: Add Layer-By-Layer Conditional Validation

**Files:**
- Modify: `src/pitcher_twin/conditional.py`
- Test: `tests/test_conditional.py`

- [ ] **Step 1: Write failing validation test**

Add a fixture-backed test proving conditional validation returns model comparisons by layer.

```python
def test_conditional_layer_validation_reports_layers_and_models() -> None:
    df = clean_pitch_features(pd.read_csv(REAL_SAMPLE), pitch_types=None)
    ranking = rank_pitcher_pitch_candidates(
        df,
        thresholds=CandidateThresholds(min_pitches=20, min_holdout=5, min_games=1),
    )
    candidate = ranking.iloc[0].to_dict()
    subset = df[(df["pitcher"] == candidate["pitcher"]) & (df["pitch_type"] == candidate["pitch_type"])]
    train, holdout = temporal_train_holdout(subset, train_fraction=0.7)

    report = validate_conditional_layers(
        train,
        holdout,
        df,
        pitcher_name=candidate["pitcher_name"],
        pitch_type=candidate["pitch_type"],
        feature_groups=["command_representation", "movement_only"],
        n_samples=20,
    )

    assert set(report["feature_group_results"]) == {"command_representation", "movement_only"}
    for row in report["feature_group_results"].values():
        assert "player_recent_weighted_game_drift_gaussian" in row["model_results"]
        assert "conditional_game_drift_copula" in row["model_results"]
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
pytest tests/test_conditional.py::test_conditional_layer_validation_reports_layers_and_models -q
```

Expected: fails because `validate_conditional_layers` is not defined.

- [ ] **Step 3: Implement validation helper**

Add `validate_conditional_layers(...)` to `conditional.py`. It should fit each requested feature group, evaluate game-drift Gaussian, game-drift copula, and conditional game-drift copula when available, and record fallback model names when copula is missing.

- [ ] **Step 4: Run conditional tests**

Run:

```bash
pytest tests/test_conditional.py -q
```

Expected: all conditional tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pitcher_twin/conditional.py tests/test_conditional.py
git commit -m "Add conditional layer validation"
```

### Task 4: Surface Conditional Comparison in the Dashboard

**Files:**
- Modify: `app/streamlit_app.py`
- Test: `tests/test_conditional.py`

- [ ] **Step 1: Write test for dashboard-safe summary objects**

Add a test proving summaries contain values the Streamlit tab can render without fitting UI code directly.

```python
def test_compare_context_distributions_returns_dashboard_payload() -> None:
    a = pd.DataFrame({"release_speed": [84.0, 85.0], "plate_x": [-0.2, 0.1], "plate_z": [2.0, 2.5]})
    b = pd.DataFrame({"release_speed": [82.0, 83.0], "plate_x": [0.4, 0.6], "plate_z": [1.2, 1.4]})
    payload = compare_context_distributions(a, b, pitcher_hand="L")
    assert payload["context_a"]["release_speed"]["mean"] == 84.5
    assert payload["context_b"]["release_speed"]["mean"] == 82.5
    assert payload["delta"]["release_speed"]["mean_delta"] == -2.0
    assert "miss_tendency" in payload["context_a"]
```

- [ ] **Step 2: Run test to verify RED or coverage gap**

Run:

```bash
pytest tests/test_conditional.py::test_compare_context_distributions_returns_dashboard_payload -q
```

Expected: fail if payload shape is missing; pass only after helper behavior is correct.

- [ ] **Step 3: Add Streamlit side-by-side tab**

In `app/streamlit_app.py`, add a "Conditional Explorer" tab. Use existing report/session assets when available. Fit the selected pitcher/pitch model from the report data path if the data file exists. Show disabled/informational UI when data is unavailable.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_conditional.py -q
```

Expected: all conditional tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/streamlit_app.py tests/test_conditional.py
git commit -m "Add conditional explorer dashboard payload"
```

### Task 5: Final Verification

**Files:**
- All changed implementation and tests.

- [ ] **Step 1: Run full test suite**

Run:

```bash
pytest -q
```

Expected: all tests pass. If an unrelated baseline failure appears, document it with the exact failing test and reason.

- [ ] **Step 2: Inspect git status**

Run:

```bash
git status --short
```

Expected: clean or only intentionally uncommitted generated artifacts.

- [ ] **Step 3: Summarize implementation**

Report changed files, model behavior, validation status, and any limitations. Do not claim full-physics success unless tests/reports support it.
