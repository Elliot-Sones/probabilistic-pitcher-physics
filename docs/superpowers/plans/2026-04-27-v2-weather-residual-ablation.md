# V2.2 Weather Residual Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether real joined weather explains the remaining V2.1 full-physics residual gap for Skubal 2025 FF.

**Architecture:** Keep V2.1 factorized physics as the default baseline. Add a weather residual layer only for downstream movement, trajectory, and command residuals, then compare V2.1 vs V2.2 by temporal C2ST AUC. Weather is accepted only if it improves `physics_core` below the V2.1 baseline without major layer regressions.

**Tech Stack:** Python, pandas, numpy, urllib/request for Open-Meteo HTTP fetches, pytest, existing factorized model and C2ST validator.

---

## Scope

Weather is a validation ablation, not a UI feature. The implementation must not fabricate weather rows. If real weather cannot be joined, the weather model variant is unavailable and the report must say so.

## Tasks

### Task 1: Real Weather Join Utilities

**Files:**
- Create: `src/pitcher_twin/weather.py`
- Create: `tests/test_weather_real.py`

- [ ] Add tests for building Open-Meteo archive URLs, normalizing hourly responses, nearest-hour selection, and joining by `game_pk`.
- [ ] Implement `WEATHER_FEATURE_COLUMNS`.
- [ ] Implement `build_open_meteo_archive_url(latitude, longitude, start_date, end_date)`.
- [ ] Implement `normalize_open_meteo_hourly(payload)`.
- [ ] Implement `nearest_hourly_weather(hourly, target_time_utc)`.
- [ ] Implement `join_weather_by_game_pitch_rows(pitches, game_weather)`.
- [ ] Verify: `pytest tests/test_weather_real.py -q`.
- [ ] Commit.

### Task 2: Weather Residual Layer

**Files:**
- Modify: `src/pitcher_twin/factorized.py`
- Modify: `tests/test_factorized.py`

- [ ] Add tests that a factorized model can fit with weather features and records a weather residual payload.
- [ ] Add optional `weather_feature_columns` to `fit_factorized_physics_model`.
- [ ] Fit a ridge adjustment from real weather features to downstream residuals after the recent-game residual offset.
- [ ] Apply that weather residual adjustment only when `sample_factorized_physics(..., use_weather=True)` receives context rows with weather columns.
- [ ] Verify weather-off output remains the current default.
- [ ] Verify: `pytest tests/test_factorized.py -q`.
- [ ] Commit.

### Task 3: Real Weather Fetch / Cache Script

**Files:**
- Create: `scripts/fetch_open_meteo_weather.py`
- Modify: `tests/test_weather_real.py`

- [ ] Add a CLI smoke test that imports `main`.
- [ ] Implement a script that reads a Statcast CSV, extracts one row per game with coordinates/time columns when present, fetches Open-Meteo hourly weather, and writes a real weather cache.
- [ ] The script must fail loudly if coordinates or game times are unavailable.
- [ ] Verify: `pytest tests/test_weather_real.py -q`.
- [ ] Commit.

### Task 4: Weather Validation Runner

**Files:**
- Create: `scripts/run_weather_residual_validation.py`
- Modify: `tests/test_factorized.py`

- [ ] Add a CLI smoke test that imports `main`.
- [ ] Implement runner that takes `--data`, `--weather-cache`, `--output-dir`, `--pitcher-id`, and `--pitch-type`.
- [ ] Fit V2.1 baseline and V2.2 weather residual model on the same temporal split.
- [ ] Write JSON and Markdown reports comparing layer AUCs.
- [ ] Verify: `pytest tests/test_factorized.py tests/test_weather_real.py -q`.
- [ ] Commit.

### Task 5: Real Skubal Weather Evaluation

**Files:**
- Generated: `outputs/weather_residual_skubal_2025_ff/weather_residual_report.json`
- Generated: `outputs/weather_residual_skubal_2025_ff/weather_residual_summary.md`
- Modify: `README.md`
- Modify: `docs/presentation.md`

- [ ] Build or locate a real weather cache for Skubal 2025 games.
- [ ] Run the V2.2 weather residual validation.
- [ ] Compare against V2.1 `physics_core = 0.611`.
- [ ] Document whether weather improves, regresses, or remains inconclusive.
- [ ] Commit artifacts and docs.

### Task 6: Full Verification

- [ ] Run `python3 -m compileall src app scripts tests`.
- [ ] Run `ruff check src tests app scripts`.
- [ ] Run `pytest -q`.
- [ ] Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`.
- [ ] Commit final fixes if needed.
