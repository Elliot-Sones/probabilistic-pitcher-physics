# V3 Generalized Validation Board Results

## Why This Was Added

The model had a strong Skubal FF story, but one validated pitch type is not enough evidence for a serious project. V3 adds a validation board that runs the same temporal tournament protocol across pitcher/pitch candidates and renders scorecards for each candidate.

This changes the question from:

```text
Can we make one Skubal fastball run look good?
```

to:

```text
Which pitcher/pitch pairs validate, which fail, and what do the failures teach us?
```

## Implementation

New files:

- `src/pitcher_twin/validation_board.py`
- `scripts/run_validation_board.py`
- `tests/test_validation_board.py`
- `outputs/validation_board_skubal_2025_top3/`
- `outputs/validation_board_latest_statcast_top3/`

The board does four things:

1. selects real pitcher/pitch candidates by pitch volume, game count, holdout size, and physics-core feature completeness
2. runs the existing model tournament for each candidate
3. summarizes the best model per layer with AUC and pass rate
4. writes leaderboard CSV/JSON, markdown scorecards, and optional rolling temporal windows

The Streamlit app now has a `Validation Board` tab that reads these generated artifacts.

## Repro Commands

Use the project virtualenv if system `python3` cannot import `sklearn`.

```bash
/Users/elliot18/Desktop/Home/Projects/trajekt-pitcher-twin/.venv/bin/python \
  scripts/run_validation_board.py \
  --data /Users/elliot18/Desktop/Home/Projects/trajekt-pitcher-twin/data/processed/skubal_2025.csv \
  --output-dir outputs/validation_board_skubal_2025_top3 \
  --top 3 \
  --repeats 3 \
  --samples 260
```

```bash
/Users/elliot18/Desktop/Home/Projects/trajekt-pitcher-twin/.venv/bin/python \
  scripts/run_validation_board.py \
  --data /Users/elliot18/assistant/projects/trajekt-scout/data/processed/latest_statcast.csv \
  --output-dir outputs/validation_board_latest_statcast_top3 \
  --top 3 \
  --min-pitches 200 \
  --min-games 4 \
  --min-holdout 40 \
  --repeats 2 \
  --samples 220
```

## Skubal 2025 Top Pitch Types

Input: real Skubal 2025 Baseball Savant CSV.

| Pitch | Pitches | Games | Holdout | Best physics model | Physics AUC | Pass rate | Status |
|---|---:|---:|---:|---|---:|---:|---|
| FF | 835 | 31 | 251 | `factorized_recent_state_anchored` | 0.533 | 1.00 | `validated_temporal_success` |
| SI | 681 | 31 | 205 | `pca_latent_residual` | 0.665 | 0.00 | `physics_core_diagnostic` |
| CH | 895 | 31 | 269 | `factorized_trend_state_anchored` | 0.680 | 0.00 | `physics_core_diagnostic` |

Layer details:

- Skubal FF validates command, movement, trajectory, and physics core in this 3-repeat board.
- Skubal FF release is still only `candidate`: AUC `0.558`, pass rate `0.67`.
- Skubal SI and CH validate command/movement pieces, but full physics fails because release/spin/trajectory relationships are still distinguishable.

Most useful classifier clues:

- FF: `release_pos_y`, `release_extension`, `spin_axis_cos`, `vy0`, `release_spin_rate`
- SI: `pfx_z`, `ay`, `az`, `release_spin_rate`, `vy0`
- CH: `spin_axis_sin`, `release_pos_y`, `spin_axis_cos`, `release_extension`, `release_pos_x`

## Latest Statcast Partial-Window Board

Input: `/Users/elliot18/assistant/projects/trajekt-scout/data/processed/latest_statcast.csv`

Strict full-season thresholds found no candidates because this cache is a partial window. With real-data thresholds of `>=200` pitches, `>=4` games, and `>=40` holdout rows:

| Pitcher | Pitch | Pitches | Games | Holdout | Best physics model | Physics AUC | Pass rate | Status |
|---|---|---:|---:|---:|---|---:|---:|---|
| Isaac Mattson | FF | 207 | 14 | 63 | `factorized_short_memory_more_uncertain` | 0.576 | 0.50 | `physics_core_candidate` |
| Freddy Peralta | FF | 292 | 6 | 88 | `factorized_short_memory_wide_residual` | 0.621 | 0.00 | `physics_core_diagnostic` |
| Taj Bradley | FF | 279 | 6 | 84 | `factorized_recent_state_anchored` | 0.649 | 0.50 | `physics_core_diagnostic` |

This is not as strong as Skubal because the cache is smaller by pitcher and season context. It is still useful because it shows the validation board can run across multiple pitchers and does not overclaim success.

## Interpretation

The project is now more serious because it can say where the model works and where it fails:

- Skubal FF is a strong, presentable validated pitch-distribution case.
- The same framework does not automatically solve Skubal SI/CH or partial-window FF candidates.
- Command and movement are usually easier to match than release/full joint physics.
- The recurring failure signals are spin axis, release geometry, acceleration, and release speed/spin relationships.

The next genuinely interesting modeling move is not another global average model. It is pitch-family-aware release and spin modeling:

```text
pitch family intent
  -> release geometry state
  -> circular spin-axis residual
  -> velocity/spin covariance
  -> movement and trajectory residual
  -> command cloud
```

## Saved Artifacts

- `outputs/validation_board_skubal_2025_top3/leaderboard.csv`
- `outputs/validation_board_skubal_2025_top3/validation_board.md`
- `outputs/validation_board_skubal_2025_top3/scorecards/skubal_tarik_ff.md`
- `outputs/validation_board_skubal_2025_top3/scorecards/skubal_tarik_si.md`
- `outputs/validation_board_skubal_2025_top3/scorecards/skubal_tarik_ch.md`
- `outputs/validation_board_latest_statcast_top3/leaderboard.csv`
- `outputs/validation_board_latest_statcast_top3/validation_board.md`
- `outputs/validation_board_latest_statcast_top3/scorecards/isaac_mattson_ff.md`
- `outputs/validation_board_latest_statcast_top3/scorecards/freddy_peralta_ff.md`
- `outputs/validation_board_latest_statcast_top3/scorecards/taj_bradley_ff.md`
