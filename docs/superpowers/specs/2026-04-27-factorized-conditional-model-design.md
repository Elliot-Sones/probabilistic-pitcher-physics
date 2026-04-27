# Factorized Conditional Pitch Model Design

## Purpose

Improve the generator by stopping the flat treatment of all context inputs and all pitch outputs. The model should respect the baseball/physics sequence:

```text
fatigue + recent form -> release / velocity / spin
release / spin -> movement
release + movement -> trajectory
intent/context + movement -> command / miss location
```

The goal is lower held-out C2ST detectability, especially for `physics_core`, while keeping layer confidence honest.

## Data Reality

The current Statcast CSV is one row per pitch. It does not contain dense frame-by-frame ball flight data. `vx0`, `vy0`, `vz0`, `ax`, `ay`, and `az` are fitted pitch-level trajectory parameters, not a time series.

Time-like columns available in the current data:

- `game_date`
- `game_pk`
- `inning`
- `at_bat_number`
- raw Statcast `pitch_number` within plate appearance
- engineered `pitcher_game_pitch_count`
- `n_thruorder_pitcher`
- `pitcher_days_since_prev_game`

Unavailable or empty in the current Skubal data:

- `sv_id`
- `tfs_zulu_deprecated`
- frame-level pitch tracking over seconds

## Model Structure

Create a new factorized generator with explicit stages:

1. **Release stage**
   - Targets: `release_speed`, `release_spin_rate`, `spin_axis_cos`, `spin_axis_sin`, `release_pos_x`, `release_pos_y`, `release_pos_z`, `release_extension`
   - Inputs: recent game state plus fatigue context only.
   - Fatigue context: `inning`, `pitcher_game_pitch_count`, `days_rest`, `times_through_order`.

2. **Movement stage**
   - Targets: `pfx_x`, `pfx_z`
   - Inputs: sampled release features.
   - No direct count/score/batter effects in v1.

3. **Trajectory stage**
   - Targets: `vx0`, `vy0`, `vz0`, `ax`, `ay`, `az`
   - Inputs: sampled release features plus sampled movement features.
   - This preserves physical coupling better than sampling trajectory independently.

4. **Command stage**
   - Targets: `plate_x`, `plate_z`
   - Inputs: intent context plus sampled movement/release speed.
   - Intent context: `balls`, `strikes`, `count_bucket_code`, `batter_stand_code`, `pitcher_score_diff`.

Each stage should use a regularized linear predictor plus residual noise as the first implementation. This is intentionally modest and testable. More complex nonlinear stages can come later after validation shows where linear structure fails.

## Validation

Add `factorized_conditional` to conditional validation reports and compare it against:

- `player_recent_weighted_game_drift_gaussian`
- `player_recent_weighted_game_drift_copula`
- `conditional_game_drift_copula`

Across:

- `command_representation`
- `movement_only`
- `release_only`
- `trajectory_only`
- `physics_core`

The expected first win is not guaranteed full-physics success. The success criterion for this implementation is that the factorized model can train, sample finite full-physics rows, and appear in the same layer-by-layer validation table so we can measure whether the staged structure improves AUC.

## Non-Goals

V1 does not:

- invent frame-level flight data;
- add weather;
- use deep learning;
- claim exact next-pitch prediction;
- force count/score context into release/movement/trajectory stages.
