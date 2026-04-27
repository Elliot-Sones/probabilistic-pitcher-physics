# Rolling Temporal Validation Board

- Pitcher: `Skubal, Tarik`
- Pitch type: `FF`
- Folds: `10`
- Repeats per fold: `4`
- Target AUC: `<=0.600`
- Target pass rate: `>=0.80`

## Primary Rolling Scoreboard

- Status: `rolling_diagnostic`
- Goals cleared: `0/3`
- Best fold physics-core AUC: `0.593`

| Metric | Current | Goal | Result | Gap |
|---|---:|---:|---|---:|
| Mean rolling physics-core AUC | 0.702 | <= 0.620 | miss | 0.082 |
| Target hit rate | 0.10 | >= 0.40 | miss | 0.300 |
| Worst fold physics-core AUC | 0.929 | < 0.800 | miss | 0.129 |

Rolling validation is the main scoreboard because it tests repeated future-game windows, not one favorable temporal split.

## Fold Results

| Fold | Train games | Test games | Train rows | Holdout rows | Best physics model | Physics AUC | Pass rate | Failures |
|---:|---|---|---:|---:|---|---:|---:|---:|
| 1 | 1-10 | 11-12 | 259 | 47 | factorized_release_state_anchored | 0.593 | 0.50 | 3 |
| 2 | 1-12 | 13-14 | 306 | 43 | factorized_short_memory_more_uncertain | 0.602 | 0.50 | 4 |
| 3 | 1-14 | 15-16 | 349 | 44 | factorized_release_state_anchored | 0.929 | 0.00 | 3 |
| 4 | 1-16 | 17-18 | 393 | 48 | factorized_trend_state_anchored | 0.639 | 0.50 | 2 |
| 5 | 1-18 | 19-20 | 441 | 57 | factorized_trend_state_anchored | 0.608 | 0.50 | 2 |
| 6 | 1-20 | 21-22 | 498 | 59 | context_neighbor_residual | 0.663 | 0.25 | 3 |
| 7 | 1-22 | 23-24 | 557 | 73 | factorized_trend_state_anchored | 0.725 | 0.00 | 2 |
| 8 | 1-24 | 25-26 | 630 | 63 | conditional_state_mixture_residual | 0.793 | 0.00 | 3 |
| 9 | 1-26 | 27-28 | 693 | 73 | factorized_trend_state_anchored | 0.765 | 0.00 | 3 |
| 10 | 1-28 | 29-30 | 766 | 39 | factorized_pitch_family_release_spin | 0.701 | 0.00 | 4 |

## Consistency

- Mean physics-core AUC: `0.702`
- Physics-core AUC range: `0.593` to `0.929`
- Target hit rate: `0.10`
- Mean pass rate: `0.23`

## Pitch-Type Failure Explainer

| Fold | Pitch | Layer | Model | AUC | Pass rate | Classifier signal | Failure mode |
|---:|---|---|---|---:|---:|---|---|
| 1 | FF | command_representation | factorized_pitch_family_release_spin | 0.574 | 0.75 | plate_z + plate_x | command/location |
| 1 | FF | release_only | factorized_trend_state_anchored | 0.631 | 0.50 | release_spin_rate + release_speed + release_pos_z + spin_axis_cos + release_pos_x | release/spin signature |
| 1 | FF | physics_core | factorized_release_state_anchored | 0.593 | 0.50 | release_pos_z + release_pos_x + spin_axis_sin + release_extension + az | release/spin signature |
| 2 | FF | command_representation | factorized_release_game_drift_gaussian | 0.570 | 0.50 | plate_z + plate_x | command/location |
| 2 | FF | release_only | factorized_physics_constrained_state | 0.558 | 0.75 | release_extension + release_pos_x + release_pos_z + spin_axis_cos + release_spin_rate | release/spin signature |
| 2 | FF | trajectory_only | factorized_short_memory_more_uncertain | 0.571 | 0.75 | vz0 + ay + az + ax + vx0 | trajectory/acceleration |
| 2 | FF | physics_core | factorized_short_memory_more_uncertain | 0.602 | 0.50 | release_spin_rate + release_extension + release_pos_x + vy0 + ay | release/spin signature |
| 3 | FF | command_representation | factorized_pitch_family_release_spin | 0.537 | 0.75 | plate_z + plate_x | command/location |
| 3 | FF | release_only | derived_joint_gaussian | 0.935 | 0.00 | release_pos_z + release_speed + release_pos_x + release_spin_rate + release_pos_y | release/spin signature |
| 3 | FF | physics_core | factorized_release_state_anchored | 0.929 | 0.00 | release_pos_z + release_pos_y + ay + release_extension + release_pos_x | release/spin signature |
| 4 | FF | release_only | factorized_release_recent_gaussian | 0.583 | 0.50 | spin_axis_cos + release_pos_z + release_extension + release_spin_rate + release_speed | release/spin signature |
| 4 | FF | physics_core | factorized_trend_state_anchored | 0.639 | 0.50 | release_pos_z + release_spin_rate + release_pos_x + release_pos_y + vy0 | release/spin signature |
| 5 | FF | release_only | factorized_trend_state_anchored | 0.608 | 0.25 | release_pos_z + release_speed + spin_axis_cos + spin_axis_sin + release_pos_x | release/spin signature |
| 5 | FF | physics_core | factorized_trend_state_anchored | 0.608 | 0.50 | release_pos_z + vy0 + release_speed + release_extension + pfx_z | acceleration/movement consistency |
| 6 | FF | release_only | context_neighbor_residual | 0.693 | 0.00 | release_pos_z + release_pos_x + release_spin_rate + release_speed + release_pos_y | release/spin signature |
| 6 | FF | trajectory_only | factorized_short_memory_wide_residual | 0.565 | 0.75 | vx0 + ax + vy0 + ay + az | trajectory/acceleration |
| 6 | FF | physics_core | context_neighbor_residual | 0.663 | 0.25 | release_pos_x + release_pos_z + release_spin_rate + az + pfx_z | acceleration/movement consistency |
| 7 | FF | release_only | factorized_release_recent_gaussian | 0.660 | 0.00 | spin_axis_cos + release_speed + release_pos_z + release_pos_y + release_pos_x | release/spin signature |
| 7 | FF | physics_core | factorized_trend_state_anchored | 0.725 | 0.00 | release_pos_z + az + ay + ax + release_spin_rate | release/spin signature |
| 8 | FF | movement_only | factorized_physics_constrained_state | 0.549 | 0.75 | pfx_x + pfx_z + plate_x + plate_z | movement |
| 8 | FF | release_only | context_neighbor_residual | 0.722 | 0.00 | release_extension + release_pos_y + spin_axis_cos + release_pos_x + release_spin_rate | release/spin signature |
| 8 | FF | physics_core | conditional_state_mixture_residual | 0.793 | 0.00 | release_spin_rate + release_extension + release_pos_y + plate_z + pfx_z | release/spin signature |
| 9 | FF | release_only | context_neighbor_residual | 0.781 | 0.00 | release_pos_z + release_pos_x + spin_axis_sin + release_speed + release_spin_rate | release/spin signature |
| 9 | FF | trajectory_only | factorized_pitch_family_release_spin | 0.590 | 0.50 | vy0 + ay + az + ax + vx0 | trajectory/acceleration |
| 9 | FF | physics_core | factorized_trend_state_anchored | 0.765 | 0.00 | release_pos_x + release_pos_z + release_spin_rate + az + ay | release/spin signature |
| 10 | FF | movement_only | factorized_short_memory_more_uncertain | 0.557 | 0.75 | pfx_x + pfx_z + plate_z + plate_x | movement |
| 10 | FF | release_only | factorized_recent_state_anchored | 0.774 | 0.00 | release_speed + release_extension + release_pos_z + release_pos_y + release_pos_x | release/spin signature |
| 10 | FF | trajectory_only | factorized_trend_state_anchored | 0.642 | 0.25 | vy0 + vz0 + vx0 + az + ax | trajectory/acceleration |
| 10 | FF | physics_core | factorized_pitch_family_release_spin | 0.701 | 0.00 | release_spin_rate + plate_z + vz0 + release_pos_x + release_pos_y | release/spin signature |
