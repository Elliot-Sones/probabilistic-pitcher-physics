# V2.5 Release-State Results

## Setup

- Data: `/Users/elliot18/Desktop/Home/Projects/trajekt-pitcher-twin/data/processed/skubal_2025.csv`
- Pitcher: `Skubal, Tarik`
- Pitch type: `FF`
- Rows: `835` Skubal FF pitches
- Split: temporal `70/30`
- Train rows: `584`
- Holdout rows: `251`
- Metric: C2ST AUC, lower is better, `0.50` is ideal
- Repeats: `30` per random-state family
- Seed families: `42`, `101`, `222`
- Target: mean AUC `<= 0.600`, pass rate `>= 0.80`

## What Was Added

V2.5 added three new diagnostics/candidates:

1. `factorized_trend_state_anchored`
   - Starts from the V2.4 short-memory factorized model.
   - Fits a bounded recent per-game trend over training games.
   - Extrapolates one game ahead with shrinkage.
   - Recenters generated physics with `alpha = 0.85`.

2. `factorized_release_state_anchored`
   - Applies an extra release-only anchor after the V2.4 state anchor.
   - Separately anchors `release_speed`, `release_spin_rate`, release position, and extension.
   - Applies a circular spin-axis angle anchor.
   - Result: diagnostic only; it over-anchors spin and hurts realism.

3. Circular spin-axis helpers
   - `fit_spin_axis_angle_anchor()`
   - `apply_spin_axis_angle_anchor()`
   - These prove the mechanics work, but the real-data tournament shows this particular anchor is too aggressive as a default.

## Physics-Core Result Across Seed Families

| Model | Seed 42 | Seed 101 | Seed 222 | 3-seed mean | Mean pass rate |
|---|---:|---:|---:|---:|---:|
| `factorized_short_memory_wide_residual` | 0.603 | 0.591 | 0.607 | 0.600 | 0.48 |
| `factorized_recent_state_anchored` | 0.577 | 0.578 | 0.580 | 0.579 | 0.77 |
| `factorized_trend_state_anchored` | 0.575 | 0.566 | 0.574 | 0.572 | 0.79 |
| `factorized_release_state_anchored` | 0.657 | 0.659 | 0.656 | 0.657 | 0.04 |
| `factorized_physics_constrained_state` | 0.619 | 0.602 | 0.606 | 0.609 | 0.43 |

## Layer Result For Best V2.5 Candidate

`factorized_trend_state_anchored`:

| Layer | 3-seed mean AUC | Mean pass rate |
|---|---:|---:|
| command_representation | 0.545 | 0.94 |
| movement_only | 0.550 | 0.90 |
| release_only | 0.591 | 0.56 |
| trajectory_only | 0.542 | 0.94 |
| physics_core | 0.572 | 0.79 |

## Interpretation

V2.5 improves full-physics realism again, but still does not justify calling the full model validated/default.

The key improvement is the trend-state anchor. V2.4 used a recency-weighted state center, which assumes the pitcher’s latest training-window state is the best future estimate. V2.5 adds a bounded trend estimate:

```text
future pitch state =
  recent game-state level
  + shrinked short-horizon game trend
  + pitch-level variation
```

That helps because Skubal’s holdout fastballs continue drifting after the training window. The trend anchor moves the generated cloud closer to that future state without using holdout means.

The release-specific spin anchor did not help. It made the classifier focus strongly on `spin_axis_cos` and `spin_axis_sin`, so it should remain diagnostic. The lesson is useful: spin axis should be modeled as circular, but not collapsed toward a single recent mean.

## Status

Best status:

```json
{
  "artifact_status": "improved_physics_core_candidate_not_default",
  "best_candidate": "factorized_trend_state_anchored",
  "validated_feature_groups": [
    "command_representation",
    "movement_only",
    "trajectory_only"
  ],
  "candidate_feature_groups": [
    "physics_core"
  ],
  "diagnostic_feature_groups": [
    "release_only_pass_rate",
    "spin_axis_over_anchor"
  ]
}
```

The physics-core mean AUC improved from V2.4 `0.579` to V2.5 `0.572`. Physics-core mean pass rate improved from `0.77` to `0.79`, just under the strict `0.80` target.

## Saved Artifacts

- `outputs/model_tournament_skubal_2025_ff_v2_5/model_tournament_report.json`
- `outputs/model_tournament_skubal_2025_ff_v2_5/model_tournament_summary.md`
- `outputs/model_tournament_skubal_2025_ff_v2_5_seed101/model_tournament_report.json`
- `outputs/model_tournament_skubal_2025_ff_v2_5_seed101/model_tournament_summary.md`
- `outputs/model_tournament_skubal_2025_ff_v2_5_seed222/model_tournament_report.json`
- `outputs/model_tournament_skubal_2025_ff_v2_5_seed222/model_tournament_summary.md`

## Next Move

The next improvement should avoid anchoring spin axis to one mean. Better next candidates:

1. sample spin-axis residuals from recent empirical angular residuals instead of mean-rotating them
2. model release geometry jointly as `release_pos_y + release_extension` plus extension residual
3. add rolling temporal validation so the trend horizon is learned/evaluated across multiple future windows
