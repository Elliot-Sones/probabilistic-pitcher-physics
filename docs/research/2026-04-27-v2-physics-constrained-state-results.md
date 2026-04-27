# V2.4 Physics-Constrained State Results

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
- Target: mean AUC `<= 0.600`, pass rate `>= 0.80`

## What Was Added

V2.4 added three extra tournament candidates:

1. `factorized_release_game_drift_copula`
   - Uses the existing recent game-drift copula release model inside the factorized generator.

2. `factorized_physics_constrained_state`
   - Starts from the short-memory factorized model.
   - Applies a release geometry constraint learned from training data:
     `release_pos_y + release_extension`.

3. `factorized_recent_state_anchored`
   - Starts from the short-memory factorized model.
   - Learns a recent game-state center from training games using recency-weighted game means.
   - Recenters generated pitch clouds toward that learned state with `alpha = 0.70`.
   - This uses training data only; it does not read holdout means.

## Physics-Core Result Across Seed Families

| Model | Seed 42 | Seed 101 | Seed 222 | 3-seed mean | Mean pass rate |
|---|---:|---:|---:|---:|---:|
| `factorized_short_memory_wide_residual` | 0.603 | 0.591 | 0.607 | 0.600 | 0.48 |
| `factorized_short_memory_more_uncertain` | 0.589 | 0.609 | 0.609 | 0.602 | 0.44 |
| `factorized_physics_constrained_state` | 0.597 | 0.587 | 0.599 | 0.594 | 0.56 |
| `factorized_recent_state_anchored` | 0.577 | 0.578 | 0.580 | 0.579 | 0.77 |

## Best Candidate Layer Means

`factorized_recent_state_anchored`:

| Layer | 3-seed mean AUC | Mean pass rate |
|---|---:|---:|
| command_representation | 0.546 | 0.94 |
| movement_only | 0.548 | 0.92 |
| release_only | 0.596 | 0.56 |
| trajectory_only | 0.544 | 0.94 |
| physics_core | 0.579 | 0.77 |

## Interpretation

This is a meaningful improvement over V2.3.

The previous best full-physics candidate, `factorized_short_memory_wide_residual`, averaged about `0.600` physics-core AUC across the three seed families. The new recent-state anchored candidate averaged `0.579`, and it was the best physics-core model in all three saved runs.

The model improved because it learned this structure:

```text
pitch distribution =
  recent pitcher state center
  + pitch-level variation
  + factorized release -> movement -> trajectory -> command dependencies
```

The important change is the recent-state anchor. Earlier models sampled a broad physical cloud, but a finite generated sample could wander away from the pitcher’s recent learned state. Anchoring keeps the generated cloud centered on the latest training-state estimate while preserving pitch-level variation.

## Caveat

This is not yet a full default model.

`factorized_recent_state_anchored` clears the physics-core mean AUC target and nearly clears the physics-core pass-rate target, but the strict default rule still fails because:

- physics-core mean pass rate across seed families is `0.77`, just below `0.80`
- release-only pass rate is `0.56`

Best status:

```json
{
  "artifact_status": "physics_core_candidate_not_default",
  "validated_feature_groups": [
    "command_representation",
    "movement_only",
    "trajectory_only"
  ],
  "candidate_feature_groups": [
    "physics_core"
  ],
  "diagnostic_feature_groups": [
    "release_only_pass_rate"
  ]
}
```

## Saved Artifacts

- `outputs/model_tournament_skubal_2025_ff_v2_4/model_tournament_report.json`
- `outputs/model_tournament_skubal_2025_ff_v2_4/model_tournament_summary.md`
- `outputs/model_tournament_skubal_2025_ff_v2_4_seed101/model_tournament_report.json`
- `outputs/model_tournament_skubal_2025_ff_v2_4_seed101/model_tournament_summary.md`
- `outputs/model_tournament_skubal_2025_ff_v2_4_seed222/model_tournament_report.json`
- `outputs/model_tournament_skubal_2025_ff_v2_4_seed222/model_tournament_summary.md`

## Next Move

The next improvement should focus on release-layer robustness, not trajectory or command. The remaining leakage is concentrated in release geometry and spin-axis fields. Best next candidates:

1. learn a separate recent-state anchor for release-only with its own alpha
2. add spin-axis circular residual modeling instead of raw `cos/sin` covariance
3. validate with both linear C2ST and a variance-sensitive validator so broad uncertainty does not hide bad physical calibration
