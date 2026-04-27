# Model Tournament

- Pitcher: `Skubal, Tarik`
- Pitch type: `FF`
- Train rows: `584`
- Holdout rows: `251`
- Repeats: `30`
- Target AUC: `<=0.600`
- Target pass rate: `>=0.80`
- Candidate default: `False`
- Best physics-core model: `factorized_trend_state_anchored`

| Layer | Best model | factorized_v2_1 | factorized_release_game_drift_gaussian | factorized_release_game_drift_copula | factorized_release_recent_gaussian | factorized_short_memory_wide_residual | factorized_short_memory_more_uncertain | factorized_recent_state_anchored | factorized_trend_state_anchored | factorized_release_state_anchored | factorized_physics_constrained_state | pca_latent_residual | context_neighbor_residual | derived_joint_gaussian |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| command_representation | factorized_release_game_drift_copula | 0.534 | 0.534 | 0.527 | 0.547 | 0.547 | 0.542 | 0.541 | 0.541 | 0.547 | 0.543 | 0.556 | 0.530 | 0.536 |
| movement_only | factorized_physics_constrained_state | 0.547 | 0.548 | 0.543 | 0.576 | 0.543 | 0.555 | 0.541 | 0.552 | 0.559 | 0.540 | 0.588 | 0.572 | 0.590 |
| release_only | factorized_trend_state_anchored | 0.626 | 0.601 | 0.602 | 0.662 | 0.612 | 0.625 | 0.605 | 0.591 | 0.677 | 0.610 | 0.725 | 0.714 | 0.730 |
| trajectory_only | factorized_trend_state_anchored | 0.616 | 0.611 | 0.610 | 0.620 | 0.549 | 0.585 | 0.547 | 0.544 | 0.548 | 0.559 | 0.648 | 0.619 | 0.639 |
| physics_core | factorized_trend_state_anchored | 0.647 | 0.639 | 0.647 | 0.685 | 0.591 | 0.609 | 0.578 | 0.566 | 0.659 | 0.602 | 0.751 | 0.741 | 0.745 |

Lower C2ST AUC is better; `0.50` is ideal real-vs-generated indistinguishability.
A model becomes a candidate default only if the same model clears the mean AUC and pass-rate targets for every layer.
