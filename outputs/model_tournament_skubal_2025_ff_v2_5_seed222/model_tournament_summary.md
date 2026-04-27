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
| command_representation | pca_latent_residual | 0.537 | 0.534 | 0.538 | 0.534 | 0.547 | 0.540 | 0.546 | 0.541 | 0.542 | 0.544 | 0.532 | 0.534 | 0.544 |
| movement_only | factorized_v2_1 | 0.533 | 0.544 | 0.544 | 0.571 | 0.539 | 0.544 | 0.560 | 0.548 | 0.558 | 0.552 | 0.604 | 0.565 | 0.591 |
| release_only | factorized_recent_state_anchored | 0.624 | 0.614 | 0.616 | 0.653 | 0.618 | 0.615 | 0.589 | 0.602 | 0.674 | 0.622 | 0.712 | 0.724 | 0.715 |
| trajectory_only | factorized_trend_state_anchored | 0.607 | 0.621 | 0.614 | 0.624 | 0.560 | 0.556 | 0.540 | 0.537 | 0.541 | 0.567 | 0.641 | 0.611 | 0.649 |
| physics_core | factorized_trend_state_anchored | 0.644 | 0.648 | 0.653 | 0.673 | 0.607 | 0.609 | 0.580 | 0.574 | 0.656 | 0.606 | 0.759 | 0.743 | 0.763 |

Lower C2ST AUC is better; `0.50` is ideal real-vs-generated indistinguishability.
A model becomes a candidate default only if the same model clears the mean AUC and pass-rate targets for every layer.
