# Model Tournament

- Pitcher: `Skubal, Tarik`
- Pitch type: `FF`
- Train rows: `584`
- Holdout rows: `251`
- Repeats: `30`
- Target AUC: `<=0.600`
- Target pass rate: `>=0.80`
- Candidate default: `False`
- Best physics-core model: `factorized_recent_state_anchored`

| Layer | Best model | factorized_v2_1 | factorized_release_game_drift_gaussian | factorized_release_game_drift_copula | factorized_release_recent_gaussian | factorized_short_memory_wide_residual | factorized_short_memory_more_uncertain | factorized_recent_state_anchored | factorized_physics_constrained_state | pca_latent_residual | context_neighbor_residual | derived_joint_gaussian |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| command_representation | context_neighbor_residual | 0.537 | 0.534 | 0.538 | 0.534 | 0.547 | 0.540 | 0.546 | 0.539 | 0.539 | 0.529 | 0.531 |
| movement_only | factorized_v2_1 | 0.533 | 0.544 | 0.544 | 0.571 | 0.539 | 0.544 | 0.560 | 0.539 | 0.598 | 0.563 | 0.589 |
| release_only | factorized_recent_state_anchored | 0.624 | 0.614 | 0.616 | 0.653 | 0.618 | 0.615 | 0.589 | 0.615 | 0.720 | 0.723 | 0.703 |
| trajectory_only | factorized_recent_state_anchored | 0.607 | 0.621 | 0.614 | 0.624 | 0.560 | 0.556 | 0.540 | 0.562 | 0.649 | 0.617 | 0.629 |
| physics_core | factorized_recent_state_anchored | 0.644 | 0.648 | 0.653 | 0.673 | 0.607 | 0.609 | 0.580 | 0.599 | 0.759 | 0.728 | 0.753 |

Lower C2ST AUC is better; `0.50` is ideal real-vs-generated indistinguishability.
A model becomes a candidate default only if the same model clears the mean AUC and pass-rate targets for every layer.
