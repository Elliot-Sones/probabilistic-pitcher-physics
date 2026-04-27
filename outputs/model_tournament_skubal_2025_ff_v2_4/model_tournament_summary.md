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
| command_representation | pca_latent_residual | 0.542 | 0.541 | 0.533 | 0.546 | 0.544 | 0.548 | 0.550 | 0.543 | 0.533 | 0.537 | 0.538 |
| movement_only | factorized_short_memory_wide_residual | 0.543 | 0.549 | 0.545 | 0.559 | 0.541 | 0.542 | 0.543 | 0.550 | 0.603 | 0.567 | 0.610 |
| release_only | factorized_physics_constrained_state | 0.618 | 0.617 | 0.615 | 0.674 | 0.613 | 0.601 | 0.593 | 0.587 | 0.711 | 0.725 | 0.712 |
| trajectory_only | factorized_recent_state_anchored | 0.612 | 0.618 | 0.632 | 0.619 | 0.562 | 0.570 | 0.544 | 0.559 | 0.645 | 0.610 | 0.657 |
| physics_core | factorized_recent_state_anchored | 0.634 | 0.664 | 0.654 | 0.680 | 0.603 | 0.589 | 0.577 | 0.597 | 0.744 | 0.748 | 0.747 |

Lower C2ST AUC is better; `0.50` is ideal real-vs-generated indistinguishability.
A model becomes a candidate default only if the same model clears the mean AUC and pass-rate targets for every layer.
