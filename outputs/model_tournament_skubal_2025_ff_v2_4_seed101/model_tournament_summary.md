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
| command_representation | factorized_release_game_drift_copula | 0.534 | 0.534 | 0.527 | 0.547 | 0.547 | 0.542 | 0.541 | 0.539 | 0.543 | 0.529 | 0.548 |
| movement_only | factorized_recent_state_anchored | 0.547 | 0.548 | 0.543 | 0.576 | 0.543 | 0.555 | 0.541 | 0.543 | 0.589 | 0.579 | 0.605 |
| release_only | factorized_physics_constrained_state | 0.626 | 0.601 | 0.602 | 0.662 | 0.612 | 0.625 | 0.605 | 0.598 | 0.722 | 0.721 | 0.729 |
| trajectory_only | factorized_recent_state_anchored | 0.616 | 0.611 | 0.610 | 0.620 | 0.549 | 0.585 | 0.547 | 0.555 | 0.651 | 0.607 | 0.655 |
| physics_core | factorized_recent_state_anchored | 0.647 | 0.639 | 0.647 | 0.685 | 0.591 | 0.609 | 0.578 | 0.587 | 0.758 | 0.740 | 0.760 |

Lower C2ST AUC is better; `0.50` is ideal real-vs-generated indistinguishability.
A model becomes a candidate default only if the same model clears the mean AUC and pass-rate targets for every layer.
