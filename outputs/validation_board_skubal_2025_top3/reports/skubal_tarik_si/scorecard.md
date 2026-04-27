# Pitcher Twin Scorecard: Skubal, Tarik SI

## Verdict

- Artifact status: `physics_core_diagnostic`
- Best physics-core model: `pca_latent_residual`
- Physics-core C2ST AUC: `0.665`
- Physics-core pass rate: `0.00`
- Target: C2ST AUC <= `0.600` and pass rate >= `0.80`

C2ST AUC is the held-out classifier two-sample score. Lower is better; `0.50` means the classifier cannot reliably distinguish real held-out pitches from generated pitches.

## Data

- Pitcher id: `669373`
- Pitch count: `681`
- Games: `31`
- Temporal train rows: `476`
- Temporal holdout rows: `205`
- Repeats: `3`

## Layer Results

| Layer | Status | Best model | Mean AUC | Pass rate |
|---|---|---|---:|---:|
| command_representation | validated | factorized_short_memory_more_uncertain | 0.516 | 1.00 |
| movement_only | validated | factorized_release_state_anchored | 0.508 | 1.00 |
| release_only | candidate | derived_joint_gaussian | 0.589 | 0.67 |
| trajectory_only | candidate | conditional_state_mixture_residual | 0.570 | 0.67 |
| physics_core | diagnostic | pca_latent_residual | 0.665 | 0.00 |

## Main Classifier Clues

- `pfx_z` importance `0.395`
- `ay` importance `0.390`
- `az` importance `0.362`
- `release_spin_rate` importance `0.306`
- `vy0` importance `0.263`
