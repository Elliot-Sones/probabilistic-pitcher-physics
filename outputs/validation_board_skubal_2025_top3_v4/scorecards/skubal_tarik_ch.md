# Pitcher Twin Scorecard: Skubal, Tarik CH

## Verdict

- Artifact status: `physics_core_diagnostic`
- Best physics-core model: `factorized_trend_state_anchored`
- Physics-core C2ST AUC: `0.680`
- Physics-core pass rate: `0.00`
- Target: C2ST AUC <= `0.600` and pass rate >= `0.80`

C2ST AUC is the held-out classifier two-sample score. Lower is better; `0.50` means the classifier cannot reliably distinguish real held-out pitches from generated pitches.

## Model Route

- Route status: `diagnostic`
- Pitch family: `changeup`
- Recommended physics model: `factorized_trend_state_anchored`
- Validated layers: `command_representation`, `movement_only`, `trajectory_only`
- Candidate layers: `none`
- Diagnostic layers: `release_only`, `physics_core`

## Data

- Pitcher id: `669373`
- Pitch count: `895`
- Games: `31`
- Temporal train rows: `626`
- Temporal holdout rows: `269`
- Repeats: `3`

## Layer Results

| Layer | Status | Best model | Mean AUC | Pass rate |
|---|---|---|---:|---:|
| command_representation | validated | factorized_release_state_anchored | 0.518 | 1.00 |
| movement_only | validated | factorized_trend_state_anchored | 0.506 | 1.00 |
| release_only | diagnostic | pca_latent_residual | 0.663 | 0.00 |
| trajectory_only | validated | factorized_physics_constrained_state | 0.539 | 1.00 |
| physics_core | diagnostic | factorized_trend_state_anchored | 0.680 | 0.00 |

## Main Classifier Clues

- `spin_axis_sin` importance `0.682`
- `release_pos_y` importance `0.423`
- `spin_axis_cos` importance `0.400`
- `release_extension` importance `0.250`
- `release_pos_x` importance `0.195`
