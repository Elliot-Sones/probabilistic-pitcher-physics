# Pitcher Twin Scorecard: Freddy Peralta FF

## Verdict

- Artifact status: `physics_core_diagnostic`
- Best physics-core model: `factorized_short_memory_wide_residual`
- Physics-core C2ST AUC: `0.621`
- Physics-core pass rate: `0.00`
- Target: C2ST AUC <= `0.600` and pass rate >= `0.80`

C2ST AUC is the held-out classifier two-sample score. Lower is better; `0.50` means the classifier cannot reliably distinguish real held-out pitches from generated pitches.

## Data

- Pitcher id: `642547`
- Pitch count: `292`
- Games: `6`
- Temporal train rows: `204`
- Temporal holdout rows: `88`
- Repeats: `2`

## Layer Results

| Layer | Status | Best model | Mean AUC | Pass rate |
|---|---|---|---:|---:|
| command_representation | validated | factorized_v2_1 | 0.508 | 1.00 |
| movement_only | validated | factorized_trend_state_anchored | 0.545 | 1.00 |
| release_only | diagnostic | factorized_short_memory_more_uncertain | 0.686 | 0.00 |
| trajectory_only | validated | factorized_short_memory_wide_residual | 0.540 | 1.00 |
| physics_core | diagnostic | factorized_short_memory_wide_residual | 0.621 | 0.00 |

## Main Classifier Clues

- `release_pos_x` importance `0.975`
- `vy0` importance `0.649`
- `ay` importance `0.542`
- `release_pos_z` importance `0.413`
- `spin_axis_cos` importance `0.349`
