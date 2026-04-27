# Pitcher Twin Scorecard: Taj Bradley FF

## Verdict

- Artifact status: `physics_core_diagnostic`
- Best physics-core model: `factorized_recent_state_anchored`
- Physics-core C2ST AUC: `0.649`
- Physics-core pass rate: `0.50`
- Target: C2ST AUC <= `0.600` and pass rate >= `0.80`

C2ST AUC is the held-out classifier two-sample score. Lower is better; `0.50` means the classifier cannot reliably distinguish real held-out pitches from generated pitches.

## Model Route

- Route status: `diagnostic`
- Pitch family: `rising_fastball`
- Recommended physics model: `factorized_recent_state_anchored`
- Validated layers: `command_representation`, `movement_only`, `trajectory_only`
- Candidate layers: `none`
- Diagnostic layers: `release_only`, `physics_core`

## Data

- Pitcher id: `671737`
- Pitch count: `279`
- Games: `6`
- Temporal train rows: `195`
- Temporal holdout rows: `84`
- Repeats: `2`

## Layer Results

| Layer | Status | Best model | Mean AUC | Pass rate |
|---|---|---|---:|---:|
| command_representation | validated | context_neighbor_residual | 0.511 | 1.00 |
| movement_only | validated | factorized_short_memory_more_uncertain | 0.511 | 1.00 |
| release_only | diagnostic | factorized_recent_state_anchored | 0.707 | 0.00 |
| trajectory_only | validated | factorized_short_memory_wide_residual | 0.510 | 1.00 |
| physics_core | diagnostic | factorized_recent_state_anchored | 0.649 | 0.50 |

## Main Classifier Clues

- `ay` importance `0.852`
- `release_pos_z` importance `0.776`
- `release_pos_x` importance `0.714`
- `spin_axis_sin` importance `0.600`
- `release_pos_y` importance `0.526`
