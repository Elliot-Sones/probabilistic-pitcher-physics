# Pitcher Twin Scorecard: Skubal, Tarik FF

## Verdict

- Artifact status: `validated_temporal_success`
- Best physics-core model: `factorized_recent_state_anchored`
- Physics-core C2ST AUC: `0.533`
- Physics-core pass rate: `1.00`
- Target: C2ST AUC <= `0.600` and pass rate >= `0.80`

C2ST AUC is the held-out classifier two-sample score. Lower is better; `0.50` means the classifier cannot reliably distinguish real held-out pitches from generated pitches.

## Model Route

- Route status: `validated`
- Pitch family: `rising_fastball`
- Recommended physics model: `factorized_recent_state_anchored`
- Validated layers: `command_representation`, `movement_only`, `trajectory_only`, `physics_core`
- Candidate layers: `release_only`
- Diagnostic layers: `none`

## Data

- Pitcher id: `669373`
- Pitch count: `835`
- Games: `31`
- Temporal train rows: `584`
- Temporal holdout rows: `251`
- Repeats: `3`

## Layer Results

| Layer | Status | Best model | Mean AUC | Pass rate |
|---|---|---|---:|---:|
| command_representation | validated | factorized_short_memory_more_uncertain | 0.506 | 1.00 |
| movement_only | validated | factorized_short_memory_more_uncertain | 0.523 | 1.00 |
| release_only | candidate | factorized_recent_state_anchored | 0.558 | 0.67 |
| trajectory_only | validated | factorized_pitch_family_release_spin | 0.511 | 1.00 |
| physics_core | validated | factorized_recent_state_anchored | 0.533 | 1.00 |

## Main Classifier Clues

- `release_pos_y` importance `0.468`
- `release_extension` importance `0.379`
- `spin_axis_cos` importance `0.336`
- `vy0` importance `0.249`
- `release_spin_rate` importance `0.104`
