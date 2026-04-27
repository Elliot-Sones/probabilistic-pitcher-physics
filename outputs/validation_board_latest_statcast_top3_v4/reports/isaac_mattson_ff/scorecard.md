# Pitcher Twin Scorecard: Isaac Mattson FF

## Verdict

- Artifact status: `physics_core_candidate`
- Best physics-core model: `factorized_short_memory_more_uncertain`
- Physics-core C2ST AUC: `0.576`
- Physics-core pass rate: `0.50`
- Target: C2ST AUC <= `0.600` and pass rate >= `0.80`

C2ST AUC is the held-out classifier two-sample score. Lower is better; `0.50` means the classifier cannot reliably distinguish real held-out pitches from generated pitches.

## Model Route

- Route status: `candidate`
- Pitch family: `rising_fastball`
- Recommended physics model: `factorized_short_memory_more_uncertain`
- Validated layers: `command_representation`, `movement_only`
- Candidate layers: `release_only`, `trajectory_only`, `physics_core`
- Diagnostic layers: `none`

## Data

- Pitcher id: `676755`
- Pitch count: `207`
- Games: `14`
- Temporal train rows: `144`
- Temporal holdout rows: `63`
- Repeats: `2`

## Layer Results

| Layer | Status | Best model | Mean AUC | Pass rate |
|---|---|---|---:|---:|
| command_representation | validated | derived_joint_gaussian | 0.521 | 1.00 |
| movement_only | validated | factorized_trend_state_anchored | 0.553 | 1.00 |
| release_only | candidate | factorized_short_memory_wide_residual | 0.586 | 0.50 |
| trajectory_only | candidate | factorized_recent_state_anchored | 0.583 | 0.50 |
| physics_core | candidate | factorized_short_memory_more_uncertain | 0.576 | 0.50 |

## Main Classifier Clues

- `release_spin_rate` importance `0.801`
- `spin_axis_cos` importance `0.495`
- `pfx_z` importance `0.408`
- `spin_axis_sin` importance `0.344`
- `az` importance `0.343`
