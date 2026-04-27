# Pitcher Twin Scorecard: Skubal, Tarik SI

## Verdict

- Artifact status: `physics_core_diagnostic`
- Best physics-core model: `conditional_state_mixture_residual`
- Physics-core C2ST AUC: `0.644`
- Physics-core pass rate: `0.33`
- Target: C2ST AUC <= `0.600` and pass rate >= `0.80`

C2ST AUC is the held-out classifier two-sample score. Lower is better; `0.50` means the classifier cannot reliably distinguish real held-out pitches from generated pitches.

## Model Route

- Route status: `diagnostic`
- Pitch family: `sinker`
- Recommended physics model: `conditional_state_mixture_residual`
- Validated layers: `command_representation`, `movement_only`
- Candidate layers: `trajectory_only`
- Diagnostic layers: `release_only`, `physics_core`

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
| release_only | diagnostic | context_neighbor_residual | 0.642 | 0.00 |
| trajectory_only | candidate | conditional_state_mixture_residual | 0.590 | 0.33 |
| physics_core | diagnostic | conditional_state_mixture_residual | 0.644 | 0.33 |

## Main Classifier Clues

- `spin_axis_sin` importance `0.961`
- `release_spin_rate` importance `0.458`
- `pfx_z` importance `0.366`
- `spin_axis_cos` importance `0.362`
- `az` importance `0.217`
