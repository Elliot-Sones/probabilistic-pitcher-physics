# Overnight Success Criteria

The overnight run is successful only if it produces real artifacts.

## Required Artifacts

- `outputs/overnight/run_status.json`
- `outputs/overnight/morning_report.md`
- `outputs/overnight/random_split_report.json`
- `outputs/overnight/temporal_split_report.json`
- candidate ranking or blocker explaining why full ranking could not be run

## Minimum Real-Result Criteria

- Uses real Statcast rows.
- Reports data path, date range, row count, and feature count.
- Evaluates random-noise, league-average, player bootstrap, and player parametric baselines.
- Runs random split and temporal split validation with a held-out classifier split.
- Records the evaluation thresholds used by the report.
- Explains whether the current result supports the project.

## Strong-Result Criteria

- At least one selected candidate has `>= 600` real pitches for a pitch type.
- Holdout has `>= 150` real pitches.
- Player-specific model is much harder to detect than league-average baseline.
- Any artifact called successful has temporal detectability C2ST AUC `<= 0.60`.
- Context/fatigue/weather ablation improves temporal validation or explains leakage.

Failure is acceptable if it is honest. The morning report must say what failed, why, and what exact next data/model step is needed.
