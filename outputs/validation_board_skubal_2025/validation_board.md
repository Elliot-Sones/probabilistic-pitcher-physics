# Pitcher Twin Validation Board

This board runs the same temporal model-tournament protocol across selected real pitcher/pitch candidates.

| Rank | Pitcher | Pitch | Games | Holdout | Best physics model | Physics AUC | Pass rate | Status |
|---:|---|---|---:|---:|---|---:|---:|---|
| 1 | Skubal, Tarik | CH | 31 | 269 | factorized_trend_state_anchored | 0.680 | 0.00 | physics_core_diagnostic |

## Interpretation

- `physics_core_diagnostic`: `1` candidate(s)

Use this board to separate one-off success from repeatable pitcher-twin generalization.
