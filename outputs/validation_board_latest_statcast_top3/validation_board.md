# Pitcher Twin Validation Board

This board runs the same temporal model-tournament protocol across selected real pitcher/pitch candidates.

| Rank | Pitcher | Pitch | Games | Holdout | Best physics model | Physics AUC | Pass rate | Status |
|---:|---|---|---:|---:|---|---:|---:|---|
| 1 | Isaac Mattson | FF | 14 | 63 | factorized_short_memory_more_uncertain | 0.576 | 0.50 | physics_core_candidate |
| 2 | Freddy Peralta | FF | 6 | 88 | factorized_short_memory_wide_residual | 0.621 | 0.00 | physics_core_diagnostic |
| 3 | Taj Bradley | FF | 6 | 84 | factorized_recent_state_anchored | 0.649 | 0.50 | physics_core_diagnostic |

## Interpretation

- `physics_core_diagnostic`: `2` candidate(s)
- `physics_core_candidate`: `1` candidate(s)

Use this board to separate one-off success from repeatable pitcher-twin generalization.
