# Real Data Smoke Test

Date run: 2026-04-27

This check uses real public Statcast rows only. No mock rows, fake players, or fabricated weather/context data were used.

## Data Used

- Source file: `/Users/elliot18/assistant/projects/trajekt-scout/data/processed/latest_statcast.csv`
- Rows after physical-feature cleaning: `115,616`
- Date range: `2026-03-26` to `2026-04-24`
- Feature set: 18 physical pitch features

Features:

- velocity and spin: `release_speed`, `release_spin_rate`, spin-axis sine/cosine
- release: `release_pos_x`, `release_pos_y`, `release_pos_z`, `release_extension`
- movement and target: `pfx_x`, `pfx_z`, `plate_x`, `plate_z`
- trajectory: `vx0`, `vy0`, `vz0`, `ax`, `ay`, `az`

## Best One-Month Candidates

The current cache is only about one month of 2026 data, so no pitcher/pitch pair reaches the full project target of 600+ pitches. It is enough for a smoke test.

| Pitcher | Pitch | Real pitches | Games |
|---|---:|---:|---:|
| Freddy Peralta | FF | 292 | 6 |
| Jacob Misiorowski | FF | 283 | 5 |
| Taj Bradley | FF | 279 | 6 |
| Framber Valdez | SI | 258 | 6 |
| Andrew Abbott | FF | 256 | 6 |
| Cristopher Sánchez | SI | 254 | 6 |
| Chase Burns | FF | 244 | 5 |
| Bubba Chandler | FF | 238 | 5 |

## Validation Method

The test uses a classifier two-sample test.

- Real class: held-out real Statcast pitches.
- Simulated class: pitches sampled from a candidate model.
- Classifier split: repeated balanced holdout; the classifier is trained on one split and scored on held-out classifier rows.
- Metric: folded detectability ROC-AUC.
- Interpretation: `0.50` means the classifier cannot tell real from simulated. Higher values mean the generated pitches are easier to detect as fake or mismatched.
- Evaluation thresholds: at least `100` train rows and `50` holdout rows per evaluated candidate; temporal success target is `<= 0.60`.

Models tested:

- `random_independent_noise`: independent noise using the player mean/std.
- `league_same_pitch_empirical`: real same-pitch-type rows from other pitchers.
- `player_empirical_bootstrap`: resampled real rows from the same pitcher/pitch type.
- `player_multivariate_gaussian`: one player-specific Gaussian with full covariance.

## Result 1: Random Split Sanity Check

Random split asks: "Can player-specific distributions be modeled at all inside the same real data window?"

| Model | Mean detectability AUC |
|---|---:|
| random independent noise | 0.556 |
| league same-pitch empirical | 0.976 |
| player empirical bootstrap | 0.568 |
| player multivariate Gaussian | 0.557 |

This is promising. Player-specific models are close to indistinguishable from real rows under a random split, while league-average rows are obviously wrong. That means the pitcher-specific variability signal is real.

## Result 2: Temporal Holdout

Temporal split asks: "If trained on earlier appearances, does the model match later appearances?"

| Model | Mean detectability AUC |
|---|---:|
| random independent noise | 0.871 |
| league same-pitch empirical | 0.982 |
| player empirical bootstrap | 0.847 |
| player multivariate Gaussian | 0.855 |

This is not good enough yet. A static one-month model does not generalize cleanly from earlier games to later games.

## Conclusion

The project is viable, but the simple static GMM is not the project.

The interesting project is a conditional pitcher-variability model:

- player-specific base distribution
- count/context conditioning
- fatigue and within-game drift
- weather/stadium context, once joined from a real external source
- temporal validation against future games

The strongest Trajekt story is:

> Real pitchers do not throw one centroid. They throw distributions that shift across count, fatigue, day, environment, and mechanics. A static distribution gets caught by the validator; the product opportunity is a realistic mode that learns and samples those conditional shifts.

## Files

- Script: `scripts/real_data_smoke_test.py`
- Temporal report: `outputs/overnight/temporal_split_report.json`
- Random report: `outputs/overnight/random_split_report.json`
- Tracked morning report: `docs/assets/overnight_morning_report.md`
