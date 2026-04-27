# Pitcher Twin Overnight Report

This report summarizes real-data validation outputs. It does not include mock data.

## Data

- Data path: `/Users/elliot18/assistant/projects/trajekt-scout/data/processed/latest_statcast.csv`
- Rows after cleaning: `115616`
- Date range: `2026-03-26` to `2026-04-24`
- Feature count: `18`
- Min train rows: `100`
- Min holdout rows: `50`
- Temporal success target: `<= 0.6`
- Classifier split: `repeated_balanced_holdout`

## Candidate Snapshot

| Pitcher | Pitch | Real pitches | Games | Train | Holdout |
|---|---:|---:|---:|---:|---:|
| Freddy Peralta | FF | 292 | 6 | 204 | 88 |
| Jacob Misiorowski | FF | 283 | 5 | 198 | 85 |
| Taj Bradley | FF | 279 | 6 | 195 | 84 |
| Framber Valdez | SI | 258 | 6 | 180 | 78 |
| Andrew Abbott | FF | 256 | 6 | 179 | 77 |
| Cristopher Sánchez | SI | 254 | 6 | 177 | 77 |
| Chase Burns | FF | 244 | 5 | 170 | 74 |
| Bubba Chandler | FF | 238 | 5 | 166 | 72 |

## Random Split

| Model | Mean detectability C2ST AUC |
|---|---:|
| league same-pitch empirical | 0.976 |
| player empirical bootstrap | 0.568 |
| player multivariate Gaussian | 0.557 |
| random independent noise | 0.556 |

## Temporal Split

| Model | Mean detectability C2ST AUC |
|---|---:|
| league same-pitch empirical | 0.982 |
| player empirical bootstrap | 0.847 |
| player multivariate Gaussian | 0.855 |
| random independent noise | 0.871 |

## Conclusion

The random split supports the project: player-specific generated pitches are hard to distinguish from held-out real pitches (`detectability AUC=0.557`).
The league-average baseline is clearly detectable, which supports the claim that pitcher-specific structure matters (`detectability AUC=0.976`).
The temporal split is the current weakness: a static model trained on earlier games does not match later games (`detectability AUC=0.855`). The next work should model count, fatigue, game-level drift, and real weather/context.

## Run Status

- OK: `True`
- Started: `2026-04-27T01:27:22.864308+00:00`
- Finished: `2026-04-27T01:27:30.411485+00:00`
