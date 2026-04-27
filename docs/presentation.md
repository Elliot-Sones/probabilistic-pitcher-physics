# Pitcher Twin Presentation

## One-Line Story

Pitcher Twin learns a real pitcher's pitch-to-pitch variability from Statcast, validates generated pitches against held-out real pitches, and exports a Trajekt-shaped session JSON.

## Current Demo

| Field | Value |
|---|---:|
| Pitcher | Tarik Skubal |
| Season | 2025 |
| Pitch | FF |
| Real FF pitches | 835 |
| Games | 31 |
| Temporal holdout rows | 251 |
| Selected model | factorized recent-state anchored |
| Selected layer | full physics, Skubal FF board |
| Selected temporal C2ST AUC | 0.533 |
| Robust V2.5 physics-core AUC | 0.572 to 0.575 |
| V3 board result | FF validated, SI/CH diagnostic |
| Rolling truth-test status | rolling_diagnostic |
| Rolling mean physics-core AUC | 0.702 |
| Rolling goal | mean <= 0.620, hit rate >= 0.40, worst fold < 0.800 |
| Success target | <= 0.60 |
| Artifact status | strong single-split FF result, rolling reliability not yet solved |

## What We Built

1. Real Statcast ingestion and cleaning.
2. Candidate ranking for viable pitcher/pitch pairs.
3. Feature groups for command, movement, release, trajectory, and full physics.
4. A generator suite with player, league, recent-window, context-weighted, GMM, game-drift, and Gaussian copula models.
5. A temporal validation split that trains on early pitches and tests on later pitches.
6. A held-out classifier two-sample test so validation does not score on the same rows it trained on.
7. Repeated-seed robustness checks.
8. Trajekt-shaped JSON export with validation metadata.
9. A Streamlit dashboard for presenting the final result.
10. A V2.1 factorized physics-residual model that samples release first, then movement, trajectory, and command with recent-game downstream residual drift.
11. A V2.3/V2.5 model tournament that compares release variants, PCA latent residuals, context-neighbor residuals, derived-feature Gaussian sampling, short-memory uncertainty-inflated factorization, and state anchoring.
12. A V3 validation board that runs the tournament across multiple pitcher/pitch candidates and creates presentation scorecards.

## Honest Result

The model now has one strong single-split full-physics pitch case and several honest failure cases. The rolling board is now the primary scoreboard because it tests repeated future-game windows rather than one favorable 70/30 temporal split.

| Board | Candidate | Pitch | Physics AUC | Pass rate | Status |
|---|---|---|---:|---:|---|
| Skubal 2025 | Skubal | FF | 0.533 | 1.00 | validated |
| Skubal 2025 | Skubal | SI | 0.665 | 0.00 | diagnostic |
| Skubal 2025 | Skubal | CH | 0.680 | 0.00 | diagnostic |
| Latest Statcast partial window | Isaac Mattson | FF | 0.576 | 0.50 | candidate |
| Latest Statcast partial window | Freddy Peralta | FF | 0.621 | 0.00 | diagnostic |
| Latest Statcast partial window | Taj Bradley | FF | 0.649 | 0.50 | diagnostic |

The strongest meeting framing:

> I can generate strong Skubal fastball distributions on a single temporal split, then use rolling validation to show the real reliability gap. The hard remaining problem is not command or movement; it is consistent release, spin, and full-joint physics across future game windows.

## Primary Rolling Scoreboard

| Metric | Current | Goal | Read |
|---|---:|---:|---|
| Mean rolling physics-core AUC | 0.702 | <= 0.620 | miss |
| Target hit rate | 0.10 | >= 0.40 | miss |
| Worst fold physics-core AUC | 0.929 | < 0.800 | miss |
| Best fold physics-core AUC | 0.593 | context only | some windows are already close |

This makes the project more credible, not weaker: it separates "we found an impressive result" from "we have a reliable production-grade generator." The next model work should be judged against this rolling gate.

## V2.1 Factorized Physics Result

V2.1 implements that next collaboration idea. It models:

```text
release / velocity / spin
  -> movement residual
  -> trajectory residual
  -> command residual
```

The useful improvement came from adding recent-game residual drift to the downstream physics layers and applying a train-only release variance floor. This attacks the temporal failure directly: later games are not just random draws from the early-season average.

| Layer | Best V1 AUC | V2.1 Factorized AUC | Result |
|---|---:|---:|---|
| command/location | 0.514 | 0.519 | still strong |
| movement only | 0.519 | 0.554 | still below 0.60 |
| release only | 0.585 | 0.575 | improved |
| trajectory only | 0.561 | 0.600 | borderline |
| physics core | 0.685 | 0.611 | improved, still diagnostic |

The honest read: V2.1 did not fully validate full physics because the target is `<= 0.60`, but it made the hard layer meaningfully better. The remaining leakage is mostly spin-axis and release-extension structure, which points to the next model refinement rather than a data-cleaning issue.

## V2.2 Weather Result

Weather was tested as a real residual ablation, not added as a product control. The pipeline joins MLB game time and venue coordinates to nearest-hour Open-Meteo historical weather by `game_pk`, then fits weather only against downstream movement/trajectory/command residuals.

Repeated over 12 sample/classifier seeds:

| Layer | V2.1 Baseline Mean | V2.2 Weather Mean | Delta |
|---|---:|---:|---:|
| command/location | 0.532 | 0.541 | +0.010 |
| movement only | 0.537 | 0.596 | +0.059 |
| release only | 0.594 | 0.584 | -0.010 |
| trajectory only | 0.619 | 0.614 | -0.006 |
| physics core | 0.646 | 0.645 | -0.001 |

The conclusion is useful even though it is not flashy: weather is technically feasible, but it does not materially improve full-physics realism for the Skubal 2025 FF test. Keep it as diagnostic evidence, not a default UI feature.

## V2.3 Model Tournament Result

V2.3 asked a different question: which model structure actually improves the hard full-physics temporal score?

Repeated over 30 sample/classifier seeds:

| Layer | V2.1 Baseline | Best V2.3 | Delta |
|---|---:|---:|---:|
| command/location | 0.542 | 0.529 | -0.013 |
| movement only | 0.543 | 0.530 | -0.013 |
| release only | 0.618 | 0.601 | -0.017 |
| trajectory only | 0.612 | 0.566 | -0.046 |
| physics core | 0.634 | 0.603 | -0.032 |

The winning family is `factorized_short_memory_wide_residual`: short-memory release drift plus explicit downstream uncertainty inflation. That says the model was not mainly missing weather; it was underestimating how much Skubal's later-game physical shape can spread around release, movement, and trajectory.

This is the best full-physics result so far, but it is not called default-ready because the physics-core pass rate is `0.47`, below the stricter robustness target of `0.80`.

## V2.5 And V3 Result

V2.5 improved Skubal FF full physics again with recent/trend state anchoring:

| Model | Physics AUC | Pass rate |
|---|---:|---:|
| `factorized_recent_state_anchored` | 0.579 | 0.77 |
| `factorized_trend_state_anchored` | 0.572 | 0.79 |

That got the robust full-physics benchmark to the edge of the strict `0.80` pass-rate target.

V3 then asked whether the result generalizes. It found:

- Skubal FF validates in the 3-repeat board at `0.533` AUC and `1.00` pass rate.
- Skubal SI and CH do not validate full physics, even though command/movement layers are strong.
- A broader partial Statcast cache produces one candidate result, Isaac Mattson FF at `0.576`, but pass rate is only `0.50`.

This is a stronger presentation artifact than a single best run because it shows the model's boundary.

## V4 Router And Release/Spin Diagnostic

V4 tried the obvious next idea: make release/spin treatment pitch-family-aware.

The result was useful, but not because the new candidate became the winner:

| Pitch | Best model after V4 | Physics AUC | V4 release/spin AUC | Read |
|---|---|---:|---:|---|
| Skubal FF | `factorized_recent_state_anchored` | 0.533 | 0.592 | keep old validated FF route |
| Skubal SI | `conditional_state_mixture_residual` | 0.644 | 0.720 | diagnostic |
| Skubal CH | `factorized_trend_state_anchored` | 0.680 | 0.701 | diagnostic |

The router is the product win. Scorecards now say:

- route status
- pitch family
- recommended physics model
- validated layers
- candidate layers
- diagnostic layers

The honest modeling conclusion:

> Pitch-family awareness is necessary, but hand-tuned spin-axis nudges are not enough. The next real model should learn release speed, spin rate, spin axis, and release geometry as a conditional mixture.

## How It Works

```text
Real Statcast rows
  -> feature engineering
  -> pitcher/pitch candidate selection
  -> early-season train split
  -> later-season holdout split
  -> generator models
  -> generated pitch samples
  -> classifier two-sample validation
  -> repeated-seed robustness
  -> Trajekt-shaped JSON
```

The selected model treats a pitch as:

```text
pitch = pitcher baseline + recent game/day drift + context effect + copula-sampled residual
```

This is the key technical step. A static model says "Skubal has one fastball distribution." The game-drift copula says "Skubal has a baseline, each appearance shifts the pitch distribution, recent appearances matter more, and the residual physics features have a non-Gaussian joint shape."

## Presentation Commands

Run the dashboard:

```bash
streamlit run app/streamlit_app.py
```

Rebuild the Skubal demo artifacts:

```bash
python3 scripts/fetch_baseball_savant_pitcher_csv.py \
  --pitcher-id 669373 \
  --season 2025 \
  --output data/processed/skubal_2025.csv

python3 scripts/build_demo_artifacts.py \
  --data data/processed/skubal_2025.csv \
  --output-dir outputs/real_demo \
  --min-pitches 450 \
  --min-holdout 120 \
  --min-games 10 \
  --samples 25 \
  --target-pitcher-id 669373 \
  --target-pitch-type FF
```

## Proof Artifacts

- `docs/assets/real_demo_morning_report.md`
- `docs/assets/real_demo_validation_report.json`
- `docs/assets/final_session.json`
- `outputs/model_tournament_skubal_2025_ff/model_tournament_report.json`
- `outputs/model_tournament_skubal_2025_ff/model_tournament_summary.md`
- `README.md`

## What To Say In The Meeting

1. The product idea is not "one perfect pitch"; it is validated pitcher variability.
2. Static full-physics models fail across time, which is the real product insight.
3. A recent-weighted game-drift copula improved command and full-physics detectability, but did not solve full joint physics.
4. The export is honest: validated layers are labeled separately from diagnostic full physics.
5. The V2.1 factorized physics model improved full-physics AUC from `0.685` to `0.611`, but full physics remains just above the strict validation target.
6. Weather was joined from real public sources and tested; it was effectively flat on full physics, so it stays diagnostic.
7. V2.5 improved the repeated full-physics mean to about `0.572`, right at the robust validation boundary.
8. V3 adds a validation board: Skubal FF validates on the single split, but SI/CH and several partial-window pitchers show the remaining frontier.
9. V4 adds a router and tests pitch-family release/spin. The router stays; the manual spin residual candidate remains diagnostic.
10. Rolling validation is now the main scoreboard: current status is `rolling_diagnostic`, with a clear target of mean AUC `<=0.620`, hit rate `>=0.40`, and worst fold `<0.800`.

```text
game latent state
  -> release + velocity + spin
  -> movement residual
  -> trajectory residual
  -> plate location
```

That is where the project becomes a deeper Trajekt collaboration rather than just a demo: the model now shows exactly which pieces are validated, which pieces are close, and where the next physics-residual refinement should focus.

The next modeling story should be pitch-family-aware:

```text
pitch family intent
  -> release geometry state
  -> circular spin-axis residual
  -> velocity/spin covariance
  -> movement and trajectory residual
  -> command cloud
```
