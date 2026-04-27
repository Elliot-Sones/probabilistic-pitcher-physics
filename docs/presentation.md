# Pitcher Twin Presentation

## One-Line Story

Pitcher Twin learns a real pitcher's pitch-to-pitch variability from Statcast, validates generated pitches against held-out real pitches, and exports a machine-session session JSON.

## Current Demo

| Field | Value |
|---|---:|
| Pitcher | Tarik Skubal |
| Season | 2025 |
| Pitch | FF |
| Real FF pitches | 835 |
| Games | 31 |
| Temporal holdout rows | 251 |
| Selected model | recent-weighted game-drift copula |
| Selected layer | command/location |
| Selected temporal C2ST AUC | 0.533 |
| Physics-core temporal C2ST AUC | 0.650 |
| Success target | <= 0.60 |
| Artifact status | validated_component_layers_physics_diagnostic |

## What We Built

1. Real Statcast ingestion and cleaning.
2. Candidate ranking for viable pitcher/pitch pairs.
3. Feature groups for command, movement, release, trajectory, and full physics.
4. A generator suite with player, league, recent-window, context-weighted, GMM, game-drift, and Gaussian copula models.
5. A temporal validation split that trains on early pitches and tests on later pitches.
6. A held-out classifier two-sample test so validation does not score on the same rows it trained on.
7. Repeated-seed robustness checks.
8. machine-session JSON export with validation metadata.
9. A Streamlit dashboard for presenting the final result.
10. A V2.1 factorized physics-residual model that samples release first, then movement, trajectory, and command with recent-game downstream residual drift.

## Honest Result

The model validates component-level realism across time:

| Layer | Mean AUC | Pass Rate <= 0.60 | Status |
|---|---:|---:|---|
| command/location | 0.539 | 0.97 | validated |
| movement only | 0.545 | 0.93 | validated |
| trajectory only | 0.592 | 0.57 | borderline |
| release only | 0.627 | 0.30 | diagnostic |
| physics core | 0.664 | 0.03 | diagnostic |

The strongest meeting framing:

> I validated Skubal's command and movement variability across time, then trained a Gaussian copula version that improved full-physics AUC from 0.683 to 0.650. That is meaningful movement, but full joint physics remains diagnostic, which points directly to the next collaboration opportunity: preserving physical relationships across release, movement, trajectory, and command.

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
  -> machine-session JSON
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
- `README.md`

## What To Say In The Meeting

1. The product idea is not "one perfect pitch"; it is validated pitcher variability.
2. Static full-physics models fail across time, which is the real product insight.
3. A recent-weighted game-drift copula improved command and full-physics detectability, but did not solve full joint physics.
4. The export is honest: validated layers are labeled separately from diagnostic full physics.
5. The V2.1 factorized physics model improved full-physics AUC from `0.685` to `0.611`, but full physics remains just above the strict validation target.

```text
game latent state
  -> release + velocity + spin
  -> movement residual
  -> trajectory residual
  -> plate location
```

That is where the project becomes a deeper target system collaboration rather than just a demo: the model now shows exactly which pieces are validated, which pieces are close, and where the next physics-residual refinement should focus.
