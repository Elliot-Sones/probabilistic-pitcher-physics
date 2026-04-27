# Pitcher Twin

Pitcher Twin is a real-data ML project for pitching-machine-style pitcher replication. The goal is not to predict one perfect next pitch. The goal is to learn a pitcher's **validated variability envelope**: given a pitcher, pitch type, and game context, what distribution of release, movement, trajectory, and command should a machine generate so that held-out real pitches cannot be easily distinguished from generated pitches?

The current strongest case study is **Tarik Skubal 2025 four-seam fastballs**:

- `2,849` total Statcast rows
- `835` Skubal FF pitches
- `31` games
- `584` train pitches
- `251` temporal holdout pitches

## Technical Idea

Two pitches can both be "Skubal FF" and still differ in small but important ways. A realistic pitching machine session needs that variation, not just a centroid.

Pitcher Twin models variation across five coupled layers:

| Layer | What it captures | Example columns |
|---|---|---|
| release | how the ball leaves the hand | `release_speed`, `release_spin_rate`, `spin_axis`, release slot |
| movement | ball movement summary | `pfx_x`, `pfx_z` |
| trajectory | fitted flight dynamics | `vx0`, `vy0`, `vz0`, `ax`, `ay`, `az` |
| command | where the pitch finishes | `plate_x`, `plate_z` |
| physics core | all of the above together | release + movement + trajectory + command |

The project validates each layer separately because a model can be excellent at command and still fail full joint physics. That distinction matters for product honesty.

## Real Pitch Variation

This is the same pitch type from the same pitcher. The spread is real Statcast variation, not generated data.

![Tarik Skubal FF real variation](docs/assets/readme/skubal_ff_variation.png)

The animation below slices the same FF pitch locations by pitcher game pitch count. This is why a single average pitch is not enough: the distribution shifts over time and context.

![Skubal FF pitch-count variation](docs/assets/readme/skubal_ff_pitch_sequence.gif)

## System Pipeline

The core loop is train, generate, then ask a held-out classifier whether it can tell generated pitches from later real pitches.

![Pitcher Twin pipeline](docs/assets/readme/pitcher_twin_pipeline.svg)

Editable Excalidraw source: [pitcher_twin_pipeline.excalidraw](docs/assets/readme/pitcher_twin_pipeline.excalidraw)

## What We Did

1. Loaded real public Statcast pitch rows.
2. Cleaned and engineered player-specific pitch features.
3. Ranked viable `(pitcher, pitch_type)` candidates by sample size, holdout size, games, completeness, and variability signal.
4. Split each candidate temporally: earlier pitches train the model, later pitches are holdout.
5. Built generator models that sample pitch distributions rather than single pitches.
6. Validated with classifier two-sample tests, so lower detectability means better realism.
7. Exported a machine-session JSON session with validation metadata.
8. Added V2.1, a factorized physics-residual model that samples pitch physics as a chain.

## How The Models Work

### Baselines

The first models prove that player-specific data matters:

- random independent noise
- league same-pitch empirical sampling
- player empirical bootstrap
- player multivariate Gaussian
- recent-window empirical models
- recent-weighted game-drift Gaussian
- recent-weighted game-drift Gaussian copula
- context-weighted Gaussian
- optional GMM when `scikit-learn` is available

The key validation lesson: static full-physics models struggle across time. That is the opening for game drift, context, and factorized physics.

### Game-Drift Copula

The game-drift copula model treats a pitch as:

```text
pitch = pitcher baseline + recent game/day drift + context effect + copula residual
```

This was the first strong component-level model. It validated command/location and movement across time under repeated-seed checks, but full joint physics remained diagnostic.

### V2.1 Factorized Physics Residual

V2.1 stops sampling one flat physics vector. It samples a pitch as a physics chain:

![V2.1 physics chain](docs/assets/readme/v21_physics_chain.svg)

Editable Excalidraw source: [v21_physics_chain.excalidraw](docs/assets/readme/v21_physics_chain.excalidraw)

V2.1 uses:

- a release model selected from the strongest existing generator suite;
- residual layers for movement, trajectory, and command;
- Gaussian-copula downstream residual sampling;
- recent-game residual drift for downstream physics;
- a train-only release variance floor so release spread does not collapse.

## Results

Validation uses a classifier two-sample test, or C2ST. Generated pitches and held-out real pitches are mixed together. A classifier tries to tell which rows are generated. The detectability AUC is folded around `0.50`.

- `0.50` means generated and real holdout pitches are hard to distinguish.
- Lower is better.
- The current validation target is `<= 0.60`.

### V2.1 Layer Results

On the Skubal 2025 FF temporal split:

![V2.1 AUC results](docs/assets/readme/v21_results_auc.png)

| Layer | V2.1 factorized | Game-drift Gaussian | Game-drift copula | Status |
|---|---:|---:|---:|---|
| command/location | 0.519 | 0.587 | 0.514 | strong |
| movement only | 0.554 | 0.588 | 0.507 | strong |
| release only | 0.575 | 0.641 | 0.612 | improved |
| trajectory only | 0.600 | 0.629 | 0.650 | borderline |
| physics core | 0.611 | 0.663 | 0.696 | improved, still diagnostic |

Main result: **V2.1 is currently the strongest full-physics model.** It brought `physics_core` to `0.611`, very close to the `<= 0.60` target, while preserving strong component-layer behavior.

### Repeated-Seed Component Results

The presentation artifact also tracks repeated-seed robustness for the strongest component models:

| Layer | Best model | Mean AUC | Pass rate <= 0.60 | Status |
|---|---|---:|---:|---|
| command/location | recent-weighted game-drift copula | 0.534 | 0.97 | validated |
| movement only | recent-weighted game-drift Gaussian | 0.545 | 0.93 | validated |
| trajectory only | player recent empirical bootstrap | 0.592 | 0.57 | borderline |
| release only | recent-weighted game-drift copula | 0.627 | 0.30 | diagnostic |
| physics core | recent-weighted game-drift copula | 0.664 | 0.03 | diagnostic |

## What The Results Mean

The project has moved from "can we generate plausible player-specific pitch variation?" to a more precise finding:

> Command and movement variability are validated. Full joint physics is close but still diagnostic. The remaining gap is mostly preserving coupled structure across spin axis, release extension, trajectory, and command.

That is exactly the layer a pitching-machine-style system would care about next: not just whether a generated pitch lands in the right place, but whether the full release-to-flight chain looks like the real pitcher.

## Reproduce

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

Fetch the Skubal 2025 data:

```bash
python3 scripts/fetch_baseball_savant_pitcher_csv.py \
  --pitcher-id 669373 \
  --season 2025 \
  --output data/processed/skubal_2025.csv
```

Build the real demo artifacts:

```bash
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

Run the V2.1 factorized validation:

```bash
python3 scripts/run_factorized_validation.py \
  --data data/processed/skubal_2025.csv \
  --output-dir outputs/factorized_skubal_2025_ff \
  --pitcher-id 669373 \
  --pitch-type FF
```

Regenerate README visuals:

```bash
python3 scripts/build_readme_visuals.py \
  --data data/processed/skubal_2025.csv \
  --output-dir docs/assets/readme
```

Run the dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Key Files

| Path | Purpose |
|---|---|
| [src/pitcher_twin/models.py](src/pitcher_twin/models.py) | generator suite, game drift, copula helpers |
| [src/pitcher_twin/factorized.py](src/pitcher_twin/factorized.py) | V2.1 factorized physics-residual model |
| [src/pitcher_twin/features.py](src/pitcher_twin/features.py) | real Statcast feature engineering |
| [src/pitcher_twin/validator.py](src/pitcher_twin/validator.py) | temporal split and C2ST validation |
| [scripts/run_factorized_validation.py](scripts/run_factorized_validation.py) | V2.1 validation runner |
| [scripts/build_readme_visuals.py](scripts/build_readme_visuals.py) | README graph, GIF, SVG, and Excalidraw generation |
| [outputs/factorized_skubal_2025_ff/factorized_validation_summary.md](outputs/factorized_skubal_2025_ff/factorized_validation_summary.md) | saved V2.1 result summary |

## Data Policy

This project is real-data only.

- No mock pitch rows.
- No synthetic weather rows.
- No fake player examples.
- No silent fallback demo data.
- Generated samples are allowed only when labeled as model output.

## Pitching Machine Framing

The public framing should stay platform-neutral:

> Pitching machines are excellent at replicating target trajectories. Pitcher Twin adds an explicit probabilistic layer for measuring, validating, and sampling each pitcher's natural variability envelope.

## Honest Status And Next Work

What is working:

- real data ingestion and candidate selection;
- command/location validation;
- movement validation;
- machine-session JSON export;
- V2.1 full-physics improvement from a flat-vector framing to a coupled physics-residual chain.

What remains diagnostic:

- full `physics_core` is close but still above the `<= 0.60` target;
- release-only robustness needs improvement;
- spin-axis and release-extension coupling still leak signal to the classifier.

Next model work:

- run repeated-seed robustness for V2.1;
- preserve spin-axis and release-extension structure more tightly;
- add count/fatigue effects inside residual-drift layers;
- test V2.1 across more pitcher/pitch pairs;
- revisit weather only after the physics-residual layer clears the validation target.
