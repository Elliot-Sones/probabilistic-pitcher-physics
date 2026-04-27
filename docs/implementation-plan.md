# Implementation Plan

## Goal

Build a credible ML/DS demo showing pitcher-specific pitch variability, realistic sampling, validation, and Trajekt-shaped export.

## Phase 1: Repo And Data

1. Create virtual environment.
2. Install package dependencies.
3. Add data-fetching function using `pybaseball.statcast_pitcher`.
4. Cache raw data to `data/raw/statcast_2024_2025.parquet`.
5. Add a small fixture dataset for tests so tests do not hit the network.

Verification:

```bash
pytest -q
```

## Phase 2: Feature Engineering

Create `src/pitcher_twin/features.py`.

Functions:

- `add_spin_axis_components(df)`
- `add_count_bucket(df)`
- `filter_pitch_types(df, pitch_types)`
- `build_model_matrix(df, feature_columns)`

Test cases:

- spin axis `0` maps to `(cos=1, sin=0)`
- spin axis `90` maps to `(cos=0, sin=1)`
- `3-2` maps to `full`
- `0-0` maps to `first_pitch`

## Phase 3: GMM Fitting

Create `src/pitcher_twin/models.py`.

Core objects:

- `PitcherGMM`
- `FittedPitchModel`
- `fit_best_gmm`
- `fit_pitcher_models`

Behavior:

- standardize features
- search `k = 1..5`
- choose lowest BIC
- store model, scaler, feature columns, sample size, BIC

Test cases:

- model refuses sample sizes below threshold
- sampled output returns original feature scale
- selected model has a valid component count

## Phase 4: Sampler

Create `src/pitcher_twin/sampler.py`.

Function:

```python
sample_pitches(
    models,
    pitcher_name: str,
    pitch_type: str,
    count_bucket: str | None,
    n: int,
    random_state: int = 42,
)
```

Behavior:

- use conditional model if available
- fall back to unconditional model
- return sampled DataFrame with model metadata

## Phase 5: Validator

Create `src/pitcher_twin/validator.py`.

Functions:

- `temporal_train_holdout(df, train_fraction=0.7)`
- `realism_auc(real_holdout, simulated, feature_columns)`
- `bootstrap_auc_ci(labels, scores)`

Model:

- logistic regression baseline
- optional random forest or XGBoost later

Report:

- AUC
- 95% bootstrap interval
- coefficients/feature leakage

## Phase 6: Latent Space

Create `src/pitcher_twin/latent.py`.

Functions:

- `build_pitcher_embedding_table(df)`
- `fit_pca_embedding(table)`
- `nearest_pitchers(embedding, pitcher_name, k=5)`

Output:

- `outputs/figures/pitcher_latent_space.html`

## Phase 7: Trajekt Export

Create `src/pitcher_twin/trajekt_format.py`.

Function:

```python
to_trajekt_json(samples, pitcher, pitch_type, count_bucket, metadata)
```

Output:

- session-level metadata
- one object per sampled pitch
- release, velocity, spin, movement, target location

## Phase 8: Demo Surface

Start with notebook:

- `notebooks/01_pitcher_twin_demo.ipynb`

Then add Streamlit if time allows:

- `app/streamlit_app.py`

Core demo screens:

- pitcher/pitch selector
- real vs simulated distributions
- trajectory plot
- validator metrics
- JSON export

## Phase 9: One-Page PDF

Create:

- `outputs/pitcher_twin_summary.pdf`

Sections:

1. Headline
2. Problem
3. Solution
4. Validation
5. Visual
6. Future work
7. Contact line

## Cut Order

Cut in this order if short on time:

1. Streamlit UI
2. XGBoost validator
3. full 50-pitcher latent space
4. multiple pitch types
5. count conditioning for every pitcher

Do not cut:

- GMM fitting
- Monte Carlo sampler
- validation
- trajectory visual
- JSON export
- one-page summary

