# Pitcher Twin Project Spec

## Thesis

Real MLB pitchers throw probability distributions, not point estimates. Pitcher Twin learns a generative model of each pitcher's real pitch-to-pitch variability, conditions that distribution on game state when sample size allows, validates realism with a classifier, and exports realistic machine-session sessions.

## Product Concept

The feature concept is `Realistic Mode`.

Canonical replication asks:

> What is the target pitch?

Realistic Mode asks:

> What does this pitcher actually look like over 15 real attempts?

The output is not just one pitch. It is a session sampled from the player's learned distribution.

## High-Level Architecture

```text
Statcast data
    |
    v
Cleaned pitch features
    |
    +--> Per-pitcher x pitch-type GMM
    |        |
    |        v
    |   Conditional sampler
    |        |
    |        v
    |   Humanized Machine Session JSON
    |
    +--> Realism validator
    |        |
    |        v
    |   AUC + feature leakage report
    |
    +--> Pitcher embedding
             |
             v
        Latent-space map
```

## Data Layer

Goal: pull public Statcast pitch-level data for recent seasons and cache it locally.

Recommended first dataset:

- 2024 full season
- 2025 full season
- 8 to 12 starting pitchers
- fastballs/sinkers first

Initial pitcher set:

- Gerrit Cole
- Kevin Gausman
- Chris Sale
- Zack Wheeler
- Tarik Skubal
- Paul Skenes
- Yoshinobu Yamamoto
- Tyler Glasnow

Required columns:

- `pitcher`
- `player_name`
- `pitch_type`
- `release_speed`
- `release_spin_rate`
- `spin_axis`
- `release_pos_x`
- `release_pos_y`
- `release_pos_z`
- `release_extension`
- `pfx_x`
- `pfx_z`
- `plate_x`
- `plate_z`
- `vx0`
- `vy0`
- `vz0`
- `ax`
- `ay`
- `az`
- `balls`
- `strikes`
- `pitch_number`
- `game_pk`
- `game_date`
- `batter`
- `stand`

## Feature Engineering

Critical preprocessing:

- Drop rows missing core physics fields.
- Filter to high-volume pitch types first: `FF`, `SI`.
- Convert `spin_axis` to circular features:
  - `spin_axis_cos`
  - `spin_axis_sin`
- Convert count to buckets:
  - `first_pitch`
  - `behind`
  - `even`
  - `ahead`
  - `full`
- Preserve raw `plate_x` and `plate_z`, but label them as command/location variables.

Feature groups:

```text
Shape features:
release_speed, release_spin_rate, spin_axis_cos, spin_axis_sin,
release_pos_x, release_pos_y, release_pos_z, release_extension,
pfx_x, pfx_z

Command features:
plate_x, plate_z
```

For the first demo, fit one combined model but report shape and command variability separately.

## Per-Pitcher GMM

Fit one Gaussian Mixture Model per `(pitcher, pitch_type)` pair.

Recommended settings:

```python
GaussianMixture(
    n_components=k,
    covariance_type="full",
    n_init=5,
    random_state=42,
)
```

Search `k = 1..5` and select by BIC.

Standardize features before fitting. Store the scaler with the model.

Minimum sample-size rule:

- Fit if `n >= 150`.
- Otherwise skip or pool using a future hierarchical model.

## Conditional GMM

Fit count-conditioned GMMs only when enough data exists.

Rule:

- Use `(pitcher, pitch_type, count_bucket)` GMM if `n >= 120`.
- Otherwise fall back to the unconditional `(pitcher, pitch_type)` GMM.

This gives the app a credible count-conditioning story without overfitting sparse buckets.

## Monte Carlo Sampler

The sampler turns a learned distribution into a pitch session.

Input:

```text
pitcher_name
pitch_type
count_bucket
n_pitches
```

Output:

```text
n realistic pitch samples with velocity, spin, release, movement, and plate location.
```

Why Monte Carlo matters:

- GMM describes the cloud.
- Monte Carlo samples actual pitches from the cloud.
- target system needs pitch targets, not just charts.

## Validation

Use two validation modes.

### Same-Season Realism

Purpose: test whether simulations resemble held-out real pitches from the same season.

Method:

1. Split 2024 data temporally.
2. Fit GMM on first 70%.
3. Simulate samples.
4. Compare simulated samples to final 30% real 2024 pitches.
5. Train a classifier to distinguish real vs simulated.
6. Report ROC-AUC.

Interpretation:

- `AUC <= 0.60`: strong realism.
- `0.60 < AUC <= 0.70`: usable, inspect leaked features.
- `AUC > 0.70`: model artifacts are too detectable.

### Next-Season Drift

Purpose: measure whether the pitcher changed.

Method:

1. Fit GMM on 2024.
2. Compare simulations to 2025.
3. Report AUC as drift/generalization, not pure realism.

This avoids incorrectly blaming the model for real year-to-year pitcher changes.

## Pitcher Latent Space

For each pitcher, build an aggregate feature vector:

- pitch-type usage
- mean velocity by pitch type
- velocity standard deviation by pitch type
- mean spin by pitch type
- mean movement by pitch type
- mean release point by pitch type
- variability metrics by pitch type

Then:

- standardize across pitchers
- apply PCA to 2D
- plot a league map
- compute nearest neighbors by Euclidean distance in standardized embedding space

## Visualizations

Must-have:

- Real vs simulated histograms
- Release-point cloud
- Plate-location cloud
- Movement cloud
- Canonical session vs humanized session trajectory plot
- Validator AUC callout
- Machine Session JSON preview

Nice-to-have:

- Count-conditioned shift plot
- Pitcher latent-space scatter
- Nearest-neighbor table

## Machine Session JSON Shape

```json
{
  "pitcher": "Gerrit Cole",
  "pitch_type": "FF",
  "count_bucket": "first_pitch",
  "model_version": "gmm-v1",
  "validator_auc": 0.56,
  "session": [
    {
      "pitch_id": 1,
      "release": {"x": -2.1, "y": 54.2, "z": 6.3},
      "velocity_mph": 96.8,
      "spin_rate_rpm": 2387,
      "spin_axis_deg": 213,
      "movement": {"pfx_x": -0.42, "pfx_z": 1.31},
      "target_location": {"plate_x": 0.4, "plate_z": 3.1}
    }
  ]
}
```

## Demo Flow

1. Show latent-space map or pitcher selector.
2. Pick Gerrit Cole fastball.
3. Show the real pitch distribution.
4. Toggle `Humanized Mode`.
5. Show 15 sampled trajectories versus 15 identical/canonical trajectories.
6. Show validation AUC.
7. Show count-conditioned distribution shift.
8. Export JSON.
9. Hand over one-page summary.

## Future Work

- Hierarchical Bayesian pooling for low-sample pitchers.
- Fatigue conditioning by pitch count and days rest.
- Distribution drift detection for mechanical or injury signals.
- Hitter-handedness conditioning.
- Pitch sequence modeling.
- Catcher target vs execution decomposition.
- Seam/active-spin residual modeling.
- Pitch-design inversion from desired movement to machine settings.

