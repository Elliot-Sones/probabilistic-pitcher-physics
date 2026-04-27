# Next Model Angles Research

Date: 2026-04-27

## Current Bottleneck

V2.3 improved Skubal 2025 FF full-physics C2ST from `0.634` to `0.603`, but the model is still not robust enough:

- target mean AUC: `<= 0.600`
- target pass rate: `>= 0.80`
- best current physics-core pass rate: `0.47`
- strongest leakage signals: `release_pos_y`, `release_extension`, `spin_axis_cos`, `pfx_z`, `release_pos_x`

The likely issue is not just model flexibility. It is that the generator is sampling raw Statcast fields directly, even though several of those fields are physically tied together.

## Sources

- Baseball Savant CSV docs define the raw columns we model, including release position, movement, plate location, initial velocity, acceleration, spin axis, and extension: https://baseballsavant.mlb.com/csv-docs
- Alan Nathan's trajectory reconstruction note explains that Statcast pitch trajectories can be represented by release point, release velocity, and average acceleration under a constant-acceleration fit: https://baseball.physics.illinois.edu/TrajectoryStudies.pdf
- Kusafuka et al. show release angle, speed, spin axis, and horizontal release point help explain pitch location, and that release parameters can covary rather than vary independently: https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2020.00036/full
- Nathan et al. give the physical motivation for spin affecting ball flight through lift/drag measurements: https://arxiv.org/abs/physics/0605041
- Masked Autoregressive Flow and Neural Spline Flow papers support richer density models when we are ready for neural conditional density estimation: https://arxiv.org/abs/1705.07057 and https://arxiv.org/abs/1906.04032
- C2ST is a valid generative-model evaluation method, but MMD/energy-style tests can add complementary diagnostics: https://arxiv.org/abs/1610.06545 and https://jmlr.csail.mit.edu/papers/v13/gretton12a.html

## Best Next Angles

### 1. Physics-Constrained Representation

Stop treating all 18 physics columns as independent model outputs.

Model a smaller latent pitch state:

```text
release point
release velocity vector
spin / movement state
plate target
```

Then derive redundant columns like `release_pos_y`, `release_extension`, velocity components, acceleration-derived path, and plate crossing from that state.

Why this is promising:

- current leakage is strongest on `release_pos_y` and `release_extension`;
- those fields are mechanically related;
- a constraint layer can prevent samples that are statistically close in each marginal but physically odd jointly.

First experiment:

- replace raw `release_pos_y` + `release_extension` sampling with a constrained pair where `release_pos_y + release_extension` follows the learned real distribution;
- derive velocity components from release speed and inferred horizontal/vertical release angles;
- validate whether physics-core AUC and release pass rate improve.

### 2. Dynamic Game-State Latent Model

Replace hand-tuned recent weighting with a simple state-space model:

```text
game_state_t = game_state_t-1 + small drift
pitch_t = baseline + game_state_t + pitch_noise
```

The state should include release side/height/extension, velocity, spin axis, movement, and trajectory offsets.

Why this is promising:

- V2.3's win came from short-memory release drift plus wider downstream uncertainty;
- that is exactly what a state-space model is supposed to learn instead of hand tuning;
- it should improve pass-rate robustness.

First experiment:

- fit per-game means for each physics layer;
- estimate game-to-game drift covariance and pitch-level noise covariance;
- predict holdout games sequentially using only prior games;
- sample from state uncertainty plus pitch noise.

### 3. Latent Intent Mixture

A single pitch type can still have multiple intentions:

```text
high fastball
arm-side fastball
glove-side fastball
waste/chase fastball
```

If we average those together, the generated cloud can look too smooth or put the wrong physics with the wrong target.

Why this is promising:

- command/location validates well, but full physics still leaks;
- the missing dependency may be "this release/movement shape belongs with this target/intention";
- count and batter hand may matter because they change intent.

First experiment:

- cluster train pitches by plate location + count + batter hand;
- fit a factorized model per intent cluster;
- sample intent first, then sample physics conditional on intent.

### 4. Hierarchical Population Prior

Skubal FF has only `584` train pitches. That is small for an 18-dimensional joint distribution.

Use league data to learn general relationships:

```text
release -> movement
movement -> trajectory
trajectory -> command
```

Then learn Skubal-specific offsets on top.

Why this is promising:

- population data can stabilize release-to-flight relationships;
- player data can still control individual style;
- this avoids asking one pitcher sample to teach every physics relationship alone.

First experiment:

- fit layer regressions on all FFs;
- fit Skubal residual offsets and residual covariance;
- compare against player-only factorized layers.

### 5. Neural Conditional Density Only After 1-4

Normalizing flows or mixture density networks are plausible, but only after the representation is better.

Good version:

```text
context + game latent state -> conditional density over constrained pitch state
```

Risky version:

```text
context -> raw 18 Statcast columns
```

The risky version may learn artifacts and overfit because the pitcher-level sample size is small.

## Recommendation

Do V2.4 as:

```text
Physics-Constrained Dynamic State Model
```

The first target should not be a neural net. It should be:

1. constrained release geometry;
2. per-game latent state drift;
3. population-prior residual layers;
4. intent clusters if the first three still miss.

Success criteria:

- physics-core mean AUC `<= 0.600`;
- physics-core pass rate `>= 0.80`;
- release-only mean AUC `<= 0.600`;
- no regression of command or movement above `0.600`;
- leakage no longer dominated by `release_pos_y` / `release_extension`.
