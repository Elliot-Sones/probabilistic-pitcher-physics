# Side-By-Side Conditional Pitch Generator Design

## Purpose

Build a v1 conditional pitch generator that lets a user compare how one pitcher/pitch type changes across two game contexts. The product question is:

> How does Player A's slider distribution change between Context A and Context B?

The model should generate calibrated probability envelopes, not claim exact next-pitch certainty. The honest output is a distribution over pitch characteristics under a selected game state.

## User Experience

The user selects one pitcher and one pitch type, then configures two side-by-side contexts.

Required controls:

- pitcher
- pitch type
- inning
- pitcher game pitch count
- balls
- strikes
- batter hand
- pitcher score differential
- sample count

Each side displays generated samples and summary ranges for:

- release speed
- spin rate
- spin axis components
- release position and extension
- movement
- trajectory
- plate location
- derived miss tendency
- layer confidence/status

The center summary highlights context-to-context differences such as velocity loss, spin change, movement shift, release drift, and plate-location tendency.

## Data Scope

V1 uses only fields already available from Statcast/current repo feature engineering:

- `inning`
- `pitcher_game_pitch_count`
- `balls`
- `strikes`
- `count_bucket_code`
- `batter_stand_code`
- `pitcher_score_diff`
- recent game ordering through `game_date` and `game_pk`

Two v1 inputs must be engineered before modeling:

- `pitcher_game_pitch_count`: cumulative pitch count for the pitcher within a game, derived by sorting each pitcher/game by `game_date`, `game_pk`, `at_bat_number`, and Statcast `pitch_number`, then applying a per-pitcher/game cumulative count. Raw Statcast `pitch_number` should not be used as the game pitch count because it resets within each plate appearance.
- `pitcher_score_diff`: score differential from the pitcher's team perspective. Positive means the pitcher's team is ahead. The current batter-perspective `score_diff = bat_score - fld_score` is not suitable for the UI label "ahead by 2."

Weather, injury/news, travel, umpire tendencies, and external park conditions are deferred. They should remain future-work hooks, not hidden assumptions.

## Model Design

Use the existing recent-weighted game-drift model as the base and make it explicitly context-controllable.

Conceptual form:

```text
conditional pitch distribution =
  recent pitcher/pitch baseline
  + game-state context effect
  + pitch-level residual variation
```

The initial implementation should prefer this order:

1. `player_recent_weighted_game_drift_copula` when enough games/residuals exist.
2. `player_recent_weighted_game_drift_gaussian` when the copula is unavailable.
3. Existing context-weighted or recent Gaussian models as fallback.

The generator should accept a small context dataframe built from UI controls. Sampling should produce a dataframe with the same feature columns expected by the existing session/export code.

Generation may use `physics_core` so the exported samples contain full release, movement, trajectory, and command fields. Reporting must still be layer-aware: full physics generation does not imply full physics validation.

## Comparison Output

For each context, compute summary statistics from generated samples:

- mean
- standard deviation
- 10th percentile
- 50th percentile
- 90th percentile

For the side-by-side delta view, compute differences in means between Context B and Context A for key baseball-readable features:

- release speed
- release spin rate
- release position x/z
- release extension
- `pfx_x`
- `pfx_z`
- `plate_x`
- `plate_z`

The UI should label this as a model-estimated shift, not a guaranteed causal effect.

Add a derived miss-tendency summary from generated `plate_x` and `plate_z` samples. This is not a new model target in v1. It is a readable interpretation layer over plate-location samples:

- arm-side versus glove-side, using pitcher handedness when available;
- up versus down;
- zone versus chase;
- spike-risk for low, noncompetitive samples.

The UI should label miss tendency as model-estimated tendency. If strike-zone height is unavailable, use an explicitly documented default zone approximation rather than hiding the assumption.

Layer confidence should be shown next to generated results. The UI should be able to say, for example:

- command: validated
- movement: validated
- release: diagnostic
- trajectory: borderline
- physics core: diagnostic

These labels should come from validation reports, not from whether samples were successfully generated.

## Validation

Validation should answer: does the conditional generator make held-out real pitches harder to distinguish from generated pitches than the current static or weakly contextual baselines, and does that improvement hold by layer?

Add a conditional validation path:

1. Split pitcher/pitch rows temporally.
2. Fit on early rows.
3. For each held-out row, build a matching context from that row.
4. Generate samples conditioned on held-out contexts.
5. Run C2ST on held-out real rows versus generated rows.
6. Compare layer-by-layer against current baselines.

The validation report should compare:

- `player_recent_weighted_game_drift_gaussian`
- `player_recent_weighted_game_drift_copula`
- conditional game-drift copula

Across:

- `command_representation`
- `movement_only`
- `release_only`
- `trajectory_only`
- `physics_core`

Report detectability AUC, top leakage features, row counts, layer status, and fallback model usage.

## Architecture

Add narrowly scoped modules rather than growing `models.py` too much:

- Existing `features.py`: engineer `pitcher_game_pitch_count` and `pitcher_score_diff` from real Statcast columns.
- `conditional.py`: context row construction, conditional sampling orchestration, summary/delta helpers.
- Existing `models.py`: keep model fit/sample primitives here.
- Existing `validator.py`: add or call a conditional validation helper without changing current C2ST semantics.
- `app/streamlit_app.py`: add a side-by-side comparison view that consumes conditional summaries.

The generation feature group should default to `physics_core`, with component-layer confidence always reported alongside it. For the first UI pass, keep the mental model simple: one pitcher, one pitch type, two contexts.

## Error Handling

The system should be explicit about fallback and confidence:

- If a candidate lacks enough games for copula residuals, fall back to game-drift Gaussian.
- If context columns are unavailable, show which controls are inactive for that dataset.
- If `at_bat_number` is unavailable, block `pitcher_game_pitch_count` engineering rather than treating raw Statcast `pitch_number` as a game pitch count.
- If a pitcher/pitch pair has too few rows, block generation with a clear message.
- If generated samples contain non-finite values, fail validation/export rather than silently cleaning them.

## Tests

Add focused tests for:

- engineering `pitcher_game_pitch_count` from rows where raw `pitch_number` resets by plate appearance;
- engineering `pitcher_score_diff` so positive means the pitcher's team is ahead;
- building context dataframes from UI-style inputs;
- conditional sampling returns expected feature columns and finite values;
- side-by-side summaries include context A, context B, and deltas;
- derived miss tendency from generated plate-location clouds;
- layer status/confidence reporting from validation reports;
- fallback behavior when copula is unavailable;
- conditional validation produces C2ST metrics and row counts by layer;
- no weather/external-data dependency is required for v1.

## Non-Goals

V1 does not:

- predict the exact next pitch;
- model pitch-type selection probability;
- join weather data;
- infer catcher target;
- use external injury/news feeds;
- claim causal effects from sliders;
- claim full-physics temporal success unless validation supports it.

## Future Extensions

After v1 is validated, useful extensions are:

- weather/air-density joins by venue and game time;
- pitcher workload windows over 3/7/14 days;
- pitch-sequence conditioning;
- batter-specific matchup conditioning;
- side-by-side real-vs-generated overlays;
- confidence bands based on local historical support.
