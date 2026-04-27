# Weather Feature Spike

## Question

Should V1 add weather as a model input for the side-by-side conditional pitch probability explorer?

## Short Answer

Not yet. Weather is technically joinable, but the first real Skubal 2025 FF spike did not improve held-out full-physics validation. Keep weather out of V1 and treat it as a future diagnostic layer.

## Data Sources Checked

- Baseball Savant CSV docs confirm Statcast includes game/date, teams, count, inning, pitch physics, and `game_pk`. They also clarify that `pitch_number` is the pitch number within the plate appearance, not pitcher workload.
- MLB Stats API live game feed provides game start time, venue, venue coordinates, roof type, and coarse in-game weather strings for each `game_pk`.
- Open-Meteo Historical Weather API provides hourly historical weather by latitude/longitude/date, including temperature, humidity, pressure, precipitation, wind speed, and wind direction.

Primary references:

- `https://baseballsavant.mlb.com/csv-docs`
- `https://open-meteo.com/en/docs/historical-weather-api`
- `https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live`

## Join Method

For each Skubal 2025 FF game:

1. Use Statcast `game_pk`.
2. Fetch MLB game metadata from the live game feed.
3. Extract game UTC start time, venue coordinates, venue name, and roof type.
4. Query Open-Meteo hourly historical weather at the venue coordinates.
5. Match the closest hourly weather row to the MLB game start time.
6. Join weather back to all pitch rows by `game_pk`.

Weather features tested:

- `temperature_2m`
- `relative_humidity_2m`
- `dew_point_2m`
- `pressure_msl`
- `surface_pressure`
- `precipitation`
- `wind_speed_10m`
- `wind_dir_sin`
- `wind_dir_cos`

Generated spike artifacts:

- `outputs/weather_spike/skubal_2025_game_weather.csv`
- `outputs/weather_spike/skubal_2025_ff_with_weather.csv`
- `outputs/weather_spike/weather_game_mean_rmse.csv`
- `outputs/weather_spike/weather_spike_summary.json`
- `outputs/weather_spike/weather_c2st_repeated.csv`
- `outputs/weather_spike/weather_repeated_summary.json`

## Experiment

Dataset:

- Pitcher: Tarik Skubal
- Season: 2025
- Pitch type: FF
- FF rows: 835
- Games: 31
- Train split: first 21 games, 518 pitches
- Holdout split: final 10 games, 317 pitches

Compared:

1. Recent-only game mean baseline.
2. Weather ridge game-mean predictor.

The weather model predicted holdout game-level physics means from game-time weather, then generated pitch samples using the same train residual covariance as the recent-only baseline.

## Results

Single-seed full-physics C2ST:

| Model | AUC |
|---|---:|
| recent-only game mean | 0.782 |
| weather ridge game mean | 0.798 |

Lower is better, so this was worse.

Repeated over 30 seeds:

| Metric | Recent Only | Weather Ridge |
|---|---:|---:|
| mean C2ST AUC | 0.787 | 0.788 |
| AUC std | 0.033 | 0.037 |

Weather beat recent-only in only `47%` of seeds.

Game-mean RMSE:

| Metric | Recent Only | Weather Ridge |
|---|---:|---:|
| mean standardized RMSE | 0.605 | 0.613 |

Weather helped some vertical movement/trajectory features:

| Feature | Recent RMSE | Weather RMSE | Improvement |
|---|---:|---:|---:|
| `az` | 0.706 | 0.502 | 0.203 |
| `vz0` | 0.546 | 0.370 | 0.176 |
| `pfx_z` | 0.591 | 0.442 | 0.149 |
| `plate_z` | 0.383 | 0.292 | 0.091 |

But it regressed spin/release features:

| Feature | Recent RMSE | Weather RMSE | Regression |
|---|---:|---:|---:|
| `release_spin_rate` | 0.599 | 0.739 | -0.140 |
| `spin_axis_sin` | 0.660 | 0.785 | -0.125 |
| `spin_axis_cos` | 0.584 | 0.692 | -0.108 |
| `release_pos_y` | 0.891 | 0.998 | -0.107 |

Roof coverage:

- Train: 18 open-air, 3 retractable.
- Holdout: 9 open-air, 1 retractable.

So this result is not only a dome/retractable artifact.

## Interpretation

Weather is technically feasible, but on this first real pitcher/pitch sample it did not improve the validation target. The apparent signal is mixed: weather helped a few vertical movement/trajectory dimensions while hurting spin/release dimensions, and the aggregate full-physics C2ST did not improve.

That means weather should not be a V1 control unless the UI clearly labels it diagnostic. The stronger V1 work remains:

1. Engineer true pitcher game pitch count.
2. Engineer pitcher-perspective score differential.
3. Build the side-by-side conditional game-state generator.
4. Validate context conditioning layer-by-layer.

## Recommendation

Do not add weather to the main V1 model yet.

Add a future `weather_diagnostic` module only after the side-by-side conditional generator exists. The right next weather test would be broader:

- multiple pitchers;
- multiple pitch types;
- outdoor-only games;
- final untouched holdout games;
- layer-specific validation for movement/trajectory rather than forcing weather into full physics.

Potential future model:

```text
pitch distribution =
  recent game-drift baseline
  + Statcast game-state context
  + optional weather residual adjustment
  + pitch-level residual noise
```

For now, weather is interesting research, not a validated product feature.
