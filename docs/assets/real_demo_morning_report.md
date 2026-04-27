# Pitcher Twin Real Demo Report

This report uses real public Statcast rows only.

## Selected Candidate

- Pitcher: `Skubal, Tarik`
- Pitch type: `FF`
- Real pitches in cache: `835`
- Games: `31`
- Holdout rows: `251`

## Model Validation

- Artifact status: `validated_component_layers_physics_diagnostic`
- Selected feature group: `command_representation`
- Selected model: `player_recent_weighted_game_drift_copula`
- Selected temporal C2ST AUC: `0.533`
- Physics-core temporal C2ST AUC: `0.650`
- Temporal success target: `<= 0.60`
- C2ST classifier split: held-out stratified classifier rows
- Export strategy: `validated_layer_over_physics_core`
- Export physics model: `player_recent_weighted_game_drift_copula`
- Scope: real-data proof-of-concept validation
- Model selection policy: prefer contextual/parametric models within 0.05 C2ST AUC of the minimum

| Feature group | Selected model | Detectability C2ST AUC | Minimum C2ST AUC |
|---|---|---:|---:|
| command_representation | player_recent_weighted_game_drift_copula | 0.533 | 0.502 |
| movement_only | player_recent_weighted_game_drift_gaussian | 0.546 | 0.502 |
| physics_batter_context | player_recent_weighted_game_drift_gaussian | 0.671 | 0.633 |
| physics_core | player_recent_weighted_game_drift_copula | 0.650 | 0.650 |
| physics_count | player_recent_weighted_game_drift_copula | 0.654 | 0.650 |
| physics_fatigue | player_context_weighted_gaussian | 0.816 | 0.779 |
| release_only | player_recent_weighted_game_drift_copula | 0.643 | 0.597 |
| shape_representation | player_recent_weighted_game_drift_copula | 0.668 | 0.625 |
| trajectory_only | player_recent_empirical_bootstrap | 0.589 | 0.577 |

Detectability AUC is folded around `0.50`; lower is better and `0.50` means a held-out classifier cannot tell generated pitches from held-out real pitches.

## Layer Status

| Status | Feature group | Model | AUC |
|---|---|---|---:|
| validated | command_representation | player_recent_weighted_game_drift_copula | 0.533 |
| validated | movement_only | player_recent_weighted_game_drift_gaussian | 0.546 |
| borderline | trajectory_only | player_recent_empirical_bootstrap | 0.589 |
| diagnostic | release_only | player_recent_weighted_game_drift_copula | 0.643 |
| diagnostic | physics_core | player_recent_weighted_game_drift_copula | 0.650 |
| diagnostic | physics_batter_context | player_recent_weighted_game_drift_gaussian | 0.671 |
| diagnostic | physics_count | player_recent_weighted_game_drift_copula | 0.654 |
| diagnostic | physics_fatigue | player_context_weighted_gaussian | 0.816 |
| diagnostic | shape_representation | player_recent_weighted_game_drift_copula | 0.668 |

## Robustness Checks

Repeated over `30` sample/classifier seeds.

| Feature group | Model | Mean AUC | Std | Pass rate <= target |
|---|---|---:|---:|---:|
| command_representation | player_recent_weighted_game_drift_copula | 0.534 | 0.032 | 0.97 |
| physics_core | player_recent_weighted_game_drift_copula | 0.664 | 0.041 | 0.03 |
| release_only | player_recent_weighted_game_drift_copula | 0.627 | 0.051 | 0.30 |
| movement_only | player_recent_weighted_game_drift_gaussian | 0.545 | 0.034 | 0.93 |
| trajectory_only | player_recent_empirical_bootstrap | 0.592 | 0.039 | 0.57 |

## Selected Generator

- `player_recent_weighted_game_drift_copula` on `command_representation`

## Caveat

Validated means repeated-seed mean AUC is at or below target and pass rate is high. Borderline layers pass a single split or mean threshold but are not stable enough yet. Full physics remains diagnostic until `physics_core` reaches the configured temporal target.
