# V4 Pitch-Family Release/Spin Results

## What Was Implemented

V4 added two pieces:

1. `factorized_pitch_family_release_spin`
   - starts from the short-memory factorized model
   - maps Statcast pitch types into physics families
   - applies pitch-family-specific release recentering
   - blends release geometry through `release_pos_y + release_extension`
   - tests empirical circular spin-axis residual sampling without collapsing spin to one mean

2. Model router
   - reads tournament reports
   - labels each layer as `validated`, `candidate`, or `diagnostic`
   - records a recommended physics model and trusted layers for each pitch
   - adds the route to scorecards

## Why This Was Worth Testing

The V3 board showed the real frontier:

- Skubal FF validates.
- Skubal SI and CH do not validate full physics.
- SI/CH failures repeatedly point at spin axis, release geometry, acceleration, and release speed/spin relationships.

So the V4 hypothesis was:

```text
pitch-family-aware release/spin treatment
  -> better release layer
  -> better full-physics realism for SI/CH
```

## Real Skubal Top-3 Board

Command:

```bash
/Users/elliot18/Desktop/Home/Projects/trajekt-pitcher-twin/.venv/bin/python \
  scripts/run_validation_board.py \
  --data /Users/elliot18/Desktop/Home/Projects/trajekt-pitcher-twin/data/processed/skubal_2025.csv \
  --output-dir outputs/validation_board_skubal_2025_top3_v4 \
  --top 3 \
  --repeats 3 \
  --samples 260
```

| Pitch | Best V4-board model | Physics AUC | Pass rate | Status |
|---|---|---:|---:|---|
| FF | `factorized_recent_state_anchored` | 0.533 | 1.00 | `validated_temporal_success` |
| SI | `conditional_state_mixture_residual` | 0.644 | 0.33 | `physics_core_diagnostic` |
| CH | `factorized_trend_state_anchored` | 0.680 | 0.00 | `physics_core_diagnostic` |

The new V4 candidate itself:

| Pitch | V4 release AUC | V4 trajectory AUC | V4 physics AUC | Read |
|---|---:|---:|---:|---|
| FF | 0.583 | 0.511 | 0.592 | useful for trajectory, not best full physics |
| SI | 0.725 | 0.709 | 0.720 | diagnostic |
| CH | 0.709 | 0.549 | 0.701 | diagnostic |

## Latest Statcast Partial-Window Board

Command:

```bash
/Users/elliot18/Desktop/Home/Projects/trajekt-pitcher-twin/.venv/bin/python \
  scripts/run_validation_board.py \
  --data /Users/elliot18/assistant/projects/trajekt-scout/data/processed/latest_statcast.csv \
  --output-dir outputs/validation_board_latest_statcast_top3_v4 \
  --top 3 \
  --min-pitches 200 \
  --min-games 4 \
  --min-holdout 40 \
  --repeats 2 \
  --samples 220
```

| Pitcher | Pitch | Best model | Physics AUC | Pass rate | Status |
|---|---|---|---:|---:|---|
| Isaac Mattson | FF | `factorized_short_memory_more_uncertain` | 0.576 | 0.50 | `physics_core_candidate` |
| Freddy Peralta | FF | `factorized_short_memory_wide_residual` | 0.621 | 0.00 | `physics_core_diagnostic` |
| Taj Bradley | FF | `factorized_recent_state_anchored` | 0.649 | 0.50 | `physics_core_diagnostic` |

The new V4 candidate did not become the best physics model on this board.

## Interpretation

This was a useful failed model hypothesis.

The model router is worth keeping because it gives the product a clean decision layer:

```text
validated pitch/layer -> can present as trusted
candidate pitch/layer -> promising but needs robustness
diagnostic pitch/layer -> show as frontier, not as solved
```

The pitch-family release/spin candidate should stay diagnostic. It proved that hand-tuned circular spin residual nudges are not enough. The classifier still finds generated samples through spin-axis and release-geometry clues, and for SI/CH it also finds velocity/acceleration inconsistencies.

## Current Product Status After V4

- Skubal FF remains the serious validated demo.
- Skubal FF trajectory layer improved with the new V4 candidate: `0.511`.
- Skubal SI improves at the board level from the V3 `0.665` result to `0.644`, but it remains diagnostic and the best model is `conditional_state_mixture_residual`, not the V4 release/spin candidate.
- Skubal CH remains diagnostic at `0.680`.
- The router scorecards are now more presentation-ready because they explain exactly what can be trusted.

## Next Modeling Move

Do not keep tuning hand-picked spin alphas. The next serious version should model release/spin as a learned conditional distribution:

```text
pitch family + recent game state + count/fatigue
  -> release speed / spin rate / spin axis mixture
  -> movement residual
  -> trajectory residual
  -> command cloud
```

Best concrete candidate:

```text
conditional release-state mixture
  with circular spin-axis component
  and velocity/spin covariance per latent state
```

That should replace the manual V4 spin nudge if we want to attack SI/CH meaningfully.

## Saved Artifacts

- `outputs/validation_board_skubal_2025_top3_v4/validation_board.md`
- `outputs/validation_board_skubal_2025_top3_v4/scorecards/skubal_tarik_ff.md`
- `outputs/validation_board_skubal_2025_top3_v4/scorecards/skubal_tarik_si.md`
- `outputs/validation_board_skubal_2025_top3_v4/scorecards/skubal_tarik_ch.md`
- `outputs/validation_board_latest_statcast_top3_v4/validation_board.md`
- `outputs/validation_board_latest_statcast_top3_v4/scorecards/isaac_mattson_ff.md`
- `outputs/validation_board_latest_statcast_top3_v4/scorecards/freddy_peralta_ff.md`
- `outputs/validation_board_latest_statcast_top3_v4/scorecards/taj_bradley_ff.md`
