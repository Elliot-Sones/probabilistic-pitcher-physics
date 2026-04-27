# Research log — Pitcher Twin model evolution

This file preserves the chronology of how the model evolved. The README
foregrounds the current honest state; this log preserves what was tried,
what worked, and what didn't.

---

## V1 baseline (static models)

The first generation tested empirical bootstrap, league-mixed empirical, and
multivariate Gaussian generators on Skubal 2025 FF.

| Layer | Best V1 AUC |
|---|---:|
| command/location | 0.514 |
| movement only | 0.519 |
| release only | 0.585 |
| trajectory only | 0.561 |
| physics core | 0.685 |

Command and movement passed the single-split target. Full physics did not.

---

## V2.1 — Factorized physics residual

V2.1 changed the generator from one flat vector into a physics chain:

```text
release / velocity / spin
  → movement residual
  → trajectory residual
  → command residual
```

That structure produced the first major full-physics jump:

| Layer | V1 best | V2.1 | Read |
|---|---:|---:|---|
| command/location | 0.514 | 0.519 | still strong |
| movement only | 0.519 | 0.554 | still realistic |
| release only | 0.585 | 0.575 | improved |
| trajectory only | 0.561 | 0.600 | reached the target edge |
| physics core | 0.685 | 0.611 | large full-physics gain |

V2.1 closed most of the full-physics gap without weather or neural nets.

---

## V2.2 — Weather residual ablation

V2.2 added real Open-Meteo weather as a trainable input source. Game-level
weather joined by `game_pk` using MLB venue coordinates and game time.

| Layer | V2.1 baseline | V2.2 weather | Δ |
|---|---:|---:|---:|
| release only | 0.594 | 0.584 | -0.010 |
| trajectory only | 0.619 | 0.614 | -0.006 |
| physics core | 0.646 | 0.645 | -0.001 |

Real weather is plumbed end to end; lift is real but small. Useful as a
reusable real-data weather path for future park/climate ablation.

---

## V2.3 — Tournament across stronger structures

The tournament moved from one candidate model to a broader search:

- V2.1 factorized baseline
- factorized release-sampler variants
- short-memory release drift plus uncertainty-inflated downstream residuals
- recent-state and trend-state anchored factorized variants
- conditional state-mixture residual sampling
- PCA latent residual sampling
- context nearest-neighbor residual sampling
- derived-feature joint Gaussian over raw physics plus geometry features

Repeated over 30 sample/classifier seeds on Skubal 2025 FF:

| Layer | V2.1 | Best | Best AUC |
|---|---:|---|---:|
| command/location | 0.542 | context neighbor residual | 0.530 |
| movement only | 0.543 | short-memory wide residual | 0.541 |
| release only | 0.618 | factorized trend-state anchored | 0.579 |
| trajectory only | 0.612 | factorized trend-state anchored | 0.543 |
| physics core | 0.634 | factorized trend-state anchored | 0.575 |

Useful conclusion: short-memory uncertainty plus recent/trend state anchoring
captures late-season pitch variation better than the V2.1 factorized baseline.

---

## V2.5 — Trend-anchored release state

V2.5 introduced a trend-state anchor (linear trend + shrinkage) on the release
layer. Improved physics-core to **0.575 AUC** with **79% pass rate** at the
≤0.60 threshold across seeds 42, 101, and 222 — the strongest robust
single-split result.

A second variant (`factorized_release_state_anchored`) over-collapsed the
spin axis and failed validation. Anchoring the trend rather than the absolute
state was the correct move.

---

## V3 — Generalized validation board

V3 turned the single-candidate tournament into a repeatable scorecard system:

```text
real Statcast cache
  → candidate pitcher/pitch selection
  → temporal train/holdout split per candidate
  → model tournament per candidate
  → leaderboard + scorecards + Streamlit board tab
```

### Skubal 2025 top pitch types

| Pitch | Pitches | Best physics model | Physics AUC | Pass rate | Status |
|---|---:|---|---:|---:|---|
| FF | 835 | factorized recent-state anchored | **0.533** | 1.00 | validated |
| SI | 681 | pca_latent_residual | 0.665 | 0.00 | diagnostic |
| CH | 895 | factorized trend-state anchored | 0.680 | 0.00 | diagnostic |

### Latest Statcast partial window

| Pitcher | Pitch | Pitches | Best physics model | Physics AUC | Pass rate | Status |
|---|---|---:|---|---:|---:|---|
| Isaac Mattson | FF | 207 | factorized short-memory more uncertain | 0.576 | 0.50 | candidate |
| Freddy Peralta | FF | 292 | factorized short-memory wide residual | 0.621 | 0.00 | diagnostic |
| Taj Bradley | FF | 279 | factorized recent-state anchored | 0.649 | 0.50 | diagnostic |

Recurring failure signal: spin axis, release geometry, acceleration, and
release speed/spin relationships.

---

## V4 — Pitch-family release/spin (negative result)

V4 tested whether pitch-family-aware release/spin treatment could close the
SI/CH gap:

```text
pitch type
  → pitch family
  → release geometry blend (release_pos_y + release_extension)
  → empirical circular spin-axis residual
  → factorized physics sample
```

Result: diagnostic, not promoted as a new default.

| Pitch | Best after V4 | Physics AUC | V4 candidate AUC | Read |
|---|---|---:|---:|---|
| Skubal FF | factorized recent-state anchored | 0.533 | 0.592 | FF still wins with the older model |
| Skubal SI | conditional state mixture residual | 0.644 | 0.720 | SI still diagnostic |
| Skubal CH | factorized trend-state anchored | 0.680 | 0.701 | CH still diagnostic |

The shipped piece worth keeping is the **model router**: every scorecard now
reports route status, pitch family, recommended physics model, validated
layers, candidate layers, and diagnostic layers.

Conclusion: hand-tuned circular spin residual nudges aren't enough. The next
candidate is a *learned conditional release-state mixture*:

```text
pitch family + recent game state + count/fatigue
  → release speed / spin rate / spin axis mixture
  → movement residual
  → trajectory residual
  → command cloud
```

---

## Rolling temporal scoreboard (current primary gate)

The 70/30 temporal split is now treated as a ceiling, not the truth. Rolling
temporal validation across cumulative future-game windows is the primary
reliability gate.

```text
train games 1-10  → test games 11-12
train games 1-12  → test games 13-14
...
train games 1-28  → test games 29-30
```

On Skubal 2025 FF, 10 rolling folds, 4 repeats per fold:

| Metric | Current | Goal | Status |
|---|---:|---:|---|
| Mean physics-core AUC | 0.702 | ≤ 0.620 | miss |
| Target hit rate | 0.10 | ≥ 0.40 | miss |
| Worst fold physics-core AUC | 0.929 | < 0.800 | miss |
| Best fold physics-core AUC | 0.593 | context only | one fold passes |
| Mean pass rate | 0.23 | supporting metric | — |

Status: `rolling_diagnostic`.

The classifier-clue explainer points the next modeling target squarely at
release/spin geometry across future game windows.

---

## Per-iteration research notes

For full ablation detail, see `docs/research/`:

- `2026-04-27-v2-physics-constrained-state-results.md`
- `2026-04-27-v2-5-release-state-results.md`
- `2026-04-27-v3-validation-board-results.md`
- `2026-04-27-v4-pitch-family-release-spin-results.md`
- `2026-04-27-rolling-main-scoreboard.md`
- `2026-04-27-next-model-angles.md`
