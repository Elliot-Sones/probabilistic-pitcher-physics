# Rolling Main Scoreboard

## Why This Matters

The earlier 70/30 temporal split is useful for model development, but it can overstate quality if the final holdout window is unusually easy. Rolling temporal validation is stricter because it repeats the same question across future game windows:

```text
train games 1-10 -> test games 11-12
train games 1-12 -> test games 13-14
...
train games 1-28 -> test games 29-30
```

That makes the rolling board the main reliability scoreboard for the project.

## Current Baseline

Skubal 2025 FF, 10 rolling folds, 4 repeats per fold:

| Metric | Current |
|---|---:|
| Mean physics-core AUC | 0.702 |
| Best fold physics-core AUC | 0.593 |
| Worst fold physics-core AUC | 0.929 |
| Target hit rate | 0.10 |
| Mean pass rate | 0.23 |

Lower C2ST AUC is better. `0.50` means the classifier struggles to tell generated samples from real future pitches.

## Presentation-Grade Goal

| Metric | Goal |
|---|---:|
| Mean rolling physics-core AUC | <= 0.620 |
| Target hit rate | >= 0.40 |
| Worst fold physics-core AUC | < 0.800 |

The current artifact is `rolling_diagnostic`, not `rolling_validated`.

## What The Scoreboard Adds

- A reusable `primary_scoreboard` block in `rolling_validation_board.json`.
- A generated markdown section that states status, goals cleared, current metrics, and gap to goal.
- A Streamlit "Rolling Scoreboard" tab with current-vs-goal metrics and fold-by-fold AUC.
- README and presentation framing that treats rolling validation as the truth test.

## Modeling Read

The best fold at `0.593` shows the model can match some future windows. The worst fold at `0.929` shows the model can still miss a game-window state badly. The main next improvement target is therefore not another single-split score; it is stability across future windows, especially release/spin geometry and full-joint physics.
