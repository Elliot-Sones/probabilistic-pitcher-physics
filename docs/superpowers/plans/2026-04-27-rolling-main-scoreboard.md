# Rolling Main Scoreboard Implementation Plan

Goal: make rolling temporal validation the primary project scoreboard, with explicit current metrics, ambitious goals, pass/fail gate logic, generated reports, and a Streamlit presentation view.

## Target Gate

Current regenerated Skubal 2025 FF rolling baseline:

| Metric | Current |
|---|---:|
| Mean rolling physics-core AUC | 0.702 |
| Best fold physics-core AUC | 0.593 |
| Worst fold physics-core AUC | 0.929 |
| Target hit rate | 0.10 |

Impressive goal:

| Metric | Goal |
|---|---:|
| Mean rolling physics-core AUC | <= 0.620 |
| Target hit rate | >= 0.40 |
| Worst fold physics-core AUC | < 0.800 |

## Tasks

1. Add tests for the rolling scoreboard gate.
   - Validate current baseline is marked diagnostic.
   - Validate a board clearing all three goals is marked validated.
   - Validate generated rolling markdown includes the primary scoreboard and goals.

2. Implement scoring in `src/pitcher_twin/rolling_validation.py`.
   - Add default goal constants.
   - Add a reusable scoring helper.
   - Attach `primary_scoreboard` to every rolling board JSON payload.
   - Update markdown so the first section answers whether the board clears the real goal.

3. Expose the board in `app/streamlit_app.py`.
   - Load `outputs/rolling_validation_skubal_2025_ff/rolling_validation_board.json`.
   - Add a "Rolling Scoreboard" tab.
   - Show current vs goal metrics and a fold-by-fold chart.

4. Update presentation docs.
   - Make README and `docs/presentation.md` say rolling validation is now the truth test.
   - Add a research note documenting the current baseline and target gate.

5. Regenerate artifacts and verify.
   - Run the rolling CLI on real Skubal 2025 FF.
   - Run focused rolling tests and full suite.
   - Compile touched Python files.
