# README Evidence Visual Assets

All displayed README visuals are generated from tracked project artifacts.

## Data Sources

- `site/data.json`: real holdout rows and generated app samples.
- `outputs/validation_board_skubal_2025_top3/leaderboard.csv`: best validated result.
- `outputs/model_tournament_skubal_2025_ff/model_tournament_report.json`: layer AUCs.
- `outputs/rolling_validation_skubal_2025_ff/rolling_validation_board.json`: rolling fold AUCs.

## Displayed Assets

- `best-result-summary.png`: top-line Skubal FF validation summary.
- `real-vs-generated-cloud.gif`: real held-out Skubal FF vs generated samples.
- `context-cloud-shift.gif`: actual pre-sampled app contexts from `site/data.json`.
- `model-architecture.png`: factorized model structure.
- `c2st-validation-workflow.png`: classifier two-sample validation flow.
- `layer-results.png`: real repeated-seed tournament layer results.
- `rolling-folds.gif`: real rolling future-window stress test.
- `model-architecture.excalidraw`: editable architecture source.
