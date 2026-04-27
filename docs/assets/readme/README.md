# README Evidence Visual Assets

All displayed README visuals are generated from tracked real-data artifacts.

## Data Sources

- `data/processed/skubal_2025.csv`: real Tarik Skubal 2025 Statcast rows.
- `site/data.json`: real holdout rows and generated app samples.
- `outputs/validation_board_skubal_2025_top3_v4/leaderboard.csv`: Skubal pitch-type board.
- `outputs/validation_board_latest_statcast_top3_v4/leaderboard.csv`: latest-Statcast candidate board.
- `outputs/model_tournament_skubal_2025_ff/model_tournament_report.json`: layer tournament.
- `outputs/rolling_validation_skubal_2025_ff/rolling_validation_board.json`: rolling future-window validation.

## Displayed Assets

- `pitch-family-inside-ff.png`: learned style families inside Skubal's FF.
- `overall-results-dashboard.png`: full project result summary.
- `real-vs-generated-diagnostics.png`: real holdout vs generated Skubal FF diagnostics.
- `family-probability-shift.gif`: context-driven family probability changes.
