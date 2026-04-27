# Pitcher Twin Overnight Agent Brief

You are working on Pitcher Twin, a real-data ML/DS project for target system Sports.

## Mission

1. Use only real public Statcast data and real public external context data.
2. Expand beyond the existing one-month 2026 smoke-test cache if network and packages allow.
3. Select the best real `(pitcher_name, pitch_type)` candidate by sample size, games, feature completeness, and holdout viability.
4. Train multiple generative model families and multiple feature groups.
5. Validate generated pitches with a classifier two-sample test against held-out real pitches and held-out classifier rows.
6. Prove the model learned player-specific variability by comparing against random-noise and league-average baselines.
7. Add count, fatigue, and real weather/context only if the real data join is available.
8. Save artifacts, metrics, failures, and a morning-readable report.

## Read First

- `docs/superpowers/plans/2026-04-26-pitcher-twin-complete-demo.md`
- `docs/real-data-smoke-test.md`
- `docs/overnight/no-mock-data-rules.md`
- `docs/overnight/success-criteria.md`

## Known Local Real Data

Use this real Statcast cache first:

```text
/Users/elliot18/assistant/projects/pitch-scout/data/processed/latest_statcast.csv
```

That cache covers roughly one month of 2026 data. It is enough for smoke validation, but not enough for the full target of `>= 600` real pitches per selected pitcher/pitch type.

## First Command

From the project root:

```bash
python scripts/run_full_real_data_pipeline.py \
  --data /Users/elliot18/assistant/projects/pitch-scout/data/processed/latest_statcast.csv \
  --output-dir outputs/overnight
```

This should produce:

- `outputs/overnight/run_status.json`
- `outputs/overnight/random_split_report.json`
- `outputs/overnight/temporal_split_report.json`
- `outputs/overnight/morning_report.md`

Build the presentation demo artifact too:

```bash
python scripts/build_demo_artifacts.py \
  --data /Users/elliot18/assistant/projects/pitch-scout/data/processed/latest_statcast.csv \
  --output-dir outputs/real_demo \
  --min-pitches 200 \
  --min-holdout 50 \
  --min-games 4
```

## Then Improve Toward The Full Plan

After the smoke runner works:

1. Install missing dependencies only if the environment allows it.
2. Pull larger real Statcast windows using `pybaseball` if network and packages allow.
3. Build candidate ranking over the larger real dataset.
4. Add model families beyond the current smoke-test baselines.
5. Add context features: count, pitch number, inning, days rest, batter handedness.
6. Add real weather only from a real historical weather source; never fabricate it.
7. Rerun temporal validation.
8. Update `outputs/overnight/morning_report.md` with the best result and blockers.

Only mark a demo artifact as successful if temporal detectability C2ST AUC is `<= 0.60`. Otherwise mark it `diagnostic_not_final` and explain what failed.

## Stop Conditions

Do not stop early because the first result is imperfect. Continue improving until one of these is true:

- full real-data validation report is complete
- missing credentials block progress
- missing packages block progress
- network failure blocks new data download
- real-data unavailability blocks a requested feature

If blocked, write:

```text
outputs/overnight/BLOCKED.md
```

Include:

- exact blocker
- command that failed
- error output summary
- exact next command Elliot should run

## Hard Rule

No mock data. If real data is missing, fail loudly and write the fetch/build command.
