# Pitcher Twin Real-Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` for inline implementation or `superpowers:subagent-driven-development` for independent finishing tasks. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a completed, real-data Pitcher Twin project that finds the best real MLB pitcher/pitch candidates, trains generative pitch-variability models on public Statcast, proves the model learned player-specific variability rather than random noise, joins real external context where available, validates realism rigorously, iterates model choice until the acceptance criteria are met or honestly reports failure, and exports machine-session JSON.

**Architecture:** Real data only. Public Statcast is the source of truth for pitches. Real venue/time weather is joined from a public historical weather source and cached. Tests use small cached slices of real public data, never fabricated rows. The modeling stack compares multiple generative models and multiple feature representations under the same validation protocol; GMM is the interpretable baseline, not the assumed winner.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn, scipy, matplotlib/seaborn, plotly, pyarrow, joblib, pytest, ruff, pybaseball, Open-Meteo or Meteostat for historical weather. Streamlit is optional polish, not a required deliverable.

---

## Absolute Data Rules

- **No mock data.**
- **No synthetic pitch rows.**
- **No fake weather rows.**
- **No fabricated player examples.**
- **No demo fallback that silently invents data.**
- Unit/integration tests may use small fixture files, but those fixture files must be sampled from real public Statcast and real public weather caches.
- If real data is missing, scripts must fail with a clear command to fetch/build the real dataset.
- Generated samples from the trained model are allowed because they are the model output. They must always be labeled as `simulated_from_model`, not real.

## Thesis

Real pitchers do not throw one perfect pitch. They throw a context-sensitive distribution.

```text
pitcher variability =
  natural pitch-to-pitch variation
+ count/game-state effects
+ fatigue/workload effects
+ batter-handedness effects
+ venue/weather effects
+ season-to-season drift
```

Pitcher Twin learns that distribution from real data, samples realistic sessions, and validates whether those samples are hard to distinguish from held-out real pitches.

## Finish Lines

### Finish Line A: Real Baseline Demo

This is the minimum project worth showing.

- Real Statcast data loaded for at least one selected pitcher/pitch pair.
- Player/pitch candidate chosen by sample size, completeness, and validation viability.
- Multiple generative model families trained on real pitch features.
- Same-season holdout validation completed.
- Player-specific model beats random-noise and league-average baselines on the real-vs-generated classifier test.
- Model-generated session exported as machine-session JSON.
- Real-vs-simulated plots, validation comparison table, leakage analysis, and JSON export are saved as artifacts.

### Finish Line B: Full Contextual ML/DS Demo

This is the stronger version.

- Real fatigue/context features included.
- Real historical weather joined by venue and game time.
- Broad per-season feature library built before modeling.
- Feature groups and model families selected by validation, not by intuition.
- Context-conditioned models tested.
- Ablation table compares base, count, fatigue, and weather/context models.
- Drift analysis compares seasons.
- Best model is selected by validation, not by preference.

## Realism Acceptance Criteria

The project does not claim success until validation supports it.

Primary metric:

- Classifier two-sample test AUC comparing held-out real pitches vs model samples.

Core proof:

```text
If the classifier easily detects random noise but struggles to detect
the player-specific model, then the model learned player-like variation.
```

Targets:

- `AUC <= 0.60`: acceptable realism.
- `0.60 < AUC <= 0.70`: borderline; inspect leakage and improve model.
- `AUC > 0.70`: not realistic enough; do not present as successful.

Secondary metrics:

- per-feature standardized Wasserstein distance
- covariance/correlation error
- marginal distribution overlap plots
- top classifier leakage features

Required comparison table:

```text
Generator                         Real-vs-generated AUC
Random independent noise           expected high / easy to detect
League-average pitch model         expected medium
Player-specific model              target <= 0.60
Player + context model             should improve or explain leakage
```

Do not only report the player model AUC. The random and league baselines are what prove the model is learning player-specific structure rather than arbitrary spread.

Model-selection rule:

1. Build a broad real feature library.
2. Define feature groups before training.
3. Train several model families on the same train split.
4. Validate every `(model family, feature group)` candidate on the same held-out real pitches.
5. Select the simplest candidate that meets realism criteria and beats random/league baselines.
6. If no candidate meets the target, report the best failure honestly with leakage features and next steps.

Do not test an uncontrolled infinite grid. The search must be broad but structured, with a final untouched holdout so the model does not win by overfitting validation.

## Model Families To Compare

GMM is only the first serious baseline. Compare:

- **Independent random-noise baseline**
  - Proves arbitrary spread is easy to detect.
- **League-average empirical model**
  - Samples real pitches of the same pitch type across pitchers.
  - Proves player-specific modeling adds value beyond generic pitch-type realism.
- **Player empirical bootstrap**
  - Resamples real training pitches from the selected pitcher/pitch.
  - Strong nonparametric baseline, but cannot generate new combinations.
- **Multivariate Gaussian**
  - Single covariance cloud.
  - Tests whether one simple distribution is enough.
- **Gaussian Mixture Model**
  - Multiple covariance clouds.
  - Good interpretable baseline for multimodal pitch shapes.
- **KDE or Gaussian copula**
  - Tests whether non-Gaussian marginals matter.
- **Conditional GMM**
  - Separate or keyed models by count/fatigue/weather buckets when sample size allows.
- **Conditional density model**
  - Optional later model, such as mixture density network or normalizing flow, only if classical models fail and sample size supports it.

## Feature Representation Search

The project should test different physics representations, not assume one.

Feature sets:

- **Release-only**
  - velocity, spin, spin axis, release position, extension
- **Movement-only**
  - `pfx_x`, `pfx_z`, plate location, zone-normalized location
- **Trajectory-only**
  - `vx0`, `vy0`, `vz0`, `ax`, `ay`, `az`
- **Endpoint representation**
  - release point plus plate crossing point
- **Shape representation**
  - release + spin + movement
- **Command representation**
  - plate location, zone distance, chase/strike-zone indicators
- **Physics core**
  - release + spin + movement + trajectory
- **Physics + count**
  - physics core plus count bucket
- **Physics + fatigue**
  - physics core plus pitch number, inning, days rest, workload when real
- **Physics + batter context**
  - physics core plus batter handedness and game state
- **Physics + weather**
  - physics core plus real joined weather fields where available
- **All available context**
  - broadest real feature group after missingness filtering

Each feature set must report:

- rows retained after missing-value filtering
- feature completeness
- C2ST AUC
- Wasserstein/covariance error
- top leakage features

The final model is the best validated pair:

```text
best_model = argmin_validation_error(model_family, feature_set)
```

## Best Player Selection

Do not hand-pick the star name first. Let the data choose the best candidate.

Create a candidate ranking over `(pitcher_name, pitch_type)` with:

- total pitch count
- number of games
- feature completeness
- same-season train/holdout split viability
- count/fatigue/context coverage
- movement/velocity variability signal
- public demo relevance

Minimum thresholds:

- `>= 600` real pitches for the selected pitcher/pitch type across the training window.
- `>= 150` real pitches in the holdout split.
- `>= 4` games.
- `>= 95%` completeness for core physics features.

Candidate pool to evaluate first:

- Kevin Gausman splitter / four-seam
- Tarik Skubal four-seam / slider
- Chris Sale slider / four-seam
- Zack Wheeler four-seam / sinker
- Logan Webb sinker
- Framber Valdez sinker
- Joe Ryan four-seam
- Freddy Peralta four-seam
- George Kirby four-seam
- Logan Gilbert four-seam

If these fail thresholds, automatically choose the highest-ranked real candidate from the data.

## Validation Principles

- Never report training-set AUC as realism.
- Same-season temporal holdout is the primary realism test.
- Next-season evaluation is drift/generalization, not pure realism.
- Use stratified cross-validation or held-out classifier splits inside the classifier two-sample test.
- Report confidence intervals.
- Report top leakage features.
- Always compare against random independent noise and a league-average model.
- The player-specific model must outperform those baselines to support the "player noise, not random noise" claim.
- Do not claim fatigue/weather helps unless ablations prove it improves out-of-sample validation or explains leakage.
- If weather does not help, say so. That is a valid finding.

## File Responsibility Map

- `src/pitcher_twin/data.py`: real Statcast loading, cache validation, candidate discovery.
- `src/pitcher_twin/schema.py`: required columns and feature groups.
- `src/pitcher_twin/context.py`: count, fatigue, rest, handedness, game-state features.
- `src/pitcher_twin/weather.py`: real weather fetch/cache/join.
- `src/pitcher_twin/features.py`: real feature matrix construction.
- `src/pitcher_twin/feature_library.py`: broad per-season feature library and feature availability report.
- `src/pitcher_twin/feature_selection.py`: validation-driven feature group/model selection.
- `src/pitcher_twin/models.py`: model registry, GMM, alternative model interfaces.
- `src/pitcher_twin/model_selection.py`: candidate/model selection by validation.
- `src/pitcher_twin/validator.py`: C2ST, distribution distance, drift, ablations.
- `src/pitcher_twin/sampler.py`: model sampling API.
- `src/pitcher_twin/machine_session_format.py`: machine-session JSON.
- `src/pitcher_twin/visualize.py`: plots.
- `scripts/fetch_real_statcast.py`: real Statcast fetch/copy/cache.
- `scripts/fetch_real_weather.py`: real historical weather cache.
- `scripts/build_real_fixtures.py`: create small real-data test fixtures from caches.
- `scripts/select_best_candidate.py`: rank real pitcher/pitch candidates.
- `scripts/build_demo_artifacts.py`: train, validate, sample, plot, export.
- `scripts/run_full_real_data_pipeline.py`: overnight-safe runner that executes real-data checks and writes machine-readable run status.
- `scripts/summarize_overnight_results.py`: convert overnight JSON outputs into a concise morning report.
- `docs/overnight/2026-04-26-overnight-agent-brief.md`: `ml-intern` prompt/brief for autonomous overnight work.
- `docs/overnight/no-mock-data-rules.md`: hard rules the overnight agent must follow.
- `docs/overnight/success-criteria.md`: objective pass/fail criteria for real results.
- `app/streamlit_app.py`: optional real artifact UI.
- `tests/fixtures/`: small real public data slices only.
- `docs/assets/`: tracked real result figures and JSON excerpts.

---

## Phase 0: Guardrails

**Purpose:** Make the no-mock-data rule executable.

**Files:**

- Create: `docs/data-policy.md`
- Create: `docs/final-acceptance-checklist.md`
- Create: `docs/assets/.gitkeep`
- Create: `tests/fixtures/.gitkeep`
- Create: `scripts/.gitkeep`

- [ ] **Step 0.1: Add data policy**

Create `docs/data-policy.md`:

```markdown
# Data Policy

Pitcher Twin is a real-data project.

Rules:

- Do not create mock pitch rows.
- Do not create synthetic weather rows.
- Do not silently fall back to demo data.
- Tests may use fixture files only if those files are sampled from real public Statcast or real public weather caches.
- Generated model samples must be labeled as simulated outputs.
- If required real data is missing, scripts must fail with a clear fetch/build command.
```

- [ ] **Step 0.2: Add acceptance checklist**

Create `docs/final-acceptance-checklist.md`:

```markdown
# Final Acceptance Checklist

- [ ] Real Statcast cache exists.
- [ ] Real weather cache exists when context/weather mode is enabled.
- [ ] No code path creates mock pitch rows.
- [ ] No code path creates fake weather rows.
- [ ] Test fixtures are real-data slices.
- [ ] Candidate ranking selects the best real pitcher/pitch pair.
- [ ] Same-season validation is complete.
- [ ] Model selected by validation meets or honestly fails the target AUC.
- [ ] Ablation table exists for full contextual mode.
- [ ] Optional Streamlit app runs from real artifacts if built.
- [ ] README shows real metrics and real player results.
```

- [ ] **Step 0.3: Commit guardrails**

Run:

```bash
mkdir -p docs/assets tests/fixtures scripts
touch docs/assets/.gitkeep tests/fixtures/.gitkeep scripts/.gitkeep
git add docs/data-policy.md docs/final-acceptance-checklist.md docs/assets/.gitkeep tests/fixtures/.gitkeep scripts/.gitkeep docs/superpowers/plans/2026-04-26-pitcher-twin-complete-demo.md
git commit -m "docs: enforce real data only project plan"
```

---

## Phase 0.5: Overnight Agent Pack

**Purpose:** Let `ml-intern` work while the user sleeps without drifting into fake data, vague research, or unvalidated claims.

**Runner choice:** Use `ml-intern`, not the `param_golf/autoresearch` repo.

Reason:

- `ml-intern` is built for autonomous ML/research coding with local file tools, dataset tools, papers, GitHub, and headless prompts.
- `autoresearch` is designed around one specific loop: modify `train.py`, train a small LLM for 5 minutes, keep/discard by `val_bpb`. That loop does not match Pitcher Twin.

**Files:**

- Create: `docs/overnight/2026-04-26-overnight-agent-brief.md`
- Create: `docs/overnight/no-mock-data-rules.md`
- Create: `docs/overnight/success-criteria.md`
- Create: `scripts/run_full_real_data_pipeline.py`
- Create: `scripts/summarize_overnight_results.py`

- [ ] **Step 0.5.1: Add the overnight brief**

Create `docs/overnight/2026-04-26-overnight-agent-brief.md` with:

```markdown
# Pitcher Twin Overnight Agent Brief

You are working on Pitcher Twin, a real-data ML/DS project for target system Sports.

Mission:

1. Use only real public Statcast data and real public external context data.
2. Expand beyond the existing one-month 2026 smoke-test cache if network and packages allow.
3. Select the best real `(pitcher_name, pitch_type)` candidate by sample size, games, feature completeness, and holdout viability.
4. Train multiple generative model families and multiple feature groups.
5. Validate generated pitches with a classifier two-sample test against held-out real pitches.
6. Prove the model learned player-specific variability by comparing against random-noise and league-average baselines.
7. Add count, fatigue, and real weather/context only if the real data join is available.
8. Save artifacts, metrics, failures, and a morning-readable report.

Start by reading:

- `docs/superpowers/plans/2026-04-26-pitcher-twin-complete-demo.md`
- `docs/real-data-smoke-test.md`
- `docs/overnight/no-mock-data-rules.md`
- `docs/overnight/success-criteria.md`

Known local real data:

- `/Users/elliot18/assistant/projects/pitch-scout/data/processed/latest_statcast.csv`

Run the current smoke-test harness first:

```bash
python scripts/run_full_real_data_pipeline.py \
  --data /Users/elliot18/assistant/projects/pitch-scout/data/processed/latest_statcast.csv \
  --output-dir outputs/overnight
```

Then improve the project toward the full plan.

Do not stop unless blocked by missing credentials, missing packages, network failure, or real-data unavailability. If blocked, write the blocker and exact next command to `outputs/overnight/BLOCKED.md`.
```

- [ ] **Step 0.5.2: Add no-mock-data overnight rules**

Create `docs/overnight/no-mock-data-rules.md`:

```markdown
# No Mock Data Rules

The overnight agent must not use mock rows, fake players, fake weather, or fabricated examples.

Allowed:

- Real public Statcast rows.
- Real public weather/stadium/context rows.
- Small fixtures sampled from real public caches with provenance metadata.
- Model-generated pitch samples labeled as `simulated_from_real_model`.

Forbidden:

- Mock pitch rows.
- Synthetic weather rows.
- Fake player examples.
- Silent fallback demo data.
- Validation on the same data used to fit the generator.
- Reporting a result without the data path, date range, row count, and validation split.

If real data is missing, fail loudly and write the fetch command. Do not invent data.
```

- [ ] **Step 0.5.3: Add success criteria**

Create `docs/overnight/success-criteria.md`:

```markdown
# Overnight Success Criteria

The overnight run is successful only if it produces real artifacts.

Required artifacts:

- `outputs/overnight/run_status.json`
- `outputs/overnight/morning_report.md`
- `outputs/overnight/random_split_report.json`
- `outputs/overnight/temporal_split_report.json`
- candidate ranking or blocker explaining why full ranking could not be run

Minimum real-result criteria:

- Uses real Statcast rows.
- Reports data path, date range, row count, and feature count.
- Evaluates random-noise, league-average, player bootstrap, and player parametric baselines.
- Runs random split and temporal split validation.
- Explains whether the current result supports the project.

Strong-result criteria:

- At least one selected candidate has `>= 600` real pitches for a pitch type.
- Holdout has `>= 150` real pitches.
- Player-specific model is much harder to detect than league-average baseline.
- Context/fatigue/weather ablation improves temporal validation or explains leakage.

Failure is acceptable if it is honest. The morning report must say what failed, why, and what exact next data/model step is needed.
```

- [ ] **Step 0.5.4: Add overnight runner and summary commands**

Create scripts that can run unattended:

```bash
python scripts/run_full_real_data_pipeline.py \
  --data /Users/elliot18/assistant/projects/pitch-scout/data/processed/latest_statcast.csv \
  --output-dir outputs/overnight

python scripts/summarize_overnight_results.py \
  --input-dir outputs/overnight \
  --output outputs/overnight/morning_report.md
```

The scripts must fail if the data path is missing. They must not create fallback data.

- [ ] **Step 0.5.5: Run with `ml-intern` while sleeping**

From the project root:

```bash
cd /Users/elliot18/assistant/projects/pitch-pitcher-twin

ml-intern --max-iterations 200 "
Read docs/overnight/2026-04-26-overnight-agent-brief.md.
Work autonomously until the full real-data validation report is complete.
Do not use mock data.
Do not use fabricated weather data.
Do not stop unless blocked by missing credentials, missing packages, network failure, or real-data unavailability.
If blocked, write outputs/overnight/BLOCKED.md with the exact blocker and next command.
"
```

Morning review:

```bash
sed -n '1,240p' outputs/overnight/morning_report.md
cat outputs/overnight/run_status.json
```

---

## Phase 1: Real Data Ingestion

**Purpose:** Build the project on real public data.

**Files:**

- Modify: `src/pitcher_twin/data.py`
- Create: `scripts/fetch_real_statcast.py`
- Create: `scripts/build_real_fixtures.py`
- Create: `tests/test_real_data_policy.py`

- [ ] **Step 1.1: Implement real cache loading**

`src/pitcher_twin/data.py` must implement:

- `REQUIRED_STATCAST_COLUMNS`
- `ensure_required_columns(df)`
- `load_statcast_cache(path)`
- `write_statcast_cache(df, path)`
- `load_existing_statcast_sources(paths)`
- `fetch_statcast_range(start_date, end_date, output_path)`

Rules:

- No generated fallback.
- If file is missing, raise `FileNotFoundError` with fetch instructions.
- Support CSV and parquet.
- Always validate required columns.

- [ ] **Step 1.2: Build or reuse real Statcast cache**

Prefer the existing real cache if available:

```bash
/Users/elliot18/assistant/projects/pitch-scout/data/processed/latest_statcast.csv
```

Then expand with pybaseball for a larger window:

```bash
python scripts/fetch_real_statcast.py \
  --start 2024-03-20 \
  --end 2026-04-25 \
  --output data/raw/statcast_2024_2026.parquet
```

If a full fetch is too slow, continue with the existing real 2026 cache but mark the validation as limited by sample size.

- [ ] **Step 1.3: Create real test fixtures**

Create `scripts/build_real_fixtures.py` that samples small real slices from the cache:

- `tests/fixtures/real_statcast_sample.parquet`
- `tests/fixtures/real_pitcher_candidates.parquet`

Rules:

- sample from actual cache
- preserve real player names
- include provenance metadata JSON:
  - source file
  - date range
  - row count
  - creation time

- [ ] **Step 1.4: Add real-data policy tests**

Tests must verify:

- fixture file exists
- fixture has provenance metadata
- fixture rows contain real Statcast columns
- missing cache raises, not fallback

Run:

```bash
pytest tests/test_real_data_policy.py -q
```

---

## Phase 2: Candidate Selection

**Purpose:** Find the best real player/pitch pair for the project.

**Files:**

- Modify: `src/pitcher_twin/data.py`
- Create: `src/pitcher_twin/candidates.py`
- Create: `scripts/select_best_candidate.py`
- Create: `tests/test_candidates.py`

- [ ] **Step 2.1: Implement candidate ranking**

Rank `(pitcher_name, pitch_type)` by:

```text
candidate_score =
  0.30 * sample_size_score
+ 0.20 * games_score
+ 0.15 * feature_completeness
+ 0.15 * holdout_viability
+ 0.10 * context_coverage
+ 0.10 * variability_signal
```

Core feature completeness columns:

- `release_speed`
- `release_spin_rate`
- `spin_axis`
- `release_pos_x`
- `release_pos_y`
- `release_pos_z`
- `release_extension`
- `pfx_x`
- `pfx_z`
- `plate_x`
- `plate_z`
- `vx0`
- `vy0`
- `vz0`
- `ax`
- `ay`
- `az`

- [ ] **Step 2.2: Run candidate selector**

Run:

```bash
python scripts/select_best_candidate.py \
  --input data/raw/statcast_2024_2026.parquet \
  --output outputs/candidate_rankings.csv
```

If using the existing 2026 cache:

```bash
python scripts/select_best_candidate.py \
  --input /Users/elliot18/assistant/projects/pitch-scout/data/processed/latest_statcast.csv \
  --output outputs/candidate_rankings.csv
```

- [ ] **Step 2.3: Lock selected real candidates**

Create `outputs/selected_candidates.json` with:

- primary candidate
- backup candidate
- sample counts
- date range
- reason selected

Do not choose manually unless the selected candidate fails validation and the reason is documented.

---

## Phase 3: Real Per-Season Feature Library

**Purpose:** Build many real, per-season feature groups first, then let validation decide which groups belong in the final model.

**Files:**

- Create: `src/pitcher_twin/schema.py`
- Create: `src/pitcher_twin/context.py`
- Create: `src/pitcher_twin/features.py`
- Create: `src/pitcher_twin/feature_library.py`
- Create: `src/pitcher_twin/feature_selection.py`
- Create: `tests/test_features_real.py`

- [ ] **Step 3.1: Build per-season feature tables**

Every feature table must include:

- `season`
- `game_year`
- `game_date`
- `pitcher_name`
- `pitcher_id`
- `pitch_type`
- `game_pk`
- `pitch_number`
- feature availability flags

Build separate feature tables per season, then a combined panel:

```text
features_2024.parquet
features_2025.parquet
features_2026.parquet
features_all.parquet
```

Per-season modeling matters because a pitcher can change mechanics, pitch mix, velocity, or command year to year.

- [ ] **Step 3.2: Build broad pitch physics features**

Features:

- spin axis as `cos/sin`
- release speed
- spin rate
- release position
- release extension
- movement
- plate location
- trajectory parameters
- effective speed when available
- zone
- strike-zone normalized plate location
- release-side features split by pitcher handedness when available

- [ ] **Step 3.3: Build command and outcome-adjacent features**

These are not all model inputs by default, but they are candidates.

Features:

- plate location spread
- distance from zone center
- chase-zone indicator
- called-strike-zone indicator
- batted-ball result fields when available
- whiff/called-strike/contact indicators when available

- [ ] **Step 3.4: Build game/fatigue context**

Features:

- count bucket
- inning
- pitch number
- times through order when available
- days rest when available
- recent workload when derivable
- batter handedness
- score differential when available
- base/out state when available
- home/away
- catcher/batter IDs when useful as grouping keys
- days since previous game when available
- rolling 7-day workload when derivable
- rolling 14-day workload when derivable

No fabricated fatigue values. If a fatigue feature cannot be derived from real data, mark it missing and exclude it from that model variant.

- [ ] **Step 3.5: Build feature availability report**

Create:

```text
outputs/feature_availability.csv
outputs/feature_groups.json
```

The report must show, by season and by candidate pitcher/pitch:

- non-null rate
- number of usable rows
- number of games
- whether feature can be used in modeling
- reason excluded if excluded

- [ ] **Step 3.6: Validation-driven feature group selection**

Do not choose one giant feature set by intuition.

Define feature groups:

```text
release_only
movement_only
trajectory_only
endpoint
shape
command
physics_core
physics_plus_count
physics_plus_fatigue
physics_plus_batter_context
physics_plus_weather
all_available_context
```

For each feature group:

1. train on the same real train split
2. generate samples
3. run real-vs-generated classifier validation
4. compute distribution distances
5. record leakage features

Select the feature group that gives the best validation result without unstable sample loss.

Save:

```text
outputs/feature_selection_report.json
outputs/feature_selection_table.csv
```

---

## Phase 4: Real Weather Join

**Purpose:** Add external variables without inventing them.

**Files:**

- Create: `src/pitcher_twin/weather.py`
- Create: `scripts/fetch_real_weather.py`
- Create: `tests/test_weather_real.py`

- [ ] **Step 4.1: Build venue/time join**

Join real weather by:

- venue/stadium
- game date/time
- nearest hourly weather record

Weather fields:

- temperature
- humidity
- pressure
- wind speed
- wind direction
- precipitation
- dome/open-roof flag when available

- [ ] **Step 4.2: Cache real weather**

Run:

```bash
python scripts/fetch_real_weather.py \
  --games data/processed/model_games.parquet \
  --output data/raw/weather_cache.parquet
```

Rules:

- If weather fetch fails, weather model variant is unavailable.
- Do not fill with fake weather.
- Dome games may have outdoor weather joined but must be flagged.

---

## Phase 5: Model And Feature Search

**Purpose:** Search model families and feature representations under one validation protocol, then select the simplest real-data model that actually works.

**Files:**

- Create: `src/pitcher_twin/models.py`
- Create: `src/pitcher_twin/model_selection.py`
- Create: `tests/test_models_real.py`

- [ ] **Step 5.1: Build model candidate grid**

Create a controlled candidate grid:

```text
model_families = [
  random_independent_noise,
  league_average_empirical,
  player_empirical_bootstrap,
  multivariate_gaussian,
  gmm,
  gaussian_copula_or_kde,
  conditional_gmm_if_sample_size_allows
]

feature_sets = [
  release_only,
  movement_only,
  trajectory_only,
  endpoint,
  shape,
  command,
  physics_core,
  physics_plus_count,
  physics_plus_fatigue,
  physics_plus_batter_context,
  physics_plus_weather,
  all_available_context
]
```

Skip any `(model_family, feature_set)` pair when:

- feature completeness is too low
- rows retained fall below threshold
- conditioning bucket has insufficient samples
- weather coverage is missing or mostly dome/unknown

- [ ] **Step 5.2: Train candidates**

For every valid pair:

1. train on the same real training split
2. sample the same number of generated pitches as the holdout set
3. store model metadata
4. store retained feature rows and missingness report

GMM-specific rules:

- standardize features
- search `n_components=1..8`
- choose by BIC within the training split only
- use full covariance when sample size supports it

- [ ] **Step 5.3: Validate candidates**

Run the same validation on every candidate:

- C2ST AUC
- confidence interval
- Wasserstein distance by feature
- covariance/correlation error
- top leakage features
- qualitative plot checks

- [ ] **Step 5.4: Select final model**

Final model must be selected by validation:

- primary realism AUC
- distribution distance
- leakage features
- sample visual quality
- feature selection report
- sample retention after missing-value filtering
- model simplicity

Save:

- `models/final_model.joblib`
- `outputs/model_selection_report.json`
- `outputs/model_feature_search.csv`

---

## Phase 6: Validation And Ablations

**Purpose:** Prove the model learned player-specific pitch variation, not arbitrary random noise, and identify which features matter.

**Files:**

- Create: `src/pitcher_twin/validator.py`
- Create: `src/pitcher_twin/ablations.py`
- Create: `tests/test_validation_real.py`

- [ ] **Step 6.1: Same-season realism**

Train model on early season/window, hold out later real pitches.

Report:

- AUC
- confidence interval
- top leakage features
- Wasserstein distances
- covariance error

- [ ] **Step 6.2: Baseline generator comparison**

Compare the selected player-specific generator against:

1. **Random independent noise baseline**
   - sample each feature independently from broad normal/uniform ranges derived from the training data
   - intentionally breaks player-specific covariance
   - classifier should detect it easily

2. **League-average pitch model**
   - train on the same pitch type across many pitchers
   - preserves pitch-type realism but not the selected player's personal signature

3. **Player-specific model**
   - train only on the selected pitcher/pitch type
   - should be harder to distinguish from held-out real pitches

Report:

```text
Generator                         Real-vs-generated AUC
Random independent noise
League-average pitch model
Player-specific model
Player-specific + context model
```

This table is the main proof that the model learned player noise rather than random noise.

- [ ] **Step 6.3: Context holdout**

Hold out a context bucket, such as:

- late fatigue
- full count
- left-handed batter
- cold/humid weather if real coverage exists

Measure whether the context model reproduces that bucket better than the base model.

- [ ] **Step 6.4: Ablation table**

Compare:

- `base_gmm`
- `gmm_count`
- `gmm_count_fatigue`
- `gmm_count_fatigue_weather`
- optional stronger model

Only claim a context feature helps if validation supports it.

---

## Phase 7: Sampling And Machine Session JSON

**Purpose:** Generate realistic sessions from the selected real model.

**Files:**

- Create: `src/pitcher_twin/sampler.py`
- Create: `src/pitcher_twin/machine_session_format.py`
- Create: `tests/test_export_real.py`

- [ ] **Step 7.1: Sample session**

Generate `n=15` pitches from the selected model.

Each pitch includes:

- release
- velocity
- spin
- movement
- plate target
- trajectory values
- context metadata
- simulated flag

- [ ] **Step 7.2: Export JSON**

Write:

- `outputs/samples/final_session.json`
- `docs/assets/final_session_excerpt.json`

Every generated pitch must be labeled:

```json
"source": "simulated_from_real_model"
```

---

## Phase 8: Visualizations

**Purpose:** Make the real model inspectable.

**Files:**

- Create: `src/pitcher_twin/visualize.py`

Figures:

- real vs simulated velocity
- real vs simulated release point
- real vs simulated movement
- real vs simulated plate location
- fatigue curve
- weather/context plot if real weather is available
- ablation results
- leakage feature chart

Save runtime outputs to `outputs/figures/`.
Copy forwardable figures to `docs/assets/`.

---

## Optional Phase 9: Streamlit App

**Purpose:** Present the completed real-data project if time allows. This is not required if the notebook/report artifacts are clear.

**Files:**

- Create/modify: `app/streamlit_app.py`

The app must show:

- selected real player and pitch
- data window and sample counts
- model selected
- validation metrics
- ablation results
- real vs simulated plots
- generated pitch table
- JSON export

If artifacts are missing, show the exact command to build them. Do not use fallback fake data.

---

## Phase 10: README And Final Project Polish

**Purpose:** Make it forwardable to target system.

README must include:

- selected real player/pitch and why
- data source and date range
- model architecture
- validation results
- whether the target AUC was met
- ablation table
- limitations
- exact commands
- screenshots/figures from real outputs

Demo line:

> I selected the best real pitcher/pitch candidate from public Statcast by sample size and feature completeness, trained a generative variability model, proved it learned player-specific variation by beating random-noise and league-average baselines in a real-vs-generated classifier test, and exported a machine-session session from the final real-data model.

---

## Final Verification

Run:

```bash
pytest -q
python -m compileall src app scripts tests
python scripts/build_demo_artifacts.py --real-data --select-best --include-context
```

If installed:

```bash
ruff check src tests app scripts
```

Optional manual UI:

```bash
streamlit run app/streamlit_app.py
```

Check:

- no mock-data code paths
- real selected player
- real date range
- validation report exists
- final model exists
- JSON export exists
- app runs if optional UI is built
- README reflects actual results
