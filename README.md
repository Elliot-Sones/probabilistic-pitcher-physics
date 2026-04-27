# Pitcher Twin

![Pitcher Twin hero: baseball trajectory probability cloud](docs/assets/readme/hero-pitcher-twin.png)

**A real pitcher doesn't throw one fastball — they throw a cloud of them, shaped by count, inning, batter, fatigue, score, and the simple fact that no human releases the ball the same way twice.** Pitcher Twin learns that cloud from public Statcast and generates pitches a classifier struggles to tell apart from real held-out ones.

> ### 🔗 [**Try the live app — pitcher-twin.vercel.app**](https://pitcher-twin.vercel.app)
> Pick a pitcher, count, and batter hand. Watch the model's pitch cloud reshape. &nbsp;·&nbsp; [Read the technical report →](https://pitcher-twin.vercel.app/report)

---

## Best validated result

The cleanest result is **Tarik Skubal 2025 four-seam fastball**. The model trains on earlier real pitches, generates new fastball samples, and validates against later real Statcast pitches.

| Metric | Result |
|---|---:|
| Real Skubal FF pitches | **835** |
| Games represented | **31** |
| Later holdout pitches | **251** |
| Classifier two-sample AUC | **0.533** |
| Repeated-seed pass rate | **100%** |

Lower AUC is better. `0.50` means the classifier has no useful signal for separating generated pitches from real held-out pitches.

![Animated best validated result for Skubal 2025 FF](docs/assets/readme/best-validated-result.gif)

---

## What it does

Pitcher Twin models the **distribution** of a pitcher's actual pitch outcomes rather than a single centroid. It learns release, velocity, spin, movement, trajectory, and command from real Statcast rows, then generates pitch clouds that can be tested against future real pitches.

The goal is a practical generator that answers: *"If I want to practice this pitch type from this pitcher in this game state, what range of pitches should I expect?"*

The output is a Trajekt-shaped session JSON of sampled pitches with **layered validation metadata** — every pitch carries the trust level of every layer (command, movement, trajectory, release, full physics) so consumers can downgrade gracefully when full physics is still diagnostic.

![Animated conditional pitch probability cloud generator](docs/assets/readme/pitch-cloud-generator.gif)

---

## How it works

![Pitcher Twin pipeline from Statcast to Trajekt export](docs/assets/readme/pipeline.png)

### The factorized physics chain (V2.1+)

Instead of modeling all pitch features as one tangled blob, the generator chains them in physical order:

```text
release point / velocity / spin
   → movement residual
   → trajectory residual
   → command residual
```

A Gaussian mixture at each layer captures the natural sub-modes (high-inside vs. low-away vs. middle-up fastballs aren't one distribution). Residual layers absorb the mechanical noise the pitcher didn't intend — the *human-error envelope*. Every layer is conditioned on game state (count, inning, batter handedness, pitch-count fatigue, score differential) and trend-anchored to capture mid-season drift in release point and stuff.

![Animated factorized physics chain](docs/assets/readme/factorized-physics-chain.gif)

### The validator (classifier two-sample test)

For every model variant, we:

1. Train on the earlier 70% of a pitcher's pitches.
2. Generate synthetic samples from the trained model.
3. Mix synthetic samples with real held-out pitches from the later 30%.
4. Train a logistic-regression classifier to tell synthetic from real.
5. Report ROC-AUC. Lower means harder to detect.

The single-split version is the **ceiling**. The rolling temporal version repeats this across many future-game windows — which is now the primary truth test.

![Animated C2ST validation loop](docs/assets/readme/c2st-validator.gif)

### The rolling temporal stress test

```text
train games  1-10  → test games 11-12
train games  1-12  → test games 13-14
...
train games  1-28  → test games 29-30
```

10 rolling folds, 4 repeats per fold. The "miss" flags are honest: the model can match some future windows (best fold 0.593) but still drifts hard on others (worst fold 0.929).

![Animated rolling temporal validation windows](docs/assets/readme/rolling-window-validation.gif)

Editable architecture sketch: [`docs/assets/readme/pitcher-twin-architecture.excalidraw`](docs/assets/readme/pitcher-twin-architecture.excalidraw)

---

## Quick start

```bash
# install
pip install -r requirements.txt

# rebuild the live app's pre-sampled data + static report
python scripts/build_interactive_data.py
python scripts/build_static_site.py

# preview the site locally (fully static — no server needed)
python -m http.server 8000 --directory site
# open http://localhost:8000

# regenerate the headline tournament
python scripts/run_model_tournament.py \
  --data data/processed/skubal_2025.csv \
  --output-dir outputs/model_tournament_skubal_2025_ff \
  --pitcher-id 669373 \
  --pitch-type FF \
  --repeats 30

# regenerate the cross-pitcher validation board
python scripts/run_validation_board.py \
  --data data/processed/skubal_2025.csv \
  --output-dir outputs/validation_board_skubal_2025_top3 \
  --top 3 --repeats 3 --samples 260

# regenerate the rolling truth test
.venv/bin/python scripts/run_rolling_temporal_board.py \
  --data data/processed/skubal_2025.csv \
  --output-dir outputs/rolling_validation_skubal_2025_ff \
  --pitcher-id 669373 --pitch-type FF \
  --initial-train-games 10 --test-games 2 --step-games 2 --repeats 4
```

Run tests: `pytest -q`.

---

## What's in the repo

| Path | What lives there |
|---|---|
| `site/` | The hosted app — `index.html` (interactive sampler), `report.html` (long-form analysis), `data.json` (pre-sampled model output) |
| `src/pitcher_twin/` | Core library — generators, factorized physics chain, validator, tournament, rolling validation |
| `scripts/build_interactive_data.py` | Pre-samples model output for the live app's pitcher × context grid |
| `scripts/build_static_site.py` | Renders the static report page with embedded Plotly charts |
| `scripts/` | Other CLI entry points: data fetch, tournaments, validation boards |
| `tests/` | pytest suite (93 tests) |
| `data/processed/skubal_2025.csv` | Public Statcast pull used by the live app |
| `docs/presentation.md` | Client-facing single-page pitch |
| `docs/research-log.md` | Full V2.1 → V4 model chronology and ablations |
| `outputs/` | Generated tournament results, validation boards, scorecards |

---

## Data policy

- **Real data only.** No mock pitches, no synthetic weather, no fake players.
- Generated samples are **always labeled** as model output, never as observed.
- Holdout pitches are split temporally (last 30% of games), never randomly.

---

## What's next

The current frontier is concentrated in release-geometry and spin-axis modeling. The detection signal — the features the C2ST classifier exploits to spot fakes — is dominated by `release_pos_x`, `release_extension`, `spin_axis_cos/sin`, `release_spin_rate`, and `vy0`. Hand-tuned circular spin residuals (V4) didn't close the gap. The next concrete candidate:

```text
conditional release-state mixture
  with circular spin-axis component
  and velocity/spin covariance per latent state
```

i.e., learn the release/spin distribution as a conditional mixture given pitch family, recent game state, and count/fatigue — instead of fixing it to a global average.

---

## Links

- 📊 [`docs/presentation.md`](docs/presentation.md) — single-page client overview
- 📓 [`docs/research-log.md`](docs/research-log.md) — full V2.1 → V4 model chronology with ablations
- 🔬 [`docs/research/`](docs/research) — individual research notes per iteration
- 🛠 [`docs/implementation-plan.md`](docs/implementation-plan.md) — engineering plan
