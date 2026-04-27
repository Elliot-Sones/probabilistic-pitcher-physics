# README Pitch-Family Results Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the README around the actual product: learned style families inside one pitch type, followed by full overall results and implementation details.

**Architecture:** `scripts/build_readme_visuals.py` will become the reproducible Matplotlib asset builder. It reads real Statcast data, app-generated samples, validation leaderboards, tournament reports, and rolling validation output, then writes README plots/GIFs under `docs/assets/readme/`.

**Tech Stack:** Python, pandas, numpy, scikit-learn, Matplotlib, Pillow, Markdown.

---

### Task 1: Pitch-Family Visuals

**Files:**
- Modify: `scripts/build_readme_visuals.py`
- Modify: `docs/assets/readme/README.md`

- [x] Build `pitch-family-inside-ff.png` from `data/processed/skubal_2025.csv`.
- [x] Cluster only Tarik Skubal FF pitches into style families using real features: `plate_x`, `plate_z`, `pfx_x`, `pfx_z`, `release_speed`, `release_spin_rate`.
- [x] Plot plate-location and movement maps with identical family colors.
- [x] Label the families using their real centroids instead of fake names.

### Task 2: Product And Prediction Visuals

**Files:**
- Modify: `scripts/build_readme_visuals.py`

- [x] Build `family-probability-shift.gif` from `site/data.json`.
- [x] For each context, infer family membership for generated Skubal FF samples using the same learned family centers.
- [x] Animate how family probabilities change across contexts.
- [x] Build `real-vs-generated-diagnostics.png` with real holdout vs generated FF plots: plate location, movement, velocity, family mix.

### Task 3: Overall Results Visuals

**Files:**
- Modify: `scripts/build_readme_visuals.py`

- [x] Build `overall-results-dashboard.png` from validation board and tournament report artifacts.
- [x] Include pitch-level AUC/pass results, layer-level AUC/pass results, and rolling validation summary.
- [x] Remove metric-card-only visual emphasis from the README.

### Task 4: README Rewrite

**Files:**
- Modify: `README.md`

- [x] Section 1: explain style families inside one pitch type.
- [x] Section 2: full overall results.
- [x] Section 3: what we built.
- [x] Section 4: how pitch families are learned.
- [x] Section 5: how conditional generation works.
- [x] Section 6: validation.
- [x] Section 7: demo / run instructions.

### Task 5: Verification And Push

**Files:**
- Modify: `docs/superpowers/plans/2026-04-27-readme-pitch-family-results-reset.md`

- [x] Run `.venv/bin/python scripts/build_readme_visuals.py`.
- [x] Validate README asset links and image dimensions.
- [x] Run `.venv/bin/python -m py_compile scripts/build_readme_visuals.py`.
- [x] Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`.
- [x] Commit README reset.
- [ ] Push to `origin/main`.
