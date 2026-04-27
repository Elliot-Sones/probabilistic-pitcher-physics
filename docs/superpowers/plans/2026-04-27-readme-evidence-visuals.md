# README Evidence Visuals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace decorative README visuals with evidence visuals built from real Statcast, generated samples, validation reports, and architecture structure.

**Architecture:** Use `scripts/build_readme_visuals.py` as the single deterministic asset builder. The script reads tracked artifacts (`site/data.json`, `data/processed/skubal_2025.csv`, tournament reports, and rolling validation JSON), writes README images/GIFs under `docs/assets/readme/`, and leaves generated assets reproducible.

**Tech Stack:** Python, Pillow, pandas, JSON artifacts, Markdown, Excalidraw JSON.

---

### Task 1: Real Evidence Visual Builder

**Files:**
- Modify: `scripts/build_readme_visuals.py`
- Modify: `docs/assets/readme/README.md`

- [x] Replace fake/demo visuals with real-data assets:
  - `real-vs-generated-cloud.gif`
  - `context-cloud-shift.gif`
  - `model-architecture.png`
  - `c2st-validation-workflow.png`
  - `layer-results.png`
  - `rolling-folds.gif`
- [x] Read `site/data.json` for generated samples and real holdout rows.
- [x] Read `outputs/model_tournament_skubal_2025_ff/model_tournament_report.json` for layer AUCs.
- [x] Read `outputs/validation_board_skubal_2025_top3/leaderboard.csv` for the best validated headline result.
- [x] Read `outputs/rolling_validation_skubal_2025_ff/rolling_validation_board.json` for rolling fold AUCs.
- [x] Run: `.venv/bin/python scripts/build_readme_visuals.py`

### Task 2: README Structure

**Files:**
- Modify: `README.md`

- [x] Lead with `real-vs-generated-cloud.gif`.
- [x] Section 1: Best validated result, with only the clear Skubal FF result.
- [x] Section 2: What the app does, using `context-cloud-shift.gif`.
- [x] Section 3: How the model works, using `model-architecture.png`.
- [x] Section 4: How validation works, using `c2st-validation-workflow.png` and `layer-results.png`.
- [x] Section 5: Stress test / limits, using `rolling-folds.gif`.
- [x] Keep quick start, repo map, data policy, and next-step sections concise.

### Task 3: Cleanup And Verification

**Files:**
- Remove unused decorative assets from `docs/assets/readme/`

- [x] Remove unreferenced decorative GIFs and AI-art hero assets.
- [x] Validate README asset links and image dimensions with Pillow.
- [x] Validate Excalidraw JSON.
- [x] Run: `.venv/bin/python -m py_compile scripts/build_readme_visuals.py`
- [x] Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`
- [x] Commit only README/evidence visual changes.
