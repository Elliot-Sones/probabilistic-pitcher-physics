# Technical-First README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the README around a technical-first explanation with accurate visuals and the V2.1 model results.

**Architecture:** Add deterministic README assets under `docs/assets/readme/`, then rewrite `README.md` to lead with the technical modeling concept, real pitch variation, model pipeline, V2.1 physics chain, results, and reproduction commands. Keep the README honest about validated component layers and diagnostic full physics. Add a reusable script so PNG/GIF/SVG/Excalidraw assets can be regenerated.

**Tech Stack:** Markdown, SVG, Excalidraw JSON, Python, pandas, matplotlib, pillow, imageio, existing Statcast CSV and validation reports.

---

### Task 1: Prepare Documentation Scratch Hygiene

**Files:**
- Modify: `.gitignore`
- Create: `docs/superpowers/specs/2026-04-27-technical-first-readme-design.md`
- Create: `docs/superpowers/plans/2026-04-27-technical-first-readme.md`

- [ ] **Step 1: Ignore visual-companion scratch**

Add `.superpowers/` to `.gitignore` so browser brainstorming files are not tracked.

- [ ] **Step 2: Commit spec and plan**

Run:

```bash
git add .gitignore docs/superpowers/specs/2026-04-27-technical-first-readme-design.md docs/superpowers/plans/2026-04-27-technical-first-readme.md
git commit -m "Plan technical-first README"
```

### Task 2: Generate README Visual Assets

**Files:**
- Modify: `requirements.txt`
- Modify: `pyproject.toml`
- Create: `scripts/build_readme_visuals.py`
- Test: `tests/test_readme_visuals.py`
- Create: `docs/assets/readme/skubal_ff_variation.png`
- Create: `docs/assets/readme/skubal_ff_pitch_sequence.gif`
- Create: `docs/assets/readme/v21_results_auc.png`
- Create: `docs/assets/readme/pitcher_twin_pipeline.svg`
- Create: `docs/assets/readme/pitcher_twin_pipeline.excalidraw`
- Create: `docs/assets/readme/v21_physics_chain.svg`
- Create: `docs/assets/readme/v21_physics_chain.excalidraw`

- [ ] **Step 1: Add visualization dependencies**

Add to both `requirements.txt` and `pyproject.toml`:

```text
pillow>=10.3
imageio>=2.34
```

- [ ] **Step 2: Write a failing script smoke test**

Create `tests/test_readme_visuals.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    script = Path(__file__).parents[1] / "scripts" / "build_readme_visuals.py"
    spec = importlib.util.spec_from_file_location("build_readme_visuals", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_readme_visuals_exposes_main_and_expected_assets() -> None:
    module = _load_script()

    assert callable(module.main)
    assert "skubal_ff_variation.png" in module.EXPECTED_ASSETS
    assert "skubal_ff_pitch_sequence.gif" in module.EXPECTED_ASSETS
    assert "pitcher_twin_pipeline.excalidraw" in module.EXPECTED_ASSETS
```

- [ ] **Step 3: Run the smoke test to verify failure**

Run:

```bash
pytest tests/test_readme_visuals.py -q
```

Expected: failure because `scripts/build_readme_visuals.py` does not exist.

- [ ] **Step 4: Create the visual generation script**

Create `scripts/build_readme_visuals.py` with functions that:

- load Skubal CSV from `data/processed/skubal_2025.csv` or `/Users/elliot18/Desktop/Home/Projects/pitch-pitcher-twin/data/processed/skubal_2025.csv`;
- write real-data PNG scatter plots using matplotlib;
- write an animated GIF using imageio;
- write Excalidraw JSON source files;
- write GitHub-renderable SVG diagrams.

- [ ] **Step 5: Generate real-data pitch variation plot**

Use `data/processed/skubal_2025.csv` if present in the repository root. If it is unavailable in the worktree, use the absolute workspace copy at `/Users/elliot18/Desktop/Home/Projects/pitch-pitcher-twin/data/processed/skubal_2025.csv`.

Create a two-panel plot:

- left panel: `plate_x` versus `plate_z` for Skubal FF, showing zone bounds;
- right panel: `pfx_x` versus `pfx_z` for Skubal FF;
- title: `Tarik Skubal FF: same pitch type, real variation`;
- subtitle/annotation: `835 FF pitches, 31 games, 2025 Statcast`.

- [ ] **Step 6: Generate pitch-sequence GIF**

Create an animated GIF that shows cumulative `plate_x`/`plate_z` samples by `pitcher_game_pitch_count` sections: `1-25`, `26-50`, `51-75`, and `76+`.

- [ ] **Step 7: Generate result graph**

Create `v21_results_auc.png` from `outputs/factorized_skubal_2025_ff/factorized_validation_report.json`, comparing V2.1 against game-drift Gaussian and game-drift copula by layer.

- [ ] **Step 8: Create pipeline SVG and Excalidraw source**

Create `pitcher_twin_pipeline.svg` showing:

```text
Statcast rows -> feature layers -> generator suite -> temporal holdout -> C2ST validation -> Machine Session JSON
```

- [ ] **Step 9: Create V2.1 SVG and Excalidraw source**

Create `v21_physics_chain.svg` showing:

```text
release / velocity / spin -> movement residual -> trajectory residual -> command residual
```

Include small callouts for recent-game residual drift and release variance floor.

### Task 3: Rewrite README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README with technical-first narrative**

Sections:

```markdown
# Pitcher Twin
## Technical Idea
## Real Pitch Variation
## System Pipeline
## What We Did
## How The Models Work
## Results
## Reproduce
## Data Policy
## Honest Status And Next Work
```

- [ ] **Step 2: Include exact result tables**

Use the V2.1 saved results:

```text
command_representation 0.519
movement_only 0.554
release_only 0.575
trajectory_only 0.600
physics_core 0.611
```

Also include the baseline comparison from `outputs/factorized_skubal_2025_ff/factorized_validation_summary.md`.

### Task 4: Verify

**Files:**
- No new files expected.

- [ ] **Step 1: Confirm README asset links exist**

Run:

```bash
test -f docs/assets/readme/skubal_ff_variation.png
test -f docs/assets/readme/skubal_ff_pitch_sequence.gif
test -f docs/assets/readme/v21_results_auc.png
test -f docs/assets/readme/pitcher_twin_pipeline.svg
test -f docs/assets/readme/pitcher_twin_pipeline.excalidraw
test -f docs/assets/readme/v21_physics_chain.svg
test -f docs/assets/readme/v21_physics_chain.excalidraw
```

- [ ] **Step 2: Run tests**

Run:

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

Run:

```bash
git add README.md requirements.txt pyproject.toml scripts/build_readme_visuals.py tests/test_readme_visuals.py docs/assets/readme/
git commit -m "Rewrite README with technical visuals"
```
