# Technical-First README Design

## Purpose

Rewrite the README so a technical reviewer can quickly understand what Pitcher Twin does, why same-pitch variability matters, how the models were built, and what the current validation results show.

## Audience

The README should primarily serve pitching-machine-style technical reviewers and collaborators. It should still be readable by a general GitHub reader, but the first screen should explain the modeling idea with engineering precision.

## Structure

Use the approved Option A: Technical First.

1. **Technical Explanation**
   - Define Pitcher Twin as a probabilistic pitcher-variability model, not a one-perfect-pitch predictor.
   - Explain why two pitches with the same pitch type differ: release, velocity, spin, movement, command, game context, and residual human variation.
   - Show real Skubal FF variation with a plot.
   - Show the data/model/validation pipeline as an Excalidraw-style diagram.

2. **What We Did**
   - Real Statcast ingestion and cleaning.
   - Candidate selection.
   - Feature engineering into command, movement, release, trajectory, and physics-core layers.
   - Generator suite and temporal validation.
   - V2.1 factorized physics-residual model.

3. **How The Models Work**
   - Explain V1/static models.
   - Explain game-drift and copula models.
   - Explain V2.1 as a physics chain.
   - Show the V2.1 chain as a visual.

4. **What We Got**
   - C2ST AUC results.
   - V1 versus V2.1 comparison.
   - Honest status labels for validated, borderline, and diagnostic layers.

5. **Reproduce**
   - Commands for tests, Skubal data fetch, demo artifacts, and factorized validation.

6. **Honest Status / Next Work**
   - Command and movement are strong.
   - Full physics improved but remains diagnostic.
   - Next work focuses on spin-axis/release-extension structure, repeated-seed V2.1 validation, and more pitcher/pitch pairs.

## Visual Assets

Create deterministic repo assets under `docs/assets/readme/`:

- `skubal_ff_variation.png`: real Skubal FF plate-location and movement variation.
- `skubal_ff_pitch_sequence.gif`: animated real pitch-location variation by pitcher game pitch-count section.
- `v21_results_auc.png`: layer-by-layer AUC result graph comparing V2.1 with baselines.
- `pitcher_twin_pipeline.svg` and `pitcher_twin_pipeline.excalidraw`: Statcast-to-validation-to-target system pipeline.
- `v21_physics_chain.svg` and `v21_physics_chain.excalidraw`: V2.1 factorized residual chain.

Add `pillow` and `imageio` to the project dependencies for GIF generation. Use generated raster imagery only if a conceptual illustration is needed. The core technical visuals should be generated from real data and diagram source so they remain accurate and reproducible.

## Acceptance Criteria

- README starts with a technical explanation and visual references.
- README says exactly what was done, how it was done, and what results were obtained.
- Visuals render from tracked files, including PNG, GIF, SVG, and Excalidraw source files.
- A reusable script can regenerate the README visuals from the tracked reports and local Skubal CSV.
- Existing tests still pass.
- No `.superpowers/` visual-companion scratch files are tracked.
