# Pitch Generator Model Improvement Research

## Research Question

How should we improve the current pitch generator so generated pitch distributions become harder to distinguish from future real pitches under temporal C2ST validation?

## Project Context

Current best saved result:

- Case study: Tarik Skubal 2025 FF.
- Train rows: 584.
- Holdout rows: 251.
- Best current full-physics model: `factorized_short_memory_wide_residual`.
- Best current physics-core mean C2ST AUC: 0.603 over 30 repeats.
- Target: <= 0.600 mean C2ST AUC with stronger pass-rate robustness.

The current model is already factorized:

```text
release / velocity / spin
  -> movement residual
  -> trajectory residual
  -> command residual
```

The tournament result says the biggest improvement came from a short-memory release model plus much wider downstream residual uncertainty. The classifier still detects the generated samples mostly through:

- `release_pos_y`
- `release_extension`
- `spin_axis_cos`
- `pfx_z`
- `release_pos_x`
- trajectory terms such as `az`, `vy0`, `vx0`, and `ax`

This points to two problems:

1. The generator is not modeling pitcher state as a time-varying latent variable.
2. The generator still treats some physically coupled outputs as if they can be sampled too independently.

## Key Findings

### Classifier Two-Sample Tests - Lopez-Paz and Oquab (2017)

- **Source:** ICLR/arXiv | **Citations:** not checked, Semantic Scholar API rate-limited | **Trust:** Tier 1
- **Link:** https://arxiv.org/abs/1610.06545
- **What they did:** Formalized classifier two-sample tests for deciding whether samples from two distributions are distinguishable.
- **Methodology:** Train a classifier on labeled real-vs-generated samples; held-out classifier accuracy near chance supports distributional similarity.
- **Key result:** C2ST is appropriate for evaluating sample quality from generative models with intractable likelihoods.
- **Relevance to us:** Our current metric is the right family of metric. The next improvement should use the leakage features from C2ST as a training signal and diagnostic loop, not just as a final report.
- **Code:** Existing repo already has a C2ST validator.

### Masked Autoregressive Flow - Papamakarios et al. (2017)

- **Source:** NeurIPS/arXiv | **Citations:** not checked, Semantic Scholar API rate-limited | **Trust:** Tier 1
- **Link:** https://arxiv.org/abs/1705.07057
- **What they did:** Proposed Masked Autoregressive Flow, a normalizing flow for flexible density estimation.
- **Methodology:** Stack autoregressive transformations so a simple base density can represent complex continuous distributions with likelihood evaluation and sampling.
- **Key result:** Autoregressive flows are strong neural density estimators.
- **Relevance to us:** A conditional flow can replace the current Gaussian/copula residual sampler for `release_only` and downstream residuals. This directly targets the non-Gaussian, nonlinear residual structure that C2ST still catches.
- **Code:** `nflows` and Pyro both support flow-based density modeling.

### Neural Spline Flows - Durkan et al. (2019)

- **Source:** NeurIPS | **Citations:** not checked, Semantic Scholar API rate-limited | **Trust:** Tier 1
- **Link:** https://papers.nips.cc/paper/8969-neural-spline-flows
- **What they did:** Introduced rational-quadratic spline transforms for more flexible normalizing flows.
- **Methodology:** Replace simple affine transforms with monotonic splines while keeping exact density evaluation and sampling.
- **Key result:** Spline flows improve density estimation and generative modeling flexibility.
- **Relevance to us:** The current model is close enough that a flexible residual density could plausibly get physics-core AUC below 0.60. Spline flows are a good fit for continuous pitch physics columns.
- **Code:** `nflows` is a PyTorch implementation; Pyro also exposes spline coupling and conditional flow patterns.

### TabDDPM - Kotelnikov et al. (2023)

- **Source:** ICML/PMLR | **Citations:** not checked, Semantic Scholar API rate-limited | **Trust:** Tier 1
- **Link:** https://proceedings.mlr.press/v202/kotelnikov23a.html
- **What they did:** Adapted denoising diffusion models to tabular data.
- **Methodology:** Handles mixed continuous/categorical tabular features with a diffusion process.
- **Key result:** Diffusion is competitive for tabular generation, especially when distributions are heterogeneous.
- **Relevance to us:** A conditional tabular diffusion model is attractive if we want one generator over context, pitch type, and physics. It is likely more work than a flow, but it may handle multimodality better once we scale beyond one pitcher/pitch pair.
- **Code:** No repo dependency yet; would require PyTorch.

### Statcast CSV Documentation - MLB/Baseball Savant

- **Source:** Official MLB documentation | **Citations:** n/a | **Trust:** Tier 1
- **Link:** https://baseballsavant.mlb.com/csv-docs
- **What they document:** Statcast pitch columns including release speed, release position, spin, extension, game IDs, and trajectory-related fields.
- **Methodology:** Official column definitions for Baseball Savant CSV downloads.
- **Key result:** `release_extension` is tracked directly, `game_pk` is a real game identifier, and post-2017 velocities are Statcast out-of-hand measurements.
- **Relevance to us:** The leakage features are official tracking outputs. We should preserve their physical relationships instead of treating them as generic tabular columns.
- **Code:** Existing feature engineering already uses these columns.

### Baseball Aerodynamics - Aguirre-Lopez et al. / Frontiers (2018)

- **Source:** Frontiers in Applied Mathematics and Statistics | **Citations:** not checked, Semantic Scholar API rate-limited | **Trust:** Tier 2
- **Link:** https://www.frontiersin.org/articles/10.3389/fams.2018.00066/full
- **What they did:** Reviewed aerodynamic forces on a baseball, including drag and Magnus effects.
- **Methodology:** Physics model of forces on spinning and non-spinning baseballs.
- **Key result:** Drag and Magnus effects can be decoupled for spinning pitches; wind and humidity matter but are harder to model robustly.
- **Relevance to us:** Weather alone should not be expected to rescue the model. More important is generating physically consistent release, spin, movement, and trajectory parameters, then using weather as a small force/residual adjustment.
- **Code:** No direct code, but this supports adding a physics-consistency layer.

### Pitch Type Prediction in MLB - Lin and Wu (2025)

- **Source:** SAGE/ATDE | **Citations:** too new to rely on count | **Trust:** Tier 3
- **Link:** https://journals.sagepub.com/doi/full/10.3233/ATDE251162
- **What they did:** Compared ML and deep learning models for pitch type prediction using pitch-by-pitch MLB data.
- **Methodology:** Engineered pitcher history, recent trends, and current game state; compared models such as LSTM, DNN, CNN, and CatBoost.
- **Key result:** The authors report that deep learning models benefit from sequential features, and that short-term recent-game trends are valuable.
- **Relevance to us:** The product should not only sample pitch physics. It should first model `P(pitch_type | game state, sequence, pitcher, batter)` and then sample physics conditional on that pitch type.
- **Code:** None used.

### Pitch Prediction with Pitcher/Count-Specific Features - Hamilton et al. (2014)

- **Source:** ICPRAM paper | **Citations:** not checked, Semantic Scholar API rate-limited | **Trust:** Tier 3
- **Link:** https://www.scitepress.org/papers/2014/47639/47639.pdf
- **What they did:** Predicted pitch type with features known before pitch release.
- **Methodology:** Used pitcher/count adaptive feature selection for pitch prediction.
- **Key result:** Pitcher/count-specific features improved prediction compared with static feature sets.
- **Relevance to us:** We should stop using one fixed context vector for all situations. A count/state-specific model or gating network should choose which context variables matter for the current pitch.
- **Code:** None used.

### Pitch Sequence Topic Models - Yoshihara and Takahashi (2020/2021)

- **Source:** SSRN | **Citations:** not checked | **Trust:** Tier 3
- **Link:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3728430
- **What they did:** Modeled pitch sequences as probabilistic topics.
- **Methodology:** Treats pitch sequences as patterns influenced by pitcher/hitter traits, game situations, and outcomes.
- **Key result:** The model extracts pitch sequence tendencies for matchups and situations.
- **Relevance to us:** A latent "sequence plan" or "game script" variable can sit above pitch-type probability and physics generation. This is especially relevant for sliders/fastballs because pitch quality is not independent of previous pitch choices.
- **Code:** None used.

### Practical Flow Implementations - nflows and Pyro

- **Source:** GitHub/Pyro documentation | **Citations:** n/a | **Trust:** Tier 2
- **Links:**
  - https://github.com/bayesiains/nflows
  - https://pyro.ai/examples/normalizing_flows_intro.html
- **What they provide:** Production-ready building blocks for normalizing flows, including spline/coupling/autoregressive transforms and conditional distribution examples.
- **Methodology:** Train invertible transforms with maximum likelihood; sample directly from learned density.
- **Key result:** These packages make conditional flow experiments practical without building all neural density machinery from scratch.
- **Relevance to us:** If we add a PyTorch dependency, V2.4 can train a conditional spline-flow residual model quickly.
- **Code:** Candidate dependency, not currently in repo.

## Synthesis

The current model is close to the target, but it is squeezing more realism out of fixed covariance scaling than out of a better learned distribution. The research points to a sharper path:

1. Add a dynamic pitcher-state layer.
2. Replace Gaussian/copula residuals with conditional density estimators.
3. Enforce physical consistency among release geometry, spin, movement, and trajectory.
4. Add a pitch-type/sequence probability model before physics generation.

The most practical next implementation is not a huge neural net. It is a V2.4 "conditional state mixture residual" model that stays mostly inside the existing stack:

```text
context + recent game state
  -> latent pitcher-state bucket / mixture component
  -> release distribution
  -> movement/trajectory/command residual mixture
  -> physics consistency projection
  -> C2ST by layer and context bucket
```

This should improve the exact leakage points:

| Leakage / weakness | Recommended fix |
|---|---|
| `release_pos_y` + `release_extension` detectable | Generate or project them jointly with a release geometry constraint, not independently. |
| `spin_axis_cos` detectable | Model spin axis as an angle distribution on the unit circle; sample angle then convert to sin/cos. |
| `pfx_z` and trajectory acceleration detectable | Model movement and trajectory residuals conditionally, then enforce physics-derived consistency checks. |
| Pass rate unstable across seeds | Learn state-specific covariance/mixture weights instead of multiplying one covariance matrix by a constant. |
| Weather barely helps | Use weather only after density/state modeling; treat it as a small physics residual modifier. |
| Product wants pitch probabilities | Add `P(pitch_type | sequence/game state)` before sampling pitch physics. |

## Recommended Implementation Path

### V2.4: Conditional State Mixture Residual Model

**Why first:** It is the highest expected gain with the least dependency risk.

Implementation:

- Add `pitcher_game_state_features`:
  - pitcher game pitch count
  - inning
  - count
  - batter hand
  - pitcher score diff
  - previous pitch type
  - previous pitch location
  - previous pitch velocity
  - recent 5/10/20 pitch rolling release-speed/spin/movement means
- Fit a small gating model that assigns each pitch to a state bucket:
  - short-memory game state
  - count family
  - fatigue band
  - recent command/movement drift
- Fit `BayesianGaussianMixture` or regularized `GaussianMixture` residual distributions per state for:
  - release
  - movement residual
  - trajectory residual
  - command residual
- Sample mixture component conditional on holdout context.
- Keep the existing C2ST validation and tournament runner.

Expected benefit:

- Better state-specific covariance.
- Less need for hand-tuned `downstream_residual_cov * 30.0`.
- Directly targets the unstable full-physics pass rate.

### V2.5: Conditional Normalizing Flow Residuals

**Why second:** Best theoretical fit, but adds PyTorch and training complexity.

Implementation:

- Add optional dependency group: `torch`, `nflows` or `pyro-ppl`.
- Train conditional spline/MAF flows for:
  - release residuals
  - downstream residuals
- Condition on:
  - game state
  - pitch count
  - recent rolling state
  - batter handedness
  - pitch type
- Evaluate against V2.4 and V2.3 in the same tournament.

Expected benefit:

- Captures nonlinear, multimodal residuals.
- Gives exact likelihood for model comparison.
- More expressive than Gaussian copula.

### V2.6: Physics Consistency Projection

**Why third:** The classifier is catching geometry and trajectory inconsistencies.

Implementation:

- Generate a smaller set of physically primary variables:
  - release speed
  - release slot
  - extension
  - spin axis angle
  - spin rate
  - movement parameters
  - command target
- Reconstruct or constrain secondary trajectory variables:
  - `vx0`, `vy0`, `vz0`
  - `ax`, `ay`, `az`
  - `pfx_x`, `pfx_z`
- Add validation metrics for:
  - release_pos_y + release_extension consistency
  - spin-axis unit-circle consistency
  - velocity/acceleration/movement physics residual
  - command/movement coherence

Expected benefit:

- Reduces physically impossible combinations.
- Should reduce top leakage features without simply adding noise.

### V2.7: Pitch-Type Probability Head

**Why fourth:** This answers the product question: "What will this player throw?"

Implementation:

- Train `P(pitch_type | pitcher, batter, count, previous pitches, game state)`.
- Start with scikit-learn models:
  - multinomial logistic regression baseline
  - random forest / histogram gradient boosting
  - calibrated probabilities
- Later add LSTM/TFT/Transformer if we scale data across pitchers.
- Pipe selected pitch type into the physics generator.

Expected benefit:

- Lets UI show pitch-type probability sliders by time/game state.
- Prevents physics generator from pretending pitch physics is independent of pitch selection.

## Relevance: HIGH

The research directly maps to the current validation failures. The next model should not be "more weather" or "more noise." It should be a conditional state-density model with physical constraints.

## Implementation Findings

Implemented first:

- recent pitcher-state feature engineering;
- `conditional_state_mixture_residual`, a V2.4 tournament candidate that clusters recent/context state and samples state-specific physics-core mixtures;
- tournament integration and candidate notes;
- regenerated the 30-repeat Skubal 2025 FF tournament.

The direct state-mixture candidate did not improve the full-physics score:

| Model | Physics-core mean AUC | Pass rate <= 0.60 |
|---|---:|---:|
| conditional_state_mixture_residual | 0.723 | 0.00 |
| previous short-memory wide residual | 0.603 | 0.43 |

The important positive finding is that the broader state-modeling direction worked through the factorized state-anchor variants already present in the tournament:

| Model | Physics-core mean AUC | Pass rate <= 0.60 |
|---|---:|---:|
| factorized_short_memory_wide_residual | 0.603 | 0.43 |
| factorized_short_memory_more_uncertain | 0.589 | 0.60 |
| factorized_recent_state_anchored | 0.577 | 0.80 |
| factorized_trend_state_anchored | 0.575 | 0.70 |

Interpretation:

- Directly sampling a full physics-core mixture is too loose; it breaks release/spin/trajectory coupling.
- The winning direction is factorized generation plus state anchoring.
- The best current full-physics score is now 0.575 C2ST AUC.
- The best robustness tradeoff is `factorized_recent_state_anchored`: 0.577 physics-core AUC with 0.80 pass rate.
- Remaining leakage is still release geometry and spin-axis structure, especially `release_pos_y`, `release_extension`, `release_speed`, `spin_axis_sin`, and `ay`.

Next implementation should refine the factorized state-anchor model rather than continue with raw full-vector mixtures.

## METHODOLOGY.md Update

No project `METHODOLOGY.md` exists. If one is added, include:

> Improve pitch-generator realism by modeling dynamic pitcher state and conditional residual densities first; use weather as a secondary physics residual input after release/movement/trajectory consistency is validated.
