# V2.1 Factorized Physics Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a factorized pitch-physics residual model that improves full-joint physics by modeling release, movement, trajectory, and command as sequential dependent layers.

**Architecture:** Keep V1's recent game-drift and conditional context machinery, but stop sampling `physics_core` as one flat vector. V2.1 samples release/velocity/spin first, then movement residuals conditioned on release/context, then trajectory residuals conditioned on release/movement/context, then plate-location residuals conditioned on the sampled upstream physics. Weather stays out of V2.1; it becomes a future residual input after this structure validates.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy, pytest, existing C2ST validator, existing Statcast feature groups.

---

## V1 Evidence Driving This Plan

Real Skubal 2025 FF conditional validation:

| Layer | Game-drift Gaussian | Game-drift Copula | Conditional Copula |
|---|---:|---:|---:|
| command | 0.514 | 0.546 | 0.537 |
| movement | 0.573 | 0.519 | 0.532 |
| release | 0.661 | 0.599 | 0.585 |
| trajectory | 0.561 | 0.660 | 0.679 |
| physics core | 0.685 | 0.721 | 0.696 |

Interpretation:

- Command is already strong, especially with the Gaussian baseline.
- Movement is strong, especially with copula residual structure.
- Conditional context helps release.
- Conditional context hurts trajectory when pushed into the flat full-vector model.
- Full physics remains diagnostic even when components are useful.

Theory validated by V1:

```text
The problem is not only missing context.
The problem is preserving cross-layer physics relationships.
```

Therefore V2.1 should not add more context everywhere. It should model the physics chain:

```text
context + recent game drift
  -> release / velocity / spin
  -> movement residual
  -> trajectory residual
  -> plate-location residual
```

---

## File Structure

Create:

- `src/pitcher_twin/factorized.py`
  - owns factorized model dataclasses, residual-layer fitting, sequential sampling, and factorized validation helpers.
- `tests/test_factorized.py`
  - owns synthetic relationship tests and fixture-backed validation contract tests.
- `scripts/run_factorized_validation.py`
  - runs the real Skubal 2025 FF evaluation and writes a report.

Modify:

- `src/pitcher_twin/models.py`
  - expose public residual copula wrappers around existing internal copula helpers.
- `src/pitcher_twin/conditional.py`
  - may call factorized validation through a small wrapper if dashboard integration needs it; do not grow it with physics-chain internals.
- `docs/presentation.md`
  - update only after real validation results exist.
- `README.md`
  - update only after real validation results exist.

Do not modify in V2.1:

- Weather join code as production model code.
- Streamlit UI controls, until factorized validation shows a useful result.

---

### Task 1: Expose Reusable Residual Copula Helpers

**Files:**
- Modify: `src/pitcher_twin/models.py`
- Test: `tests/test_models_validation_real.py`

- [ ] **Step 1: Write the failing tests**

Append these tests to `tests/test_models_validation_real.py`:

```python
def test_public_residual_copula_helpers_sample_finite_residuals() -> None:
    from pitcher_twin.models import (
        fit_residual_gaussian_copula,
        sample_residual_gaussian_copula,
    )

    residuals = np.array(
        [
            [-1.0, -0.5, 0.2],
            [-0.2, -0.1, 0.0],
            [0.0, 0.1, 0.1],
            [0.5, 0.4, -0.2],
            [1.2, 0.8, -0.1],
        ]
        * 5,
        dtype=float,
    )

    payload = fit_residual_gaussian_copula(residuals)
    samples = sample_residual_gaussian_copula(payload, n=20, random_state=7)

    assert payload["copula_kind"] == "gaussian_empirical_margins"
    assert samples.shape == (20, 3)
    assert np.isfinite(samples).all()


def test_public_residual_copula_returns_none_for_tiny_training_sets() -> None:
    from pitcher_twin.models import fit_residual_gaussian_copula

    residuals = np.ones((3, 2), dtype=float)
    assert fit_residual_gaussian_copula(residuals) is None
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
pytest tests/test_models_validation_real.py::test_public_residual_copula_helpers_sample_finite_residuals tests/test_models_validation_real.py::test_public_residual_copula_returns_none_for_tiny_training_sets -q
```

Expected: fail because `fit_residual_gaussian_copula` and `sample_residual_gaussian_copula` are not exported.

- [ ] **Step 3: Implement public wrappers**

Add these functions near the existing copula helpers in `src/pitcher_twin/models.py`:

```python
def fit_residual_gaussian_copula(residuals: np.ndarray) -> dict[str, object] | None:
    """Fit empirical Gaussian-copula residual margins for downstream residual layers."""
    return _gaussian_copula_residual_payload(residuals)


def sample_residual_gaussian_copula(
    payload: dict[str, object],
    n: int,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample residuals from a fitted empirical Gaussian-copula payload."""
    return _sample_copula_residuals(payload, n=n, rng=_rng(random_state))
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
pytest tests/test_models_validation_real.py::test_public_residual_copula_helpers_sample_finite_residuals tests/test_models_validation_real.py::test_public_residual_copula_returns_none_for_tiny_training_sets -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/pitcher_twin/models.py tests/test_models_validation_real.py
git commit -m "Expose residual copula helpers"
```

---

### Task 2: Add Residual Ridge Layer Primitive

**Files:**
- Create: `src/pitcher_twin/factorized.py`
- Test: `tests/test_factorized.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_factorized.py` with:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from pitcher_twin.factorized import fit_residual_layer, sample_residual_layer


def test_residual_layer_learns_linear_relationship_and_returns_finite_samples() -> None:
    frame = pd.DataFrame(
        {
            "release_speed": np.linspace(90.0, 99.0, 40),
            "release_spin_rate": np.linspace(2200.0, 2400.0, 40),
            "inning": np.tile([1.0, 5.0], 20),
        }
    )
    frame["pfx_z"] = 0.04 * frame["release_speed"] + 0.001 * frame["release_spin_rate"]
    frame["pfx_x"] = -0.02 * frame["release_speed"] + 0.03 * frame["inning"]

    layer = fit_residual_layer(
        frame,
        name="movement",
        conditioning_columns=["release_speed", "release_spin_rate", "inning"],
        target_columns=["pfx_x", "pfx_z"],
        ridge=1.0,
    )
    context = frame[["release_speed", "release_spin_rate", "inning"]].head(8)
    samples = sample_residual_layer(layer, context, random_state=9)

    assert samples.shape == (8, 2)
    assert samples.columns.tolist() == ["pfx_x", "pfx_z"]
    assert np.isfinite(samples.to_numpy(float)).all()
    assert abs(samples["pfx_z"].mean() - frame["pfx_z"].head(8).mean()) < 0.4
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
pytest tests/test_factorized.py::test_residual_layer_learns_linear_relationship_and_returns_finite_samples -q
```

Expected: fail because `pitcher_twin.factorized` does not exist.

- [ ] **Step 3: Implement residual layer primitive**

Create `src/pitcher_twin/factorized.py` with:

```python
"""Factorized physics-residual pitch models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitcher_twin.models import (
    fit_residual_gaussian_copula,
    sample_residual_gaussian_copula,
)


@dataclass
class ResidualLayer:
    name: str
    conditioning_columns: list[str]
    target_columns: list[str]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    beta: np.ndarray
    residual_cov: np.ndarray
    residual_copula: dict[str, object] | None
    source_row_count: int


def _matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    values = frame[columns].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Residual layer input contains non-finite values.")
    return values


def fit_residual_layer(
    frame: pd.DataFrame,
    *,
    name: str,
    conditioning_columns: list[str],
    target_columns: list[str],
    ridge: float = 10.0,
) -> ResidualLayer:
    keep = conditioning_columns + target_columns
    fit_frame = frame[keep].dropna().reset_index(drop=True)
    if len(fit_frame) < max(20, len(conditioning_columns) + 3):
        raise ValueError(f"Not enough rows to fit residual layer {name}.")

    x = _matrix(fit_frame, conditioning_columns)
    y = _matrix(fit_frame, target_columns)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    xz = (x - x_mean) / x_std
    y_mean = y.mean(axis=0)
    yc = y - y_mean
    xtx = xz.T @ xz + np.eye(xz.shape[1]) * ridge
    beta = np.linalg.solve(xtx, xz.T @ yc)
    pred = y_mean + xz @ beta
    residuals = y - pred
    residual_cov = np.cov(residuals, rowvar=False)
    if residual_cov.ndim == 0:
        residual_cov = np.eye(y.shape[1]) * float(residual_cov)
    residual_cov = residual_cov + np.eye(y.shape[1]) * 1e-5
    return ResidualLayer(
        name=name,
        conditioning_columns=conditioning_columns,
        target_columns=target_columns,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        beta=beta,
        residual_cov=residual_cov,
        residual_copula=fit_residual_gaussian_copula(residuals),
        source_row_count=int(len(fit_frame)),
    )


def _predict_layer_mean(layer: ResidualLayer, context: pd.DataFrame) -> np.ndarray:
    x = _matrix(context, layer.conditioning_columns)
    xz = (x - layer.x_mean) / layer.x_std
    return layer.y_mean + xz @ layer.beta


def sample_residual_layer(
    layer: ResidualLayer,
    context: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    mean = _predict_layer_mean(layer, context)
    if layer.residual_copula is not None:
        residuals = sample_residual_gaussian_copula(
            layer.residual_copula,
            n=len(context),
            random_state=random_state,
        )
    else:
        rng = np.random.default_rng(random_state)
        residuals = rng.multivariate_normal(
            np.zeros(len(layer.target_columns)),
            layer.residual_cov,
            size=len(context),
            check_valid="ignore",
        )
    values = mean + residuals
    return pd.DataFrame(values, columns=layer.target_columns)
```

- [ ] **Step 4: Run test to verify GREEN**

Run:

```bash
pytest tests/test_factorized.py::test_residual_layer_learns_linear_relationship_and_returns_finite_samples -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/pitcher_twin/factorized.py tests/test_factorized.py
git commit -m "Add factorized residual layer primitive"
```

---

### Task 3: Fit and Sample the Factorized Physics Chain

**Files:**
- Modify: `src/pitcher_twin/factorized.py`
- Test: `tests/test_factorized.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
from pitcher_twin.features import PITCH_PHYSICS_FEATURES, TRAJECTORY_FEATURES
from pitcher_twin.factorized import fit_factorized_physics_model, sample_factorized_physics


def _synthetic_factorized_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    release_speed = rng.normal(95.0, 1.0, n)
    spin = rng.normal(2350.0, 80.0, n)
    frame = pd.DataFrame(
        {
            "pitcher": 1,
            "pitcher_name": "Pitcher, Test",
            "pitch_type": "FF",
            "game_pk": np.repeat(np.arange(1, 9), n // 8),
            "game_date": np.repeat(pd.date_range("2026-04-01", periods=8).astype(str), n // 8),
            "at_bat_number": np.arange(n) // 4,
            "pitch_number": np.tile([1, 2, 3, 4], n // 4),
            "balls": rng.integers(0, 4, n),
            "strikes": rng.integers(0, 3, n),
            "count_bucket_code": 2.0,
            "inning": rng.integers(1, 8, n).astype(float),
            "pitcher_game_pitch_count": np.arange(1, n + 1).astype(float),
            "batter_stand_code": rng.integers(0, 2, n).astype(float),
            "pitcher_score_diff": rng.normal(0.0, 2.0, n),
            "release_speed": release_speed,
            "release_spin_rate": spin,
            "spin_axis_cos": rng.normal(-0.95, 0.02, n),
            "spin_axis_sin": rng.normal(0.25, 0.03, n),
            "release_pos_x": rng.normal(1.4, 0.08, n),
            "release_pos_y": rng.normal(54.0, 0.2, n),
            "release_pos_z": rng.normal(6.1, 0.1, n),
            "release_extension": rng.normal(6.5, 0.15, n),
        }
    )
    frame["pfx_x"] = -0.02 * frame["release_speed"] + rng.normal(0, 0.05, n)
    frame["pfx_z"] = 0.002 * frame["release_spin_rate"] + rng.normal(0, 0.05, n)
    frame["vx0"] = -0.05 * frame["release_speed"] + rng.normal(0, 0.1, n)
    frame["vy0"] = -1.45 * frame["release_speed"] + rng.normal(0, 0.2, n)
    frame["vz0"] = -4.0 + 0.2 * frame["pfx_z"] + rng.normal(0, 0.1, n)
    frame["ax"] = 15 * frame["pfx_x"] + rng.normal(0, 0.2, n)
    frame["ay"] = 25 + rng.normal(0, 0.4, n)
    frame["az"] = -20 + 4 * frame["pfx_z"] + rng.normal(0, 0.2, n)
    frame["plate_x"] = 0.5 * frame["pfx_x"] + 0.03 * frame["pitcher_score_diff"] + rng.normal(0, 0.1, n)
    frame["plate_z"] = 2.2 + 0.25 * frame["pfx_z"] - 0.004 * frame["pitcher_game_pitch_count"] + rng.normal(0, 0.1, n)
    return frame


def test_factorized_physics_model_samples_full_physics_core_columns() -> None:
    frame = _synthetic_factorized_frame()
    model = fit_factorized_physics_model(
        frame,
        frame,
        pitcher_name="Pitcher, Test",
        pitch_type="FF",
        random_state=5,
    )
    context = frame.head(12)
    samples = sample_factorized_physics(model, n=12, context_df=context, random_state=6)

    expected = PITCH_PHYSICS_FEATURES + ["pfx_x", "pfx_z"] + TRAJECTORY_FEATURES + ["plate_x", "plate_z"]
    assert samples.columns.tolist() == expected
    assert samples.shape == (12, len(expected))
    assert np.isfinite(samples.to_numpy(float)).all()
    assert model.movement_layer.source_row_count >= 20
    assert model.trajectory_layer.source_row_count >= 20
    assert model.command_layer.source_row_count >= 20
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
pytest tests/test_factorized.py::test_factorized_physics_model_samples_full_physics_core_columns -q
```

Expected: fail because `fit_factorized_physics_model` and `sample_factorized_physics` do not exist.

- [ ] **Step 3: Implement factorized model dataclass and fit/sample functions**

Add to `src/pitcher_twin/factorized.py`:

```python
from pitcher_twin.features import CONTEXT_FEATURES, PITCH_PHYSICS_FEATURES, TRAJECTORY_FEATURES
from pitcher_twin.models import GeneratorModel, fit_generator_suite, sample_generator

MOVEMENT_FLIGHT_COLUMNS = ["pfx_x", "pfx_z"]
COMMAND_COLUMNS = ["plate_x", "plate_z"]
FACTORIZED_PHYSICS_COLUMNS = (
    PITCH_PHYSICS_FEATURES + MOVEMENT_FLIGHT_COLUMNS + TRAJECTORY_FEATURES + COMMAND_COLUMNS
)


@dataclass
class FactorizedPhysicsModel:
    model_name: str
    pitcher_name: str
    pitch_type: str
    release_model: GeneratorModel
    release_model_name: str
    movement_layer: ResidualLayer
    trajectory_layer: ResidualLayer
    command_layer: ResidualLayer
    context_columns: list[str]
    feature_columns: list[str]
    source_row_count: int


def _available_context_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in CONTEXT_FEATURES if column in frame.columns]


def _select_release_model(suite: dict[str, GeneratorModel]) -> tuple[str, GeneratorModel]:
    for name in (
        "player_recent_weighted_game_drift_copula",
        "player_recent_weighted_game_drift_gaussian",
        "player_context_weighted_gaussian",
        "player_recent_multivariate_gaussian",
        "player_multivariate_gaussian",
    ):
        if name in suite:
            return name, suite[name]
    if not suite:
        raise ValueError("No release generator models were fitted.")
    name = next(iter(suite))
    return name, suite[name]


def fit_factorized_physics_model(
    player_train: pd.DataFrame,
    league_df: pd.DataFrame,
    *,
    pitcher_name: str,
    pitch_type: str,
    random_state: int = 42,
) -> FactorizedPhysicsModel:
    context_columns = _available_context_columns(player_train)
    keep = FACTORIZED_PHYSICS_COLUMNS + context_columns
    frame = player_train[keep].dropna().reset_index(drop=True)
    if len(frame) < 40:
        raise ValueError("At least 40 complete rows are required for factorized physics.")

    release_suite = fit_generator_suite(
        player_train,
        league_df,
        pitcher_name=pitcher_name,
        pitch_type=pitch_type,
        feature_group="release_only",
        random_state=random_state,
    )
    release_name, release_model = _select_release_model(release_suite)
    movement_layer = fit_residual_layer(
        frame,
        name="movement",
        conditioning_columns=PITCH_PHYSICS_FEATURES + context_columns,
        target_columns=MOVEMENT_FLIGHT_COLUMNS,
        ridge=10.0,
    )
    trajectory_layer = fit_residual_layer(
        frame,
        name="trajectory",
        conditioning_columns=PITCH_PHYSICS_FEATURES + MOVEMENT_FLIGHT_COLUMNS + context_columns,
        target_columns=TRAJECTORY_FEATURES,
        ridge=10.0,
    )
    command_layer = fit_residual_layer(
        frame,
        name="command",
        conditioning_columns=(
            PITCH_PHYSICS_FEATURES
            + MOVEMENT_FLIGHT_COLUMNS
            + TRAJECTORY_FEATURES
            + context_columns
        ),
        target_columns=COMMAND_COLUMNS,
        ridge=10.0,
    )
    return FactorizedPhysicsModel(
        model_name="player_factorized_physics_residual",
        pitcher_name=pitcher_name,
        pitch_type=pitch_type,
        release_model=release_model,
        release_model_name=release_name,
        movement_layer=movement_layer,
        trajectory_layer=trajectory_layer,
        command_layer=command_layer,
        context_columns=context_columns,
        feature_columns=FACTORIZED_PHYSICS_COLUMNS,
        source_row_count=int(len(frame)),
    )


def _context_for_sampling(model: FactorizedPhysicsModel, context_df: pd.DataFrame | None, n: int) -> pd.DataFrame:
    if context_df is None or not model.context_columns:
        return pd.DataFrame(index=range(n))
    context = context_df[model.context_columns].dropna().reset_index(drop=True)
    if context.empty:
        return pd.DataFrame(index=range(n))
    if len(context) >= n:
        return context.head(n).reset_index(drop=True)
    repeats = int(np.ceil(n / len(context)))
    return pd.concat([context] * repeats, ignore_index=True).head(n)


def sample_factorized_physics(
    model: FactorizedPhysicsModel,
    n: int,
    context_df: pd.DataFrame | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    context = _context_for_sampling(model, context_df, n)
    release = sample_generator(
        model.release_model,
        n=n,
        random_state=random_state,
        context_df=context if not context.empty else None,
    ).reset_index(drop=True)
    stage = pd.concat([release, context], axis=1)
    movement = sample_residual_layer(model.movement_layer, stage, random_state=random_state + 1)
    stage = pd.concat([stage, movement], axis=1)
    trajectory = sample_residual_layer(model.trajectory_layer, stage, random_state=random_state + 2)
    stage = pd.concat([stage, trajectory], axis=1)
    command = sample_residual_layer(model.command_layer, stage, random_state=random_state + 3)
    output = pd.concat([release, movement, trajectory, command], axis=1)
    return output[model.feature_columns]
```

- [ ] **Step 4: Run factorized tests**

Run:

```bash
pytest tests/test_factorized.py -q
```

Expected: all factorized tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pitcher_twin/factorized.py tests/test_factorized.py
git commit -m "Add factorized physics sampling chain"
```

---

### Task 4: Add Layer-by-Layer Factorized Validation

**Files:**
- Modify: `src/pitcher_twin/factorized.py`
- Test: `tests/test_factorized.py`

- [ ] **Step 1: Write the failing validation test**

Append:

```python
from pitcher_twin.factorized import validate_factorized_physics


def test_factorized_validation_reports_baselines_and_layer_metrics() -> None:
    frame = _synthetic_factorized_frame()
    train = frame.iloc[:56].copy()
    holdout = frame.iloc[56:].copy()

    report = validate_factorized_physics(
        train,
        holdout,
        frame,
        pitcher_name="Pitcher, Test",
        pitch_type="FF",
        n_samples=40,
        random_state=8,
    )

    assert report["model_name"] == "player_factorized_physics_residual"
    assert set(report["layer_results"]) == {
        "command_representation",
        "movement_only",
        "release_only",
        "trajectory_only",
        "physics_core",
    }
    for layer, row in report["layer_results"].items():
        assert "factorized_auc" in row
        assert "game_drift_gaussian_auc" in row
        assert "game_drift_copula_auc" in row
        assert 0.5 <= row["factorized_auc"] <= 1.0
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
pytest tests/test_factorized.py::test_factorized_validation_reports_baselines_and_layer_metrics -q
```

Expected: fail because `validate_factorized_physics` does not exist.

- [ ] **Step 3: Implement validation helper**

Add to `factorized.py`:

```python
from pitcher_twin.features import FEATURE_GROUPS
from pitcher_twin.validator import classifier_two_sample_test

VALIDATION_LAYERS = [
    "command_representation",
    "movement_only",
    "release_only",
    "trajectory_only",
    "physics_core",
]


def _fit_baseline_model(
    train: pd.DataFrame,
    league_df: pd.DataFrame,
    pitcher_name: str,
    pitch_type: str,
    feature_group: str,
    requested: str,
    random_state: int,
) -> GeneratorModel:
    suite = fit_generator_suite(
        train,
        league_df,
        pitcher_name=pitcher_name,
        pitch_type=pitch_type,
        feature_group=feature_group,
        random_state=random_state,
    )
    if requested in suite:
        return suite[requested]
    _, fallback = _select_release_model(suite)
    return fallback


def validate_factorized_physics(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    league_df: pd.DataFrame,
    *,
    pitcher_name: str,
    pitch_type: str,
    n_samples: int = 300,
    random_state: int = 42,
) -> dict[str, object]:
    factorized = fit_factorized_physics_model(
        train,
        league_df,
        pitcher_name=pitcher_name,
        pitch_type=pitch_type,
        random_state=random_state,
    )
    sample_count = max(n_samples, len(holdout))
    factorized_samples = sample_factorized_physics(
        factorized,
        n=sample_count,
        context_df=holdout,
        random_state=random_state + 10,
    )

    layer_results: dict[str, dict[str, object]] = {}
    for index, feature_group in enumerate(VALIDATION_LAYERS):
        columns = FEATURE_GROUPS[feature_group]
        factorized_metrics = classifier_two_sample_test(
            holdout,
            factorized_samples,
            columns,
            random_state=random_state + 100 + index,
        )
        gaussian = _fit_baseline_model(
            train,
            league_df,
            pitcher_name,
            pitch_type,
            feature_group,
            "player_recent_weighted_game_drift_gaussian",
            random_state,
        )
        copula = _fit_baseline_model(
            train,
            league_df,
            pitcher_name,
            pitch_type,
            feature_group,
            "player_recent_weighted_game_drift_copula",
            random_state,
        )
        gaussian_samples = sample_generator(
            gaussian,
            n=sample_count,
            random_state=random_state + 200 + index,
            context_df=holdout,
        )
        copula_samples = sample_generator(
            copula,
            n=sample_count,
            random_state=random_state + 300 + index,
            context_df=holdout,
        )
        gaussian_metrics = classifier_two_sample_test(
            holdout,
            gaussian_samples,
            columns,
            random_state=random_state + 400 + index,
        )
        copula_metrics = classifier_two_sample_test(
            holdout,
            copula_samples,
            columns,
            random_state=random_state + 500 + index,
        )
        layer_results[feature_group] = {
            "features": columns,
            "factorized_auc": float(factorized_metrics["auc"]),
            "game_drift_gaussian_auc": float(gaussian_metrics["auc"]),
            "game_drift_copula_auc": float(copula_metrics["auc"]),
            "factorized_top_leakage": factorized_metrics["top_leakage_features"],
        }

    return {
        "model_name": factorized.model_name,
        "pitcher_name": pitcher_name,
        "pitch_type": pitch_type,
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "release_model": factorized.release_model_name,
        "layer_results": layer_results,
    }
```

- [ ] **Step 4: Run validation tests**

Run:

```bash
pytest tests/test_factorized.py -q
```

Expected: all factorized tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pitcher_twin/factorized.py tests/test_factorized.py
git commit -m "Add factorized physics validation"
```

---

### Task 5: Add Real Skubal Factorized Evaluation Script

**Files:**
- Create: `scripts/run_factorized_validation.py`
- Test: `tests/test_factorized.py`

- [ ] **Step 1: Write the failing CLI smoke test**

Append to `tests/test_factorized.py`:

```python
import importlib.util
from pathlib import Path


def test_factorized_validation_script_exposes_main() -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_factorized_validation.py"
    spec = importlib.util.spec_from_file_location("run_factorized_validation", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.main)
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
pytest tests/test_factorized.py::test_factorized_validation_script_exposes_main -q
```

Expected: fail because `scripts/run_factorized_validation.py` does not exist.

- [ ] **Step 3: Implement script**

Create `scripts/run_factorized_validation.py`:

```python
#!/usr/bin/env python3
"""Run real-data factorized physics validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pitcher_twin.data import load_statcast_cache  # noqa: E402
from pitcher_twin.factorized import validate_factorized_physics  # noqa: E402
from pitcher_twin.features import clean_pitch_features  # noqa: E402
from pitcher_twin.validator import temporal_train_holdout  # noqa: E402


def run(data_path: Path, output_dir: Path, pitcher_id: int, pitch_type: str) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_statcast_cache(data_path)
    clean = clean_pitch_features(raw, pitch_types=None)
    subset = clean[(clean["pitcher"] == pitcher_id) & (clean["pitch_type"] == pitch_type)].copy()
    if subset.empty:
        raise RuntimeError(f"No rows found for pitcher={pitcher_id}, pitch_type={pitch_type}.")
    pitcher_name = str(subset["pitcher_name"].dropna().iloc[0])
    train, holdout = temporal_train_holdout(subset, train_fraction=0.70)
    report = validate_factorized_physics(
        train,
        holdout,
        clean,
        pitcher_name=pitcher_name,
        pitch_type=pitch_type,
        n_samples=max(300, len(holdout)),
        random_state=42,
    )
    report.update(
        {
            "data_path": str(data_path),
            "pitcher_id": int(pitcher_id),
            "rows_raw": int(len(raw)),
            "rows_clean": int(len(clean)),
            "rows_subset": int(len(subset)),
        }
    )
    report_path = output_dir / "factorized_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    summary_lines = [
        "# Factorized Physics Validation",
        "",
        f"- Pitcher: `{pitcher_name}`",
        f"- Pitch type: `{pitch_type}`",
        f"- Train rows: `{report['n_train']}`",
        f"- Holdout rows: `{report['n_holdout']}`",
        "",
        "| Layer | Factorized | Game-Drift Gaussian | Game-Drift Copula |",
        "|---|---:|---:|---:|",
    ]
    for layer, row in report["layer_results"].items():
        summary_lines.append(
            f"| {layer} | {row['factorized_auc']:.3f} | "
            f"{row['game_drift_gaussian_auc']:.3f} | {row['game_drift_copula_auc']:.3f} |"
        )
    summary_path = output_dir / "factorized_validation_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    return {"report": str(report_path), "summary": str(summary_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/factorized_skubal_2025_ff"))
    parser.add_argument("--pitcher-id", type=int, default=669373)
    parser.add_argument("--pitch-type", type=str, default="FF")
    args = parser.parse_args()
    outputs = run(args.data, args.output_dir, args.pitcher_id, args.pitch_type)
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI smoke test**

Run:

```bash
pytest tests/test_factorized.py::test_factorized_validation_script_exposes_main -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_factorized_validation.py tests/test_factorized.py
git commit -m "Add factorized validation runner"
```

---

### Task 6: Run Real Skubal Evaluation and Decide Whether V2.1 Helps

**Files:**
- Generated: `outputs/factorized_skubal_2025_ff/factorized_validation_report.json`
- Generated: `outputs/factorized_skubal_2025_ff/factorized_validation_summary.md`
- Modify if useful: `docs/presentation.md`
- Modify if useful: `README.md`

- [ ] **Step 1: Run real evaluation**

Run:

```bash
python3 scripts/run_factorized_validation.py \
  --data data/processed/skubal_2025.csv \
  --output-dir outputs/factorized_skubal_2025_ff \
  --pitcher-id 669373 \
  --pitch-type FF
```

Expected: writes JSON and markdown reports.

- [ ] **Step 2: Compare against V1 result table**

Use this baseline table from V1:

| Layer | Best V1 AUC |
|---|---:|
| command | 0.514 |
| movement | 0.519 |
| release | 0.585 |
| trajectory | 0.561 |
| physics core | 0.685 |

Decision rules:

- If factorized `physics_core` is below `0.685`, V2.1 improved the main failure.
- If factorized `physics_core` is at or below `0.600`, V2.1 becomes a candidate validated full-physics model.
- If factorized improves physics core but regresses command/movement above `0.600`, report it as diagnostic, not production.
- If factorized does not improve physics core, keep it as a research branch and move to a different structure.

- [ ] **Step 3: Generate the docs table from the report**

Run this command to produce the exact markdown table for any docs update:

```bash
python3 - <<'PY'
import json
from pathlib import Path

baseline = {
    "command_representation": 0.514,
    "movement_only": 0.519,
    "release_only": 0.585,
    "trajectory_only": 0.561,
    "physics_core": 0.685,
}
report = json.loads(Path("outputs/factorized_skubal_2025_ff/factorized_validation_report.json").read_text())
print("## V2.1 Factorized Physics Residual Result")
print()
print("The factorized model was added because V1 showed strong component validation but weak full-physics validation. It models release first, then movement, trajectory, and command as residual layers.")
print()
print("| Layer | Best V1 AUC | Factorized AUC | Result |")
print("|---|---:|---:|---|")
for layer, v1_auc in baseline.items():
    factorized_auc = report["layer_results"][layer]["factorized_auc"]
    result = "improved" if factorized_auc < v1_auc else "regressed"
    print(f"| {layer} | {v1_auc:.3f} | {factorized_auc:.3f} | {result} |")
PY
```

If V2.1 does not improve `physics_core`, add only a short diagnostic note to `docs/presentation.md` and keep the main demo result unchanged.

- [ ] **Step 4: Commit artifacts and docs**

If artifacts are worth tracking:

```bash
git add docs/presentation.md README.md outputs/factorized_skubal_2025_ff/factorized_validation_report.json outputs/factorized_skubal_2025_ff/factorized_validation_summary.md
git commit -m "Evaluate factorized physics residual model"
```

If artifacts are not worth tracking:

```bash
git add docs/presentation.md README.md
git commit -m "Document factorized physics residual findings"
```

---

### Task 7: Full Verification

**Files:**
- All touched code and docs.

- [ ] **Step 1: Compile**

Run:

```bash
python3 -m compileall src app scripts tests
```

Expected: exit code `0`.

- [ ] **Step 2: Lint**

Run:

```bash
ruff check src tests app scripts
```

Expected: `All checks passed!`

- [ ] **Step 3: Test**

Run:

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Test with plugin autoload disabled**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit any final fixes**

```bash
git status --short
git add <changed-files>
git commit -m "Verify factorized physics residual model"
```

Only commit if there are final fixes or docs after the validation commit.

---

## Self-Review

Spec coverage:

- Uses V1 validation results as the reason for factorization.
- Keeps V1 conditional context and game-drift foundation.
- Does not add weather to V2.1.
- Adds layer-by-layer validation rather than only physics-core validation.
- Requires real Skubal evaluation before presentation claims.

Known risks:

- Factorized sampling may improve physics core but regress command or movement. The plan explicitly reports this rather than hiding it.
- Synthetic tests prove mechanics, not real-world performance. Task 6 is the real decision gate.
- The first implementation uses ridge residual layers, not a neural model. This is intentional: interpretable first, more complex only if V2.1 fails.
