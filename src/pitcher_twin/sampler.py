"""Monte Carlo sampling from fitted real pitcher distributions."""

from __future__ import annotations

import pandas as pd

from pitcher_twin.models import GeneratorModel, sample_generator


def sample_pitch_session(
    model: GeneratorModel,
    n: int = 15,
    random_state: int = 42,
    context_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    samples = sample_generator(model, n=n, random_state=random_state, context_df=context_df)
    samples.insert(0, "sample_index", range(1, len(samples) + 1))
    samples["pitcher_name"] = model.pitcher_name
    samples["pitch_type"] = model.pitch_type
    samples["model_name"] = model.model_name
    samples["feature_group"] = model.feature_group
    samples["source"] = "simulated_from_real_model"
    return samples
