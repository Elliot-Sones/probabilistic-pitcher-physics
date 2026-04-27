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
