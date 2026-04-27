from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from urllib.parse import parse_qs, urlparse


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "fetch_baseball_savant_pitcher_csv.py"


def _load_fetcher():
    spec = spec_from_file_location("fetch_baseball_savant_pitcher_csv", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_baseball_savant_csv_url_targets_real_pitcher_season() -> None:
    module = _load_fetcher()
    url = module.build_baseball_savant_csv_url(pitcher_id=592332, season=2025)
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    assert parsed.netloc == "baseballsavant.mlb.com"
    assert parsed.path == "/statcast_search/csv"
    assert query["player_type"] == ["pitcher"]
    assert query["pitchers_lookup[]"] == ["592332"]
    assert query["hfSea"] == ["2025|"]
    assert query["type"] == ["details"]
