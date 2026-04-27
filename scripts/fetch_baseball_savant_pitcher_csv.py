#!/usr/bin/env python3
"""Fetch real Baseball Savant Statcast CSV rows for one pitcher/season."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASEBALL_SAVANT_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"


def build_baseball_savant_csv_url(
    pitcher_id: int,
    season: int,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    params = {
        "all": "true",
        "hfPT": "",
        "hfAB": "",
        "hfBBT": "",
        "hfPR": "",
        "hfZ": "",
        "stadium": "",
        "hfBBL": "",
        "hfNewZones": "",
        "hfGT": "R|",
        "hfC": "",
        "hfSea": f"{season}|",
        "hfSit": "",
        "player_type": "pitcher",
        "hfOuts": "",
        "opponent": "",
        "pitcher_throws": "",
        "batter_stands": "",
        "hfSA": "",
        "game_date_gt": start_date or f"{season}-03-01",
        "game_date_lt": end_date or f"{season}-11-30",
        "hfInfield": "",
        "team": "",
        "position": "",
        "hfOutfield": "",
        "hfRO": "",
        "home_road": "",
        "hfFlag": "",
        "hfPull": "",
        "metric_1": "",
        "hfInn": "",
        "min_pitches": "0",
        "min_results": "0",
        "group_by": "name",
        "sort_col": "pitches",
        "player_event_sort": "api_p_release_speed",
        "sort_order": "desc",
        "min_pas": "0",
        "type": "details",
        "pitchers_lookup[]": str(pitcher_id),
    }
    return f"{BASEBALL_SAVANT_CSV_URL}?{urlencode(params)}"


def fetch_csv(url: str, output_path: Path) -> Path:
    request = Request(url, headers={"User-Agent": "pitch-pitcher-twin/1.0"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(request, timeout=90) as response:
        output_path.write_bytes(response.read())
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pitcher-id", type=int, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    url = build_baseball_savant_csv_url(
        pitcher_id=args.pitcher_id,
        season=args.season,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    output = fetch_csv(url, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
