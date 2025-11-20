from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import requests

from .config import AppConfig, get_config

FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/mmz4281"


def build_download_url(season_code: str, league_code: str) -> str:
    return f"{FOOTBALL_DATA_BASE}/{season_code}/{league_code}.csv"


def download_file(url: str, destination: Path) -> None:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    destination.write_bytes(response.content)


def download_league(cfg: AppConfig, season_code: str, league_code: str) -> Path | None:
    destination = cfg.raw_dir / f"{league_code}_{season_code}.csv"
    url = build_download_url(season_code, league_code)
    try:
        download_file(url, destination)
    except requests.HTTPError as exc:
        print(f"Failed to download {url}: {exc}")
        return None
    print(f"Saved {destination.relative_to(cfg.base_dir)}")
    time.sleep(1)
    return destination


def sync_all(cfg: AppConfig) -> list[Path]:
    downloaded: list[Path] = []
    for league in cfg.leagues:
        for season_code in league.season_codes:
            file_path = download_league(cfg, season_code, league.code)
            if file_path:
                downloaded.append(file_path)
    return downloaded


def main() -> None:
    cfg = get_config()
    files = sync_all(cfg)
    if files:
        print(f"Downloaded/updated {len(files)} files.")
    else:
        print("No files downloaded. Check league codes/seasons.")


if __name__ == "__main__":
    main()
