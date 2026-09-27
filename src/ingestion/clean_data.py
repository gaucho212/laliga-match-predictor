"""Module responsible for Entity Resolution, Data Cleaning, and merging raw datasets.

Combines market/match data from football-data.co.uk with expected goals (xG)
from Understat into a single unified source of truth.
"""

import logging
from pathlib import Path
import re
from typing import Dict
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DataCleaning")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1-do-1 Canonical Entity Mapping (football-data -> Understat)
FOOTBALL_DATA_TO_UNDERSTAT_TEAMS: dict[str, str] = {
    "Ath Bilbao": "Athletic Club",
    "Ath Madrid": "Atletico Madrid",
    "Betis": "Real Betis",
    "Celta": "Celta Vigo",
    "Espanol": "Espanyol",
    "Oviedo": "Real Oviedo",
    "Sociedad": "Real Sociedad",
    "Valladolid": "Real Valladolid",
    "Vallecano": "Rayo Vallecano",
}


def standardize_team_names(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Ujednolica nazwy zespołów przy użyciu słownika mapującego."""
    return series.map(lambda name: mapping.get(str(name).strip(), str(name).strip()))


def standardize_season_format(series: pd.Series) -> pd.Series:
    """Konwertuje różne formaty zapisu sezonu ('2223', '2022', '2022-2023') do formatu 'YYYY-YYYY'.

    Args:
        series: Kolumna z oznaczeniem sezonu.

    Returns:
        pd.Series: Zunifikowany ciąg znaków w formacie '2023-2024'.
    """

    def _normalize(val: str) -> str:
        s = str(val).strip()
        # Jeśli już jest w formacie '2023-2024'
        if re.match(r"^\d{4}-\d{4}$", s):
            return s
        # Jeśli jest w formacie 4 cyfr '2324'
        if (
            re.match(r"^\d{4}$", s)
            and abs(int(s[2:]) - int(s[:2])) == 1
            and int(s[:2]) < 50
        ):
            start_yr = 2000 + int(s[:2])
            end_yr = 2000 + int(s[2:])
            return f"{start_yr}-{end_yr}"
        # Jeśli jest pojedynczym rokiem startowym '2023'
        if re.match(r"^\d{4}$", s) and int(s) >= 2000:
            start_yr = int(s)
            return f"{start_yr}-{start_yr + 1}"
        return s

    return series.map(_normalize)


def clean_and_merge_datasets(
    df_fd: pd.DataFrame,
    df_xg: pd.DataFrame,
    team_mapping: dict[str, str],
) -> pd.DataFrame:
    """Czyści, normalizuje i bezstratnie łączy zbiór football-data ze zbiorem Understat."""
    logger.info("Rozpoczynanie Entity Resolution i czyszczenia danych...")

    fd = df_fd.copy()
    xg = df_xg.copy()

    # 1. Normalizacja nazw zespołów
    fd["home_team"] = standardize_team_names(fd["HomeTeam"], team_mapping)
    fd["away_team"] = standardize_team_names(fd["AwayTeam"], team_mapping)
    xg["home_team"] = xg["home_team"].astype(str).str.strip()
    xg["away_team"] = xg["away_team"].astype(str).str.strip()

    # 2. Normalizacja formatu sezonu do standardu 'YYYY-YYYY'
    fd["season"] = standardize_season_format(fd["Season"])
    xg["season"] = standardize_season_format(xg["season"])

    logger.info(
        "Unikalne sezony po unifikacji w FD: %s", fd["season"].unique().tolist()
    )
    logger.info(
        "Unikalne sezony po unifikacji w XG: %s", xg["season"].unique().tolist()
    )

    # 3. Dynamiczne wykrywanie kolumn xG z Understat
    xg_cols_candidates_home = [c for c in xg.columns if c in ["home_xg", "xg_home"]]
    xg_cols_candidates_away = [c for c in xg.columns if c in ["away_xg", "xg_away"]]

    if not xg_cols_candidates_home or not xg_cols_candidates_away:
        raise KeyError(
            f"Nie znaleziono kolumn xG w Understat. Dostępne: {list(xg.columns)}"
        )

    home_xg_col = xg_cols_candidates_home[0]
    away_xg_col = xg_cols_candidates_away[0]

    xg_subset = xg[
        ["season", "home_team", "away_team", home_xg_col, away_xg_col]
    ].copy()
    xg_subset.rename(
        columns={home_xg_col: "home_xg", away_xg_col: "away_xg"}, inplace=True
    )

    # 4. Sprawdzenie integralności klucza złożonego
    join_keys = ["season", "home_team", "away_team"]
    dup_fd = fd.duplicated(subset=join_keys).sum()
    dup_xg = xg_subset.duplicated(subset=join_keys).sum()

    if dup_fd > 0 or dup_xg > 0:
        raise ValueError(
            f"Klucz złączenia nie jest unikalny! Duplikaty w FD: {dup_fd}, w XG: {dup_xg}"
        )

    # 5. Bezstratne złączenie wewnętrzne (Inner Join)
    merged_df = pd.merge(fd, xg_subset, on=join_keys, how="inner")
    logger.info(
        "Liczba wierszy po złączeniu: %d (oczekiwano: %d)", len(merged_df), len(df_fd)
    )

    # 6. Quality Gate: weryfikacja utraty rekordów
    if len(merged_df) != len(df_fd):
        missing_count = len(df_fd) - len(merged_df)
        logger.error("Wykryto utratę %d meczów podczas złączenia!", missing_count)

        merged_keys = set(
            zip(merged_df["season"], merged_df["home_team"], merged_df["away_team"])
        )
        fd_keys = set(zip(fd["season"], fd["home_team"], fd["away_team"]))
        missing_matches = fd_keys - merged_keys
        logger.error("Przykładowe niedopasowane mecze: %s", list(missing_matches)[:5])
        raise RuntimeError(
            "Błąd Entity Resolution: Niespójność uniemożliwiła pełne złączenie."
        )

    # 7. Parsowanie i sortowanie chronologiczne dat
    merged_df["match_date"] = pd.to_datetime(merged_df["Date"], dayfirst=True)
    merged_df.sort_values(by="match_date", inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df


def run_clean_pipeline() -> None:
    """Orkiestruje wczytanie danych z data/raw, czyszczenie i zapis do data/processed."""
    logger.info("=== START: DATA CLEANING & MERGING ===")

    fd_path = RAW_DATA_DIR / "football_data_raw.parquet"
    xg_path = RAW_DATA_DIR / "understat_xg_raw.parquet"

    if not fd_path.exists() or not xg_path.exists():
        raise FileNotFoundError(
            "Brak wymaganych plików surowych w data/raw/. Uruchom najpierw fetch_raw_data.py."
        )

    df_fd = pd.read_parquet(fd_path)
    df_xg = pd.read_parquet(xg_path)

    merged_df = clean_and_merge_datasets(df_fd, df_xg, FOOTBALL_DATA_TO_UNDERSTAT_TEAMS)

    output_path = PROCESSED_DATA_DIR / "matches_merged.parquet"
    merged_df.to_parquet(output_path, index=False)
    logger.info("Zapisano oczyszczoną tabelę meczową do: %s", output_path)
    logger.info("=== SUKCES: DATA CLEANING ZAKOŃCZONY ===")


if __name__ == "__main__":
    run_clean_pipeline()
