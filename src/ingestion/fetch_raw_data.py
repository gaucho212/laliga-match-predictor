"""Module responsible for fetching raw match data and advanced metrics (xG).

Sources:
- football-data.co.uk (Match statistics, scores, referee data, closing betting odds)
- Understat via soccerdata (Expected Goals - xG / xGA)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import requests
import soccerdata as sd

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DataIngestion")

# Ścieżki katalogów
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Kody sezonów dla football-data.co.uk (np. '2324' = 2023/2024)
FOOTBALL_DATA_SEASONS: dict[str, str] = {
    "2022-2023": "2223",
    "2023-2024": "2324",
    "2024-2025": "2425",
    "2025-2026": "2526",
}

# Sezony dla biblioteki soccerdata (Understat)
SOCCERDATA_SEASONS: list[str] = ["2022-2023", "2023-2024", "2024-2025", "2025-2026"]


def fetch_football_data_co_uk(
    seasons: dict[str, str], league_code: str = "SP1"
) -> pd.DataFrame:
    """Pobiera historyczne dane meczowe oraz kursy z portalu football-data.co.uk.

    Args:
        seasons: Słownik mapujący czytelną nazwę sezonu na kod URL (np. {'2023-2024': '2324'}).
        league_code: Kod ligi ('SP1' oznacza hiszpańską Primera Division / La Liga).

    Returns:
        pd.DataFrame: Połączona tabela surowych meczów z wybranych sezonów.
    """
    logger.info("Rozpoczynanie pobierania danych z football-data.co.uk...")
    dfs: list[pd.DataFrame] = []

    for season_name, season_code in seasons.items():
        url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"
        logger.info(f"Pobieranie sezonu {season_name} z: {url}")

        try:
            # Niektóre starsze pliki CSV mają specyficzne kodowanie
            df_season = pd.read_csv(url, encoding="unicode_escape")
            # Usuwamy puste wiersze pojawiające się na końcu plików football-data
            df_season = df_season.dropna(how="all")
            df_season["Season"] = season_name
            dfs.append(df_season)
            logger.info(f"-> Pobrano {len(df_season)} meczów dla sezonu {season_name}.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Błąd sieci podczas pobierania {url}: {e}")
        except Exception as e:
            logger.error(f"Nieoczekiwany błąd przy parsowaniu {url}: {e}")

    if not dfs:
        raise RuntimeError("Nie udało się pobrać żadnych danych z football-data.co.uk.")

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Pomyślnie zintegrowano football-data. Łącznie wierszy: {len(combined_df)}")
    return combined_df


def fetch_understat_xg_data(seasons: list[str]) -> pd.DataFrame:
    """Pobiera dane xG (Expected Goals) z Understat za pośrednictwem soccerdata.

    Args:
        seasons: Lista sezonów w formacie ['2023-2024', ...].

    Returns:
        pd.DataFrame: Zbiór danych meczowych zawierający metryki xG i xGA.
    """
    logger.info("Inicjalizacja pobierania zaawansowanych metryk xG z Understat...")
    try:
        understat = sd.Understat(leagues="ESP-La Liga", seasons=seasons)
        df_schedule = understat.read_schedule()

        # soccerdata zwraca MultiIndex, spłaszczamy go dla ułatwienia późniejszych transformacji
        df_schedule = df_schedule.reset_index()
        logger.info(f"Pomyślnie pobrano harmonogram i xG z Understat. Liczba wierszy: {len(df_schedule)}")
        return df_schedule
    except Exception as e:
        logger.error(f"Błąd podczas odpytywania Understat via soccerdata: {e}")
        raise


def run_pipeline() -> None:
    """Orkiestruje proces pobierania surowych danych i zapisuje pliki Parquet."""
    logger.info("=== DATA INGESTION ===")

    # 1. Pobranie danych bazowych i kursów
    raw_fd_df = fetch_football_data_co_uk(FOOTBALL_DATA_SEASONS)
    fd_output_path = RAW_DATA_DIR / "football_data_raw.parquet"
    raw_fd_df.to_parquet(fd_output_path, index=False)
    logger.info(f"Zapisano plik meczowy: {fd_output_path}")

    # 2. Pobranie danych xG z Understat
    raw_xg_df = fetch_understat_xg_data(SOCCERDATA_SEASONS)
    xg_output_path = RAW_DATA_DIR / "understat_xg_raw.parquet"
    raw_xg_df.to_parquet(xg_output_path, index=False)
    logger.info(f"Zapisano plik xG: {xg_output_path}")

    logger.info("=== INGESTION ZAKOŃCZONY SUKCESEM ===")


if __name__ == "__main__":
    run_pipeline()