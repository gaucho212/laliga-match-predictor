"""Automated data quality checks for raw ingested data.

Validates that raw Parquet files exist, are non-empty, and contain
critical columns required for subsequent feature engineering.
"""

from pathlib import Path
import pandas as pd
import pytest

# Dynamiczne wyznaczenie katalogu głównego projektu
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


@pytest.fixture(scope="session")
def raw_data_paths() -> dict[str, Path]:
    """Zwraca ścieżki do plików surowych."""
    return {
        "football_data": RAW_DATA_DIR / "football_data_raw.parquet",
        "understat": RAW_DATA_DIR / "understat_xg_raw.parquet",
    }


def test_raw_files_exist(raw_data_paths: dict[str, Path]) -> None:
    """Weryfikuje, czy potok ETL faktycznie wygenerował oba pliki Parquet."""
    for source_name, path in raw_data_paths.items():
        assert path.exists(), f"Plik dla źródła '{source_name}' nie istnieje: {path}"


def test_football_data_structure(raw_data_paths: dict[str, Path]) -> None:
    """Sprawdza integralność tabeli wyników i kursów (football-data.co.uk)."""
    df = pd.read_parquet(raw_data_paths["football_data"])

    # 1. Sprawdzenie, czy dane nie są puste
    assert not df.empty, "Tabela football_data_raw jest pusta!"
    assert len(df) > 1000, f"Oczekiwano >1000 meczów z 4 sezonów, pobrano tylko: {len(df)}"

    # 2. Sprawdzenie krytycznych kolumn biznesowych
    required_columns = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season"}
    missing_columns = required_columns - set(df.columns)
    assert not missing_columns, f"Brakujące kolumny w football-data: {missing_columns}"


def test_understat_xg_structure(raw_data_paths: dict[str, Path]) -> None:
    """Sprawdza integralność tabeli metryk xG (Understat)."""
    df = pd.read_parquet(raw_data_paths["understat"])

    assert not df.empty, "Tabela understat_xg_raw jest pusta!"

    # Sprawdzamy obecność xG (zabezpieczenie przed zmianą schematu API)
    xg_cols = [c for c in df.columns if "xg" in c.lower()]
    assert len(xg_cols) >= 2, f"Nie znaleziono kolumn xG/xGA w Understat. Dostępne: {list(df.columns)}"