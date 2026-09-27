"""Feature Engineering Module for La Liga Match Prediction Engine.

Transforms wide match facts into a granular Team-Match Long format,
allowing for temporal rolling aggregations strictly eliminating look-ahead bias.
"""

import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FeatureBuilder")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def calculate_match_points(row: pd.Series, is_home: bool) -> int:
    """Zwraca liczbę punktów zdobytych przez drużynę (3 za wygraną, 1 za remis, 0 za porażkę)."""
    ftr = row["FTR"]
    if ftr == "D":
        return 1
    if is_home:
        return 3 if ftr == "H" else 0
    else:
        return 3 if ftr == "A" else 0


def create_team_match_long_table(df_matches: pd.DataFrame) -> pd.DataFrame:
    """Transformuje tabelę meczów (Wide, 1 wiersz = 1 mecz) do formatu Team-Match (Long, 2 wiersze = 1 mecz).

    Args:
        df_matches: Oczyszczona i zintegrowana ramka danych z matches_merged.parquet.

    Returns:
        pd.DataFrame: Posortowana chronologicznie tabela Long o wymiarze 2*N wierszy.
    """
    logger.info("Rozpoczynanie transformacji tabeli Wide -> Long...")

    # Nadajemy unikalny, syntetyczny identyfikator meczu, jeśli nie istnieje
    df = df_matches.copy()
    if "match_id" not in df.columns:
        df["match_id"] = [f"match_{i:05d}" for i in range(len(df))]

    # 1. Ekstrakcja perspektywy Gospodarzy
    home_records = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "match_date": pd.to_datetime(df["match_date"]),
            "season": df["season"],
            "team": df["home_team"],
            "opponent": df["away_team"],
            "is_home": 1,
            "goals_scored": df["FTHG"].astype(int),
            "goals_conceded": df["FTAG"].astype(int),
            "xg_for": df["home_xg"].astype(float),
            "xg_against": df["away_xg"].astype(float),
            "shots_for": df["HS"].astype(int),
            "shots_against": df["AS"].astype(int),
            "sot_for": df["HST"].astype(int),
            "sot_against": df["AST"].astype(int),
            "points": [calculate_match_points(row, is_home=True) for _, row in df.iterrows()],
        }
    )

    # 2. Ekstrakcja perspektywy Gości
    away_records = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "match_date": pd.to_datetime(df["match_date"]),
            "season": df["season"],
            "team": df["away_team"],
            "opponent": df["home_team"],
            "is_home": 0,
            "goals_scored": df["FTAG"].astype(int),
            "goals_conceded": df["FTHG"].astype(int),
            "xg_for": df["away_xg"].astype(float),
            "xg_against": df["home_xg"].astype(float),
            "shots_for": df["AS"].astype(int),
            "shots_against": df["HS"].astype(int),
            "sot_for": df["AST"].astype(int),
            "sot_against": df["HST"].astype(int),
            "points": [calculate_match_points(row, is_home=False) for _, row in df.iterrows()],
        }
    )

    # 3. Złączenie wierszy i sortowanie osi czasu
    df_long = pd.concat([home_records, away_records], ignore_index=True)

    # Sortowanie: najpierw po nazwie zespołu, a w ramach zespołu chronologicznie po dacie meczu
    df_long.sort_values(by=["team", "match_date"], ascending=[True, True], inplace=True)
    df_long.reset_index(drop=True, inplace=True)

    # 4. Asercje integralności
    expected_rows = len(df_matches) * 2
    assert (
        len(df_long) == expected_rows
    ), f"Błąd wymiaru tabeli Long! Oczekiwano {expected_rows}, otrzymano {len(df_long)}"

    logger.info(
        "Pomyślnie utworzono tabelę Long: %d wierszy, %d unikalnych drużyn.",
        len(df_long),
        df_long["team"].nunique(),
    )
    return df_long


if __name__ == "__main__":
    merged_input_path = PROCESSED_DATA_DIR / "matches_merged.parquet"
    if not merged_input_path.exists():
        raise FileNotFoundError(f"Brak pliku bazowego: {merged_input_path}")

    df_clean = pd.read_parquet(merged_input_path)
    df_team_match = create_team_match_long_table(df_clean)

    # Zapis do warstwy interim/processed pod dalszy Feature Engineering
    long_output_path = PROCESSED_DATA_DIR / "team_matches_long.parquet"
    df_team_match.to_parquet(long_output_path, index=False)
    logger.info("Zapisano tabelę bazową Long do: %s", long_output_path)