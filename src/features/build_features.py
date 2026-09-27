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

def calculate_rest_days(
    df_long: pd.DataFrame, max_rest_cap: int = 21
) -> pd.DataFrame:
    """Oblicza liczbę dni odpoczynku od poprzedniego oficjalnego meczu dla każdej drużyny.

    Dla pierwszego meczu w sezonie (lub po długiej przerwie letniej) wartość jest
    ograniczana z góry (capping), aby uniknąć anomalii numerycznych (outliers).

    Args:
        df_long: Posortowana chronologicznie tabela Long (team, match_date).
        max_rest_cap: Maksymalna liczba dni odpoczynku (wartość nasycenia).

    Returns:
        pd.DataFrame: Ramka z dodaną kolumną 'rest_days'.
    """
    logger.info("Kalkulacja dni odpoczynku (Rest Days)...")
    df = df_long.copy()

    # Różnica dat w dniach względem poprzedniego meczu TEJ SAMEJ drużyny
    df["prev_match_date"] = df.groupby("team")["match_date"].shift(1)
    df["rest_days"] = (df["match_date"] - df["prev_match_date"]).dt.days

    # Pierwszy mecz w sezonie lub przerwa letnia: uzupełniamy medianą lub wartością graniczną
    # 7 dni to standardowy mikrocykl tygodniowy w lidze
    df["rest_days"] = df["rest_days"].fillna(max_rest_cap)

    # Capping – 90 dni przerwy wakacyjnej nie oznacza 10x lepszej regeneracji niż 14 dni
    df["rest_days"] = df["rest_days"].clip(lower=2, upper=max_rest_cap)

    df.drop(columns=["prev_match_date"], inplace=True)
    return df

def add_rolling_metrics(
    df_long: pd.DataFrame, windows: list[int] = [3, 5]
) -> pd.DataFrame:
    """Generuje metryki kroczące formy sportowej z rygorystyczną eliminacją Data Leakage.

    Każda metryka jest przesuwana o 1 mecz wstecz (.shift(1)) przed wykonaniem .rolling().

    Args:
        df_long: Tabela Long z podstawowymi statystykami meczowymi.
        windows: Lista rozmiarów okien czasowych (liczba poprzednich meczów).

    Returns:
        pd.DataFrame: Tabela Long wzbogacona o cechy kroczące.
    """
    logger.info("Generowanie metryk kroczących (Rolling Features)...")
    df = df_long.copy()

    # 1. Metryki bazowe na poziomie pojedynczego meczu
    df["xg_diff"] = df["xg_for"] - df["xg_against"]

    # SOTR: obsługa dzielenia przez zero, gdy w meczu nie padł żaden strzał celny
    total_sot = df["sot_for"] + df["sot_against"]
    df["sotr"] = np.where(total_sot > 0, df["sot_for"] / total_sot, 0.5)

    # 2. Obliczanie średnich kroczących w pętli po oknach
    for w in windows:
        # Rolling xG Differential
        df[f"roll_xg_diff_{w}"] = (
            df.groupby("team")["xg_diff"]
            .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
        )

        # Rolling xG For (jakość ataku)
        df[f"roll_xg_for_{w}"] = (
            df.groupby("team")["xg_for"]
            .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
        )

        # Rolling xG Against (szczelność defensywy)
        df[f"roll_xg_against_{w}"] = (
            df.groupby("team")["xg_against"]
            .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
        )

        # Rolling Shots on Target Ratio
        df[f"roll_sotr_{w}"] = (
            df.groupby("team")["sotr"]
            .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
        )

        # Rolling Points Per Game (Forma punktowa)
        df[f"roll_ppg_{w}"] = (
            df.groupby("team")["points"]
            .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
        )

    return df


if __name__ == "__main__":
    merged_input_path = PROCESSED_DATA_DIR / "matches_merged.parquet"
    if not merged_input_path.exists():
        raise FileNotFoundError(f"Brak pliku bazowego: {merged_input_path}")

    df_clean = pd.read_parquet(merged_input_path)
    df_team_match = create_team_match_long_table(df_clean)
    
    df_with_rest = calculate_rest_days(df_team_match)
    df_features_long = add_rolling_metrics(df_with_rest)    

    # Zapis do warstwy interim/processed pod dalszy Feature Engineering
    long_output_path = PROCESSED_DATA_DIR / "team_matches_long.parquet"
    df_features_long.to_parquet(long_output_path, index=False)
    logger.info("Zapisano tabelę bazową Long do: %s", long_output_path)