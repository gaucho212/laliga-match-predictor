"""Dynamic Elo Rating Engine tailored for Association Football (La Liga).

Incorporates Home Pitch Advantage, Goal Difference Multipliers (World Football Elo standard),
and guarantees zero data leakage by extracting pre-match ratings before state updates.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EloEngine")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def calculate_goal_margin_multiplier(home_goals: int, away_goals: int) -> float:
    """Oblicza mnożnik marginesu bramkowego (G) wg reguł World Football Elo.

    Większa różnica bramek generuje większy transfer punktów Elo między zespołami.

    Args:
        home_goals: Liczba goli strzelonych przez gospodarza.
        away_goals: Liczba goli strzelonych przez gościa.

    Returns:
        float: Mnożnik G skalujący współczynnik K.
    """
    diff = abs(home_goals - away_goals)
    if diff <= 1:
        return 1.0
    elif diff == 2:
        return 1.5
    else:
        # Skalowanie nieliniowe dla pogromów (3+ bramek)
        return (11.0 + diff) / 8.0


def calculate_expected_outcome(
    home_elo: float, away_elo: float, home_advantage: float = 65.0
) -> Tuple[float, float]:
    """Wyznacza prawdopodobieństwo oczekiwane wyniku meczu na bazie różnicy Elo.

    Args:
        home_elo: Aktualny rating Elo gospodarza.
        away_elo: Aktualny rating Elo gościa.
        home_advantage: Stała punktowa przewagi własnego boiska (handicap domowy).

    Returns:
        Tuple[float, float]: Oczekiwany wynik (W_e_home, W_e_away) w przedziale [0, 1].
    """
    # Efektywna różnica z uwzględnieniem atutu własnego stadionu
    elo_diff = (home_elo + home_advantage) - away_elo
    we_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    we_away = 1.0 - we_home
    return we_home, we_away


def compute_historical_elo(
    df_matches: pd.DataFrame,
    initial_elo: float = 1500.0,
    k_factor: float = 20.0,
    home_advantage: float = 65.0,
) -> pd.DataFrame:
    """Przeprowadza chronologiczną symulację rankingu Elo dla wszystkich spotkań.

    Kluczowa cecha architektury: gwarantuje brak wycieku danych (Look-Ahead Bias)
    poprzez rejestrację ratingów PRZED wykonaniem aktualizacji pomeczowej.

    Args:
        df_matches: Oczyszczona tabela meczowa (Wide format) z kolumnami:
            ['match_date', 'home_team', 'away_team', 'FTHG', 'FTAG', 'FTR'].
        initial_elo: Bazowy rating początkowy dla każdego klubu.
        k_factor: Współczynnik dynamiki zmian rankingu.
        home_advantage: Premia punktowa za grę u siebie.

    Returns:
        pd.DataFrame: Tabela wejściowa wzbogacona o cechy:
            ['home_elo_pre', 'away_elo_pre', 'elo_diff_pre'].
    """
    logger.info("Rozpoczynanie chronologicznej symulacji rankingu Elo...")

    # Bezwzględny wymóg: sortowanie chronologiczne
    df = df_matches.sort_values(by="match_date", ascending=True).copy()
    df.reset_index(drop=True, inplace=True)

    # Stan globalny: słownik przechowujący aktualny rating każdego klubu
    current_elos: dict[str, float] = {}

    home_elo_pre = []
    away_elo_pre = []

    for idx, row in df.iterrows():
        h_team = row["home_team"]
        a_team = row["away_team"]
        h_goals = int(row["FTHG"])
        a_goals = int(row["FTAG"])
        result = row["FTR"]  # 'H', 'D', 'A'

        # Inicjalizacja beniaminka lub nowego zespołu w bazie
        r_home = current_elos.get(h_team, initial_elo)
        r_away = current_elos.get(a_team, initial_elo)

        # 1. ZAPISUJEMY CECHY PRZED MECZEM (Zero Data Leakage)
        home_elo_pre.append(r_home)
        away_elo_pre.append(r_away)

        # 2. KALKULACJA AKTUALIZACJI PO MECZU
        we_home, we_away = calculate_expected_outcome(r_home, r_away, home_advantage)
        margin_multiplier = calculate_goal_margin_multiplier(h_goals, a_goals)

        # Rzeczywisty wynik w formacie numerycznym
        if result == "H":
            actual_home, actual_away = 1.0, 0.0
        elif result == "A":
            actual_home, actual_away = 0.0, 1.0
        else:  # Remis 'D'
            actual_home, actual_away = 0.5, 0.5

        # Aktualizacja stanu dla obu zespołów
        delta_home = k_factor * margin_multiplier * (actual_home - we_home)
        delta_away = k_factor * margin_multiplier * (actual_away - we_away)

        current_elos[h_team] = r_home + delta_home
        current_elos[a_team] = r_away + delta_away

    # 3. Dodanie cech do ramki
    df["home_elo_pre"] = home_elo_pre
    df["away_elo_pre"] = away_elo_pre
    # Kluczowa cecha interakcyjna dla modeli ML:
    df["elo_diff_pre"] = (df["home_elo_pre"] + home_advantage) - df["away_elo_pre"]

    logger.info("Symulacja Elo zakończona. Przetworzono %d meczów.", len(df))
    return df

if __name__ == "__main__":
    input_path = PROCESSED_DATA_DIR / "matches_merged.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Brak pliku: {input_path}")

    df_matches = pd.read_parquet(input_path)
    df_with_elo = compute_historical_elo(df_matches)

    # Diagnostyka stanu końcowego (Power Ranking po 4 sezonach)
    print("\n--- DIAGNOSTYKA: KOŃCOWY POWER RANKING ELO ---")
    latest_matches = df_with_elo.tail(50)  # Ostatnie mecze w bazie
    all_teams = set(df_with_elo["home_team"].unique())

    # Pobieramy ostatni znany pre-elo dla każdego zespołu
    final_ratings = {}
    for team in all_teams:
        team_matches = df_with_elo[
            (df_with_elo["home_team"] == team) | (df_with_elo["away_team"] == team)
        ].iloc[-1]

        if team_matches["home_team"] == team:
            final_ratings[team] = team_matches["home_elo_pre"]
        else:
            final_ratings[team] = team_matches["away_elo_pre"]

    ranking_df = (
        pd.DataFrame(list(final_ratings.items()), columns=["Team", "Final_Elo"])
        .sort_values(by="Final_Elo", ascending=False)
        .reset_index(drop=True)
    )

    print(ranking_df.head(8).to_string())
    print("...")
    print(ranking_df.tail(4).to_string())