"""Poisson Regression Benchmark Model for Soccer Match Outcome Prediction.

Estimates expected goals (lambda_home, lambda_away) using pre-match rolling xG
and Elo differentials, then maps the score probability matrix onto the 1/X/2 market.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from evaluate import (
    TARGET_MAPPING,
    evaluate_probability_predictions,
    temporal_train_val_test_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PoissonModel")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def calculate_match_probabilities_from_lambdas(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    max_goals: int = 10,
) -> np.ndarray:
    """Konwertuje wektory oczekiwanych goli (lambda) na macierz prawdopodobieństw 1/X/2.

    Dla każdego spotkania generuje siatkę wyników (Score Grid) od 0:0 do max_goals:max_goals,
    sumując prawdopodobieństwa dla wygranej gospodarza (H), remisu (D) i wygranej gościa (A).

    Args:
        lambda_home: Wektor przewidywanych goli gospodarzy (N,).
        lambda_away: Wektor przewidywanych goli gości (N,).
        max_goals: Maksymalna liczba rozważanych bramek na zespół (zabezpieczenie numeryczne).

    Returns:
        np.ndarray: Macierz prawdopodobieństw o wymiarach (N, 3) dla klas [H, D, A].
    """
    n_matches = len(lambda_home)
    probs = np.zeros((n_matches, 3))
    goals_range = np.arange(max_goals + 1)

    for i in range(n_matches):
        l_h = max(lambda_home[i], 1e-4)
        l_a = max(lambda_away[i], 1e-4)

        # Rozkłady prawdopodobieństwa bramek dla obu zespołów
        p_home_goals = poisson.pmf(goals_range, l_h)
        p_away_goals = poisson.pmf(goals_range, l_a)

        # Iloczyn zewnętrzny: macierz prawdopodobieństw dokładnych wyników (max_goals+1, max_goals+1)
        score_matrix = np.outer(p_home_goals, p_away_goals)

        # Agregacja prawdopodobieństw 1/X/2
        p_draw = np.sum(np.diag(score_matrix))
        p_home = np.sum(np.tril(score_matrix, k=-1))
        p_away = np.sum(np.triu(score_matrix, k=1))

        # Normalizacja wiersza do sumy 1.0
        total_p = p_home + p_draw + p_away
        probs[i] = [p_home / total_p, p_draw / total_p, p_away / total_p]

    return probs


class PoissonGoalModel:
    """Opakowanie na dwa modele regresji Poissona estymujące gole obu zespołów."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Inicjalizuje regresory z regularyzacją L2 (Ridge)."""
        self.model_home = PoissonRegressor(alpha=alpha, max_iter=500)
        self.model_away = PoissonRegressor(alpha=alpha, max_iter=500)
        self.home_feature_cols: list[str] = []
        self.away_feature_cols: list[str] = []

    def fit(self, df_train: pd.DataFrame, df_raw_matches: pd.DataFrame) -> "PoissonGoalModel":
        """Trenuje modele na danych historycznych bez wycieku danych."""
        logger.info("Trenowanie modeli regresji Poissona dla bramek...")

        # Złączenie po sprawdzonym kluczu naturalnym (season, home_team, away_team)
        join_keys = ["season", "home_team", "away_team"]
        target_goals = df_raw_matches[join_keys + ["FTHG", "FTAG"]].copy()

        train_merged = df_train.merge(target_goals, on=join_keys, how="inner")

        if len(train_merged) != len(df_train):
            raise ValueError(
                f"Utrata rekordów przy złączeniu celów! Oczekiwano {len(df_train)}, otrzymano {len(train_merged)}"
            )

        self.home_feature_cols = [
            "home_roll_xg_for_5",
            "away_roll_xg_against_5",
            "elo_diff_pre",
            "home_rest_days",
        ]
        self.away_feature_cols = [
            "away_roll_xg_for_5",
            "home_roll_xg_against_5",
            "elo_diff_pre",
            "away_rest_days",
        ]

        X_home = train_merged[self.home_feature_cols]
        y_home = train_merged["FTHG"].astype(int)

        X_away = train_merged[self.away_feature_cols]
        y_away = train_merged["FTAG"].astype(int)

        self.model_home.fit(X_home, y_home)
        self.model_away.fit(X_away, y_away)
        logger.info("Modele Poissona wytrenowane pomyślnie na %d meczach.", len(train_merged))
        return self

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        """Generuje prawdopodobieństwa 1/X/2 dla zadanej ramki cech."""
        X_home = df_features[self.home_feature_cols]
        X_away = df_features[self.away_feature_cols]

        lambda_h = self.model_home.predict(X_home)
        lambda_a = self.model_away.predict(X_away)

        return calculate_match_probabilities_from_lambdas(lambda_h, lambda_a)


def run_poisson_benchmark() -> None:
    """Orkiestruje ewaluację modelu Poissona na zbiorze testowym Out-of-Time."""
    logger.info("=== START: POISSON BENCHMARK EVALUATION ===")

    mart_path = PROCESSED_DATA_DIR / "analytical_mart_wide.parquet"
    raw_path = PROCESSED_DATA_DIR / "matches_merged.parquet"

    df_mart = pd.read_parquet(mart_path)
    df_raw = pd.read_parquet(raw_path)

    # 1. Kodowanie targetu (H: 0, D: 1, A: 2)
    df_mart["target"] = df_mart["FTR"].map(TARGET_MAPPING).astype(int)

    # 2. Podział chronologiczny
    train_seasons = ["2022-2023", "2023-2024"]
    val_seasons = ["2024-2025"]
    test_seasons = ["2025-2026"]

    df_train, df_val, df_test = temporal_train_val_test_split(
        df_mart, train_seasons, val_seasons, test_seasons
    )

    # 3. Trening modelu
    model = PoissonGoalModel(alpha=1.0)
    model.fit(df_train, df_raw)

    # 4. Predykcja na zbiorze testowym Out-of-Time
    y_test = df_test["target"].to_numpy()
    y_prob_poisson = model.predict_proba(df_test)

    # 5. Ewaluacja probabilistyczna
    metrics = evaluate_probability_predictions(
        y_true=y_test,
        y_prob=y_prob_poisson,
        model_name="Poisson GLM (xG & Elo Calibrated)",
    )


if __name__ == "__main__":
    run_poisson_benchmark()