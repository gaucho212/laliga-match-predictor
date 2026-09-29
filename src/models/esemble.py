"""Ensemble Probability Blending Module for La Liga Match Prediction.

Optimizes blending weights between Poisson GLM and LightGBM GBDT on the
validation set, then benchmarks the ensemble on the out-of-time test set.
"""

import logging
from pathlib import Path
from typing import Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from baseline_poisson import PoissonGoalModel
from evaluate import (
    TARGET_MAPPING,
    evaluate_probability_predictions,
    temporal_train_val_test_split,
)
from train_lightgbm import extract_features_and_target

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EnsembleOptimizer")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def find_optimal_blend_weight(
    y_true_val: np.ndarray,
    prob_poisson_val: np.ndarray,
    prob_lgb_val: np.ndarray,
    steps: int = 101,
) -> Tuple[float, float]:
    """Wyszukuje optymalną wagę w (grid search) minimalizującą Log Loss na zbiorze walidacyjnym.

    P_ensemble = w * P_poisson + (1 - w) * P_lgbm

    Args:
        y_true_val: Rzeczywiste etykiety ze zbioru walidacyjnego.
        prob_poisson_val: Prawdopodobieństwa Poissona na walidacji.
        prob_lgb_val: Prawdopodobieństwa LightGBM na walidacji.
        steps: Gęstość siatki poszukiwań w przedziale [0, 1].

    Returns:
        Tuple[float, float]: Optymalna waga Poissona (w) oraz najlepszy uzyskany Log Loss.
    """
    best_weight = 0.5
    best_loss = float("inf")
    weights = np.linspace(0.0, 1.0, steps)

    for w in weights:
        blended = w * prob_poisson_val + (1.0 - w) * prob_lgb_val
        # Normalizacja wierszowa
        blended = blended / blended.sum(axis=1, keepdims=True)
        loss = log_loss(y_true_val, blended)

        if loss < best_loss:
            best_loss = loss
            best_weight = w

    logger.info(
        "Optymalizacja walidacyjna: Najlepsza waga Poissona = %.2f (LGBM = %.2f), Val Log Loss = %.4f",
        best_weight,
        1.0 - best_weight,
        best_loss,
    )
    return float(best_weight), float(best_loss)


def run_ensemble_pipeline() -> None:
    """Orkiestruje proces trenowania, optymalizacji wag i ślepego testu modelu hybrydowego."""
    logger.info("=== START: ENSEMBLE PROBABILITY BLENDING ===")

    # 1. Wczytanie danych
    df_mart = pd.read_parquet(PROCESSED_DATA_DIR / "analytical_mart_wide.parquet")
    df_raw = pd.read_parquet(PROCESSED_DATA_DIR / "matches_merged.parquet")
    df_mart["target"] = df_mart["FTR"].map(TARGET_MAPPING).astype(int)

    # 2. Ścisły podział chronologiczny
    train_seasons = ["2022-2023", "2023-2024"]
    val_seasons = ["2024-2025"]
    test_seasons = ["2025-2026"]

    df_train, df_val, df_test = temporal_train_val_test_split(
        df_mart, train_seasons, val_seasons, test_seasons
    )

    X_val, y_val, _ = extract_features_and_target(df_val)
    X_test, y_test, _ = extract_features_and_target(df_test)

    # 3. Trening i predykcje Poissona
    poisson_model = PoissonGoalModel(alpha=1.0)
    poisson_model.fit(df_train, df_raw)
    p_poisson_val = poisson_model.predict_proba(df_val)
    p_poisson_test = poisson_model.predict_proba(df_test)

    # 4. Predykcje LightGBM
    lgb_booster = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm_classifier.txt"))
    p_lgb_val = lgb_booster.predict(X_val)
    p_lgb_test = lgb_booster.predict(X_test)

    # 5. Optymalizacja wagi na zbiorze walidacyjnym (Zero Test Leakage)
    opt_w, val_loss = find_optimal_blend_weight(y_val, p_poisson_val, p_lgb_val)

    # 6. Ślepy test na zbiorze 2025-2026 (Out-of-Time)
    p_ensemble_test = opt_w * p_poisson_test + (1.0 - opt_w) * p_lgb_test
    p_ensemble_test = p_ensemble_test / p_ensemble_test.sum(axis=1, keepdims=True)

    # 7. Ewaluacja probabilistyczna modelu hybrydowego
    metrics = evaluate_probability_predictions(
        y_true=y_test,
        y_prob=p_ensemble_test,
        model_name=f"Hybrid Ensemble (Poisson {opt_w:.2f} + LGBM {1.0 - opt_w:.2f})",
    )


if __name__ == "__main__":
    run_ensemble_pipeline()