"""LightGBM Multi-class Outcome Predictor for La Liga Matches.

Trains a gradient boosted decision tree ensemble to forecast 1/X/2 match probabilities
using pre-match rolling xG, Elo ratings, and fatigue differentials with early stopping.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd

from evaluate import (
    CLASS_NAMES,
    TARGET_MAPPING,
    evaluate_probability_predictions,
    temporal_train_val_test_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("LightGBMTrainer")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def extract_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Izoluje macierz cech predykcyjnych (X) od metadanych i wektora celu (y).

    Gwarantuje, że żadne identyfikatory ani zmienne post-match nie wejdą do modelu.

    Args:
        df: Zbiór analityczny (analytical_mart_wide).

    Returns:
        Tuple[pd.DataFrame, np.ndarray, list[str]]: Macierz X, wektor y oraz lista nazw cech.
    """
    excluded_cols = [
        "match_id",
        "match_date",
        "season",
        "home_team",
        "away_team",
        "FTR",
        "target",
    ]

    feature_cols = [c for c in df.columns if c not in excluded_cols]
    X = df[feature_cols].copy()
    y = df["target"].to_numpy().astype(int)

    return X, y, feature_cols


def train_lightgbm_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> lgb.LGBMClassifier:
    """Trenuje model LightGBM Classifier z regularizacją dostosowaną do małych prób.

    Wykorzystuje early stopping na zbiorze walidacyjnym, monitorując wieloklasowy Log Loss.

    Args:
        X_train: Cechy treningowe (sezony 2022-2024).
        y_train: Etykiety treningowe.
        X_val: Cechy walidacyjne (sezon 2024-2025).
        y_val: Etykiety walidacyjne.

    Returns:
        lgb.LGBMClassifier: Wytrenowany i zoptymalizowany model.
    """
    logger.info("Konfiguracja i inicjalizacja estymatora LightGBM Classifier...")

    # Ścisła regularyzacja pod zbiór ~750 wierszy
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        boosting_type="gbdt",
        learning_rate=0.02,
        n_estimators=1000,
        num_leaves=15,
        max_depth=4,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,  # Regularyzacja L1 (Lasso)
        reg_lambda=1.0,  # Regularyzacja L2 (Ridge)
        random_state=42,
        verbosity=-1,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=40, verbose=False),
        lgb.log_evaluation(period=0),  # Wyciszenie logów iteracji
    ]

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=callbacks,
    )

    logger.info(
        "Optymalna liczba drzew decyzyjnych (best_iteration): %d",
        clf.best_iteration_,
    )
    return clf


def analyze_feature_importance(
    model: lgb.LGBMClassifier, feature_names: list[str], top_n: int = 10
) -> pd.DataFrame:
    """Wylicza i prezentuje ważność cech na bazie całkowitego zysku informacyjnego (Gain).

    Args:
        model: Wytrenowany estymator LightGBM.
        feature_names: Lista analizowanych cech.
        top_n: Liczba najważniejszych zmiennych do wyświetlenia.

    Returns:
        pd.DataFrame: Posortowana tabela ważności cech.
    """
    importance_gain = model.booster_.feature_importance(importance_type="gain")
    df_imp = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance_Gain": importance_gain,
        }
    )
    df_imp["Relative_Weight_%"] = (
        df_imp["Importance_Gain"] / df_imp["Importance_Gain"].sum()
    ) * 100
    df_imp.sort_values(by="Importance_Gain", ascending=False, inplace=True)
    df_imp.reset_index(drop=True, inplace=True)

    logger.info("=== TOP %d NAJWAŻNIEJSZYCH CECH MODELU (Gain) ===", top_n)
    for idx, row in df_imp.head(top_n).iterrows():
        logger.info(
            "%2d. %-24s: %6.1f%% całkowitego zysku",
            idx + 1,
            row["Feature"],
            row["Relative_Weight_%"],
        )

    return df_imp


def run_lightgbm_pipeline() -> None:
    """Orkiestruje pełny proces uczenia, doboru liczby iteracji i ewaluacji out-of-time."""
    logger.info("=== START: TRENING I EWALUACJA MODELU LIGHTGBM ===")

    mart_path = PROCESSED_DATA_DIR / "analytical_mart_wide.parquet"
    if not mart_path.exists():
        raise FileNotFoundError(f"Brak pliku martu analitycznego: {mart_path}")

    df_mart = pd.read_parquet(mart_path)
    df_mart["target"] = df_mart["FTR"].map(TARGET_MAPPING).astype(int)

    # 1. Ścisły podział chronologiczny
    train_seasons = ["2022-2023", "2023-2024"]
    val_seasons = ["2024-2025"]
    test_seasons = ["2025-2026"]

    df_train, df_val, df_test = temporal_train_val_test_split(
        df=df_mart,
        train_seasons=train_seasons,
        val_seasons=val_seasons,
        test_seasons=test_seasons,
    )

    # 2. Izolacja macierzy cech i wektorów celu
    X_train, y_train, feature_cols = extract_features_and_target(df_train)
    X_val, y_val, _ = extract_features_and_target(df_val)
    X_test, y_test, _ = extract_features_and_target(df_test)

    logger.info("Wymiary macierzy cech: %d kolumn predykcyjnych.", len(feature_cols))

    # 3. Trening z early stopping na zbiorze walidacyjnym
    lgbm_model = train_lightgbm_classifier(X_train, y_train, X_val, y_val)

    # 4. Predykcja prawdopodobieństw na ślepym zbiorze testowym Out-of-Time
    y_prob_test = lgbm_model.predict_proba(X_test)

    # 5. Ewaluacja probabilistyczna
    metrics = evaluate_probability_predictions(
        y_true=y_test,
        y_prob=y_prob_test,
        model_name="LightGBM GBDT (Tuned & Regularized)",
    )

    # 6. Analiza ważności cech
    df_importance = analyze_feature_importance(lgbm_model, feature_cols, top_n=10)

    # 7. Zapis modelu i ważności cech na dysk
    model_save_path = MODELS_DIR / "lightgbm_classifier.txt"
    lgbm_model.booster_.save_model(str(model_save_path))
    df_importance.to_parquet(MODELS_DIR / "feature_importance.parquet", index=False)
    logger.info("Zapisano model do: %s", model_save_path)


if __name__ == "__main__":
    run_lightgbm_pipeline()