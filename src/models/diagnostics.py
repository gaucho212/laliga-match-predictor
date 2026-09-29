"""Model Diagnostics: Confusion Matrix and Class Probability Calibration.

Compares classification distributions and probability calibration for
Poisson GLM vs. LightGBM on the Out-of-Time Test Set.
"""

from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from baseline_poisson import PoissonGoalModel
from evaluate import (
    CLASS_NAMES,
    TARGET_MAPPING,
    temporal_train_val_test_split,
)
from train_lightgbm import extract_features_and_target

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def run_comparative_diagnostics() -> None:
    """Generuje macierze pomyłek i raporty klasyfikacji dla obu modeli."""
    # 1. Wczytanie danych
    df_mart = pd.read_parquet(PROCESSED_DATA_DIR / "analytical_mart_wide.parquet")
    df_raw = pd.read_parquet(PROCESSED_DATA_DIR / "matches_merged.parquet")
    df_mart["target"] = df_mart["FTR"].map(TARGET_MAPPING).astype(int)

    train_seasons = ["2022-2023", "2023-2024"]
    val_seasons = ["2024-2025"]
    test_seasons = ["2025-2026"]

    df_train, df_val, df_test = temporal_train_val_test_split(
        df_mart, train_seasons, val_seasons, test_seasons
    )

    X_test, y_test, _ = extract_features_and_target(df_test)

    # 2. Predykcje LightGBM
    lgb_booster = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm_classifier.txt"))
    y_prob_lgb = lgb_booster.predict(X_test)
    y_pred_lgb = np.argmax(y_prob_lgb, axis=1)

    # 3. Predykcje Poisson
    poisson_model = PoissonGoalModel(alpha=1.0)
    poisson_model.fit(df_train, df_raw)
    y_prob_poisson = poisson_model.predict_proba(df_test)
    y_pred_poisson = np.argmax(y_prob_poisson, axis=1)

    # 4. Generowanie i prezentacja macierzy pomyłek
    cm_lgb = confusion_matrix(y_test, y_pred_lgb)
    cm_poisson = confusion_matrix(y_test, y_pred_poisson)

    print("\n" + "=" * 55)
    print("ROZKŁAD FAKTYCZNY W ZBIORZE TESTOWYM (378 MECZÓW):")
    counts = np.bincount(y_test, minlength=3)
    for name, cnt in zip(CLASS_NAMES, counts):
        print(f"  {name:10s}: {cnt:3d} ({cnt / len(y_test):.1%})")

    print("\n" + "=" * 55)
    print("MACIERZ POMYŁEK: LIGHTGBM (Argmax)")
    df_cm_lgb = pd.DataFrame(
        cm_lgb,
        index=[f"True_{c}" for c in CLASS_NAMES],
        columns=[f"Pred_{c}" for c in CLASS_NAMES],
    )
    print(df_cm_lgb.to_string())

    print("\n" + "=" * 55)
    print("MACIERZ POMYŁEK: POISSON GLM (Argmax)")
    df_cm_poisson = pd.DataFrame(
        cm_poisson,
        index=[f"True_{c}" for c in CLASS_NAMES],
        columns=[f"Pred_{c}" for c in CLASS_NAMES],
    )
    print(df_cm_poisson.to_string())
    print("=" * 55)


if __name__ == "__main__":
    run_comparative_diagnostics()