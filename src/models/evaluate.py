"""Evaluation and Temporal Validation Module for Match Outcome Predictions.

Implements strict chronological train-test splitting and probabilistic evaluation
metrics (Multi-class Log Loss, Brier Score) tailored for 1/X/2 soccer forecasting.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ModelEvaluation")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Konwencja kodowania klas wynikowych
TARGET_MAPPING: dict[str, int] = {"H": 0, "D": 1, "A": 2}
CLASS_NAMES: list[str] = ["Home_Win", "Draw", "Away_Win"]


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_seasons: list[str],
    val_seasons: list[str],
    test_seasons: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dzieli zbiór analityczny na podzbiory train, validation i test w oparciu o sezony.

    Gwarantuje brak wycieku danych pomiędzy przeszłością a przyszłością.

    Args:
        df: Zintegrowana tabela facts mart (zawiera kolumnę 'season').
        train_seasons: Lista sezonów do nauki modeli (np. ['2022-2023', '2023-2024']).
        val_seasons: Lista sezonów do strojenia hiperparametrów (np. ['2024-2025']).
        test_seasons: Sezon out-of-time do ewaluacji końcowej (np. ['2025-2026']).

    Returns:
        Tuple zawierająca ramki (df_train, df_val, df_test).
    """
    logger.info("Rozpoczynanie chronologicznego podziału zbioru danych...")

    df_train = df[df["season"].isin(train_seasons)].copy()
    df_val = df[df["season"].isin(val_seasons)].copy()
    df_test = df[df["season"].isin(test_seasons)].copy()

    # Weryfikacja integralności podziału
    total_split = len(df_train) + len(df_val) + len(df_test)
    if total_split != len(df):
        logger.warning(
            "Część rekordów (%d) nie została przypisana do żadnego zbioru! Sprawdź nazwy sezonów.",
            len(df) - total_split,
        )

    logger.info(
        "Podział ukończony: Train=%d, Validation=%d, Test=%d meczów.",
        len(df_train),
        len(df_val),
        len(df_test),
    )
    return df_train, df_val, df_test


def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Wylicza wieloklasowy Brier Score.

    Args:
        y_true: Rzeczywiste etykiety klas (wartości całkowite 0, 1, 2).
        y_prob: Macierz przewidywanych prawdopodobieństw o wymiarach (N, 3).

    Returns:
        float: Średni Brier Score dla całego zbioru.
    """
    n_samples, n_classes = y_prob.shape
    # Konwersja etykiet do reprezentacji One-Hot Encoding
    y_true_one_hot = np.zeros((n_samples, n_classes))
    for i, label in enumerate(y_true):
        y_true_one_hot[i, label] = 1.0

    squared_diff = (y_prob - y_true_one_hot) ** 2
    return float(np.mean(np.sum(squared_diff, axis=1)))


def evaluate_probability_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> dict[str, float]:
    """Dokonuje pełnej ewaluacji jakości probabilistycznej prognoz meczowych.

    Args:
        y_true: Wektor prawdziwych klas (0, 1, 2).
        y_prob: Macierz prawdopodobieństw (N, 3).
        model_name: Nazwa identyfikacyjna modelu dla raportu.

    Returns:
        dict: Słownik zawierający metryki: Log Loss, Brier Score oraz Accuracy.
    """
    # Zabezpieczenie numeryczne przed log(0)
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    # Renormalizacja do sumy równej 1.0 w wierszu
    y_prob_clipped = y_prob_clipped / y_prob_clipped.sum(axis=1, keepdims=True)

    loss = float(log_loss(y_true, y_prob_clipped))
    brier = calculate_brier_score(y_true, y_prob_clipped)

    # Obliczenie klasycznego Accuracy wyłącznie w celach porównawczych
    y_pred = np.argmax(y_prob_clipped, axis=1)
    acc = float(np.mean(y_pred == y_true))

    logger.info("=== RAPORT EWALUACJI: %s ===", model_name)
    logger.info("-> Log Loss:    %.4f (im niższy, tym lepszy)", loss)
    logger.info("-> Brier Score: %.4f (im niższy, tym lepszy)", brier)
    logger.info("-> Accuracy:    %.2f%% (metryka pomocnicza)", acc * 100)

    return {"log_loss": loss, "brier_score": brier, "accuracy": acc}


def generate_naive_prior_benchmark(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Generuje naiwny model bazowy oparty wyłącznie na historycznym rozkładzie klas z treningu.

    Każdy zaawansowany model musi uzyskać niższy Log Loss niż ten benchmark.
    """
    # Wyznaczenie proporcji H, D, A w zbiorze treningowym
    class_counts = np.bincount(y_train, minlength=3)
    priors = class_counts / len(y_train)

    logger.info(
        "Naiwne prawdopodobieństwa a priori (Train): H=%.3f, D=%.3f, A=%.3f",
        priors[0],
        priors[1],
        priors[2],
    )

    # Rozszerzenie stałego wektora na cały zbiór testowy
    y_prob_naive = np.tile(priors, (len(y_test), 1))
    return evaluate_probability_predictions(
        y_test, y_prob_naive, model_name="Naive Historical Prior"
    )


if __name__ == "__main__":

    input_path = PROCESSED_DATA_DIR / "analytical_mart_wide.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Brak pliku analitycznego: {input_path}")

    # 1. Wczytanie zbioru danych
    df = pd.read_parquet(input_path)

    # 2. Wektorowe zakodowanie targetu (H: 0, D: 1, A: 2)
    df["target"] = df["FTR"].map(TARGET_MAPPING).astype(int)

    # 3. Definicja sezonów podziału chronologicznego (Time-Series Split)
    train_seasons: list[str] = ["2022-2023", "2023-2024"]
    validation_seasons: list[str] = ["2024-2025"]
    test_seasons: list[str] = ["2025-2026"]

    logger.info("Rozdzielenie danych w oparciu o oś czasu...")
    train_data, val_data, test_data = temporal_train_val_test_split(
        df=df,
        train_seasons=train_seasons,
        val_seasons=validation_seasons,
        test_seasons=test_seasons,
    )

    # 4. Wyizolowanie wektorów targetu (NumPy Arrays)
    y_train = train_data["target"].to_numpy()
    y_test = test_data["target"].to_numpy()

    logger.info("Uruchomienie ewaluacji benchmarku naiwnego (Naive Prior Baseline)...")
    naive_metrics = generate_naive_prior_benchmark(y_train=y_train, y_test=y_test)
    print("\nWyniki benchmarku naiwnego:")
    print(naive_metrics)