"""Financial Backtesting & Market Efficiency Benchmark Module.

Evaluates market closing odds calibration, de-margins implied probabilities,
and simulates an out-of-time Value Betting strategy (EV >= 5%) for the 2025-2026 season.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.baseline_poisson import PoissonGoalModel
from src.models.evaluate import TARGET_MAPPING, evaluate_probability_predictions, temporal_train_val_test_split



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BacktestEngine")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

def extract_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Izoluje macierz cech predykcyjnych (X) od metadanych, kursów i wektora celu (y).

    Gwarantuje zgodność wymiarów (dokładnie 28 cech) nawet po złączeniu z kursami bukmacherskimi.
    """
    excluded_cols = [
        "match_id",
        "match_date",
        "season",
        "home_team",
        "away_team",
        "FTR",
        "target",
        # Kolumny rynkowe (kursy) wykluczane z macierzy X
        "B365H",
        "B365D",
        "B365A",
    ]

    feature_cols = [c for c in df.columns if c not in excluded_cols]
    X = df[feature_cols].copy()
    y = df["target"].to_numpy().astype(int)

    return X, y, feature_cols


def calculate_demargined_market_probabilities(
    df_raw: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Odmarżowuje kursy bukmacherskie metodą normalizacji proporcjonalnej.

    Args:
        df_raw: Ramka danych z kursami B365H, B365D, B365A.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Czyste prawdopodobieństwa rynkowe (N, 3) oraz macierz surowych kursów (N, 3).
    """
    odds = df_raw[["B365H", "B365D", "B365A"]].to_numpy().astype(float)
    
    # Odwrotności kursów (prawdopodobieństwa z marżą)
    raw_probs = 1.0 / odds
    
    # Suma prawdopodobieństw w wierszu (1.0 + overround)
    overround = raw_probs.sum(axis=1, keepdims=True)
    
    # Renormalizacja proporcjonalna
    demargined_probs = raw_probs / overround
    return demargined_probs, odds


def run_value_betting_simulation(
    df_test_matches: pd.DataFrame,
    y_true: np.ndarray,
    model_probs: np.ndarray,
    odds_matrix: np.ndarray,
    min_ev: float = 0.05,
    stake_size: float = 100.0,
) -> dict:
    """Symuluje chronologiczną strategię Value Bettingu (Flat Stake) na zbiorze testowym.

    Args:
        df_test_matches: Metadane meczów testowych.
        y_true: Prawdziwe wyniki meczów (0: H, 1: D, 2: A).
        model_probs: Prawdopodobieństwa wygenerowane przez model hybrydowy (N, 3).
        odds_matrix: Kursy bukmacherskie [B365H, B365D, B365A] (N, 3).
        min_ev: Minimalny próg wartości oczekiwanej kwalifikujący zakład (np. 0.05 = +5%).
        stake_size: Stała wielkość stawki w jednostkach walutowych.

    Returns:
        dict: Raport podsumowujący metryki finansowe strategii.
    """
    logger.info("Uruchamianie symulacji Value Betting z minimalnym EV = +%.1f%%...", min_ev * 100)

    n_matches = len(y_true)
    bets = []
    capital = 0.0
    equity_curve = [capital]

    # Mapowanie indeksów klas
    outcome_labels = ["H", "D", "A"]

    for i in range(n_matches):
        match_date = df_test_matches.iloc[i]["match_date"]
        h_team = df_test_matches.iloc[i]["home_team"]
        a_team = df_test_matches.iloc[i]["away_team"]
        actual_result = y_true[i]

        for c in range(3):
            p = model_probs[i, c]
            odd = odds_matrix[i, c]
            ev = (p * odd) - 1.0

            # Kwalifikacja zakładu
            if ev >= min_ev:
                won = (actual_result == c)
                pnl = (stake_size * (odd - 1.0)) if won else -stake_size
                capital += pnl
                equity_curve.append(capital)

                bets.append(
                    {
                        "match_date": match_date,
                        "match": f"{h_team} vs {a_team}",
                        "outcome": outcome_labels[c],
                        "model_prob": p,
                        "odds": odd,
                        "ev": ev,
                        "won": won,
                        "pnl": pnl,
                        "capital_after": capital,
                    }
                )

    df_bets = pd.DataFrame(bets)

    if df_bets.empty:
        logger.warning("Brak zakładów spełniających kryterium EV >= %.2f!", min_ev)
        return {"total_bets": 0}

    # Wyliczanie metryk finansowych
    total_bets = len(df_bets)
    total_staked = total_bets * stake_size
    total_profit = df_bets["pnl"].sum()
    roi = (total_profit / total_staked) * 100.0
    win_rate = (df_bets["won"].mean()) * 100.0

    # Kalkulacja Maximum Drawdown (MDD)
    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdowns = peak - equity
    max_drawdown = np.max(drawdowns)

    logger.info("=== RAPORT FINANSOWY STRATEGII VALUE BETTING ===")
    logger.info("-> Łączna liczba zakładów: %d (spośród %d rozegranych meczów)", total_bets, n_matches)
    logger.info("-> Skuteczność (Win Rate): %.2f%%", win_rate)
    logger.info("-> Suma obrotu (Turnover): %.2f j.", total_staked)
    logger.info("-> Wynik finansowy (Net PnL): %+.2f j.", total_profit)
    logger.info("-> Zwrot z inwestycji (ROI / Yield): %+.2f%%", roi)
    logger.info("-> Maksymalne obsunięcie kapitału (Max Drawdown): %.2f j.", max_drawdown)

    return {
        "df_bets": df_bets,
        "total_bets": total_bets,
        "roi": roi,
        "pnl": total_profit,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
    }


def execute_full_market_audit() -> None:
    """Orkiestruje audyt rynkowy oraz symulację Value Betting na zbiorze 2025-2026."""
    logger.info("=== START: AUDYT RYNKOWY I FINANCIAL BACKTEST ===")

    # 1. Wczytanie danych
    mart_path = PROCESSED_DATA_DIR / "analytical_mart_wide.parquet"
    raw_path = PROCESSED_DATA_DIR / "matches_merged.parquet"

    df_mart = pd.read_parquet(mart_path)
    df_raw = pd.read_parquet(raw_path)
    df_mart["target"] = df_mart["FTR"].map(TARGET_MAPPING).astype(int)

    # 2. Spójny podział chronologiczny
    train_seasons = ["2022-2023", "2023-2024"]
    val_seasons = ["2024-2025"]
    test_seasons = ["2025-2026"]

    df_train, df_val, df_test = temporal_train_val_test_split(
        df_mart, train_seasons, val_seasons, test_seasons
    )

    # 3. Pobranie kursów rynkowych dla zbioru testowego przez złączenie
    join_keys = ["season", "home_team", "away_team"]
    test_merged = df_test.merge(
        df_raw[join_keys + ["B365H", "B365D", "B365A"]],
        on=join_keys,
        how="inner",
    )

    y_test = test_merged["target"].to_numpy()
    p_market_test, odds_test = calculate_demargined_market_probabilities(test_merged)

    # 4. EWALUACJA POZIOMU 3: BENCHMARK RYNKOWY (Closing Odds)
    logger.info("--- EWALUACJA BENCHMARKU RYNKOWEGO (BET365 CLOSING) ---")
    evaluate_probability_predictions(
        y_true=y_test,
        y_prob=p_market_test,
        model_name="Market Closing Odds (Demargined Bet365)",
    )

    # 5. Generowanie predykcji modelu hybrydowego
    poisson = PoissonGoalModel(alpha=1.0).fit(df_train, df_raw)
    p_poisson_test = poisson.predict_proba(test_merged)

    X_test, _, _ = extract_features_and_target(test_merged)
    lgb_booster = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm_classifier.txt"))
    p_lgb_test = lgb_booster.predict(X_test)

    # Zastosowanie optymalnej wagi ze Sprintu 3 (w = 0.93)
    p_ensemble_test = 0.93 * p_poisson_test + 0.07 * p_lgb_test
    p_ensemble_test = p_ensemble_test / p_ensemble_test.sum(axis=1, keepdims=True)

    # 6. Symulacja strategii finansowej
    results = run_value_betting_simulation(
        df_test_matches=test_merged,
        y_true=y_test,
        model_probs=p_ensemble_test,
        odds_matrix=odds_test,
        min_ev=0.05,  # +5% przewagi matematycznej
        stake_size=100.0,
    )

    if results["total_bets"] > 0:
        results_path = MODELS_DIR / "backtest_bets.parquet"
        results["df_bets"].to_parquet(results_path, index=False)
        logger.info("Zapisano historię zawartych zakładów do: %s", results_path)


if __name__ == "__main__":
    execute_full_market_audit()