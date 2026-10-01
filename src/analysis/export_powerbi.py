"""Power BI Data Export & Semantic Layer Preparation.

Transforms backtest logs, model features, and feature importance tables
into relational CSV files ready for a Star Schema in Power BI Desktop.
"""

import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PowerBIExporter")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
POWERBI_EXPORT_DIR = BASE_DIR / "data" / "powerbi_export"
POWERBI_EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def export_powerbi_tables() -> None:
    """Tworzy zbiory wymiarów i faktów pod schemat gwiazdy w Power BI."""
    logger.info("Wczytywanie artefaktów modeli i martu analitycznego...")

    bets_path = MODELS_DIR / "backtest_bets.parquet"
    mart_path = PROCESSED_DATA_DIR / "analytical_mart_wide.parquet"
    feat_imp_path = MODELS_DIR / "feature_importance.parquet"

    df_bets = pd.read_parquet(bets_path)
    df_mart = pd.read_parquet(mart_path)
    df_feat_imp = pd.read_parquet(feat_imp_path)

    # 1. Tabela Faktów: Fact_Bets (Transakcje zawieranych zakładów)
    df_bets["bet_id"] = range(1, len(df_bets) + 1)
    df_bets["stake"] = 100.0
    df_bets["match_date"] = pd.to_datetime(df_bets["match_date"])
    
    # Rozbicie meczu na kluby pod relacje
    df_bets["home_team"] = df_bets["match"].apply(lambda x: x.split(" vs ")[0])
    df_bets["away_team"] = df_bets["match"].apply(lambda x: x.split(" vs ")[1])
    
    # 2. Tabela Wymiaru: Dim_Outcome (Słownik rynków)
    dim_outcome = pd.DataFrame(
        {
            "outcome_code": ["H", "D", "A"],
            "outcome_name": ["Gospodarz (Home Win)", "Remis (Draw)", "Gość (Away Win)"],
            "market_category": ["Strona domowa", "Podział punktów", "Strona wyjazdowa"],
        }
    )

    # 3. Tabela Wymiaru: Dim_Teams (Unikalna lista klubów)
    all_teams = sorted(list(set(df_mart["home_team"].unique()) | set(df_mart["away_team"].unique())))
    dim_teams = pd.DataFrame({"team_id": range(1, len(all_teams) + 1), "team_name": all_teams})

    # 4. Tabela Wymiaru Czasu: Dim_Date
    min_date = df_bets["match_date"].min()
    max_date = df_bets["match_date"].max()
    date_range = pd.date_range(start=min_date, end=max_date, freq="D")
    
    dim_date = pd.DataFrame({"Date": date_range})
    dim_date["Year"] = dim_date["Date"].dt.year
    dim_date["Month"] = dim_date["Date"].dt.month
    dim_date["MonthName"] = dim_date["Date"].dt.strftime("%B")
    dim_date["WeekOfYear"] = dim_date["Date"].dt.isocalendar().week
    dim_date["DayOfWeek"] = dim_date["Date"].dt.strftime("%A")

    # Zapis do plików CSV (uniwersalny import w Power BI bez konieczności instalacji bibliotek Parquet)
    df_bets.to_csv(POWERBI_EXPORT_DIR / "Fact_Bets.csv", index=False)
    dim_outcome.to_csv(POWERBI_EXPORT_DIR / "Dim_Outcome.csv", index=False)
    dim_teams.to_csv(POWERBI_EXPORT_DIR / "Dim_Teams.csv", index=False)
    dim_date.to_csv(POWERBI_EXPORT_DIR / "Dim_Date.csv", index=False)
    df_feat_imp.to_csv(POWERBI_EXPORT_DIR / "Fact_FeatureImportance.csv", index=False)

    logger.info("Pomyślnie wyeksportowano 5 tabel do katalogu: %s", POWERBI_EXPORT_DIR)


if __name__ == "__main__":
    export_powerbi_tables()