from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd()
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

df_fd = pd.read_parquet(RAW_DATA_DIR / "football_data_raw.parquet")
df_xg = pd.read_parquet(RAW_DATA_DIR / "understat_xg_raw.parquet")

teams_fd=set(df_fd["HomeTeam"].unique())
teams_xg=set(df_xg["home_team"].unique())

print(f"Liczba unikalnych klubów w football-data: {len(teams_fd)}")
print(f"Liczba unikalnych klubów w Understat: {len(teams_xg)}")

mismatched_teams = sorted(list(teams_fd - teams_xg))
print("\nDrużyny wymagające zmapowania (są w football-data, brak w Understat):")
for team in mismatched_teams:
    print(f" - '{team}'")