from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd()
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
df = pd.read_parquet(PROCESSED_DATA_DIR / "analytical_mart_wide.parquet")
barca_name = [t for t in df["home_team"].unique() if "barca" in t.lower() or "barcelona" in t.lower()][0]

cols_to_inspect = [
    "match_date", 
    "home_team", 
    "away_team",
    "FTR", 
    "elo_diff_pre", 
    "diff_roll_sotr_5",
    "diff_rest_days"
]

barca_home = df[df["home_team"] == barca_name]
barca_away = df[df["away_team"] == barca_name]

df_barca = pd.concat([barca_home,barca_away],ignore_index=True)
df_barca.reset_index(drop=True, inplace=True)

    
df_barca.sort_values(by=["match_date"], ascending=[True], inplace=True)
print(df_barca[-5:][cols_to_inspect])
    


