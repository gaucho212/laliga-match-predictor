from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd()
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
df = pd.read_parquet(PROCESSED_DATA_DIR / "team_matches_long.parquet")
barca_name = [t for t in df["team"].unique() if "barca" in t.lower() or "barcelona" in t.lower()][0]

cols_to_inspect = [
    "match_date", 
    "opponent", 
    "is_home",
    "xg_for", 
    "xg_against", 
    "xg_diff",
    "roll_xg_diff_3", 
    "rest_days"
]

df_barca = (
    df[df["team"] == barca_name]
    .sort_values(by="match_date")
    [cols_to_inspect]
    
)

print(df_barca.to_string())