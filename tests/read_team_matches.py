from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd()
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
df = pd.read_parquet(PROCESSED_DATA_DIR / "team_matches_long.parquet")
print(df[:5])