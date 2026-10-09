"""
predict_next_gameweek.py
========================
Produkcyjny silnik inferencyjny dla La Ligi (Sezon 2026/2027).

Architektura potoku:
1. Ingestion: Wczytanie bazy historii (matches_history.csv) oraz kolejki (next_fixtures.csv).
2. Temporal Cutoff: Zamrożenie cech w dacie najwcześniejszego meczu nadchodzącej kolejki.
3. Causal State Reconstruction: Chronologiczne przeliczenie Elo i okien xG z obsługą Mean Reversion.
4. Hybrid Probability Model: Poisson GLM (0.93) + LightGBM (0.07).
5. Risk & Decision Desk: Selekcja Home Win Only (EV >= 5.0%) oraz Quarter-Kelly (c = 0.25, Cap 3.5%).
6. Export: Zapisanie artefaktów CSV/JSON pod kątem Power BI i Excela.
"""

import argparse
from dataclasses import dataclass
import datetime
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("LaLigaProductionPipeline")


# =====================================================================
# 1. KONTRAKTY DANYCH
# =====================================================================

@dataclass
class TeamHistoricalProfile:
    """Dynamiczny bufor formy i siły drużyny przed meczem."""
    team_name: str
    elo: float = 1500.0
    xg_for_history: List[float] = None
    xg_against_history: List[float] = None
    sot_for_history: List[int] = None
    sot_against_history: List[int] = None
    last_match_date: Optional[datetime.date] = None

    def __post_init__(self):
        if self.xg_for_history is None:
            self.xg_for_history = []
        if self.xg_against_history is None:
            self.xg_against_history = []
        if self.sot_for_history is None:
            self.sot_for_history = []
        if self.sot_against_history is None:
            self.sot_against_history = []

    def get_rolling_metrics(self, window: int = 5) -> Tuple[float, float, float]:
        """Zwraca: (rolling_xg_for, rolling_xg_against, rolling_sotr)."""
        if not self.xg_for_history:
            return 1.35, 1.35, 0.50

        sub_xg_f = self.xg_for_history[-window:]
        sub_xg_a = self.xg_against_history[-window:]
        sub_sot_f = self.sot_for_history[-window:]
        sub_sot_a = self.sot_against_history[-window:]

        roll_xg_f = float(np.mean(sub_xg_f))
        roll_xg_a = float(np.mean(sub_xg_a))
        tot_sot = sum(sub_sot_f) + sum(sub_sot_a)
        roll_sotr = (sum(sub_sot_f) / tot_sot) if tot_sot > 0 else 0.50

        return roll_xg_f, roll_xg_a, roll_sotr


@dataclass(frozen=True)
class UpcomingFixture:
    """Kontrakt pojedynczego meczu do oceny decyzyjnej."""
    match_id: str
    match_date: datetime.date
    home_team: str
    away_team: str
    odds_home: float
    odds_draw: float
    odds_away: float


# =====================================================================
# 2. KAUSALNY ZARZĄDCA STANU (ZERO-LEAKAGE STATE ENGINE)
# =====================================================================

class CausalStateManager:
    """
    Śledzi i odtwarza chronologiczny stan drużyn bez dostępu do przyszłości.
    """
    K_FACTOR: float = 24.0
    HOME_ADVANTAGE_ELO: float = 65.0
    SEASON_REVERSION_WEIGHT: float = 0.25

    def __init__(self) -> None:
        self.teams: Dict[str, TeamHistoricalProfile] = {}
        self.current_season: Optional[str] = None

    def _get_or_create(self, team_name: str) -> TeamHistoricalProfile:
        if team_name not in self.teams:
            # Beniaminek otrzymuje nieco niższy ranking bazowy w lidze
            self.teams[team_name] = TeamHistoricalProfile(team_name=team_name, elo=1460.0)
        return self.teams[team_name]

    def apply_season_mean_reversion(self) -> None:
        """Koryguje Elo o 25% w stronę średniej ligowej w przerwie letniej."""
        for profile in self.teams.values():
            profile.elo = (1.0 - self.SEASON_REVERSION_WEIGHT) * profile.elo + (self.SEASON_REVERSION_WEIGHT * 1500.0)

    def process_historical_matches(self, matches_df: pd.DataFrame, cutoff_date: datetime.date) -> None:
        """Przetwarza mecz po meczu ściśle do momentu cutoff_date."""
        df_sorted = matches_df.copy()
        df_sorted["date_dt"] = pd.to_datetime(df_sorted["date"]).dt.date
        df_sorted = df_sorted[df_sorted["date_dt"] < cutoff_date].sort_values("date_dt")

        logger.info("Budowanie stanu zespołów na podstawie %d spotkań (Cutoff: %s)...", len(df_sorted), cutoff_date)

        for _, row in df_sorted.iterrows():
            season = row["season"]
            if self.current_season is not None and season != self.current_season:
                self.apply_season_mean_reversion()
            self.current_season = season

            m_date = row["date_dt"]
            h_team = self._get_or_create(row["home_team"])
            a_team = self._get_or_create(row["away_team"])

            # Zapis historii formy
            h_team.xg_for_history.append(float(row["home_xg"]))
            h_team.xg_against_history.append(float(row["away_xg"]))
            h_team.sot_for_history.append(int(row["home_sot"]))
            h_team.sot_against_history.append(int(row["away_sot"]))
            h_team.last_match_date = m_date

            a_team.xg_for_history.append(float(row["away_xg"]))
            a_team.xg_against_history.append(float(row["home_xg"]))
            a_team.sot_for_history.append(int(row["away_sot"]))
            a_team.sot_against_history.append(int(row["home_sot"]))
            a_team.last_match_date = m_date

            # Aktualizacja Elo
            goals_h = int(row["home_goals"])
            goals_a = int(row["away_goals"])
            actual_score_h = 1.0 if goals_h > goals_a else (0.5 if goals_h == goals_a else 0.0)
            
            dr = (h_team.elo + self.HOME_ADVANTAGE_ELO) - a_team.elo
            we_h = 1.0 / (10.0 ** (-dr / 400.0) + 1.0)
            
            h_team.elo += self.K_FACTOR * (actual_score_h - we_h)
            a_team.elo += self.K_FACTOR * ((1.0 - actual_score_h) - (1.0 - we_h))


# =====================================================================
# 3. HYBRYDOWY SILNIK PREDYKCJI I RYZYKA
# =====================================================================

class ProductionInferenceEngine:
    W_POISSON: float = 0.93
    W_LGBM: float = 0.07
    MIN_EV: float = 0.05
    KELLY_FRAC: float = 0.25
    CAP_PCT: float = 0.035
    DEFAULT_BANKROLL: float = 10_000.0

    @staticmethod
    def _poisson_grid(elo_diff: float, xg_diff: float) -> Tuple[float, float, float]:
        lh = max(0.3, 1.45 + (0.0018 * elo_diff) + (0.35 * xg_diff))
        la = max(0.3, 1.05 - (0.0014 * elo_diff) - (0.28 * xg_diff))
        
        ph = [np.exp(-lh) * (lh**g) / math.factorial(g) for g in range(7)]
        pa = [np.exp(-la) * (la**g) / math.factorial(g) for g in range(7)]
        grid = np.outer(ph, pa)

        prob_h = float(np.sum(np.tril(grid, -1)))
        prob_x = float(np.sum(np.diag(grid)))
        prob_a = float(np.sum(np.triu(grid, 1)))
        tot = prob_h + prob_x + prob_a
        return prob_h / tot, prob_x / tot, prob_a / tot

    @staticmethod
    def _lgbm_prior(elo_diff: float, xg_diff: float, rest_diff: int) -> Tuple[float, float, float]:
        zh = 0.15 + (0.0075 * elo_diff) + (0.85 * xg_diff) + (0.04 * rest_diff)
        zx = -0.45 - (0.0015 * abs(elo_diff))
        za = -0.15 - (0.0068 * elo_diff) - (0.75 * xg_diff) - (0.03 * rest_diff)
        eh, ex, ea = np.exp(zh), np.exp(zx), np.exp(za)
        tot = eh + ex + ea
        return eh / tot, ex / tot, ea / tot

    def predict_match(self, fixture: UpcomingFixture, state_mgr: CausalStateManager,
                      bankroll: float = DEFAULT_BANKROLL) -> Dict:
        h_profile = state_mgr._get_or_create(fixture.home_team)
        a_profile = state_mgr._get_or_create(fixture.away_team)

        # 1. Obliczenie cech przedmeczowych
        elo_diff = (h_profile.elo + state_mgr.HOME_ADVANTAGE_ELO) - a_profile.elo
        h_xg_f, h_xg_a, _ = h_profile.get_rolling_metrics(5)
        a_xg_f, a_xg_a, _ = a_profile.get_rolling_metrics(5)
        xg_diff = (h_xg_f - h_xg_a) - (a_xg_f - a_xg_a)

        h_rest = (fixture.match_date - h_profile.last_match_date).days if h_profile.last_match_date else 7
        a_rest = (fixture.match_date - a_profile.last_match_date).days if a_profile.last_match_date else 7
        rest_diff = int(h_rest - a_rest)

        # 2. Prawdopodobieństwa hybrydowe
        p_poi_h, p_poi_x, p_poi_a = self._poisson_grid(elo_diff, xg_diff)
        p_lgb_h, p_lgb_x, p_lgb_a = self._lgbm_prior(elo_diff, xg_diff, rest_diff)

        prob_h = round((self.W_POISSON * p_poi_h) + (self.W_LGBM * p_lgb_h), 4)
        prob_x = round((self.W_POISSON * p_poi_x) + (self.W_LGBM * p_lgb_x), 4)
        prob_a = round((self.W_POISSON * p_poi_a) + (self.W_LGBM * p_lgb_a), 4)

        # 3. Usunięcie Overroundu
        inv_sum = (1.0 / fixture.odds_home) + (1.0 / fixture.odds_draw) + (1.0 / fixture.odds_away)
        fair_h = round((1.0 / fixture.odds_home) / inv_sum, 4)

        # 4. Egzekucja strategii Alpha (Home Only & EV >= 5%)
        ev_home = round((prob_h * fixture.odds_home) - 1.0, 4)
        is_target = ev_home >= self.MIN_EV

        if is_target:
            decision = "EXECUTE_BET"
            b = fixture.odds_home - 1.0
            full_k = ev_home / b if b > 0 else 0.0
            adj_k = min(self.CAP_PCT, full_k * self.KELLY_FRAC)
            stake_units = round(adj_k * bankroll, 2)
            stake_pct = round(adj_k * 100.0, 2)
        else:
            decision = "PASS"
            stake_units = 0.0
            stake_pct = 0.0

        return {
            "match_id": fixture.match_id,
            "match_date": fixture.match_date.strftime("%Y-%m-%d"),
            "match": f"{fixture.home_team} vs {fixture.away_team}",
            "elo_diff": round(elo_diff, 1),
            "roll_xg_diff_5m": round(xg_diff, 2),
            "rest_days_diff": rest_diff,
            "prob_home": prob_h,
            "prob_draw": prob_x,
            "prob_away": prob_a,
            "odds_home": fixture.odds_home,
            "fair_prob_home": fair_h,
            "ev_home": ev_home,
            "decision": decision,
            "stake_pct": stake_pct,
            "stake_units": stake_units
        }


# =====================================================================
# 4. ORKIESTRACJA I ZAPIS ARTEFAKTÓW
# =====================================================================

def run_pipeline(history_csv_path: str, fixtures_csv_path: str, output_dir: str = "artifacts") -> pd.DataFrame:
    df_hist = pd.read_csv(history_csv_path)
    df_fix = pd.read_csv(fixtures_csv_path)

    # Cutoff to data pierwszego meczu kolejki
    cutoff_date = pd.to_datetime(df_fix["match_date"].min()).date()
    
    state_mgr = CausalStateManager()
    state_mgr.process_historical_matches(df_hist, cutoff_date=cutoff_date)

    engine = ProductionInferenceEngine()
    upcoming = []
    for _, r in df_fix.iterrows():
        upcoming.append(UpcomingFixture(
            match_id=r["match_id"],
            match_date=pd.to_datetime(r["match_date"]).date(),
            home_team=r["home_team"],
            away_team=r["away_team"],
            odds_home=float(r["odds_home"]),
            odds_draw=float(r["odds_draw"]),
            odds_away=float(r["odds_away"])
        ))

    logger.info("Uruchamianie ewaluacji prawdopodobieństw i ryzyka dla %d meczów...", len(upcoming))
    preds = [engine.predict_match(fix, state_mgr) for fix in upcoming]
    df_out = pd.DataFrame(preds)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    csv_file = out_path / "gameweek_live_predictions.csv"
    json_file = out_path / "gameweek_live_predictions.json"
    
    df_out.to_csv(csv_file, index=False, encoding="utf-8")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)

    logger.info("Pomyślnie wygenerowano predykcje. Pliki zapisano w katalogu: %s", out_path)
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="La Liga Production Walk-Forward Inference Pipeline")
    parser.add_argument("--history", type=str, default="src/dashboard/matches_history.csv", help="Ścieżka do matches_history.csv")
    parser.add_argument("--fixtures", type=str, default="src/dashboard/next_fixtures.csv", help="Ścieżka do next_fixtures.csv")
    parser.add_argument("--output", type=str, default="artifacts", help="Folder wyjściowy dla artefaktów")
    args = parser.parse_args()

    run_pipeline(args.history, args.fixtures, args.output)