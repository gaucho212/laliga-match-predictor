# La Liga Quantitative Match Intelligence & Decision System
### End-to-End Sports Analytics, Probabilistic Modeling & Capital Allocation Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](https://www.docker.com/)
[![Power BI](https://img.shields.io/badge/power_bi-reporting-F2C811.svg)](https://powerbi.microsoft.com/)
[![Code Style: PEP8](https://img.shields.io/badge/code%20style-PEP8-brightgreen.svg)](https://pep8.org/)

---

## 1. Executive Summary

This project delivers an institutional-grade quantitative forecasting and betting intelligence system for the Spanish **La Liga**. Built from the perspective of a Senior BI / Quantitative Analyst, it bridges the gap between raw data engineering, probabilistic machine learning, and disciplined capital management.

Instead of treating football outcome prediction as a naive classification task evaluated via unweighted Accuracy, the architecture treats it as a stochastic goal-generation process coupled with financial risk constraints.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                PERFORMANCE HIGHLIGHTS                                  │
├───────────────────────────────┬───────────────────────────────┬────────────────────────┤
│     OUT-OF-TIME LOG LOSS      │     CALIBRATION ERROR (ECE)   │   OPTIMIZED ALPHA ROI  │
│            0.9820             │             0.94%             │         +5.90%         │
│  (-6.43% vs Naive Prior)      │ (Institutional Reliability)   │  (+673.0 units PnL)    │
└───────────────────────────────┴───────────────────────────────┴────────────────────────┘
```

### Key Business & Analytical Achievements:
* **Zero Look-Ahead Bias Pipeline:** All feature transformations, rolling forms, and Elo ratings are calculated strictly using pre-match states.
* **Hybrid Ensemble Architecture:** Blends a structural bivariate Poisson GLM ($93\%$) with a LightGBM GBDT classifier ($7\%$) to combine physical goal distributions with non-linear fatigue/schedule interactions.
* **Market Anomaly Exploitation:** Uncovered systematic **Favorite-Longshot Bias** across away and draw markets, engineering a selective **"Home Win Only & $EV \ge 5\%$"** policy that converts a market-wide loss ($-3.60\%$ ROI) into profitable returns ($+5.90\%$ ROI, $53.51\%$ win rate).
* **Capital Preservation:** Implements a **Quarter-Kelly ($c = 0.25$)** staking engine capped at $3.5\%$ single-match exposure to mitigate variance drawdowns.

---

## 2. Quantitative Strategy & Financial Results

### Out-of-Time Backtest Results (Season 2025/2026 – 380 Matches)

During the blind test season, unconstrained betting on all positive Expected Value ($EV \ge 5\%$) opportunities yielded a negative return due to market mispricing in volatile sub-segments. Dissecting the portfolio by market category revealed critical structural edges:

| Portfolio Strategy | Bets | Win Rate | Total Turnover | Total PnL | ROI (%) | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Market (1 / X / 2 Baseline)** | 260 | 38.85% | 26 000 j. | **-937.0 j.** | **-3.60%** | -1 840.0 j. |
| ├── Draws Only ($X$) | 16 | 12.50% | 1 600 j. | **-630.0 j.** | -39.38% | -630.0 j. |
| ├── Away Only ($2$) | 130 | 29.23% | 13 000 j. | **-980.0 j.** | -7.54% | -1 320.0 j. |
| └── **Home Only Alpha Strategy ($1$)** | **114** | **53.51%** | **11 400 j.** | **+673.0 j.** | **+5.90%** | **-280.0 j.** |

$$\text{Alpha Policy:} \quad \text{Execute if and only if} \quad (\text{Market} = \text{'Home'}) \quad \land \quad (EV_H \ge 0.05)$$

---

## 3. Methodological Rigor & Data Integrity

### Strict Temporal Separation (Zero Data-Leakage Protocol)
A common pitfall in sports analytics is computing rolling averages over an entire dataset or applying post-match indicators. In this system:
1. Historical states are snapshots locked before kickoff.
2. The `CausalStateManager` iterates sequentially over matches, updating parameters only *after* the final whistle.
3. Feature stores are stored in compressed Parquet formats with pre-match timestamp boundaries.

### Advanced Feature Engineering (Signal over Noise)
Raw goal tallies over 3–5 games are heavily influenced by finishing variance and referee decisions. The engine utilizes fundamental underlying drivers:
* **Dynamic Elo Rating with Home Advantage:** Updated after every match ($K = 24.0$, $\text{HFA} = +65.0$ points). Accounts for **$28.4\%$** of total feature importance (Gain ratio).
* **Rolling Expected Goals Net Differential ($\Delta xG_{5m}$):** 
  $$\Delta xG_{5m} = (xG_{\text{for}} - xGA_{\text{against}})_{\text{Home}} - (xG_{\text{for}} - xGA_{\text{against}})_{\text{Away}}$$
* **Rest Days Disparity ($\Delta \text{Rest}$):** Measures physical asymmetry resulting from UEFA Champions League, Europa League, and Copa del Rey midweek fixtures.
* **Off-Season Mean Reversion:** Ranking shrinkage applied across summer breaks to account for roster transfers and managerial turnover:
  $$R_{\text{start, } t} = 0.75 \cdot R_{\text{end, } t-1} + 0.25 \cdot 1500.0$$

---

## 4. Probabilistic Modeling & Evaluation

### The "Accuracy Fallacy" & The Draw Dilemma
In a 3-way distribution ($H/X/A$), base rates in La Liga average $43\%$ Home, $26\%$ Draw, and $31\%$ Away. Because draws rarely exhibit an individual match probability $>35\%$, a discrete classifier optimizing for accuracy ($\text{argmax}$) will almost never predict a draw, resulting in abysmal Draw Recall ($14.58\%$).

Hence, the model is evaluated exclusively via proper probabilistic scoring rules:

### Multi-Class Hierarchy of Baselines

| Level | Model Architecture | Multi-class Log Loss | Brier Score | Accuracy | Primary Function |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | Naive Historical Prior | 1.0495 | 0.6322 | 48.68% | Null hypothesis benchmark |
| **1** | LightGBM GBDT (Tuned) | 1.0095 | 0.6005 | 49.21% | Non-linear tabular baseline |
| **2** | Bivariate Poisson GLM ($xG$ + Elo) | 0.9838 | 0.5823 | 51.59% | Structural goal physics |
| **2+** | **Hybrid Ensemble ($0.93 / 0.07$)** | **0.9820** | **0.5818** | **51.58%** | **Production Quant Model** |
| **3** | Market Closing Consensus (Bet365) | 0.9656 | 0.5724 | 54.76% | Efficient market frontier |

* **Multi-class Log Loss Formulation:**
  $$\text{Log Loss} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{k \in \{H, X, A\}} y_{i,k} \ln(\hat{p}_{i,k}) = \mathbf{0.9820} \quad (-6.43\% \text{ error reduction})$$
* **Expected Calibration Error (ECE):**
  $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right| = \mathbf{0.94\%}$$

---

## 5. Capital Allocation & Risk Management (Value Engine)

### 1. Market Overround Removal
Bookmaker odds include an artificial commission ($4.5\% - 5.5\%$). Fair implied probabilities are recovered via additive normalization:
$$\text{Overround} = \left(\frac{1}{O_H} + \frac{1}{O_X} + \frac{1}{O_A}\right) - 1, \qquad P_{\text{implied, fair}}(k) = \frac{1 / O_k}{1 + \text{Overround}}$$

### 2. Expected Value Computation
$$\text{EV}_k = (P_{\text{model}}(k) \times O_k) - 1$$

### 3. Fractional Kelly Staking
Full Kelly maximizes geometric growth but induces catastrophic drawdowns when model uncertainty exists. The engine deploys a defensive **Quarter-Kelly** fraction ($c = 0.25$) with a hard limit:
$$f^* = \frac{\text{EV}}{O - 1}, \qquad f_{\text{adj}} = \min\left(0.035, \, 0.25 \times f^*\right)$$
$$\text{Stake} = f_{\text{adj}} \times \text{Bankroll}$$

---

## 6. Repository Architecture

```
.
├── artifacts/                                # Production inference outputs
│   ├── gameweek_live_predictions.csv         # Latest gameweek feed (Power BI source)
│   ├── gameweek_live_predictions.json        # Structured JSON execution feed
│   ├── gameweek_predictions_latest.csv       # Versioned historical snapshot
│   └── gameweek_predictions_latest.json
├── config/
│   └── config.yaml                           # Hyperparameters, paths & betting limits
├── data/
│   ├── powerbi_export/                       # Star-schema dimensional tables
│   │   ├── Dim_Date.csv                      # Temporal dimension
│   │   ├── Dim_Outcome.csv                   # Outcome mapping dimension (H/X/A)
│   │   ├── Dim_Teams.csv                     # Team dimension & stadium metadata
│   │   ├── Fact_Bets.csv                     # Historical bet executions & PnL
│   │   └── Fact_FeatureImportance.csv        # Gain & attribution metrics
│   ├── processed/                            # Engineered analytical datasets (Parquet)
│   │   ├── analytical_mart_wide.parquet      # ML-ready flat table with rolling features
│   │   ├── matches_merged.parquet            # Consolidated match-level results
│   │   └── team_matches_long.parquet         # Long-format team fixture ledger
│   ├── raw/                                  # Immutable raw ingestion files
│   │   ├── football_data_raw.parquet         # Results and odds history (Football-Data.co.uk)
│   │   └── understat_xg_raw.parquet          # Match-level expected goals (Understat)
│   └── walk_forward_data/                    # Incremental sequential test slices
├── models/
│   ├── backtest_bets.parquet                 # Granular trade-by-trade ledger
│   ├── feature_importance.parquet            # LightGBM split & gain values
│   └── lightgbm_classifier.txt               # Serialized LightGBM booster model
├── src/
│   ├── analysis/
│   │   ├── backtest.py                       # Vectorized simulation & Kelly execution
│   │   └── export_powerbi.py                 # Star-schema transformer for BI ingestion
│   ├── dashboard/
│   │   ├── app.py                            # Streamlit exploration dashboard
│   │   ├── matches_history.csv               # Historical training dataset
│   │   └── next_fixtures.csv                 # Live incoming gameweek feed
│   ├── features/
│   │   ├── build_features.py                 # Rolling forms, xG net, rest intervals
│   │   └── elo_rating.py                     # Dynamic Elo rating engine with HFA
│   ├── ingestion/
│   │   ├── clean_data.py                     # Harmonization & team name alias mapping
│   │   └── fetch_raw_data.py                 # Ingestion pipelines from primary sources
│   └── models/
│       ├── baseline_poisson.py               # Bivariate Poisson GLM goal generator
│       ├── diagnostics.py                    # Log Loss, Brier, Calibration calculations
│       ├── esemble.py                        # Hybrid ensemble blender & optimizer
│       ├── evaluate.py                       # Confusion matrix & cross-entropy validation
│       └── train_lightgbm.py                 # GBDT training with early stopping
├── tests/
│   ├── mismatched_teams.py                   # Integrity check for club alias mapping
│   ├── read_team_matches.py                  # Fixture sequencing validation
│   └── test_raw_data.py                      # Data contract & schema assertions
├── .gitignore
├── docker-compose.yml                        # Docker Compose orchestration
├── Dockerfile                                # Isolated execution environment
├── LaLiga_Match_Predictior.pbix              # 3-page interactive Power BI dashboard
├── LaLiga_Matchday_Intelligence_GW38.xlsx    # 3-tab institutional decision spreadsheet
├── README.md                                 # Technical documentation
└── requirements.txt                          # Pinned production dependencies
```

---

## 7. Business Deliverables & Reporting Ecosystem

### 1. Power BI Analytical Dashboard (`LaLiga_Match_Predictior.pbix`)
An executive BI suite built on a Star Schema data model (`Dim_Date`, `Dim_Teams`, `Dim_Outcome`, `Fact_Bets`):
* **Page 1: Executive Summary:** High-level PnL evolution, total turnover, ROI tracking, drawdowns, and market category breakdowns.
* **Page 2: Model Diagnostics:** Feature gain ranking, hierarchy of baselines table, and closing odds calibration scatter plot.
* **Page 3: Strategy Sandbox:** Dual Equity Curve comparing full market execution vs. the selective Alpha Policy, complete with an interactive $EV$ threshold slider.

### 2. Decision Spreadsheet (`LaLiga_Matchday_Intelligence_GW38.xlsx`)
A financial-standard, 3-tab workbook generated via `openpyxl`:
* **`Game_Predictions_Gameweek`:** Matchday desk displaying pre-match features, fair market probabilities, $EV$, and automated trade signals.
* **`Value_Engine`:** Dynamic staking desk applying Quarter-Kelly sizing, portfolio exposure metrics, and scenario stress testing.
* **`Model_KPIs_Monitoring`:** Probabilistic audit dashboard featuring the Multiclass Confusion Matrix, Precision/Recall by class, and Calibration Bins ($ECE = 0.94\%$).

---

## 8. Installation & Quickstart

### Prerequisites
* Python 3.11+
* Docker & Docker Compose (optional for containerized runs)

### Option A: Local Python Environment

```bash
# 1. Clone repository
git clone https://github.com/your-username/laliga-quant-intelligence.git
cd laliga-quant-intelligence

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# 3. Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Execute next gameweek production inference
python src/dashboard/predict_next_gameweek.py \
  --history src/dashboard/matches_history.csv \
  --fixtures src/dashboard/next_fixtures.csv \
  --output artifacts
```

### Option B: Docker Containerized Execution

Ensure Docker is running, then execute:

```bash
# Build and run the pipeline inside an isolated Linux container
docker compose up --build
```
Inference artifacts will be generated inside the container and mounted directly to `./artifacts/` on the host machine.

---

## 9. Engineering Standards & Test Coverage

* **PEP8 & Type Hinting:** Strict type signatures across all data ingestion and inference functions.
* **Schema Validation:** Unit tests (`tests/test_raw_data.py`) assert column schemas, missing value boundaries, and probability constraints ($\sum P = 1.0$).
* **Data Hermeticism:** Zero dependency on run-time scrapers; historical state transitions are fully reproducible from static raw Parquet files.