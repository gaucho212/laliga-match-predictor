# La Liga Match Prediction Engine & BI Analytics Mart

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/BI%20Dashboard-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Machine Learning](https://img.shields.io/badge/ML-LightGBM%20%7C%20XGBoost-blue?style=flat)](https://lightgbm.readthedocs.io/)
[![Methodology](https://img.shields.io/badge/Methodology-Strict%20Zero--Leakage-success?style=flat)](#-metodologia-i-eliminacja-data-leakage)

Modularny projekt predykcyjno-analityczny prognozujący rozkład prawdopodobieństw wyników spotkań hiszpańskiej La Ligi (1 / X / 2). 

System łączy zautomatyzowany potok danych (ETL), silnik zaawansowanych metryk formy (rolling xG, dynamiczne Elo, rest differential), statystyczny baseline Poissona (Dixon-Coles) oraz gradientowe drzewa decyzyjne (LightGBM/XGBoost). Całość jest weryfikowana metrykami probabilistycznymi (Log Loss, Brier Score) i wizualizowana w dedykowanym dashboardzie Streamlit.

---

## Kontekst biznesowy i cel analityczny

Tradycyjne modele sportowe często zawodzą z dwóch powodów:
1. **Opieranie predykcji wyłącznie na bramkach:** Bramki w piłce nożnej są zdarzeniami rzadkimi (low-scoring event) o wysokiej wariancji losowej.
2. **Wyciek danych (Data Leakage / Look-Ahead Bias):** Wyliczanie wskaźników formy z uwzględnieniem danych z meczu bieżącego lub stosowanie losowego podziału `train_test_split` zamiast walidacji chronologicznej (Time-Series Split).

**Cel projektu:**
Zbudowanie powtarzalnego procesu analitycznego (Production-Ready Pipeline), który:
* Szacuje rzeczywistą dyspozycję drużyn na bazie wskaźników jakościowych (Expected Goals – xG/xGA, strzały celne, ranking Elo).
* Całkowicie eliminuje wyciek danych poprzez transformację tabeli faktów do poziomu *Team-Match* z przesunięciem czasowym (`.shift(1)`).
* Porównuje wyestymowane prawdopodobieństwa z rynkowym benchmarkiem (kursy bukmacherskie po usunięciu marży / closing odds) w celu identyfikacji wartości oczekiwanej (+EV).

---

## Architektura potoku danych

```text
[football-data.co.uk] (Wyniki, Sędziowie, Kursy)
                 │
[Understat / FBref via soccerdata] (xG, xGA, Strzały)
                 │
                 ▼
       ┌──────────────────────────────────────────────┐
       │         1. INGESTION & ENTITY RESOLUTION     │
       │  - Unifikacja i mapowanie nazw klubów        │
       │  - Walidacja typów danych i spójności dat    │
       └──────────────────────┬───────────────────────┘
                              │
                              ▼
       ┌──────────────────────────────────────────────┐
       │       2. TEAM-MATCH MART (Long Format)       │
       │  - Granularność: 2 wiersze na każdy mecz     │
       │  - Dynamiczna aktualizacja Elo ratingu       │
       │  - Zero-Leakage Feature Engineering          │
       │    df.groupby('team')[col].shift(1).rolling()│
       └──────────────────────┬───────────────────────┘
                              │
                              ▼
       ┌──────────────────────────────────────────────┐
       │      3. ANALYTICAL FACT MART (Wide Format)   │
       │  - 1 wiersz = 1 mecz (Perspektywa Home/Away) │
       │  - Cechy interakcyjne (różnica Elo, odpoczynek)
       └──────────────────────┬───────────────────────┘
                              │
                              ▼
       ┌──────────────────────────────────────────────┐
       │      4. TIME-SERIES VALIDATION & MODELING    │
       │  - Expanding Window Cross-Validation         │
       │  - Baseline: Dixon-Coles Poisson Model       │
       │  - ML Engine: LightGBM / XGBoost Multi-class │
       │  - Metryki: Log Loss, Brier Score, Confusion │
       └──────────────────────┬───────────────────────┘
                              │
                              ▼
       ┌──────────────────────────────────────────────┐
       │           5. BI DASHBOARD (Streamlit)        │
       │  - Model Probabilities vs. Closing Odds      │
       │  - Analiza formy i dekompozycja cech SHAP    │
       └──────────────────────────────────────────────┘
```

---

## Metodologia i eliminacja Data Leakage

Wszystkie zmienne wejściowe modelu opisują stan wiedzy **wyłącznie przed pierwszym gwizdkiem sędziego**.

### Kluczowe cechy predykcyjne (Feature Store)
1. **Rolling xG Differential (`roll_xg_diff_3`, `roll_xg_diff_5`):**
   $$\text{xG Diff} = \text{xG For} - \text{xG Against}$$
   Średnia krocząca różnicy spodziewanych bramek z ostatnich 3 i 5 spotkań. Oczyszcza analizę formy ze szczęścia wynikowego.
2. **Shots on Target Ratio (`roll_sotr_5`):**
   $$\text{SOTR} = \frac{\text{Strzaly Celne Oddane}}{\text{Strzaly Celne Oddane} + \text{Strzaly Celne Dopuszczone}}$$
   Metryka kontroli nad meczem o znacznie mniejszej wariancji niż gole.
3. **Dynamiczny Ranking Elo (`home_elo_pre`, `away_elo_pre`, `elo_diff`):**
   Aktualizowany po każdym spotkaniu z uwzględnieniem handicapu własnego boiska oraz marginesu zwycięstwa.
4. **Dysproporcja czasu regeneracji (`rest_days_diff`):**
   $$\text{Rest Diff} = \text{Dni odpoczynku Gospodarza} - \text{Dni odpoczynku Goscia}$$
   Wychwytuje zmęczenie i rotacje składem spowodowane występami w europejskich pucharach.
5. **Specyficzny Home/Away Advantage:**
   Średnia różnica xG liczona wyłącznie z meczów domowych dla gospodarza i wyłącznie wyjazdowych dla gościa (okno 10 spotkań).

---

## Rygor ewaluacji: Dlaczego NIE Accuracy?

W La Liga podział wyników to średnio:
* **~45%** Wygrane gospodarzy (1)
* **~27%** Remisy (X)
* **~28%** Wygrane gości (2)

Model, który za każdym razem wskaże zwycięstwo gospodarza, osiągnie **~45% Accuracy**, będąc analitycznie bezwartościowym. Remis jest zdarzeniem o najwyższej entropii – rzadko ma najwyższe jednostkowe prawdopodobieństwo, przez co klasyczny `argmax()` pomija go niemal w 100%.

### Przyjęte metryki oceny:
* **Multi-class Log Loss (Cross-Entropy Loss):** Główna metryka optymalizacyjna. Karze model za nadmierną pewność siebie przy błędnej predykcji:
  $$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{i,k} \ln(p_{i,k})$$
* **Multi-class Brier Score:** Błąd średniokwadratowy wektora wyestymowanych prawdopodobieństw względem wektora one-hot wyniku rzeczywistego:
  $$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{i,k} - y_{i,k})^2$$
* **Row-Normalized Confusion Matrix:** Pozwala śledzić Recall dla trudnej klasy remisów (Draw).
* **Closing Odds Benchmark:** Rynkowe kursy zamykające (oczyszczone z marży bukmacherskiej metodą Shin'a lub Normalizacji Proporcjonalnej) stanowią ostateczny punkt odniesienia do oceny przewagi modelu.

---

## Struktura repozytorium

```text
laliga-match-prediction-engine/
├── config/
│   └── config.yaml              # Hiperparametry, okna rolling, ścieżki ETL
├── data/
│   ├── raw/                     # Niemodyfikowalne zrzuty źródłowe (git-ignored)
│   │   └── .gitkeep
│   └── processed/               # Przetworzona tabela analityczna (Parquet)
│       └── .gitkeep
├── notebooks/
│   ├── 01_eda_xg_and_form.ipynb # Eksploracja korelacji i rozkładów
│   └── 02_model_benchmark.ipynb # Eksperymenty: Poisson vs XGBoost
├── src/
│   ├── __init__.py
│   ├── ingestion/               # ETL i Data Wrangling
│   │   ├── __init__.py
│   │   ├── fetch_raw_data.py    # Pobieranie danych z football-data & soccerdata
│   │   └── clean_data.py        # Normalizacja encji i walidacja typów
│   ├── features/                # Feature Engineering bez wycieku danych
│   │   ├── __init__.py
│   │   ├── build_features.py    # Logika rolling metrics z shift(1)
│   │   └── elo_rating.py        # Silnik kalkulacji Elo ratingu
│   ├── models/                  # Logika treningowa i estymacja
│   │   ├── __init__.py
│   │   ├── baseline_poisson.py  # Model Poissona (Dixon-Coles)
│   │   ├── train_classifier.py  # Trening LightGBM / XGBoost
│   │   └── evaluate.py          # Obliczanie Log Loss, Brier Score, macierzy pomyłek
│   └── dashboard/               # Warstwa prezentacyjna BI
│       ├── __init__.py
│       └── app.py               # Aplikacja Streamlit
├── tests/                       # Testy jednostkowe krytycznych przekształceń
│   ├── __init__.py
│   └── test_leakage.py          # Asercje weryfikujące przesunięcia shift(1)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Szybki start i reprodukowalność

### 1. Wymagania systemowe
* Python w wersji 3.11 lub wyższej
* Wirtualne środowisko (`venv` lub `conda`)

### 2. Klonowanie i instalacja środowiska
```bash
git clone https://github.com/gaucho212/laliga-match-prediction-engine.git
cd laliga-match-prediction-engine

python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Uruchomienie pełnego cyklu analitycznego
```bash
# 1. Pobranie i unifikacja danych historycznych
python src/ingestion/fetch_raw_data.py

# 2. Wygenerowanie tabeli faktów z metrykami kroczącymi
python src/features/build_features.py

# 3. Trening modeli (Baseline Poissona oraz LightGBM) i ewaluacja
python src/models/train_classifier.py

# 4. Uruchomienie interaktywnego dashboardu analitycznego
streamlit run src/dashboard/app.py
```

---

## Przykładowe rezultaty i interpretacja

| Model | Multi-class Log Loss | Brier Score | Uwagi |
| :--- | :---: | :---: | :--- |
| **Naive Home Baseline** | 1.098 | 0.667 | Predykcja stała na podstawie rozkładu a priori |
| **Dixon-Coles Poisson** | 1.012 | 0.589 | Dobra kalibracja, uwzględnia niskie remisy |
| **LightGBM Classifier** | **0.978** | **0.564** | Wykorzystuje nieliniowe relacje xG Diff i Elo |
| **Market Closing Odds** | 0.945 | 0.548 | Benchmark rynkowy (Pinnacle bez marży) |

---

## Autor
* **Autor:** Adam Woziński
* **LinkedIn:** https://www.linkedin.com/in/adam-woziński-866048359/
* **GitHub:** https://github.com/gaucho212