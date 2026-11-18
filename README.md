# Short-Term Electricity Demand Forecasting for the Indian Power Grid

Forecasting India's hourly national electricity demand **1 hour and 24 hours ahead** using demand history, calendar effects and weather, with an early-warning system for high-demand hours.

CSE570: Machine Learning with Python.

| | 
|---|---|
| **Task** | Supervised time-series regression on hourly national demand (MW) |
| **Horizons** | 1 hour ahead and 24 hours ahead |
| **Data** | 46,728 hourly observations, 1 Jan 2019 to 30 Apr 2024, plus hourly weather for five regional cities |
| **Models** | Linear Regression, Random Forest, Gradient Boosting, against three naive baselines |
| **Evaluation** | Chronological split, time-series cross-validation, one held-out test period (Jan 2023 to Apr 2024) |
| **Extra** | Adaptive high-demand (peak) warnings with hour-level and episode-level metrics |
| **Interactive dashboard** | React app in a separate repository (see [Dashboard](#dashboard)) |

---

## Table of contents

1. [What this project does](#what-this-project-does)
2. [Headline results](#headline-results)
3. [Data](#data)
4. [Method](#method)
5. [Peak-demand warnings](#peak-demand-warnings)
6. [Findings from the analysis](#findings-from-the-analysis)
7. [Repository structure](#repository-structure)
8. [How to reproduce](#how-to-reproduce)
9. [Testing and leakage control](#testing-and-leakage-control)
10. [Dashboard](#dashboard)
11. [Limitations](#limitations)
12. [Future work](#future-work)
13. [Tech stack](#tech-stack)

---

## What this project does

Grid operators must schedule generation before demand arrives, so they need a forecast of how much power the grid will be asked to supply. This project builds and evaluates such a forecast for the Indian national grid.

It forecasts **demand** (what consumers draw from the grid, in megawatts), not generation.

Because the data ends in April 2024, the evaluation is a **backtest**. For every hour of the test period the model sees only information available at that moment, predicts demand 1 or 24 hours later, and only afterwards is the prediction compared with what actually happened. The test period was never used for training or tuning, so the reported accuracy is an honest estimate of how the approach would have performed in real operation.

The project answers five questions:

1. How accurately can hourly national demand be forecast 1 and 24 hours ahead?
2. Do machine-learning models beat simple baselines such as "same hour yesterday"?
3. How much do calendar effects and weather add on top of demand history?
4. Which inputs matter most?
5. Can the forecast be turned into a useful early warning for unusually high demand?

---

## Headline results

All numbers below are on the **held-out test set** (Jan 2023 to Apr 2024, 11,664 hourly forecasts per horizon). Skill is the reduction in MAE compared with the best naive baseline.

### 1 hour ahead

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | R² | Skill (%) |
|---|---:|---:|---:|---:|---:|
| Persistence (baseline) | 4,174 | 5,321 | 2.34 | 0.929 | 0.0 |
| Previous day (baseline) | 4,767 | 6,499 | 2.67 | 0.894 | -14.2 |
| Previous week (baseline) | 8,261 | 10,864 | 4.64 | 0.704 | -97.9 |
| **Linear Regression** | **916** | **1,252** | **0.50** | **0.996** | **78.1** |
| Random Forest (tuned) | 1,326 | 1,811 | 0.73 | 0.992 | 68.2 |
| Gradient Boosting (tuned) | 1,089 | 1,462 | 0.60 | 0.995 | 73.9 |

### 24 hours ahead

| Model | MAE (MW) | RMSE (MW) | MAPE (%) | R² | Skill (%) |
|---|---:|---:|---:|---:|---:|
| Persistence / previous day (identical at 24 h) | 4,767 | 6,499 | 2.67 | 0.894 | 0.0 |
| Previous week (baseline) | 8,261 | 10,864 | 4.64 | 0.704 | -73.3 |
| Linear Regression | 3,845 | 5,243 | 2.16 | 0.931 | 19.3 |
| Random Forest (tuned) | 3,481 | 4,881 | 1.96 | 0.940 | 27.0 |
| **Gradient Boosting (tuned)** | **3,479** | **4,877** | **1.95** | **0.940** | **27.0** |

What the numbers say:

- Every learned model clearly beats every naive baseline at both horizons.
- At **1 hour ahead** the error is about **0.5 to 0.6 %** of demand. Demand changes smoothly from one hour to the next, so recent history explains almost everything, and a simple linear model is already very strong. It edged out the tuned tree models on the test set.
- At **24 hours ahead** the problem is harder (about **1.95 %** error) because the model must anticipate the daily cycle and weather a full day in advance. The tree models pull ahead of the linear model here.
- The 24-hour persistence and previous-day baselines are the same forecast by construction (the same hour yesterday), which is why they have identical scores.

For reference, hyperparameters were selected by time-series cross-validation, which chose Gradient Boosting for the 1-hour model and Random Forest for the 24-hour model. The full tuning leaderboard is in `results/tables/tuning_top5_cv.csv` and the chosen settings are in `results/tables/best_params.json`.

---

## Data

| File | Content | Role |
|---|---|---|
| `data/raw/india_hourly_demand.xlsx` | National and five regional demand in MW, hourly, 2019-01-01 to 2024-04-30 | Main dataset |
| `data/raw/india_weather_hourly.csv` | Hourly weather for five regional cities, fetched from the Open-Meteo archive | Weather inputs |
| `data/processed/india_demand_weather.csv` | Cleaned demand merged with weather, with anomaly flags | Model input |
| `data/raw/load_dataset.csv` and `data/processed/load_dataset_clean.csv` | Hourly load with co-located weather, Nov 2021 to Nov 2022 | Weather-impact case study |
| `data/external/panama_continuous_dataset.csv` | Panama national demand and weather | External benchmark only, **not Indian data and not used in any result** |

Source details and licences for each raw file are in [`data/README.md`](data/README.md).

**Demand series.** 46,728 hourly rows with no missing or duplicate timestamps. National demand averages about 160 GW (minimum 95 GW, maximum 237 GW), split across the northern, western, eastern, southern and north-eastern regional grids.

**Weather.** Temperature, relative humidity, dew point, wind speed, solar radiation and precipitation for one representative city per region (Delhi, Mumbai, Kolkata, Chennai and Guwahati), combined into national weather columns, plus per-region temperatures.

**Cleaning (see `results/tables/cleaning_report.csv`).**

- Demand had no gaps or duplicates.
- 20 hours deviated by more than 15 % from neighbouring same-hour values. They were **flagged but kept**, because holidays, curfews and events are real demand rather than errors.
- Weather had no missing values after merging, and gaps of up to 3 hours would have been interpolated.
- In the secondary load dataset, 24 irradiance sentinel values (-999) were set to missing and filled from neighbouring days, and 79 non-positive meter readings were treated as drop-outs and interpolated.

**Time resolution.** All available data is hourly, so 15-minute forecasting is not possible and is not attempted.

---

## Method

### Forecasting setup

For each forecast origin time `t` and horizon `h` (1 or 24 hours), the model predicts demand at `t + h` using only information available at `t`.

The target is divided by the **recent demand level** (the mean of the previous 168 hours). India's demand grows roughly 5 % a year, and tree models cannot extrapolate beyond the range they saw in training, so predicting a ratio instead of a raw megawatt value lets them handle growth. Predictions are multiplied back by the level to return to MW.

### Chronological splits

Splits are defined on the **target time**, so no training target ever falls inside the validation or test period.

| Split | Period | Hourly samples (1 h / 24 h) |
|---|---|---|
| Train | 2019-01-01 to 2021-12-31 | 25,968 / 25,945 |
| Validation | 2022-01-01 to 2022-12-31 | 8,760 / 8,760 |
| Test | 2023-01-01 to 2024-04-30 | 11,664 / 11,664 |

### Features

Features are built in `src/features.py` and grouped in three nested sets so their value can be measured separately (`results/tables/feature_catalog.csv`):

| Group | Examples |
|---|---|
| Demand lags | Demand 0, 1, 2, 3, 6, 12, 24, 48 and 168 hours before the origin, same hour yesterday, same hour last week |
| Rolling demand | 24-hour mean, standard deviation, max and min, weekly growth rate |
| Calendar (of the target hour) | Hour, weekday and day of year as sine/cosine pairs, weekend flag |
| Events | Indian public holidays, COVID lockdown period (22 Mar to 7 Jun 2020) |
| Weather (at the target hour) | Temperature, humidity, dew point, wind, solar radiation, precipitation, cooling degrees above 25 °C, 24-hour mean temperature, temperature change from the origin |

Feature sets: `demand` (history only) is contained in `calendar`, which is contained in `full` (adds weather).

### Models

| Model | Purpose |
|---|---|
| Persistence | Baseline: demand now equals demand at the target |
| Previous day | Baseline: the same hour yesterday |
| Previous week | Baseline: the same hour last week |
| Linear Regression | Simple, fast, interpretable benchmark |
| Random Forest | Non-linear ensemble of trees |
| Gradient Boosting (histogram-based) | Non-linear boosted trees |

### Tuning and final evaluation

1. Random Forest and Gradient Boosting hyperparameters were tuned with **expanding-window time-series cross-validation** on 2019 to 2022 data only, with a 24-hour gap between training and validation folds to prevent leakage.
2. The selected models were refit and evaluated **once** on the test set.
3. Metrics: MAE, RMSE, MAPE, R², and skill relative to the best baseline.

Selected hyperparameters:

| Horizon | Selected model | Cross-validated error | Key settings |
|---|---|---:|---|
| 1 h | Gradient Boosting | 0.67 % | 600 iterations, learning rate 0.08, max depth 6, 15 leaf nodes, min 20 samples per leaf, L2 = 1.0 |
| 24 h | Random Forest | 1.90 % | 100 trees, min 5 samples per leaf, 70 % of features per split |

---

## Peak-demand warnings

A forecast is most valuable when it warns operators about unusually high demand, so the project also evaluates the forecast as an early-warning system.

**Definition of a high-demand hour.** Demand above the **95th percentile of the previous 365 days**. The threshold is adaptive and looks only backwards, so it is known when the forecast is made. A fixed threshold would fail because demand keeps growing.

**Warning rule.** A warning is issued when the forecast exceeds `threshold × (1 - margin)`. The safety margin is selected on **validation data only** (model trained on 2019 to 2021), then applied once to the test set. The validation search picked a margin of 0 % in every case, because a larger margin raised recall but lowered precision by more.

**Episodes.** Consecutive high-demand hours at most 6 hours apart are merged into one episode, so the system is also scored on whether it catches each real peak event, not only each hour.

### Test-set warning performance

| Horizon | Method | Precision | Recall | F1 | False-alarm rate | Episode recall |
|---|---|---:|---:|---:|---:|---:|
| 1 h | Gradient Boosting | 0.954 | 0.919 | 0.936 | 0.7 % | 0.951 |
| 1 h | Previous-day baseline | 0.745 | 0.738 | 0.741 | 3.7 % | 0.807 |
| 24 h | Random Forest | 0.836 | 0.773 | 0.804 | 2.2 % | 0.856 |
| 24 h | Previous-day baseline | 0.745 | 0.742 | 0.743 | 3.7 % | 0.807 |

- About 12.7 % of test hours (1,476 to 1,486 hours) count as high-demand, forming 243 separate episodes.
- At 1 hour ahead the model catches about 92 % of high-demand hours and 95 % of episodes, and 95 % of its warnings are correct.
- At 24 hours ahead it catches about 77 % of hours and 86 % of episodes, still clearly better than the baseline.
- The high-demand share is above the nominal 5 % because a trailing 365-day threshold lags India's demand growth.

---

## Findings from the analysis

**Feature value (validation set, Gradient Boosting).**

| Horizon | Demand history only | + calendar | + weather (full) |
|---|---:|---:|---:|
| 1 h, MAE | 1,075 MW | 1,033 MW | 1,034 MW |
| 24 h, MAE | 3,498 MW | 2,993 MW | 2,902 MW |

- Calendar information helps at both horizons, and helps a lot at 24 hours ahead (about 14 % lower error).
- Weather adds a small further gain at 24 hours (about 3 %) and essentially nothing at 1 hour, where the latest demand already contains the weather's effect.

**Most important inputs (permutation importance).** The most recent demand value dominates at both horizons, followed by same-hour-yesterday and same-hour-last-week demand, with hour of day and day of week next. See `results/figures/05_permutation_importance.png`.

**Weather relationships.** Correlations with national demand are strongest for temperature (+0.42 hourly), humidity (-0.35) and solar radiation (+0.31), and negligible for precipitation (see `results/tables/demand_weather_correlations.csv`).

**COVID-19.** April 2020 demand was 24.5 % below April 2019, which is why a lockdown indicator is included. By April 2024 demand was about 21 % above April 2019.

**Where the model is weakest.** At 1 hour ahead, error is fairly even across months (about 850 to 1,240 MW) and is highest around the morning (6 to 9 h) and evening (19 to 20 h) ramps. At 24 hours ahead, error is highest in the hot pre-monsoon months (April averages about 5,300 MW, May about 4,600 MW) and in the afternoon (12 to 16 h), and lowest in winter (February about 1,800 MW). See `results/figures/06_error_by_hour_month.png` and `results/figures/06_test_peak_week.png`.

---

## Repository structure

```
.
├── data/
│   ├── raw/                  Original demand and weather files
│   ├── external/             Panama benchmark (not used in results)
│   ├── processed/            Cleaned, merged dataset used for modelling
│   └── README.md             Source and licence notes for every file
├── src/
│   ├── fetch_weather.py      Downloads hourly weather from Open-Meteo
│   ├── features.py           Leakage-safe feature and target construction
│   ├── evaluate.py           Chronological splits and metrics
│   └── peaks.py              Peak-warning metrics, episodes, margin sweep
├── notebooks/
│   ├── 03_eda.ipynb                          Exploratory analysis
│   ├── 04_feature_engineering.ipynb          Features and ablations
│   ├── 06_tuning_and_final_evaluation.ipynb  Cross-validated tuning, one-time test evaluation
│   └── 07_peak_demand_warnings.ipynb         Warning thresholds, margins, evaluation
├── tests/
│   ├── test_features.py      Look-ahead leakage and target-alignment tests
│   └── test_peaks.py         Episode and warning-metric tests
├── results/
│   ├── tables/               Every metric table (CSV, JSON)
│   ├── figures/              Every figure (PNG)
│   ├── predictions_test_h1.csv, predictions_test_h24.csv
│   └── peak_warnings_test_h1.csv, peak_warnings_test_h24.csv
└── requirements.txt
```

---

## How to reproduce

**Requirements:** Python 3.10 or newer.

```bash
git clone https://github.com/tanishmaity03/India-Demand-Forecasting-ML.git
cd India-Demand-Forecasting-ML
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt holidays
```

**1. Run the tests**

```bash
python -m unittest discover -s tests -t .
```

**2. (Optional) re-download the weather**

```bash
python -m src.fetch_weather
```

The processed dataset is already included, so this step is only needed to rebuild it from scratch.

**3. Run the notebooks in order** (locally with Jupyter, or in Google Colab):

| Notebook | Produces |
|---|---|
| `03_eda.ipynb` | EDA figures and correlation tables |
| `04_feature_engineering.ipynb` | Feature catalogue, ablation and importance tables |
| `06_tuning_and_final_evaluation.ipynb` | Tuned parameters, test predictions, final results table |
| `07_peak_demand_warnings.ipynb` | Warning margins, warning performance, timelines |

Every result table in `results/` is regenerated by these notebooks. Models use fixed random seeds, so reruns match the numbers in this README.

---

## Testing and leakage control

The most common failure in forecasting projects is information leaking from the future into the features. This project guards against it in three ways:

1. **Features use only the past.** Everything in `src/features.py` is computed from data at or before the forecast origin, except calendar values and weather for the target hour, which are known or forecastable in advance (see [Limitations](#limitations) for the weather caveat).
2. **Splits are chronological and defined on target time**, and cross-validation uses an expanding window with a gap.
3. **Automated tests** in `tests/` check this directly:
   - no future demand appears in any feature,
   - the target is aligned with the correct hour,
   - baselines are computable at the forecast origin,
   - no missing values reach the models,
   - episode merging and warning metrics behave correctly on hand-built cases.

---

## Dashboard

An interactive dashboard presents the results for review: forecast against actual demand, model comparison, demand history, peak warnings, and a Method page that links each result back to the code that produced it.

- **Live site:** https://tanishmaity03.github.io/Dahsboard/
- **Source:** https://github.com/tanishmaity03/Dahsboard

It is a React, TypeScript and Tailwind application that reads precomputed CSV results exported from this repository, so it shows backtest results rather than live data.

---

## Limitations

- **Weather at the target hour is actual weather**, standing in for a weather forecast that would really be used. Weather gains are therefore somewhat optimistic. A production system would feed in forecast weather and accuracy would drop slightly at 24 hours.
- **Backtest, not live deployment.** The data ends in April 2024, so there is no live forecasting of unseen hours.
- **One grid, about 16 test months.** Results may not transfer to other grids or other periods.
- **Peak definition is a modelling choice.** The 95th-percentile rolling threshold and the 6-hour episode gap are reasonable but not the only options. The resulting peak rate is about 12.7 %, not 5 %, because the threshold lags demand growth.
- **National level only.** Regional demand is analysed but not forecast separately.
- **Hourly resolution.** Intra-hour variation cannot be captured.
- **Holiday and event handling** relies on a public-holiday calendar and a fixed COVID period. Local festivals, strikes and unusual events are not modelled.

---

## Future work

- Replace actual weather with archived weather **forecasts** for an honest operational estimate.
- Forecast regional demand and combine the regional forecasts.
- Add probabilistic forecasts (prediction intervals) so operators see uncertainty, not just a point estimate.
- Try sequence models (LSTM, temporal convolution, transformer) and compare them with the tree models.
- Connect a live demand feed so the dashboard can show genuinely forward-looking forecasts.
- Calibrate warning margins by the real cost of a missed peak versus a false alarm.

---

## Tech stack

**Modelling:** Python, pandas, NumPy, scikit-learn, statsmodels, `holidays`
**Analysis and plots:** matplotlib, seaborn
**Data access:** Open-Meteo historical weather API
**Dashboard:** React, TypeScript, Tailwind CSS, Recharts, GSAP, deployed with GitHub Pages

---

## Author

GitHub: [@tanishmaity03](https://github.com/tanishmaity03). CSE570: Machine Learning with Python.
