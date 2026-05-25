# Hierarchical Time Series Forecasting

Forecasting retail sales across a 4-level product hierarchy, comparing ARIMA, Prophet, and LSTM baselines, and reconciling them with MinTrace-WLS — the statistically optimal reconciliation method from Wickramasuriya et al. (2019, JASA).

This is the type of system DS teams at retail, finance, and energy companies run in production: forecasts must be *coherent* (product forecasts must sum to category, category to department, department to total) and *accurate* at the level where planning decisions are made.

---

## Hierarchy

```
TOTAL
├── FOOD
│   ├── FOOD_1  →  FOOD_1_P1 … FOOD_1_P5
│   ├── FOOD_2  →  FOOD_2_P1 … FOOD_2_P5
│   └── FOOD_3  →  FOOD_3_P1 … FOOD_3_P5
└── HOUSEHOLD
    ├── HH_1    →  HH_1_P1 … HH_1_P5
    ├── HH_2    →  HH_2_P1 … HH_2_P5
    └── HH_3    →  HH_3_P1 … HH_3_P5
```

- **39 time series** total (1 + 2 + 6 + 30)
- **3 years of daily sales** (2020–2022), M5-style synthetic retail data
- **28-day test horizon** (held-out evaluation)

---

## Models

### ARIMA
AIC-selected ARIMA(p,d,q) via statsmodels. Differencing order chosen by ADF test; p and q searched over [0,3]. Fit independently per series.

### Prophet
Facebook Prophet with multiplicative weekly + yearly seasonality. Handles trend changepoints automatically. Fit independently per series.

### LSTM
2-layer LSTM (PyTorch) with 64 hidden units, 60-day input window, recursive multi-step decoding. Min-max scaled per series, trained for 50 epochs with Adam + gradient clipping.

---

## Reconciliation Methods

Raw model forecasts are *incoherent* — the 30 product forecasts won't sum to the category and total forecasts. Three reconciliation strategies are compared:

### Top-Down
Forecast at the total level, then distribute to products using each product's historical average share. Simple but throws away all product-level signal.

### MinTrace-OLS
Project base forecasts onto the coherent subspace via:

```
P = S(S'S)⁻¹S'
```

where **S** is the summing matrix mapping bottom-level products to all aggregates. Treats all series equally regardless of forecast quality.

### MinTrace-WLS *(Wickramasuriya et al. 2019)*
The statistically optimal approach. Weights each series by the inverse of its in-sample residual variance:

```
W   = diag(σ²₁, σ²₂, ..., σ²ₙ)
P   = S(S'W⁻¹S)⁻¹S'W⁻¹
```

Series where the model fits poorly (high σ²) contribute less to the reconciled forecast. This is the version used in production systems and cited in the hierarchical forecasting literature.

---

## Results

### Base model accuracy by level (Mean RMSE)

| Model | Total | Department | Category | Product |
|---|---|---|---|---|
| ARIMA | 38.08 | 21.39 | 10.68 | **3.59** |
| Prophet | **30.57** | **16.66** | **7.92** | 3.86 |
| LSTM | 152.36 | 30.86 | 14.21 | 3.97 |

Prophet dominates at aggregate levels — its trend+seasonality decomposition handles smooth rolled-up signals well. ARIMA wins at the noisy individual product level where parsimony beats expressiveness. The LSTM's recursive decoder compounds errors over 28 steps, making it struggle on the downward-trending aggregate series.

### Reconciliation accuracy (product level, Mean RMSE)

| Method | RMSE | vs best base |
|---|---|---|
| TopDown-ARIMA | 10.75 | +3.0× |
| TopDown-LSTM | 12.00 | +3.3× |
| OLS-ARIMA | 3.74 | +4% |
| OLS-LSTM | 5.20 | **+31%** |
| **WLS-ARIMA** | **3.57** | **−0.6%** ✓ |
| WLS-Prophet | 3.86 | neutral |
| WLS-LSTM | 3.83 | **−3.5%** ✓ |

**Key findings:**

1. **Top-Down is always 3× worse.** Discarding product-level signals and relying on proportion shares is a bad trade at every level of model quality.

2. **OLS MinTrace hurts LSTM badly (+31%).** Because OLS weights all series equally, the total-level LSTM forecast (RMSE 152) corrupts the product reconciliation. This is the canonical failure mode OLS MinTrace is known for.

3. **WLS MinTrace fixes it.** WLS identifies that LSTM's upper-level residual variance is enormous (σ² ≈ 152²) and down-weights it near-zero automatically. Product-level LSTM accuracy recovers to 3.83 — better than the unreconciled baseline.

4. **WLS-ARIMA (3.57) is the best result overall** — reconciliation *improved* on the best base model, which is the ideal outcome: coherent forecasts that are also more accurate.

5. **WLS-Prophet = base Prophet (3.86).** Prophet is well-calibrated at every level, so residual variances are similar across the hierarchy. WLS has nothing to correct — which is the correct behavior.

---

## Plots

**`01_total_forecast.png`** — All three models vs actuals at total level. Prophet and ARIMA track the weekly seasonality; LSTM trends down due to recursive error accumulation.

**`02_category_forecasts.png`** — 6-panel grid of category-level forecasts. Prophet wins consistently.

**`03_reconciliation_comparison.png`** — Bar chart of all 12 configurations by product RMSE. The story in one image: Top-Down is red and tall, WLS is purple and short.

**`04_wls_improvement_heatmap.png`** — Per-product RMSE improvement of WLS over the base model. WLS-LSTM shows the most variance (some products improve by 17%, others regress) because LSTM training is stochastic. On average, WLS wins.

**`05_hierarchy_cascade.png`** — The full top-down cascade for one product family, with WLS reconciliation overlaid at the product panel.

---

## Project structure

```
├── data_generator.py       # Synthetic M5-style retail data (30 products, 3 years)
├── models/
│   ├── arima_model.py      # AIC-selected ARIMA via statsmodels
│   ├── prophet_model.py    # Facebook Prophet wrapper
│   └── lstm_model.py       # 2-layer LSTM (PyTorch), recursive decoding
├── reconciliation.py       # Summing matrix S, Top-Down, MinTrace-OLS, MinTrace-WLS
├── main.py                 # End-to-end pipeline
├── requirements.txt
└── results/                # Generated plots and CSVs
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Runtime: ~5 minutes on CPU (ARIMA grid search + 39 Prophet fits + 39 LSTM training runs).

---

## Stack

`statsmodels` · `prophet` · `PyTorch` · `hierarchicalforecast` · `pandas` · `numpy` · `matplotlib` · `seaborn`

---

## Reference

Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization. *Journal of the American Statistical Association*, 114(526), 804–819.
