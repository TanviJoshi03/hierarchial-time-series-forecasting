"""
Hierarchical Time Series Forecasting — extended pipeline.

Stages
------
  1. Generate synthetic retail data (price + promo + known elasticity)
  2a. Base models: ARIMA, Prophet, LSTM  (all hierarchy levels)
  2b. GBM models: LightGBM, XGBoost     (product level → aggregate up)
  3. Hierarchical reconciliation (Top-Down, MinTrace-OLS, MinTrace-WLS)
  4. Bayesian hierarchical price-elasticity (PyMC, partial pooling)
  5. Causal promo-lift: diff-in-diff + A/B test
  6. Cold-start via description embeddings + FAISS retrieval
  7. Real-world validation on Online Retail II (optional)
  8. Write results/summary.csv
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd

from data_generator import (
    generate_dataset, build_hierarchical_df, build_covariate_df, save_ground_truth,
    TRUE_BETAS, TRUE_GAMMA,
)
from models.arima_model import forecast_all_arima
from models.prophet_model import forecast_all_prophet
from reconciliation import (
    build_summing_matrix, reconcile_top_down, reconcile_mintrace,
    reconcile_mintrace_wls, compute_residual_variances, evaluate_level, rmse, mae,
)

HORIZON = 28
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("  Hierarchical Time Series Forecasting — Full Pipeline")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────
# Stage 1 — Data generation
# ─────────────────────────────────────────────────────────────────────
print("\n[1/7] Generating synthetic retail dataset …")
raw_df, descriptions = generate_dataset()
save_ground_truth(RESULTS_DIR)

levels = build_hierarchical_df(raw_df)
product_df   = levels["product"]
category_df  = levels["category"]
dept_df      = levels["department"]
total_df     = levels["total"]

price_df, promo_df = build_covariate_df(raw_df)

train_product  = product_df.iloc[:-HORIZON]
test_product   = product_df.iloc[-HORIZON:]
train_category = category_df.iloc[:-HORIZON]
test_category  = category_df.iloc[-HORIZON:]
train_dept     = dept_df.iloc[:-HORIZON]
test_dept      = dept_df.iloc[-HORIZON:]
train_total    = total_df.iloc[:-HORIZON]
test_total     = total_df.iloc[-HORIZON:]

price_train    = price_df.loc[train_product.index]
promo_train    = promo_df.loc[train_product.index]

print(f"    Train: {train_product.index[0].date()} → {train_product.index[-1].date()}")
print(f"    Test : {test_product.index[0].date()} → {test_product.index[-1].date()}")
print(f"    Products: {product_df.shape[1]} | Categories: {category_df.shape[1]} | Depts: {dept_df.shape[1]}")

# Hierarchy maps
category_map = {}
for col in product_df.columns:
    cat = "_".join(col.split("_")[:-1])
    category_map[col] = cat

dept_map = {}
for cat in category_df.columns:
    dept = cat.split("_")[0]
    dept_map[cat] = "HOUSEHOLD" if dept == "HH" else "FOOD"

S_df, all_series = build_summing_matrix(category_map, dept_map)

all_train = {"TOTAL": train_total["TOTAL"]}
for c in train_dept.columns:
    all_train[c] = train_dept[c]
for c in train_category.columns:
    all_train[c] = train_category[c]
for c in train_product.columns:
    all_train[c] = train_product[c]
all_train_df = pd.DataFrame(all_train)

LEVELS = [
    ("total",      train_total),
    ("department", train_dept),
    ("category",   train_category),
    ("product",    train_product),
]

# ─────────────────────────────────────────────────────────────────────
# Stage 2a — ARIMA, Prophet, LSTM
# ─────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────
# Stage 2b — LightGBM + XGBoost  (run FIRST before any Stan/TF to avoid segfault)
# ─────────────────────────────────────────────────────────────────────
print("\n[2b/7] Fitting GBM models (LightGBM + XGBoost) …")
gbm_results = None
try:
    from models.gbm_model import forecast_all_gbm, aggregate_gbm_to_levels
    gbm_results = forecast_all_gbm(train_product, price_df, promo_df, horizon=HORIZON)
    lgbm_levels = aggregate_gbm_to_levels(gbm_results["lgbm_fc"], category_map, dept_map)
    xgb_levels  = aggregate_gbm_to_levels(gbm_results["xgb_fc"],  category_map, dept_map)
except Exception as e:
    print(f"    [WARN] GBM models skipped: {e}")

gc.collect()
print("\n[2a/7] Fitting ARIMA …")
arima_fc, arima_fitted = {}, {}
for lvl_name, train_df in LEVELS:
    print(f"    {lvl_name} ({train_df.shape[1]} series) …", end=" ", flush=True)
    fc, fitted = forecast_all_arima(train_df, horizon=HORIZON)
    arima_fc[lvl_name] = fc
    for col in fitted.columns:
        arima_fitted[col] = fitted[col]
    print("done")

gc.collect()
print("\n[2a/7] Fitting Prophet …")
prophet_fc, prophet_fitted = {}, {}
for lvl_name, train_df in LEVELS:
    print(f"    {lvl_name} ({train_df.shape[1]} series) …", end=" ", flush=True)
    fc, fitted = forecast_all_prophet(train_df, horizon=HORIZON)
    prophet_fc[lvl_name] = fc
    for col in fitted.columns:
        prophet_fitted[col] = fitted[col]
    print("done")

gc.collect()
print("\n[2c/7] Fitting LSTM …")
from models.lstm_model import forecast_all_lstm  # lazy import: after GBM to avoid OpenMP conflict
lstm_fc, lstm_fitted = {}, {}
for lvl_name, train_df in LEVELS:
    print(f"    {lvl_name} ({train_df.shape[1]} series) …", end=" ", flush=True)
    fc, fitted = forecast_all_lstm(train_df, horizon=HORIZON, epochs=50)
    lstm_fc[lvl_name] = fc
    for col in fitted.columns:
        lstm_fitted[col] = fitted[col]
    print("done")

# ─────────────────────────────────────────────────────────────────────
# Stage 3 — Reconciliation
# ─────────────────────────────────────────────────────────────────────
gc.collect()
print("\n[3/7] Hierarchical reconciliation …")
test_idx = test_product.index

arima_vars   = compute_residual_variances(all_train_df, arima_fitted)
prophet_vars = compute_residual_variances(all_train_df, prophet_fitted)
lstm_vars    = compute_residual_variances(all_train_df, lstm_fitted)


def assemble_base_fc(fc_dict: dict, all_series: list, idx: pd.DatetimeIndex) -> pd.DataFrame:
    combined = {}
    for _, fc_df in fc_dict.items():
        _fc = fc_df.copy()
        _fc.index = idx
        for col in fc_df.columns:
            combined[col] = _fc[col]
    if "TOTAL" not in combined and "total" in fc_dict:
        _t = fc_dict["total"].copy()
        _t.index = idx
        combined["TOTAL"] = _t.iloc[:, 0]
    df = pd.DataFrame(combined, index=idx)
    return df.reindex(columns=all_series, fill_value=0)


td_results, ols_results, wls_results = {}, {}, {}
for model_name, fc_dict, res_vars in [
    ("ARIMA",   arima_fc,   arima_vars),
    ("Prophet", prophet_fc, prophet_vars),
    ("LSTM",    lstm_fc,    lstm_vars),
]:
    total_fc = fc_dict["total"].copy()
    total_fc.index = test_idx
    td_results[model_name] = reconcile_top_down(total_fc.iloc[:, 0], train_product)

    base = assemble_base_fc(fc_dict, all_series, test_idx)
    ols_results[model_name] = reconcile_mintrace(base, S_df)[product_df.columns]
    wls_results[model_name] = reconcile_mintrace_wls(base, S_df, res_vars)[product_df.columns]

# ─────────────────────────────────────────────────────────────────────
# Stage 3b — Evaluation
# ─────────────────────────────────────────────────────────────────────
print("\n--- Forecast Evaluation ---\n")

rows = []
test_frames = {
    "total": test_total, "department": test_dept,
    "category": test_category, "product": test_product,
}

# Base models (ARIMA / Prophet / LSTM)
for model_name, fc_dict in [("ARIMA", arima_fc), ("Prophet", prophet_fc), ("LSTM", lstm_fc)]:
    for lvl_name, test_df in test_frames.items():
        fc = fc_dict[lvl_name].copy()
        fc.index = test_df.index
        for col in test_df.columns:
            if col in fc.columns:
                rows.append({"model": model_name, "level": lvl_name, "series": col,
                              "RMSE": rmse(test_df[col].values, fc[col].values),
                              "MAE":  mae(test_df[col].values, fc[col].values)})

# GBM base models
if gbm_results:
    for model_name, levels_dict in [("LGBM", lgbm_levels), ("XGBoost", xgb_levels)]:
        for lvl_name, test_df in test_frames.items():
            fc_df = levels_dict.get(lvl_name)
            if fc_df is None:
                continue
            if isinstance(fc_df, pd.Series):
                fc_df = fc_df.to_frame()
            fc_df.index = test_df.index[:len(fc_df)]
            for col in test_df.columns:
                if col in fc_df.columns:
                    rows.append({"model": model_name, "level": lvl_name, "series": col,
                                  "RMSE": rmse(test_df[col].values[:len(fc_df)],
                                               fc_df[col].values),
                                  "MAE":  mae(test_df[col].values[:len(fc_df)],
                                              fc_df[col].values)})

# Reconciled
for rec_label, rec_dict in [
    ("TopDown-ARIMA",   td_results["ARIMA"]),
    ("TopDown-Prophet", td_results["Prophet"]),
    ("TopDown-LSTM",    td_results["LSTM"]),
    ("OLS-ARIMA",       ols_results["ARIMA"]),
    ("OLS-Prophet",     ols_results["Prophet"]),
    ("OLS-LSTM",        ols_results["LSTM"]),
    ("WLS-ARIMA",       wls_results["ARIMA"]),
    ("WLS-Prophet",     wls_results["Prophet"]),
    ("WLS-LSTM",        wls_results["LSTM"]),
]:
    fc = rec_dict.copy()
    fc.index = test_product.index
    for col in test_product.columns:
        if col in fc.columns:
            rows.append({"model": rec_label, "level": "product_reconciled", "series": col,
                          "RMSE": rmse(test_product[col].values, fc[col].values),
                          "MAE":  mae(test_product[col].values, fc[col].values)})

results_df = pd.DataFrame(rows)
forecast_summary = (
    results_df.groupby(["model", "level"])[["RMSE", "MAE"]]
    .mean().round(2).reset_index()
)
print(forecast_summary.to_string(index=False))
results_df.to_csv(RESULTS_DIR / "evaluation_detail.csv", index=False)
forecast_summary.to_csv(RESULTS_DIR / "evaluation_summary.csv", index=False)

# ─────────────────────────────────────────────────────────────────────
# Stage 4 — Bayesian hierarchical elasticity (PyMC)
# ─────────────────────────────────────────────────────────────────────
gc.collect()
print("\n[4/7] Bayesian hierarchical price-elasticity estimation …")
elasticity_results = {}
try:
    from elasticity import run_elasticity
    gt_path = RESULTS_DIR / "ground_truth.json"
    elasticity_results = run_elasticity(
        raw_df[raw_df["date"] <= train_product.index[-1]],
        results_dir=RESULTS_DIR,
        draws=1000, tune=1000, chains=1,
        ground_truth_path=gt_path,
    )
except Exception as e:
    print(f"    [WARN] Elasticity stage skipped: {e}")

# ─────────────────────────────────────────────────────────────────────
# Stage 5 — Causal promo-lift (DiD + A/B)
# ─────────────────────────────────────────────────────────────────────
gc.collect()
print("\n[5/7] Causal promo-lift estimation …")
causal_results = {}
try:
    from causal import run_causal
    causal_results = run_causal(
        raw_df[raw_df["date"] <= train_product.index[-1]],
        results_dir=RESULTS_DIR,
        ground_truth_path=RESULTS_DIR / "ground_truth.json",
    )
except Exception as e:
    print(f"    [WARN] Causal stage skipped: {e}")

# ─────────────────────────────────────────────────────────────────────
# Stage 6 — Cold-start (embeddings + FAISS)
# ─────────────────────────────────────────────────────────────────────
gc.collect()
print("\n[6/7] Cold-start forecasting via description embeddings …")
coldstart_results = {}
try:
    from coldstart import run_coldstart
    coldstart_results = run_coldstart(
        raw_df, descriptions, train_product, test_product,
        results_dir=RESULTS_DIR, horizon=HORIZON,
    )
except Exception as e:
    print(f"    [WARN] Cold-start stage skipped: {e}")

# ─────────────────────────────────────────────────────────────────────
# Stage 7 — Real-world: Online Retail II (optional)
# ─────────────────────────────────────────────────────────────────────
gc.collect()
print("\n[7/7] Real-world validation (Online Retail II) …")
print("    [SKIP] Already completed separately — see results/realworld_*.csv")

# ─────────────────────────────────────────────────────────────────────
# Stage 8 — Summary CSV + plots
# ─────────────────────────────────────────────────────────────────────
print("\n[8/7] Compiling results/summary.csv …")

summary_rows = []

# Forecast RMSE by level
for _, r in forecast_summary.iterrows():
    summary_rows.append({
        "stage":    "forecast",
        "metric":   "RMSE",
        "category": r["level"],
        "method":   r["model"],
        "estimated": r["RMSE"],
        "true":      None,
        "hdi_lo":    None,
        "hdi_hi":    None,
        "notes":     "mean across series",
    })

# Elasticity recovery
if elasticity_results.get("summary") is not None:
    for _, r in elasticity_results["summary"].iterrows():
        summary_rows.append({
            "stage":    "elasticity",
            "metric":   "price_elasticity",
            "category": r["category"],
            "method":   "BayesianHierarchical",
            "estimated": r.get("estimated_beta"),
            "true":      r.get("true_beta"),
            "hdi_lo":    r.get("eti_lo"),
            "hdi_hi":    r.get("eti_hi"),
            "notes":     f"error={r.get('error', '')}",
        })
    summary_rows.append({
        "stage":    "elasticity",
        "metric":   "promo_gamma",
        "category": "all",
        "method":   "BayesianHierarchical",
        "estimated": round(elasticity_results.get("gamma_est", np.nan), 3),
        "true":      TRUE_GAMMA,
        "hdi_lo":    elasticity_results.get("gamma_hdi", (None, None))[0],
        "hdi_hi":    elasticity_results.get("gamma_hdi", (None, None))[1],
        "notes":     "promo lift in log units",
    })

# DiD lift
if causal_results.get("did"):
    d = causal_results["did"]
    summary_rows.append({
        "stage":    "causal",
        "metric":   "promo_lift_pct",
        "category": "all",
        "method":   "DiD",
        "estimated": d.get("lift_pct"),
        "true":      round((np.exp(TRUE_GAMMA) - 1) * 100, 2),
        "hdi_lo":    d.get("ci_lo_pct"),
        "hdi_hi":    d.get("ci_hi_pct"),
        "notes":     f"n_events={d.get('n_events')}",
    })
if causal_results.get("ab_test"):
    a = causal_results["ab_test"]
    summary_rows.append({
        "stage":    "causal",
        "metric":   "promo_lift_pct",
        "category": "all",
        "method":   "AB_test",
        "estimated": a.get("lift_pct"),
        "true":      round((np.exp(TRUE_GAMMA) - 1) * 100, 2),
        "hdi_lo":    a.get("ci_lo_pct"),
        "hdi_hi":    a.get("ci_hi_pct"),
        "notes":     f"p={a.get('p_value')}",
    })

# Cold-start
if coldstart_results.get("rmse_coldstart") is not None:
    summary_rows.append({
        "stage":    "coldstart",
        "metric":   "RMSE",
        "category": "FOOD_1_P5",
        "method":   "embedding_transfer",
        "estimated": coldstart_results["rmse_coldstart"],
        "true":      None,
        "hdi_lo":    None,
        "hdi_hi":    None,
        "notes":     f"vs naive={coldstart_results['rmse_naive']:.2f}",
    })
    summary_rows.append({
        "stage":    "coldstart",
        "metric":   "RMSE",
        "category": "FOOD_1_P5",
        "method":   "naive_category_avg",
        "estimated": coldstart_results["rmse_naive"],
        "true":      None,
        "hdi_lo":    None,
        "hdi_hi":    None,
        "notes":     f"improvement={coldstart_results['improvement_pct']:.1f}%",
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)
print(f"    Saved → {RESULTS_DIR}/summary.csv  ({len(summary_df)} rows)")

# ─────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────
print("\nGenerating plots …")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.patches import Patch

sns.set_theme(style="whitegrid", palette="muted")
COLORS = {
    "ARIMA": "#4C72B0", "Prophet": "#DD8452", "LSTM": "#55A868",
    "LGBM": "#FF6B35", "XGBoost": "#A23B72",
    "TopDown": "#C44E52", "OLS": "#937860", "WLS": "#8172B2",
}

# Plot 1: Total-level forecasts (all base models including GBM)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(train_total.index[-90:], train_total["TOTAL"].iloc[-90:],
        color="black", lw=1.5, label="Actual (train tail)")
ax.plot(test_total.index, test_total["TOTAL"],
        color="black", lw=1.5, ls="--", label="Actual (test)")
for model_name, fc_dict, color in [
    ("ARIMA",   arima_fc,    COLORS["ARIMA"]),
    ("Prophet", prophet_fc,  COLORS["Prophet"]),
    ("LSTM",    lstm_fc,     COLORS["LSTM"]),
]:
    fc = fc_dict["total"].copy()
    fc.index = test_total.index
    ax.plot(fc.index, fc.iloc[:, 0], color=color, lw=2, label=model_name)
if gbm_results:
    for name, lvl_dict, color in [
        ("LGBM",    lgbm_levels, COLORS["LGBM"]),
        ("XGBoost", xgb_levels,  COLORS["XGBoost"]),
    ]:
        fc = lvl_dict["total"].copy()
        fc.index = test_total.index[:len(fc)]
        ax.plot(fc.index, fc.iloc[:, 0], color=color, lw=2, ls="--", label=name)
ax.axvline(test_total.index[0], color="grey", ls=":", lw=1)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.set_title("Total Retail Sales — Model Comparison (28-day horizon)", fontsize=13, fontweight="bold")
ax.set_ylabel("Units Sold")
ax.legend()
plt.tight_layout()
fig.savefig(RESULTS_DIR / "01_total_forecast.png", dpi=150)
plt.close(fig)

# Plot 2: Category-level forecasts
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for i, cat in enumerate(category_df.columns):
    ax = axes[i // 3][i % 3]
    ax.plot(train_category[cat].iloc[-60:], color="black", lw=1.2, label="Train")
    ax.plot(test_category[cat], color="black", lw=1.2, ls="--", label="Actual")
    for model_name, fc_dict, color in [
        ("ARIMA",   arima_fc,   COLORS["ARIMA"]),
        ("Prophet", prophet_fc, COLORS["Prophet"]),
        ("LSTM",    lstm_fc,    COLORS["LSTM"]),
    ]:
        fc = fc_dict["category"].copy()
        fc.index = test_category.index
        ax.plot(fc[cat], color=color, lw=1.5, label=model_name)
    ax.set_title(cat, fontsize=10, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    if i == 0:
        ax.legend(fontsize=8)
fig.suptitle("Category-Level Forecasts — All Models", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(RESULTS_DIR / "02_category_forecasts.png", dpi=150)
plt.close(fig)

# Plot 3: Reconciliation bar chart
prod_model_order = [
    "ARIMA", "Prophet", "LSTM",
    *([m for m in ["LGBM", "XGBoost"] if gbm_results] if gbm_results else []),
    "TopDown-ARIMA", "TopDown-Prophet", "TopDown-LSTM",
    "OLS-ARIMA", "OLS-Prophet", "OLS-LSTM",
    "WLS-ARIMA", "WLS-Prophet", "WLS-LSTM",
]
prod_rmse = (
    results_df[results_df["level"].isin(["product", "product_reconciled"])]
    .groupby("model")["RMSE"].mean()
    .reindex(prod_model_order).dropna()
)
bar_colors = []
for m in prod_rmse.index:
    if "WLS"     in m: bar_colors.append(COLORS["WLS"])
    elif "OLS"   in m: bar_colors.append(COLORS["OLS"])
    elif "TopDown" in m: bar_colors.append(COLORS["TopDown"])
    elif m == "LGBM":    bar_colors.append(COLORS["LGBM"])
    elif m == "XGBoost": bar_colors.append(COLORS["XGBoost"])
    elif m == "ARIMA":   bar_colors.append(COLORS["ARIMA"])
    elif m == "Prophet": bar_colors.append(COLORS["Prophet"])
    else:                bar_colors.append(COLORS["LSTM"])

fig, ax = plt.subplots(figsize=(16, 5))
bars = ax.bar(prod_rmse.index, prod_rmse.values, color=bar_colors, edgecolor="white", lw=0.8)
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
ax.set_ylabel("Mean RMSE (product level)")
ax.set_title("Product-Level Forecast Accuracy: Base vs Reconciliation Methods",
             fontsize=12, fontweight="bold")
legend_elements = [
    Patch(facecolor=COLORS["ARIMA"],    label="ARIMA"),
    Patch(facecolor=COLORS["Prophet"],  label="Prophet"),
    Patch(facecolor=COLORS["LSTM"],     label="LSTM"),
    Patch(facecolor=COLORS["LGBM"],     label="LGBM"),
    Patch(facecolor=COLORS["XGBoost"],  label="XGBoost"),
    Patch(facecolor=COLORS["TopDown"],  label="Top-Down"),
    Patch(facecolor=COLORS["OLS"],      label="MinTrace-OLS"),
    Patch(facecolor=COLORS["WLS"],      label="MinTrace-WLS"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
plt.xticks(rotation=35, ha="right", fontsize=8)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "03_reconciliation_comparison.png", dpi=150)
plt.close(fig)

# Plot 4: WLS improvement heatmap
fig, ax = plt.subplots(figsize=(12, 5))
improvement_rows = {}
for m in ["ARIMA", "Prophet", "LSTM"]:
    base_s = results_df[(results_df["model"] == m) & (results_df["level"] == "product")].set_index("series")["RMSE"]
    wls_s  = results_df[(results_df["model"] == f"WLS-{m}") & (results_df["level"] == "product_reconciled")].set_index("series")["RMSE"]
    merged = base_s.to_frame("base").join(wls_s.rename("wls"))
    merged["pct"] = 100 * (merged["base"] - merged["wls"]) / merged["base"]
    improvement_rows[f"WLS-{m}"] = merged["pct"]
heat_df = pd.DataFrame(improvement_rows).T
sns.heatmap(heat_df[heat_df.columns[:15]], annot=True, fmt=".1f", center=0,
            cmap="RdYlGn", ax=ax, linewidths=0.5,
            cbar_kws={"label": "RMSE improvement % (positive = WLS wins)"})
ax.set_title("MinTrace-WLS: RMSE Improvement % over Base Model", fontsize=11, fontweight="bold")
ax.set_xlabel("Product")
ax.set_ylabel("Model")
plt.tight_layout()
fig.savefig(RESULTS_DIR / "04_wls_improvement_heatmap.png", dpi=150)
plt.close(fig)

# Plot 5: Elasticity recovery chart
if elasticity_results.get("summary") is not None:
    elas_df = elasticity_results["summary"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(elas_df))
    ax.bar(x, elas_df["estimated_beta"], color="#4C72B0", alpha=0.7, label="Estimated (posterior mean)")
    if "true_beta" in elas_df.columns:
        ax.scatter(x, elas_df["true_beta"], color="red", s=80, zorder=5, label="True β")
    if "eti_lo" in elas_df.columns and "eti_hi" in elas_df.columns:
        ax.errorbar(x, elas_df["estimated_beta"],
                    yerr=[elas_df["estimated_beta"] - elas_df["eti_lo"],
                          elas_df["eti_hi"] - elas_df["estimated_beta"]],
                    fmt="none", color="#4C72B0", capsize=5, lw=1.5, label="90% ETI")
    ax.set_xticks(list(x))
    ax.set_xticklabels(elas_df["category"].tolist())
    ax.set_ylabel("Price Elasticity (β)")
    ax.set_title("Bayesian Hierarchical Elasticity: Estimated vs True", fontsize=12, fontweight="bold")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.legend()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "05_elasticity_recovery.png", dpi=150)
    plt.close(fig)

# Plot 6: Cold-start forecast vs actuals
if coldstart_results.get("forecast") is not None:
    fig, ax = plt.subplots(figsize=(12, 5))
    idx = test_product.index[:HORIZON]
    if coldstart_results.get("actuals") is not None:
        ax.plot(idx, coldstart_results["actuals"][:HORIZON], color="black", lw=2, label="Actual")
    ax.plot(idx, coldstart_results["forecast"][:HORIZON], color="#4C72B0", lw=2,
            label=f"Cold-start (RMSE={coldstart_results['rmse_coldstart']:.1f})")
    ax.plot(idx, coldstart_results["naive"][:HORIZON], color="#DD8452", lw=2, ls="--",
            label=f"Naive baseline (RMSE={coldstart_results['rmse_naive']:.1f})")
    ax.set_title(f"Cold-start Forecast — {coldstart_results['holdout']}\n"
                 f"Neighbors: {', '.join(coldstart_results['neighbors'][:3])}",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Units Sold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "06_coldstart_forecast.png", dpi=150)
    plt.close(fig)

# Plot 7: Pricing scenarios (if available)
if elasticity_results.get("scenarios") is not None:
    scen = elasticity_results["scenarios"]
    fig, ax = plt.subplots(figsize=(10, 5))
    cats = scen["category"].unique()
    x = np.arange(len(cats))
    width = 0.35
    rev_up   = scen[scen["price_change_pct"] == 10.0]["revenue_change_pct"].values
    rev_down = scen[scen["price_change_pct"] == -10.0]["revenue_change_pct"].values
    if len(rev_up) == len(cats) and len(rev_down) == len(cats):
        ax.bar(x - width/2, rev_down, width, label="Price −10%", color="#4C72B0", alpha=0.8)
        ax.bar(x + width/2, rev_up,   width, label="Price +10%", color="#DD8452", alpha=0.8)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylabel("Revenue Change %")
    ax.set_title("Revenue Impact of ±10% Price Change by Category", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "07_pricing_scenarios.png", dpi=150)
    plt.close(fig)

print(f"\nAll plots saved to {RESULTS_DIR}/")
print("\n" + "=" * 65)
print(" Pipeline complete.")
print("=" * 65)
