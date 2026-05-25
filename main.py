"""
Hierarchical Time Series Forecasting — main pipeline.

Steps:
  1. Generate synthetic M5-style retail data
  2. Split train / test (last 28 days held out)
  3. Fit ARIMA, Prophet, LSTM on all hierarchy levels
  4. Reconcile with Top-Down, MinTrace-OLS, and MinTrace-WLS
  5. Evaluate and compare all models + reconciliation methods
  6. Save results and plots
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

from data_generator import generate_dataset, build_hierarchical_df
from models.arima_model import forecast_all_arima
from models.prophet_model import forecast_all_prophet
from models.lstm_model import forecast_all_lstm
from reconciliation import (
    build_summing_matrix,
    reconcile_top_down,
    reconcile_mintrace,
    reconcile_mintrace_wls,
    compute_residual_variances,
    evaluate_level,
    rmse, mae,
)

HORIZON = 28
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print(" Hierarchical Time Series Forecasting Pipeline")
print("=" * 60)

# ── 1. Data ────────────────────────────────────────────────────────────────
print("\n[1/5] Generating synthetic retail dataset …")
raw_df = generate_dataset()
levels = build_hierarchical_df(raw_df)

product_df   = levels["product"]
category_df  = levels["category"]
dept_df      = levels["department"]
total_df     = levels["total"]

train_product  = product_df.iloc[:-HORIZON]
test_product   = product_df.iloc[-HORIZON:]
train_category = category_df.iloc[:-HORIZON]
test_category  = category_df.iloc[-HORIZON:]
train_dept     = dept_df.iloc[:-HORIZON]
test_dept      = dept_df.iloc[-HORIZON:]
train_total    = total_df.iloc[:-HORIZON]
test_total     = total_df.iloc[-HORIZON:]

print(f"    Train: {train_product.index[0].date()} → {train_product.index[-1].date()}")
print(f"    Test : {test_product.index[0].date()} → {test_product.index[-1].date()}")
print(f"    Products: {product_df.shape[1]} | Categories: {category_df.shape[1]} | Depts: {dept_df.shape[1]}")

# ── 2. Hierarchy metadata ──────────────────────────────────────────────────
category_map = {}
for col in product_df.columns:
    cat = "_".join(col.split("_")[:-1])
    category_map[col] = cat

dept_map = {}
for cat in category_df.columns:
    dept = cat.split("_")[0]
    dept_map[cat] = "HOUSEHOLD" if dept == "HH" else "FOOD"

S_df, all_series = build_summing_matrix(category_map, dept_map)

# All training actuals keyed by series name (for residual variance computation)
all_train = {}
all_train["TOTAL"] = train_total["TOTAL"]
for c in train_dept.columns:
    all_train[c] = train_dept[c]
for c in train_category.columns:
    all_train[c] = train_category[c]
for c in train_product.columns:
    all_train[c] = train_product[c]
all_train_df = pd.DataFrame(all_train)

# ── 3. Forecasting ─────────────────────────────────────────────────────────
LEVELS = [
    ("total",      train_total),
    ("department", train_dept),
    ("category",   train_category),
    ("product",    train_product),
]

print("\n[2/5] Fitting ARIMA on all hierarchy levels …")
arima_fc, arima_fitted = {}, {}
for lvl_name, train_df in LEVELS:
    print(f"    {lvl_name} ({train_df.shape[1]} series) …", end=" ", flush=True)
    fc, fitted = forecast_all_arima(train_df, horizon=HORIZON)
    arima_fc[lvl_name] = fc
    for col in fitted.columns:
        arima_fitted[col if col != "TOTAL" else "TOTAL"] = fitted[col]
    print("done")

print("\n[3/5] Fitting Prophet on all hierarchy levels …")
prophet_fc, prophet_fitted = {}, {}
for lvl_name, train_df in LEVELS:
    print(f"    {lvl_name} ({train_df.shape[1]} series) …", end=" ", flush=True)
    fc, fitted = forecast_all_prophet(train_df, horizon=HORIZON)
    prophet_fc[lvl_name] = fc
    for col in fitted.columns:
        prophet_fitted[col] = fitted[col]
    print("done")

print("\n[4/5] Fitting LSTM on all hierarchy levels …")
lstm_fc, lstm_fitted = {}, {}
for lvl_name, train_df in LEVELS:
    print(f"    {lvl_name} ({train_df.shape[1]} series) …", end=" ", flush=True)
    fc, fitted = forecast_all_lstm(train_df, horizon=HORIZON, epochs=50)
    lstm_fc[lvl_name] = fc
    for col in fitted.columns:
        lstm_fitted[col] = fitted[col]
    print("done")

# ── 4. Reconciliation ──────────────────────────────────────────────────────
print("\n[5/5] Hierarchical reconciliation (Top-Down + OLS + WLS) …")

test_idx = test_product.index

# Residual variances per series for each model
arima_vars   = compute_residual_variances(all_train_df, arima_fitted)
prophet_vars = compute_residual_variances(all_train_df, prophet_fitted)
lstm_vars    = compute_residual_variances(all_train_df, lstm_fitted)

def assemble_base_fc(fc_dict: dict, all_series: list, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Stack per-level forecast dicts into one wide DataFrame of all series."""
    combined = {}
    for lvl_name, fc_df in fc_dict.items():
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

# Top-Down (total forecast → products via historical shares)
td_results = {}
for model_name, fc_dict in [("ARIMA", arima_fc), ("Prophet", prophet_fc), ("LSTM", lstm_fc)]:
    total_fc = fc_dict["total"].copy()
    total_fc.index = test_idx
    td_results[model_name] = reconcile_top_down(total_fc.iloc[:, 0], train_product)

# OLS MinTrace (baseline — ignores error covariance)
ols_results = {}
for model_name, fc_dict in [("ARIMA", arima_fc), ("Prophet", prophet_fc), ("LSTM", lstm_fc)]:
    base = assemble_base_fc(fc_dict, all_series, test_idx)
    rec = reconcile_mintrace(base, S_df)
    ols_results[model_name] = rec[product_df.columns]

# WLS MinTrace (Wickramasuriya et al. 2019 — weights by residual variance)
wls_results = {}
for model_name, fc_dict, res_vars in [
    ("ARIMA",   arima_fc,   arima_vars),
    ("Prophet", prophet_fc, prophet_vars),
    ("LSTM",    lstm_fc,    lstm_vars),
]:
    base = assemble_base_fc(fc_dict, all_series, test_idx)
    rec = reconcile_mintrace_wls(base, S_df, res_vars)
    wls_results[model_name] = rec[product_df.columns]

# ── 5. Evaluation ──────────────────────────────────────────────────────────
print("\n--- Evaluation Results ---\n")

rows = []
test_frames = {
    "total": test_total, "department": test_dept,
    "category": test_category, "product": test_product,
}

for model_name, fc_dict in [("ARIMA", arima_fc), ("Prophet", prophet_fc), ("LSTM", lstm_fc)]:
    for lvl_name, test_df in test_frames.items():
        fc = fc_dict[lvl_name].copy()
        fc.index = test_df.index
        for col in test_df.columns:
            if col in fc.columns:
                rows.append({"model": model_name, "level": lvl_name, "series": col,
                              "RMSE": rmse(test_df[col].values, fc[col].values),
                              "MAE":  mae(test_df[col].values, fc[col].values)})

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
summary = (results_df.groupby(["model", "level"])[["RMSE", "MAE"]]
           .mean().round(2).reset_index())

print(summary.to_string(index=False))
results_df.to_csv(RESULTS_DIR / "evaluation_detail.csv", index=False)
summary.to_csv(RESULTS_DIR / "evaluation_summary.csv", index=False)

# ── 6. Visualizations ─────────────────────────────────────────────────────
print("\nGenerating plots …")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
COLORS = {
    "ARIMA": "#4C72B0", "Prophet": "#DD8452", "LSTM": "#55A868",
    "TopDown": "#C44E52", "OLS": "#937860", "WLS": "#8172B2",
}

# Plot 1: Total-level forecasts
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(train_total.index[-90:], train_total["TOTAL"].iloc[-90:],
        color="black", lw=1.5, label="Actual (train tail)")
ax.plot(test_total.index, test_total["TOTAL"],
        color="black", lw=1.5, ls="--", label="Actual (test)")
for model_name, fc_dict, color in [
    ("ARIMA",   arima_fc,   COLORS["ARIMA"]),
    ("Prophet", prophet_fc, COLORS["Prophet"]),
    ("LSTM",    lstm_fc,    COLORS["LSTM"]),
]:
    fc = fc_dict["total"].copy()
    fc.index = test_total.index
    ax.plot(fc.index, fc.iloc[:, 0], color=color, lw=2, label=model_name)
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

# Plot 3: Reconciliation comparison bar chart (all methods)
prod_model_order = [
    "ARIMA", "Prophet", "LSTM",
    "TopDown-ARIMA", "TopDown-Prophet", "TopDown-LSTM",
    "OLS-ARIMA", "OLS-Prophet", "OLS-LSTM",
    "WLS-ARIMA", "WLS-Prophet", "WLS-LSTM",
]
prod_rmse = (
    results_df[results_df["level"].isin(["product", "product_reconciled"])]
    .groupby("model")["RMSE"].mean()
    .reindex(prod_model_order)
    .dropna()
)

bar_colors = []
for m in prod_rmse.index:
    if "WLS" in m:       bar_colors.append(COLORS["WLS"])
    elif "OLS" in m:     bar_colors.append(COLORS["OLS"])
    elif "TopDown" in m: bar_colors.append(COLORS["TopDown"])
    elif m == "ARIMA":   bar_colors.append(COLORS["ARIMA"])
    elif m == "Prophet": bar_colors.append(COLORS["Prophet"])
    else:                bar_colors.append(COLORS["LSTM"])

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(prod_rmse.index, prod_rmse.values, color=bar_colors,
              edgecolor="white", linewidth=0.8)
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
ax.set_ylabel("Mean RMSE (product level)")
ax.set_title(
    "Product-Level Forecast Accuracy:\nBase vs Top-Down vs MinTrace-OLS vs MinTrace-WLS",
    fontsize=12, fontweight="bold",
)
# legend patches
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS["ARIMA"],   label="Base models"),
    Patch(facecolor=COLORS["TopDown"], label="Top-Down"),
    Patch(facecolor=COLORS["OLS"],     label="MinTrace-OLS"),
    Patch(facecolor=COLORS["WLS"],     label="MinTrace-WLS (optimal)"),
]
ax.legend(handles=legend_elements, loc="upper right")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
fig.savefig(RESULTS_DIR / "03_reconciliation_comparison.png", dpi=150)
plt.close(fig)

# Plot 4: WLS vs OLS vs Base improvement heatmap
fig, ax = plt.subplots(figsize=(12, 5))
improvement_rows = {}
for m in ["ARIMA", "Prophet", "LSTM"]:
    base_s  = results_df[(results_df["model"] == m) & (results_df["level"] == "product")].set_index("series")["RMSE"]
    wls_s   = results_df[(results_df["model"] == f"WLS-{m}") & (results_df["level"] == "product_reconciled")].set_index("series")["RMSE"]
    merged  = base_s.to_frame("base").join(wls_s.rename("wls"))
    merged["pct"] = 100 * (merged["base"] - merged["wls"]) / merged["base"]
    improvement_rows[f"WLS-{m}"] = merged["pct"]

heat_df = pd.DataFrame(improvement_rows).T
sample_cols = heat_df.columns[:15]
sns.heatmap(heat_df[sample_cols], annot=True, fmt=".1f", center=0,
            cmap="RdYlGn", ax=ax, linewidths=0.5,
            cbar_kws={"label": "RMSE improvement % (positive = WLS wins)"})
ax.set_title(
    "MinTrace-WLS: RMSE Improvement % over Base Model\n(positive = WLS reconciliation beats independent base forecast)",
    fontsize=11, fontweight="bold",
)
ax.set_xlabel("Product")
ax.set_ylabel("Model")
plt.tight_layout()
fig.savefig(RESULTS_DIR / "04_wls_improvement_heatmap.png", dpi=150)
plt.close(fig)

# Plot 5: Hierarchy cascade (Prophet base + WLS reconciled)
sample_prod = product_df.columns[0]
sample_cat  = category_map[sample_prod]
sample_dept = dept_map[sample_cat]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
pairs = [
    (axes[0][0], "TOTAL",       train_total["TOTAL"],    test_total["TOTAL"],
     prophet_fc["total"].copy(), test_total.index, None),
    (axes[0][1], sample_dept,   train_dept[sample_dept], test_dept[sample_dept],
     prophet_fc["department"].copy(), test_dept.index, None),
    (axes[1][0], sample_cat,    train_category[sample_cat], test_category[sample_cat],
     prophet_fc["category"].copy(), test_category.index, None),
    (axes[1][1], sample_prod,   train_product[sample_prod], test_product[sample_prod],
     prophet_fc["product"].copy(), test_product.index, wls_results["Prophet"][sample_prod]),
]
for ax, label, tr, te, fc_df, idx, wls_col in pairs:
    fc_df.index = idx
    fc_col = fc_df[label] if label in fc_df.columns else fc_df.iloc[:, 0]
    ax.plot(tr.iloc[-60:], color="black", lw=1.2, label="Train")
    ax.plot(te, color="black", lw=1.2, ls="--", label="Actual")
    ax.plot(idx, fc_col.values, color=COLORS["Prophet"], lw=2, label="Prophet (base)")
    if wls_col is not None:
        ax.plot(idx, wls_col.values, color=COLORS["WLS"], lw=2, ls="-.", label="WLS reconciled")
    ax.axvline(idx[0], color="grey", ls=":", lw=1)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(fontsize=8)
fig.suptitle(
    "Hierarchy Cascade: Total → Department → Category → Product\n"
    "(Prophet base forecasts; product panel shows WLS reconciliation)",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "05_hierarchy_cascade.png", dpi=150)
plt.close(fig)

print(f"\nAll plots saved to {RESULTS_DIR}/")
print("\n" + "=" * 60)
print(" Pipeline complete.")
print("=" * 60)
