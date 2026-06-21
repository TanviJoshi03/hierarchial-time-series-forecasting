"""
Bayesian hierarchical price-elasticity estimation via PyMC.

Model (partial pooling across categories):
  log(units_it) = alpha[product_i]
                + beta_cat[cat_i] * log(price_it / ref_price_i)   ← elasticity
                + gamma * promo_it                                  ← promo control
                + dow_effect[dow_t]                                 ← controls
                + month_effect[month_t]
                + N(0, sigma)

After sampling:
  - Compare posterior beta_cat vs TRUE_BETAS (recovery check)
  - Revenue-optimal pricing scenario: ±10% price move per category
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ── Data preparation ──────────────────────────────────────────────────────────

def _prepare_elasticity_data(df: pd.DataFrame) -> dict:
    """
    Convert long-form retail df to arrays for PyMC.
    Uses ALL data (promo as control variable) to maximise information.
    """
    # Work in log space; need a reference price per product
    ref_prices = df.groupby("product")["price"].median()

    df = df.copy()
    df["log_units"] = np.log(np.maximum(df["sales"], 0.1))
    df["log_price_ratio"] = np.log(df["price"] / df["product"].map(ref_prices))
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month - 1  # 0-indexed

    products = sorted(df["product"].unique())
    categories = sorted(df["category"].unique())
    prod2idx = {p: i for i, p in enumerate(products)}
    cat2idx  = {c: i for i, c in enumerate(categories)}

    # Product → category mapping
    prod_cat = df[["product", "category"]].drop_duplicates().set_index("product")["category"]

    return {
        "log_units":       df["log_units"].values.astype(np.float64),
        "log_price_ratio": df["log_price_ratio"].values.astype(np.float64),
        "promo":           df["promo"].values.astype(np.float64),
        "dow":             df["dow"].values.astype(int),
        "month":           df["month"].values.astype(int),
        "prod_idx":        np.array([prod2idx[p] for p in df["product"]], dtype=int),
        "cat_idx":         np.array([cat2idx[prod_cat[p]] for p in df["product"]], dtype=int),
        "n_prod":          len(products),
        "n_cat":           len(categories),
        "products":        products,
        "categories":      categories,
        "cat2idx":         cat2idx,
    }


# ── PyMC model ────────────────────────────────────────────────────────────────

def fit_elasticity_model(data: dict, draws: int = 1000, tune: int = 1000, chains: int = 2):
    """Fit the hierarchical elasticity model and return inference data."""
    import pymc as pm

    with pm.Model() as model:
        # Hyperpriors for category elasticities
        mu_beta    = pm.Normal("mu_beta", mu=-1.0, sigma=0.5)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.5)
        beta_cat   = pm.Normal("beta_cat", mu=mu_beta, sigma=sigma_beta,
                               shape=data["n_cat"])

        # Product baselines
        alpha = pm.Normal("alpha", mu=0.0, sigma=5.0, shape=data["n_prod"])

        # Promo control
        gamma = pm.Normal("gamma", mu=0.0, sigma=1.0)

        # Calendar controls (absorb DOW + monthly seasonality)
        dow_eff   = pm.Normal("dow_effect", mu=0.0, sigma=0.5, shape=7)
        month_eff = pm.Normal("month_effect", mu=0.0, sigma=0.5, shape=12)

        # Likelihood
        sigma = pm.HalfNormal("sigma", sigma=0.3)

        mu = (alpha[data["prod_idx"]]
              + beta_cat[data["cat_idx"]] * data["log_price_ratio"]
              + gamma * data["promo"]
              + dow_eff[data["dow"]]
              + month_eff[data["month"]])

        pm.Normal("obs", mu=mu, sigma=sigma, observed=data["log_units"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(
                draws=draws, tune=tune, chains=chains,
                target_accept=0.9, random_seed=42, progressbar=True,
            )
    return model, idata


# ── Pricing scenarios ─────────────────────────────────────────────────────────

def pricing_scenarios(
    idata,
    categories: list[str],
    price_changes: tuple[float, float] = (-0.10, +0.10),
) -> pd.DataFrame:
    """
    For ±10% price moves, compute expected %Δdemand and %Δrevenue per category.
    Revenue-optimal direction: inelastic (|β| < 1) → raise price; elastic → lower.
    """
    import arviz as az
    summary = az.summary(idata, var_names=["beta_cat"], ci_prob=0.9)
    # ArviZ 1.x may return numeric columns as dtype object — cast to float
    summary["mean"] = pd.to_numeric(summary["mean"], errors="coerce")
    betas = summary["mean"].values.astype(float)   # shape (n_cat,)

    rows = []
    for i, cat in enumerate(categories):
        b = betas[i]
        for dp in price_changes:
            # exact: demand_ratio = (1+dp)^beta
            demand_ratio = (1 + dp) ** b - 1
            revenue_ratio = (1 + dp) * (1 + demand_ratio) - 1
            rows.append({
                "category": cat,
                "price_change_pct": round(dp * 100, 0),
                "beta_posterior_mean": round(b, 3),
                "demand_change_pct": round(demand_ratio * 100, 2),
                "revenue_change_pct": round(revenue_ratio * 100, 2),
            })
    df = pd.DataFrame(rows)
    # Mark optimal direction per category
    def optimal(grp):
        best = grp.loc[grp["revenue_change_pct"].idxmax()]
        grp["optimal_direction"] = (grp["price_change_pct"] == best["price_change_pct"])
        return grp
    return df.groupby("category", group_keys=False).apply(optimal)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_elasticity(
    df: pd.DataFrame,
    results_dir: Path,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    ground_truth_path: Path | None = None,
) -> dict:
    """
    Full elasticity pipeline: prep → sample → compare to ground truth → scenarios.

    Parameters
    ----------
    df : long-form DataFrame with [date, product, category, sales, price, promo]
    results_dir : where to write elasticity_summary.csv + scenarios.csv

    Returns
    -------
    dict with keys: 'summary', 'scenarios', 'idata'
    """
    import arviz as az

    print("    Preparing data …")
    data = _prepare_elasticity_data(df)

    print(f"    Fitting PyMC model  ({len(data['log_units']):,} obs, "
          f"{data['n_cat']} categories, {data['n_prod']} products) …")
    _, idata = fit_elasticity_model(data, draws=draws, tune=tune, chains=chains)

    # Posterior summary for beta_cat  (ArviZ 1.x uses ci_prob, not hdi_prob)
    summary = az.summary(idata, var_names=["beta_cat", "gamma"], ci_prob=0.9)
    cats = data["categories"]

    # ArviZ 1.x: shaped vars indexed as "beta_cat[0]", "beta_cat[1]", ...
    # dtype may be 'O' (object) in ArviZ 1.2 — cast numeric columns
    numeric_cols = ["mean", "sd", "mcse_mean", "mcse_sd"]
    for c in summary.columns:
        if c not in ("ess_bulk", "ess_tail", "r_hat"):
            summary[c] = pd.to_numeric(summary[c], errors="coerce")

    beta_rows = summary.loc[summary.index.str.startswith("beta_cat")].copy()
    beta_rows.index = cats

    # Column names in ArviZ 1.x: eti90_lb / eti90_ub (not hdi_3% / hdi_97%)
    lo_col = [c for c in beta_rows.columns if c.endswith("_lb")][0]
    hi_col = [c for c in beta_rows.columns if c.endswith("_ub")][0]

    gamma_row = summary.loc[summary.index.str.startswith("gamma")]

    # Load ground truth if available
    gt_betas, gt_gamma = {}, None
    if ground_truth_path and Path(ground_truth_path).exists():
        with open(ground_truth_path) as f:
            gt = json.load(f)
        gt_betas = gt.get("true_betas", {})
        gt_gamma = gt.get("true_gamma")

    # Build comparison table
    recovery_rows = []
    for cat in cats:
        row = beta_rows.loc[cat]
        r = {
            "category": cat,
            "estimated_beta": round(row["mean"], 3),
            "eti_lo":  round(row[lo_col], 3),
            "eti_hi":  round(row[hi_col], 3),
        }
        if cat in gt_betas:
            r["true_beta"] = gt_betas[cat]
            r["error"] = round(row["mean"] - gt_betas[cat], 3)
        recovery_rows.append(r)

    gamma_est = float(gamma_row["mean"].iloc[0])
    gamma_hdi_lo = float(gamma_row[lo_col].iloc[0])
    gamma_hdi_hi = float(gamma_row[hi_col].iloc[0])

    recovery_df = pd.DataFrame(recovery_rows)

    # Pricing scenarios
    scenarios_df = pricing_scenarios(idata, cats)

    # Save
    results_dir.mkdir(exist_ok=True)
    recovery_df.to_csv(results_dir / "elasticity_recovery.csv", index=False)
    scenarios_df.to_csv(results_dir / "pricing_scenarios.csv", index=False)

    print("\n    ── Elasticity Recovery ──")
    print(recovery_df.to_string(index=False))
    if gt_gamma is not None:
        print(f"\n    gamma (promo lift): estimated={gamma_est:.3f}  "
              f"90%ETI=[{gamma_hdi_lo:.3f}, {gamma_hdi_hi:.3f}]  true={gt_gamma}")
    print("\n    ── Pricing Scenarios (±10% price) ──")
    print(scenarios_df[["category","price_change_pct","demand_change_pct",
                         "revenue_change_pct","optimal_direction"]].to_string(index=False))

    return {
        "summary": recovery_df,
        "gamma_est": gamma_est,
        "gamma_hdi": (gamma_hdi_lo, gamma_hdi_hi),
        "scenarios": scenarios_df,
        "idata": idata,
    }
