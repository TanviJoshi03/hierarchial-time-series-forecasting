"""
Causal promo-lift estimation via Difference-in-Differences and A/B test.

DiD design
----------
For each promo event (product p, window [s, e]):
  - Pre-period: [s-14, s-1]
  - Post-period: [s, e]
  - Treated unit: product p
  - Controls: same-category products with NO promo in [s-7, e+7]

Regression:
  log_units ~ treat + post + treat:post + C(product) + C(event_week)
  clustered SEs on product

The treat:post coefficient → exp(coef) - 1 = % causal lift

A/B framing
-----------
Treatment group: all (product, day) pairs with promo=1
Control group: same product's non-promo days (matched by rolling window)
Welch t-test + 95% CI
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


# ── Promo window detection ────────────────────────────────────────────────────

def _find_promo_windows(promo_series: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return list of (start_date, end_date) for each contiguous promo window."""
    windows = []
    in_promo = False
    start_dt = None
    for dt, val in promo_series.items():
        if val > 0 and not in_promo:
            start_dt = dt
            in_promo = True
        elif val == 0 and in_promo:
            windows.append((start_dt, dt - pd.Timedelta("1D")))
            in_promo = False
    if in_promo:
        windows.append((start_dt, promo_series.index[-1]))
    return windows


# ── Event panel construction ──────────────────────────────────────────────────

def _build_event_panel(
    df: pd.DataFrame,
    pre_days: int = 14,
    post_buffer: int = 7,
    max_events: int = 200,
) -> pd.DataFrame:
    """
    Build stacked event-study panel for DiD regression.

    Each promo event contributes rows for [treated + controls] × [pre + post window].
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["log_units"] = np.log(np.maximum(df["sales"], 0.1))

    # Build per-product promo schedule
    promo_schedule: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for prod, grp in df.groupby("product"):
        series = grp.set_index("date")["promo"].sort_index()
        promo_schedule[prod] = _find_promo_windows(series)

    # Category → products mapping for control selection
    cat_products: dict[str, list[str]] = {}
    for prod, cat in df[["product", "category"]].drop_duplicates().values:
        cat_products.setdefault(cat, []).append(prod)

    df_indexed = df.set_index(["product", "date"])

    records = []
    event_counter = 0

    for prod, windows in promo_schedule.items():
        cat = df.loc[df["product"] == prod, "category"].iloc[0]
        all_promo_dates = set()
        for p2, wins in promo_schedule.items():
            for s, e in wins:
                all_promo_dates.update(pd.date_range(s, e))

        for win_start, win_end in windows:
            if event_counter >= max_events:
                break

            pre_start  = win_start - pd.Timedelta(days=pre_days)
            post_end   = win_end
            buffer_lo  = win_start - pd.Timedelta(days=post_buffer)
            buffer_hi  = win_end   + pd.Timedelta(days=post_buffer)

            # Controls: same category, no promo overlap in buffered window
            controls = []
            for p2 in cat_products[cat]:
                if p2 == prod:
                    continue
                conflict = any(
                    not (e2 < buffer_lo or s2 > buffer_hi)
                    for s2, e2 in promo_schedule.get(p2, [])
                )
                if not conflict:
                    controls.append(p2)

            if not controls:
                continue

            window_dates = pd.date_range(pre_start, post_end, freq="D")
            event_id = f"E{event_counter:04d}"
            event_counter += 1

            for date in window_dates:
                is_post = int(date >= win_start)
                event_week = (date - pre_start).days // 7

                # Treated
                key = (prod, date)
                if key in df_indexed.index:
                    row = df_indexed.loc[key].squeeze()
                    records.append({
                        "event_id":   event_id,
                        "product":    prod,
                        "category":   cat,
                        "date":       date,
                        "log_units":  float(row["log_units"]),
                        "log_price":  float(np.log(max(float(row.get("price", 1.0)), 0.01))),
                        "treat":      1,
                        "post":       is_post,
                        "event_week": event_week,
                        "week":       date.isocalendar()[1],
                    })

                # Controls
                for ctrl in controls:
                    key = (ctrl, date)
                    if key in df_indexed.index:
                        row = df_indexed.loc[key].squeeze()
                        records.append({
                            "event_id":   event_id,
                            "product":    ctrl,
                            "category":   cat,
                            "date":       date,
                            "log_units":  float(row["log_units"]),
                            "log_price":  float(np.log(max(float(row.get("price", 1.0)), 0.01))),
                            "treat":      0,
                            "post":       is_post,
                            "event_week": event_week,
                            "week":       date.isocalendar()[1],
                        })

    return pd.DataFrame(records)


# ── DiD regression ────────────────────────────────────────────────────────────

def run_did(event_df: pd.DataFrame) -> dict:
    """
    OLS diff-in-diff with product + event-week FEs, clustered SEs on product.

    Runs two specifications:
      - raw: total observed lift (price reduction + promotional effect)
      - price_controlled: adds log_price as covariate → isolates pure gamma
    """
    def _fit_and_extract(formula, df):
        m = smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["product"]},
        )
        coef  = m.params.get("treat:post", np.nan)
        se    = m.bse.get("treat:post", np.nan)
        pval  = m.pvalues.get("treat:post", np.nan)
        return coef, se, pval

    # Raw DiD: total promo effect (includes price discount)
    coef_raw, se_raw, pval_raw = _fit_and_extract(
        "log_units ~ treat + post + treat:post + C(product) + C(event_week)", event_df
    )
    ci_lo_raw = coef_raw - 1.96 * se_raw
    ci_hi_raw = coef_raw + 1.96 * se_raw

    # Price-controlled DiD: isolates direct promotional effect (≈ gamma)
    price_coef = np.nan
    price_ci_lo = price_ci_hi = price_pval = np.nan
    if "log_price" in event_df.columns:
        coef_pc, se_pc, pval_pc = _fit_and_extract(
            "log_units ~ treat + post + treat:post + log_price + C(product) + C(event_week)",
            event_df,
        )
        price_ci_lo = coef_pc - 1.96 * se_pc
        price_ci_hi = coef_pc + 1.96 * se_pc
        price_coef = coef_pc
        price_pval = pval_pc

    return {
        # Total observed lift (raw DiD)
        "coef_log":   round(coef_raw, 4),
        "lift_pct":   round((np.exp(coef_raw) - 1) * 100, 2),
        "ci_lo_pct":  round((np.exp(ci_lo_raw) - 1) * 100, 2),
        "ci_hi_pct":  round((np.exp(ci_hi_raw) - 1) * 100, 2),
        "p_value":    round(pval_raw, 4),
        # Price-controlled (≈ pure gamma)
        "coef_log_price_ctrl":  round(price_coef, 4) if not np.isnan(price_coef) else None,
        "lift_pct_price_ctrl":  round((np.exp(price_coef) - 1) * 100, 2) if not np.isnan(price_coef) else None,
        "ci_lo_price_ctrl":     round((np.exp(price_ci_lo) - 1) * 100, 2) if not np.isnan(price_ci_lo) else None,
        "ci_hi_price_ctrl":     round((np.exp(price_ci_hi) - 1) * 100, 2) if not np.isnan(price_ci_hi) else None,
        "p_value_price_ctrl":   round(float(price_pval), 4) if not np.isnan(price_pval) else None,
        "n_events":   event_df["event_id"].nunique(),
        "n_obs":      len(event_df),
    }


# ── Simple A/B test ───────────────────────────────────────────────────────────

def run_ab_test(df: pd.DataFrame) -> dict:
    """
    A/B framing: promo=1 (treatment) vs promo=0 (control) log-units.
    Welch t-test + 95% CI on the mean difference.
    """
    df = df.copy()
    df["log_units"] = np.log(np.maximum(df["sales"], 0.1))

    treatment = df.loc[df["promo"] == 1, "log_units"].values
    control   = df.loc[df["promo"] == 0, "log_units"].values

    t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=False)
    diff = treatment.mean() - control.mean()
    se = np.sqrt(treatment.var(ddof=1) / len(treatment) + control.var(ddof=1) / len(control))

    return {
        "mean_diff_log":    round(diff, 4),
        "lift_pct":         round((np.exp(diff) - 1) * 100, 2),
        "ci_lo_pct":        round((np.exp(diff - 1.96 * se) - 1) * 100, 2),
        "ci_hi_pct":        round((np.exp(diff + 1.96 * se) - 1) * 100, 2),
        "t_stat":           round(t_stat, 3),
        "p_value":          round(p_val, 6),
        "n_treatment_obs":  len(treatment),
        "n_control_obs":    len(control),
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def run_causal(
    df: pd.DataFrame,
    results_dir: Path,
    true_gamma: float | None = None,
    ground_truth_path: Path | None = None,
) -> dict:
    """
    Run DiD + A/B promo-lift estimation and write results.

    Parameters
    ----------
    df : long-form DataFrame with [date, product, category, sales, promo]
    results_dir : output directory

    Returns
    -------
    dict with 'did', 'ab_test' result dicts
    """
    if ground_truth_path and Path(ground_truth_path).exists():
        import json
        with open(ground_truth_path) as f:
            gt = json.load(f)
        true_gamma = gt.get("true_gamma", true_gamma)

    print("    Building event-study panel …", end=" ", flush=True)
    event_df = _build_event_panel(df)
    print(f"{event_df['event_id'].nunique()} events, {len(event_df):,} rows")

    if len(event_df) == 0:
        print("    WARNING: no events found — skipping causal estimation")
        return {"did": {}, "ab_test": {}}

    print("    Running diff-in-diff regression …", end=" ", flush=True)
    did_result = run_did(event_df)
    print("done")

    print("    Running A/B test …", end=" ", flush=True)
    ab_result = run_ab_test(df)
    print("done")

    # Save
    results_dir.mkdir(exist_ok=True)
    event_df.to_csv(results_dir / "did_event_panel.csv", index=False)

    summary = pd.DataFrame([
        {"method": "DiD",  **did_result},
        {"method": "A/B",  **ab_result},
    ])
    summary.to_csv(results_dir / "causal_summary.csv", index=False)

    print("\n    ── Causal Lift Estimates ──")
    true_str = f"  (true γ lift={round((np.exp(true_gamma)-1)*100,2)}%)" if true_gamma else ""
    print(f"    DiD (raw, incl. price)    : lift={did_result['lift_pct']:.2f}%  "
          f"95%CI=[{did_result['ci_lo_pct']:.2f}%, {did_result['ci_hi_pct']:.2f}%]")
    if did_result.get("lift_pct_price_ctrl") is not None:
        print(f"    DiD (price-controlled ≈ γ): lift={did_result['lift_pct_price_ctrl']:.2f}%  "
              f"95%CI=[{did_result['ci_lo_price_ctrl']:.2f}%, {did_result['ci_hi_price_ctrl']:.2f}%]"
              f"{true_str}")
    print(f"    A/B (raw)                 : lift={ab_result['lift_pct']:.2f}%  "
          f"95%CI=[{ab_result['ci_lo_pct']:.2f}%, {ab_result['ci_hi_pct']:.2f}%]")

    return {"did": did_result, "ab_test": ab_result, "event_df": event_df}
