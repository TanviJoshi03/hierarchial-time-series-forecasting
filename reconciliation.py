"""
Hierarchical reconciliation: Top-Down, MinTrace-OLS, and MinTrace-WLS methods.

Hierarchy structure:
  Total (1)
    └─ Department (2)
         └─ Category (6)
              └─ Product (30)

Top-Down  : distribute total forecast proportionally using historical shares.
MinTrace-OLS: OLS projection — P = S(S'S)⁻¹S'  (ignores error covariance)
MinTrace-WLS: WLS projection weighted by per-series residual variance
              (Wickramasuriya et al. 2019, JASA) — the statistically optimal method.
              W = diag(σ²_i),  P = S(S'W⁻¹S)⁻¹S'W⁻¹
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Summing matrix S
# ---------------------------------------------------------------------------

def build_summing_matrix(
    category_map: dict[str, str],   # {product: category}
    dept_map: dict[str, str],        # {category: department}
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the summing matrix S where  y_all = S @ y_bottom.
    Rows = all series (total, depts, cats, products).
    Cols = bottom-level products.
    Returns (S DataFrame, ordered list of all series names).
    """
    products = sorted(category_map.keys())
    cats = sorted(set(category_map.values()))
    depts = sorted(set(dept_map.values()))

    all_series = ["TOTAL"] + depts + cats + products
    n_bottom = len(products)
    n_all = len(all_series)

    S = np.zeros((n_all, n_bottom), dtype=float)
    prod_idx = {p: i for i, p in enumerate(products)}
    series_idx = {s: i for i, s in enumerate(all_series)}

    # total row — sum of all products
    S[series_idx["TOTAL"], :] = 1.0

    # department rows
    for p, cat in category_map.items():
        dept = dept_map[cat]
        S[series_idx[dept], prod_idx[p]] = 1.0

    # category rows
    for p, cat in category_map.items():
        S[series_idx[cat], prod_idx[p]] = 1.0

    # product rows (identity block)
    for p in products:
        S[series_idx[p], prod_idx[p]] = 1.0

    S_df = pd.DataFrame(S, index=all_series, columns=products)
    return S_df, all_series


# ---------------------------------------------------------------------------
# Top-Down reconciliation
# ---------------------------------------------------------------------------

def reconcile_top_down(
    total_fc: pd.Series,
    product_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Distribute the total forecast proportionally using each product's
    average historical share of total sales.

    Returns a DataFrame with rows = forecast dates, cols = products.
    """
    avg_shares = product_history.mean() / product_history.mean().sum()
    reconciled = pd.DataFrame(
        np.outer(total_fc.values, avg_shares.values),
        index=total_fc.index,
        columns=product_history.columns,
    )
    return reconciled


# ---------------------------------------------------------------------------
# MinTrace (OLS) reconciliation
# ---------------------------------------------------------------------------

def reconcile_mintrace(
    base_forecasts: pd.DataFrame,   # rows=dates, cols=all series (ordered as S)
    S: pd.DataFrame,
) -> pd.DataFrame:
    """
    OLS MinTrace: P = S(S'S)^{-1}S'
    Reconciled = P @ base_forecasts.T  → shape (all_series, horizon)
    Returns reconciled DataFrame matching base_forecasts shape.
    """
    Sm = S.values  # (n_all, n_bottom)
    STS = Sm.T @ Sm
    P = Sm @ np.linalg.solve(STS, Sm.T)   # (n_all, n_all)

    Y_hat = base_forecasts[S.index].values.T   # (n_all, horizon)
    Y_rec = P @ Y_hat                           # (n_all, horizon)
    Y_rec = np.maximum(Y_rec, 0)

    return pd.DataFrame(Y_rec.T, index=base_forecasts.index, columns=S.index)


# ---------------------------------------------------------------------------
# MinTrace (WLS) reconciliation  — Wickramasuriya et al. 2019
# ---------------------------------------------------------------------------

def compute_residual_variances(
    actuals: pd.DataFrame,
    fitted_dict: dict[str, pd.Series],   # {series_name: fitted Series}
    min_var: float = 1e-4,
) -> pd.Series:
    """
    Estimate σ²_i = Var(y_i - ŷ_i) for each series using in-sample residuals.
    `fitted_dict` may cover a subset of the training window (e.g. LSTM skips first `window` steps).
    """
    variances = {}
    for name, fitted in fitted_dict.items():
        if name not in actuals.columns:
            continue
        actual = actuals[name].reindex(fitted.index).dropna()
        residuals = actual.values - fitted.reindex(actual.index).values
        variances[name] = max(float(np.var(residuals)), min_var)
    return pd.Series(variances)


def reconcile_mintrace_wls(
    base_forecasts: pd.DataFrame,   # rows=dates, cols=all series (ordered as S)
    S: pd.DataFrame,
    residual_vars: pd.Series,       # σ²_i per series, indexed by series name
) -> pd.DataFrame:
    """
    WLS MinTrace (Wickramasuriya et al. 2019):
      W   = diag(σ²_i)  for i in all series
      P   = S (S'W⁻¹S)⁻¹ S'W⁻¹
      Ŷ_r = P Ŷ_base

    Series with high residual variance get down-weighted, so a well-calibrated
    product-level model isn't corrupted by a poor total-level model.
    """
    Sm = S.values   # (n_all, n_bottom)

    # align residual variances to S row order; fill unknowns with column median
    median_var = float(residual_vars.median())
    w_diag = np.array([
        residual_vars.get(name, median_var) for name in S.index
    ])
    w_inv = 1.0 / w_diag                    # (n_all,)

    # P = S (S'W⁻¹S)⁻¹ S'W⁻¹
    StWinv = Sm.T * w_inv                   # (n_bottom, n_all) — broadcast
    StWinvS = StWinv @ Sm                   # (n_bottom, n_bottom)
    P = Sm @ np.linalg.solve(StWinvS, StWinv)  # (n_all, n_all)

    Y_hat = base_forecasts[S.index].values.T    # (n_all, horizon)
    Y_rec = P @ Y_hat                            # (n_all, horizon)
    Y_rec = np.maximum(Y_rec, 0)

    return pd.DataFrame(Y_rec.T, index=base_forecasts.index, columns=S.index)


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def mase(actual: np.ndarray, predicted: np.ndarray, naive: np.ndarray) -> float:
    """Mean Absolute Scaled Error — scale by naive (seasonal) forecast error."""
    denom = np.mean(np.abs(actual - naive))
    return float(np.mean(np.abs(actual - predicted)) / (denom + 1e-8))


def evaluate_level(actuals: pd.DataFrame, forecasts: pd.DataFrame,
                   label: str = "") -> pd.DataFrame:
    """Return per-series RMSE and MAE for a given hierarchy level."""
    rows = []
    for col in actuals.columns:
        if col in forecasts.columns:
            a = actuals[col].values
            p = forecasts[col].values[: len(a)]
            rows.append({"series": col, "level": label,
                          "RMSE": rmse(a, p), "MAE": mae(a, p)})
    return pd.DataFrame(rows)
