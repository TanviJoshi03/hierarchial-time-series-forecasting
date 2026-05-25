"""
ARIMA baseline using statsmodels auto_arima (AIC-selected orders).
Fits one ARIMA per time series, returns h-step-ahead forecasts.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


def _select_d(series: np.ndarray, max_d: int = 2) -> int:
    """Choose differencing order d via ADF test."""
    for d in range(max_d + 1):
        s = np.diff(series, n=d) if d > 0 else series
        try:
            p = adfuller(s, autolag="AIC")[1]
        except Exception:
            return d
        if p < 0.05:
            return d
    return max_d


def fit_arima(series: pd.Series, horizon: int = 28, freq: str = "D") -> dict:
    """
    Fit ARIMA(p,d,q) to `series` and forecast `horizon` steps.
    Returns dict with keys: model_name, forecasts (Series), fitted (Series).
    """
    vals = series.dropna().values.astype(float)
    d = _select_d(vals)

    best_aic = np.inf
    best_order = (1, d, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(0, 4):
            for q in range(0, 4):
                try:
                    res = SARIMAX(vals, order=(p, d, q),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False).fit(disp=False)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(vals, order=best_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)

    last_date = series.index[-1]
    future_idx = pd.date_range(last_date + pd.Timedelta("1D"), periods=horizon, freq=freq)
    fc = model.forecast(steps=horizon)
    fc = np.maximum(fc, 0)

    return {
        "model_name": f"ARIMA{best_order}",
        "forecasts": pd.Series(fc, index=future_idx, name=series.name),
        "fitted": pd.Series(model.fittedvalues, index=series.index, name=series.name),
        "aic": best_aic,
    }


def forecast_all_arima(
    level_df: pd.DataFrame, horizon: int = 28
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run ARIMA for every column in `level_df`. Returns (forecasts_df, fitted_df)."""
    fc_results, fitted_results = {}, {}
    for col in level_df.columns:
        out = fit_arima(level_df[col], horizon=horizon)
        fc_results[col] = out["forecasts"]
        fitted_results[col] = out["fitted"]
    fc_df = pd.DataFrame(fc_results)
    fc_df.columns.name = level_df.columns.name
    fitted_df = pd.DataFrame(fitted_results)
    fitted_df.columns.name = level_df.columns.name
    return fc_df, fitted_df
