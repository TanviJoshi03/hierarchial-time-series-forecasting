"""
Facebook Prophet wrapper.
Handles weekly + yearly seasonality automatically; fits one model per series.
"""

import logging
import pandas as pd
import numpy as np

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def fit_prophet(series: pd.Series, horizon: int = 28) -> dict:
    """
    Fit Prophet to `series` (DatetimeIndex). Returns forecast dict.
    """
    from prophet import Prophet  # lazy import to avoid Stan init overhead

    df_train = pd.DataFrame({"ds": series.index, "y": series.values})
    df_train = df_train.dropna()

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    m.fit(df_train)

    future = m.make_future_dataframe(periods=horizon, freq="D")
    forecast = m.predict(future)

    fitted_vals = forecast[forecast["ds"].isin(series.index)]["yhat"].values
    fc_vals = forecast[~forecast["ds"].isin(series.index)]["yhat"].values
    fc_vals = np.maximum(fc_vals, 0)

    last_date = series.index[-1]
    future_idx = pd.date_range(last_date + pd.Timedelta("1D"), periods=horizon, freq="D")

    return {
        "model_name": "Prophet",
        "forecasts": pd.Series(fc_vals, index=future_idx, name=series.name),
        "fitted": pd.Series(fitted_vals, index=series.index, name=series.name),
    }


def forecast_all_prophet(
    level_df: pd.DataFrame, horizon: int = 28
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Prophet for every column in `level_df`. Returns (forecasts_df, fitted_df)."""
    fc_results, fitted_results = {}, {}
    for col in level_df.columns:
        out = fit_prophet(level_df[col], horizon=horizon)
        fc_results[col] = out["forecasts"]
        fitted_results[col] = out["fitted"]
    fc_df = pd.DataFrame(fc_results)
    fc_df.columns.name = level_df.columns.name
    fitted_df = pd.DataFrame(fitted_results)
    fitted_df.columns.name = level_df.columns.name
    return fc_df, fitted_df
