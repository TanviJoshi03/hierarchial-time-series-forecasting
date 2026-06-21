"""
Global LightGBM + XGBoost forecasters via Darts.

One model is fit across all 30 products simultaneously (global model) using:
  - Lagged sales: [-1, -7, -14, -28]
  - Lagged price + promo covariates: [-1, -7]
  - Calendar covariates (DOW, month) as future covariates

The global model learns cross-product patterns and incorporates price/promo
signals that tie directly to the elasticity and causal estimators.
"""

import numpy as np
import pandas as pd


def _build_calendar_df(dates: pd.DatetimeIndex) -> pd.DataFrame:
    dow = dates.dayofweek
    month = dates.month
    return pd.DataFrame({
        "dow_sin":   np.sin(2 * np.pi * dow / 7),
        "dow_cos":   np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    }, index=dates)


def forecast_all_gbm(
    product_df: pd.DataFrame,
    price_df: pd.DataFrame,
    promo_df: pd.DataFrame,
    horizon: int = 28,
) -> dict[str, pd.DataFrame]:
    """
    Fit global LightGBM and XGBoost on all products then generate forecasts.

    Parameters
    ----------
    product_df : training-period sales pivot (dates × products)
    price_df   : full-period price pivot (covers train + test; used as past covariates)
    promo_df   : full-period promo pivot
    horizon    : forecast horizon in days

    Returns
    -------
    dict with keys 'lgbm_fc' and 'xgb_fc' — each a DataFrame (horizon × n_products)
    """
    from darts import TimeSeries
    from darts.models import LightGBMModel, XGBModel

    products = list(product_df.columns)
    train_idx = product_df.index
    full_idx = price_df.index

    # One target TimeSeries per product (training period)
    series_list = [TimeSeries.from_series(product_df[col]) for col in products]

    # Past covariates: price + promo per product (full period so Darts can pull lags)
    past_cov_list = []
    for col in products:
        cov = pd.DataFrame({
            "price": price_df[col],
            "promo": promo_df[col],
        }, index=full_idx)
        past_cov_list.append(TimeSeries.from_dataframe(cov))

    # Future covariates: calendar features (deterministically known; same for all products)
    cal_df = _build_calendar_df(full_idx)
    future_cov_single = TimeSeries.from_dataframe(cal_df)
    future_cov_list = [future_cov_single] * len(products)

    forecast_start = train_idx[-1] + pd.Timedelta("1D")
    fc_index = pd.date_range(forecast_start, periods=horizon, freq="D")

    results = {}
    for model_cls, name in [(LightGBMModel, "lgbm"), (XGBModel, "xgb")]:
        print(f"      Fitting {name.upper()} global model …", end=" ", flush=True)
        model = model_cls(
            lags=[-1, -7, -14, -28],
            lags_past_covariates=[-1, -7],
            lags_future_covariates=[0],
            output_chunk_length=horizon,
            random_state=42,
            verbose=-1,
        )
        model.fit(series_list, past_covariates=past_cov_list, future_covariates=future_cov_list)
        preds = model.predict(
            horizon,
            series=series_list,
            past_covariates=past_cov_list,
            future_covariates=future_cov_list,
        )
        fc_dict = {}
        for ts, col in zip(preds, products):
            vals = np.maximum(ts.to_series().values[:horizon], 0)
            fc_dict[col] = pd.Series(vals, index=fc_index, name=col)
        results[f"{name}_fc"] = pd.DataFrame(fc_dict)
        print("done")

    return results


def aggregate_gbm_to_levels(
    product_fc: pd.DataFrame,
    category_map: dict[str, str],
    dept_map: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """
    Bottom-up aggregation of product-level GBM forecasts to category / dept / total.
    Returns dict matching the structure of arima_fc / prophet_fc / lstm_fc.
    """
    cat_fc = {}
    dept_fc = {}

    for cat in set(category_map.values()):
        prods = [p for p, c in category_map.items() if c == cat and p in product_fc.columns]
        if prods:
            cat_fc[cat] = product_fc[prods].sum(axis=1)

    for dept in set(dept_map.values()):
        cats = [c for c, d in dept_map.items() if d == dept and c in cat_fc]
        if cats:
            dept_fc[dept] = pd.concat([cat_fc[c] for c in cats], axis=1).sum(axis=1)

    total_fc = product_fc.sum(axis=1).rename("TOTAL")

    return {
        "product":    product_fc,
        "category":   pd.DataFrame(cat_fc),
        "department": pd.DataFrame(dept_fc),
        "total":      total_fc.to_frame(),
    }
