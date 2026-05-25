"""
Synthetic M5-style retail dataset generator.

Hierarchy:  Total > Department > Category > Product
- 2 departments (Food, Household)
- 6 categories (3 per department)
- 30 products (5 per category)
- 3 years of daily sales (2020-2022)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

DEPARTMENTS = ["FOOD", "HOUSEHOLD"]
CATEGORIES = {
    "FOOD": ["FOOD_1", "FOOD_2", "FOOD_3"],
    "HOUSEHOLD": ["HH_1", "HH_2", "HH_3"],
}
PRODUCTS_PER_CAT = 5

# Base demand and multiplicative seasonality params per category
CAT_PARAMS = {
    "FOOD_1":  {"base": 120, "trend": 0.08, "weekly_amp": 0.25, "yearly_amp": 0.15},
    "FOOD_2":  {"base": 80,  "trend": 0.05, "weekly_amp": 0.30, "yearly_amp": 0.20},
    "FOOD_3":  {"base": 60,  "trend": 0.03, "weekly_amp": 0.20, "yearly_amp": 0.25},
    "HH_1":    {"base": 50,  "trend": 0.06, "weekly_amp": 0.15, "yearly_amp": 0.10},
    "HH_2":    {"base": 40,  "trend": 0.04, "weekly_amp": 0.12, "yearly_amp": 0.08},
    "HH_3":    {"base": 30,  "trend": 0.02, "weekly_amp": 0.18, "yearly_amp": 0.12},
}


def _generate_product_series(cat: str, product_idx: int, dates: pd.DatetimeIndex) -> np.ndarray:
    p = CAT_PARAMS[cat]
    t = np.arange(len(dates))
    n = len(dates)

    trend = p["base"] * (1 + p["trend"] * t / 365)
    week_season = 1 + p["weekly_amp"] * np.sin(2 * np.pi * t / 7 + product_idx)
    year_season = 1 + p["yearly_amp"] * np.sin(2 * np.pi * t / 365.25 - np.pi / 2)

    # product-level scaling factor
    scale = RNG.uniform(0.5, 1.5)
    noise = RNG.normal(0, 0.05 * p["base"], n)
    sales = scale * trend * week_season * year_season + noise
    return np.maximum(sales, 0).round(2)


def generate_dataset(start="2020-01-01", end="2022-12-31") -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    records = []
    for dept in DEPARTMENTS:
        for cat in CATEGORIES[dept]:
            for k in range(PRODUCTS_PER_CAT):
                product_id = f"{cat}_P{k+1}"
                sales = _generate_product_series(cat, k, dates)
                for i, d in enumerate(dates):
                    records.append({
                        "date": d,
                        "department": dept,
                        "category": cat,
                        "product": product_id,
                        "sales": sales[i],
                    })
    df = pd.DataFrame(records)
    return df


def build_hierarchical_df(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return aggregated sales at each level of the hierarchy."""
    product = df.pivot_table(index="date", columns="product", values="sales", aggfunc="sum")
    category = df.pivot_table(index="date", columns="category", values="sales", aggfunc="sum")
    department = df.pivot_table(index="date", columns="department", values="sales", aggfunc="sum")
    total = df.groupby("date")["sales"].sum().rename("TOTAL").to_frame()
    return {"total": total, "department": department, "category": category, "product": product}


if __name__ == "__main__":
    out = Path("data")
    out.mkdir(exist_ok=True)
    df = generate_dataset()
    df.to_csv(out / "retail_sales.csv", index=False)
    print(f"Generated {len(df):,} rows | {df['product'].nunique()} products | {df['date'].min()} – {df['date'].max()}")
    levels = build_hierarchical_df(df)
    for lvl, frame in levels.items():
        frame.to_csv(out / f"{lvl}.csv")
    print("Saved hierarchy CSVs to data/")
