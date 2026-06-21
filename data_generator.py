"""
Synthetic M5-style retail dataset generator — extended with price, promotions,
known price elasticity, and promo lift so estimators can be scored against ground truth.

Hierarchy:  Total > Department > Category > Product
- 2 departments (Food, Household)
- 6 categories (3 per department)
- 30 products (5 per category)
- 3 years of daily sales (2020-2022)

Demand model (log-linear, per-product):
  log(demand_it) = log(base_seasonal_it)
                 + beta[cat] * log(price_it / ref_price_i)   # price elasticity
                 + gamma * promo_it                            # promo lift
                 + eps_it,   eps ~ N(0, 0.05)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

DEPARTMENTS = ["FOOD", "HOUSEHOLD"]
CATEGORIES = {
    "FOOD": ["FOOD_1", "FOOD_2", "FOOD_3"],
    "HOUSEHOLD": ["HH_1", "HH_2", "HH_3"],
}
PRODUCTS_PER_CAT = 5

CAT_PARAMS = {
    "FOOD_1":  {"base": 120, "trend": 0.08, "weekly_amp": 0.25, "yearly_amp": 0.15},
    "FOOD_2":  {"base": 80,  "trend": 0.05, "weekly_amp": 0.30, "yearly_amp": 0.20},
    "FOOD_3":  {"base": 60,  "trend": 0.03, "weekly_amp": 0.20, "yearly_amp": 0.25},
    "HH_1":    {"base": 50,  "trend": 0.06, "weekly_amp": 0.15, "yearly_amp": 0.10},
    "HH_2":    {"base": 40,  "trend": 0.04, "weekly_amp": 0.12, "yearly_amp": 0.08},
    "HH_3":    {"base": 30,  "trend": 0.02, "weekly_amp": 0.18, "yearly_amp": 0.12},
}

# Known true price elasticities (log-log) — FOOD more elastic than HOUSEHOLD
TRUE_BETAS = {
    "FOOD_1": -1.3,
    "FOOD_2": -1.1,
    "FOOD_3": -0.9,
    "HH_1":   -0.6,
    "HH_2":   -0.5,
    "HH_3":   -0.7,
}
TRUE_GAMMA = 0.18   # promo lift in log units (~+20% demand)
NOISE_SIGMA = 0.05  # log-space noise std

CAT_REF_PRICES = {
    "FOOD_1": 2.50, "FOOD_2": 1.80, "FOOD_3": 3.20,
    "HH_1":   4.50, "HH_2":   6.00, "HH_3":   3.80,
}

_DESC_ADJECTIVES = {
    "FOOD_1": ["organic", "premium", "natural", "wholegrain", "artisan"],
    "FOOD_2": ["fresh", "seasonal", "locally-sourced", "classic", "light"],
    "FOOD_3": ["gourmet", "luxury", "specialty", "traditional", "handcrafted"],
    "HH_1":   ["heavy-duty", "economy", "professional", "eco-friendly", "multipurpose"],
    "HH_2":   ["ultra", "concentrated", "long-lasting", "biodegradable", "industrial"],
    "HH_3":   ["compact", "portable", "refillable", "scented", "hypoallergenic"],
}
_DESC_NOUNS = {
    "FOOD_1": "snack pack", "FOOD_2": "produce bundle", "FOOD_3": "meal kit",
    "HH_1":   "cleaning kit", "HH_2": "detergent", "HH_3": "dispenser",
}
_DESC_SIZES = ["100g", "200g", "500g", "1kg", "250ml", "500ml", "1L"]
_CAT_ORDER = list(CAT_PARAMS.keys())


def _make_description(cat: str, k: int) -> str:
    adj = _DESC_ADJECTIVES[cat][k % 5]
    noun = _DESC_NOUNS[cat]
    size = _DESC_SIZES[(k * 3 + _CAT_ORDER.index(cat)) % len(_DESC_SIZES)]
    return f"{cat}_P{k+1} {adj} {noun} {size}"


def _generate_promo_mask(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sprinkle 6–10 non-overlapping promo windows of 3–7 days over n days."""
    mask = np.zeros(n, dtype=bool)
    n_windows = int(rng.integers(6, 11))
    attempts = 0
    placed = 0
    while placed < n_windows and attempts < 200:
        attempts += 1
        dur = int(rng.integers(3, 8))
        start = int(rng.integers(0, n - dur))
        # Skip if overlaps with an existing window (+ 3-day gap)
        if mask[max(0, start - 3): start + dur + 3].any():
            continue
        mask[start: start + dur] = True
        placed += 1
    return mask


def _generate_product_series(
    cat: str,
    product_idx: int,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (sales, price, promo_flag).

    Demand model in log space:
      log(demand) = log(base_seasonal) + beta*log(price/ref_price) + gamma*promo + eps
    """
    p = CAT_PARAMS[cat]
    t = np.arange(len(dates))
    n = len(dates)

    trend = p["base"] * (1 + p["trend"] * t / 365)
    week_season = 1 + p["weekly_amp"] * np.sin(2 * np.pi * t / 7 + product_idx)
    year_season = 1 + p["yearly_amp"] * np.sin(2 * np.pi * t / 365.25 - np.pi / 2)
    scale = rng.uniform(0.5, 1.5)
    base_seasonal = np.maximum(scale * trend * week_season * year_season, 1.0)

    # Price: ref_price × slow random walk, clipped to ±40%
    ref_price = CAT_REF_PRICES[cat] * rng.uniform(0.85, 1.15)
    log_walk = np.cumsum(rng.normal(0, 0.005, n))
    price = ref_price * np.exp(log_walk - log_walk[0])
    price = np.clip(price, ref_price * 0.6, ref_price * 1.4)

    # Promo: 6–10 windows per product per year, stacked across years
    n_years = max(1, n // 365)
    promo = np.zeros(n, dtype=bool)
    for _ in range(n_years):
        promo |= _generate_promo_mask(n, rng)
    promo = promo.astype(float)

    # On promo days: price drops ~20%
    price_actual = price * (1 - 0.20 * promo)

    # Log-linear demand
    beta = TRUE_BETAS[cat]
    log_base = np.log(base_seasonal)
    log_price_ratio = np.log(price_actual / ref_price)
    eps = rng.normal(0, NOISE_SIGMA, n)
    log_demand = log_base + beta * log_price_ratio + TRUE_GAMMA * promo + eps
    sales = np.maximum(np.exp(log_demand), 0).round(2)

    return sales, price_actual.round(4), promo


def generate_dataset(
    start: str = "2020-01-01",
    end: str = "2022-12-31",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Generate synthetic retail dataset with price and promo channels.

    Returns
    -------
    df : DataFrame with columns [date, department, category, product, sales, price, promo]
    descriptions : dict {product_id: natural-language description}
    """
    dates = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(42)
    records = []
    descriptions: dict[str, str] = {}

    for dept in DEPARTMENTS:
        for cat in CATEGORIES[dept]:
            for k in range(PRODUCTS_PER_CAT):
                product_id = f"{cat}_P{k+1}"
                descriptions[product_id] = _make_description(cat, k)
                sales, price, promo = _generate_product_series(cat, k, dates, rng)
                for i, d in enumerate(dates):
                    records.append({
                        "date": d,
                        "department": dept,
                        "category": cat,
                        "product": product_id,
                        "sales": sales[i],
                        "price": price[i],
                        "promo": promo[i],
                    })

    df = pd.DataFrame(records)
    return df, descriptions


def build_hierarchical_df(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return aggregated sales at each level of the hierarchy."""
    product = df.pivot_table(index="date", columns="product", values="sales", aggfunc="sum")
    category = df.pivot_table(index="date", columns="category", values="sales", aggfunc="sum")
    department = df.pivot_table(index="date", columns="department", values="sales", aggfunc="sum")
    total = df.groupby("date")["sales"].sum().rename("TOTAL").to_frame()
    return {"total": total, "department": department, "category": category, "product": product}


def build_covariate_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (price_pivot, promo_pivot) indexed by date, one column per product.
    Use these as past-covariates for the GBM forecasters.
    """
    price_df = df.pivot_table(index="date", columns="product", values="price", aggfunc="mean")
    promo_df = df.pivot_table(index="date", columns="product", values="promo", aggfunc="max")
    return price_df, promo_df


def save_ground_truth(results_dir: Path) -> None:
    results_dir.mkdir(exist_ok=True)
    gt = {
        "true_betas": TRUE_BETAS,
        "true_gamma": TRUE_GAMMA,
        "category_ref_prices": CAT_REF_PRICES,
    }
    with open(results_dir / "ground_truth.json", "w") as f:
        json.dump(gt, f, indent=2)
    print(f"    Ground truth saved → {results_dir}/ground_truth.json")


if __name__ == "__main__":
    out = Path("data")
    out.mkdir(exist_ok=True)
    df, descs = generate_dataset()
    df.to_csv(out / "retail_sales.csv", index=False)
    print(f"Generated {len(df):,} rows | {df['product'].nunique()} products | "
          f"{df['date'].min().date()} – {df['date'].max().date()}")
    levels = build_hierarchical_df(df)
    for lvl, frame in levels.items():
        frame.to_csv(out / f"{lvl}.csv")
    print("Saved hierarchy CSVs to data/")
    save_ground_truth(Path("results"))
