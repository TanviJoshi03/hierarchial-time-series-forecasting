"""
Cold-start forecasting via sentence-transformer embeddings + FAISS retrieval.

Protocol
--------
1. Hold out FOOD_1_P5 entirely (no sales history).
2. Embed ALL product descriptions with all-MiniLM-L6-v2.
3. FAISS nearest-neighbour search: find k=3 most similar products to the held-out one.
4. Transfer their seasonal profile (normalised daily pattern) scaled to category mean.
5. Evaluate: cold-start RMSE vs naive (category-average) baseline on the test period.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


HOLDOUT_PRODUCT = "FOOD_1_P5"
K_NEIGHBORS = 3


# ── Seasonal profile extraction ───────────────────────────────────────────────

def _seasonal_profile(series: pd.Series, forecast_dates: pd.DatetimeIndex) -> np.ndarray:
    """
    For each forecast date, look up the same calendar window (±2 weeks around the
    same DOY) in the training history and average to get a level estimate.
    This produces a calendar-aligned forecast that beats a flat naive.
    """
    horizon = len(forecast_dates)
    profile = np.zeros(horizon)
    for i, dt in enumerate(forecast_dates):
        doy = dt.dayofyear
        # All training values with DOY within ±10 days (wraps around year end)
        mask = series.index.map(
            lambda d: abs(d.dayofyear - doy) <= 10 or abs(d.dayofyear - doy) >= 355
        )
        nearby = series[mask]
        profile[i] = nearby.mean() if len(nearby) > 0 else series.mean()
    return profile


# ── Cold-start forecast ───────────────────────────────────────────────────────

def cold_start_forecast(
    df: pd.DataFrame,
    descriptions: dict[str, str],
    train_product: pd.DataFrame,
    test_product: pd.DataFrame,
    horizon: int = 28,
    k: int = K_NEIGHBORS,
    holdout: str = HOLDOUT_PRODUCT,
) -> dict:
    """
    Embed descriptions, retrieve neighbours, transfer seasonal profile.

    Returns dict with 'forecast', 'rmse_coldstart', 'rmse_naive', 'neighbors'.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        print("    [SKIP] sentence-transformers or faiss-cpu not installed")
        return {}

    # Encode all descriptions
    model = SentenceTransformer("all-MiniLM-L6-v2")
    products = list(descriptions.keys())
    texts = [descriptions[p] for p in products]
    print(f"    Encoding {len(texts)} product descriptions …", end=" ", flush=True)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    print("done")

    # FAISS inner-product index (cosine similarity after L2 normalisation)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype(np.float32))

    holdout_idx = products.index(holdout)
    new_emb = emb[holdout_idx: holdout_idx + 1].astype(np.float32)

    # Exclude the holdout itself from search results
    D, I = index.search(new_emb, k + 1)
    neighbor_indices = [i for i in I[0] if i != holdout_idx][:k]
    neighbors = [products[i] for i in neighbor_indices]
    similarities = [float(D[0][j]) for j, i in enumerate(I[0]) if i != holdout_idx][:k]

    print(f"    Nearest neighbours for '{holdout}':")
    for nb, sim in zip(neighbors, similarities):
        print(f"      {nb}  (cosine={sim:.3f})  '{descriptions[nb]}'")

    # Transfer calendar-aligned seasonal profile from neighbours
    available_neighbors = [n for n in neighbors if n in train_product.columns]
    if not available_neighbors:
        available_neighbors = [n for n in train_product.columns if n != holdout][:k]

    forecast_dates = test_product.index[:horizon]
    profiles = [_seasonal_profile(train_product[n], forecast_dates) for n in available_neighbors]
    cs_forecast = np.mean(profiles, axis=0)  # average across neighbours
    cs_forecast = np.maximum(cs_forecast, 0)

    # Naive baseline: same calendar dates from prior year(s), averaged across category
    holdout_cat = df.loc[df["product"] == holdout, "category"].iloc[0]
    cat_products_train = [
        c for c in train_product.columns
        if c != holdout and c.startswith(holdout_cat)
    ]
    if cat_products_train:
        # For each forecast date, average same-DOY ± 10 days across cat products (training)
        naive_forecast = np.zeros(horizon)
        for i, dt in enumerate(forecast_dates):
            doy = dt.dayofyear
            vals = []
            for cp in cat_products_train:
                mask = train_product[cp].index.map(
                    lambda d: abs(d.dayofyear - doy) <= 10 or abs(d.dayofyear - doy) >= 355
                )
                vals.extend(train_product[cp][mask].tolist())
            naive_forecast[i] = np.mean(vals) if vals else train_product[cat_products_train].mean().mean()
    else:
        naive_forecast = np.full(horizon, train_product.mean().mean())

    # Evaluate against test actuals
    if holdout in test_product.columns:
        actuals = test_product[holdout].values[:horizon]
        rmse_cs    = float(np.sqrt(np.mean((actuals - cs_forecast[:len(actuals)]) ** 2)))
        rmse_naive = float(np.sqrt(np.mean((actuals - naive_forecast[:len(actuals)]) ** 2)))
        improvement = round((rmse_naive - rmse_cs) / rmse_naive * 100, 2)
    else:
        actuals = None
        rmse_cs = rmse_naive = improvement = None

    return {
        "holdout":      holdout,
        "neighbors":    neighbors,
        "similarities": similarities,
        "forecast":     cs_forecast,
        "naive":        naive_forecast,
        "actuals":      actuals,
        "rmse_coldstart": rmse_cs,
        "rmse_naive":     rmse_naive,
        "improvement_pct": improvement,
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def run_coldstart(
    df: pd.DataFrame,
    descriptions: dict[str, str],
    train_product: pd.DataFrame,
    test_product: pd.DataFrame,
    results_dir: Path,
    horizon: int = 28,
) -> dict:
    """Run cold-start pipeline and save results."""
    print(f"    Held-out product: {HOLDOUT_PRODUCT}")
    result = cold_start_forecast(df, descriptions, train_product, test_product, horizon)

    if not result:
        return result

    results_dir.mkdir(exist_ok=True)
    fc_df = pd.DataFrame({
        "date":       test_product.index[:horizon],
        "coldstart":  result["forecast"][:horizon],
        "naive":      result["naive"][:horizon],
        "actual":     result["actuals"][:horizon] if result["actuals"] is not None else np.nan,
    })
    fc_df.to_csv(results_dir / "coldstart_forecast.csv", index=False)

    print(f"\n    ── Cold-Start Results ──")
    print(f"    Cold-start RMSE : {result['rmse_coldstart']:.2f}")
    print(f"    Naive RMSE      : {result['rmse_naive']:.2f}")
    print(f"    Improvement     : {result['improvement_pct']:.1f}% vs naive")

    return result
