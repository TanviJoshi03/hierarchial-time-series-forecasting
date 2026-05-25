"""
Univariate LSTM forecaster built with PyTorch.
Architecture: 2-layer LSTM → Linear output
Input window: 60 days → forecast horizon steps (recursive decoding)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def _make_sequences(vals: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(vals) - window):
        X.append(vals[i: i + window])
        y.append(vals[i + window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def fit_lstm(series: pd.Series, horizon: int = 28, window: int = 60,
             epochs: int = 50, lr: float = 1e-3) -> dict:
    vals = series.dropna().values.astype(np.float32)

    # min-max scale to [0, 1]
    vmin, vmax = vals.min(), vals.max()
    scaled = (vals - vmin) / (vmax - vmin + 1e-8)

    X, y = _make_sequences(scaled, window)
    X_t = torch.tensor(X).unsqueeze(-1)   # (N, window, 1)
    y_t = torch.tensor(y)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = LSTMForecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # recursive multi-step forecast
    model.eval()
    with torch.no_grad():
        seed = torch.tensor(scaled[-window:], dtype=torch.float32)
        preds = []
        for _ in range(horizon):
            inp = seed.unsqueeze(0).unsqueeze(-1)   # (1, window, 1)
            pred = model(inp).item()
            preds.append(pred)
            seed = torch.cat([seed[1:], torch.tensor([pred])])

    fc_scaled = np.array(preds)
    fc = fc_scaled * (vmax - vmin + 1e-8) + vmin
    fc = np.maximum(fc, 0)

    # in-sample fitted values
    model.eval()
    with torch.no_grad():
        fitted_scaled = model(X_t).numpy()
    fitted = fitted_scaled * (vmax - vmin + 1e-8) + vmin

    last_date = series.index[-1]
    future_idx = pd.date_range(last_date + pd.Timedelta("1D"), periods=horizon, freq="D")
    fitted_idx = series.index[window:]

    return {
        "model_name": "LSTM",
        "forecasts": pd.Series(fc, index=future_idx, name=series.name),
        "fitted": pd.Series(fitted, index=fitted_idx, name=series.name),
    }


def forecast_all_lstm(
    level_df: pd.DataFrame, horizon: int = 28, window: int = 60, epochs: int = 50
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run LSTM for every column in `level_df`. Returns (forecasts_df, fitted_df)."""
    fc_results, fitted_results = {}, {}
    for col in level_df.columns:
        out = fit_lstm(level_df[col], horizon=horizon, window=window, epochs=epochs)
        fc_results[col] = out["forecasts"]
        fitted_results[col] = out["fitted"]
    fc_df = pd.DataFrame(fc_results)
    fc_df.columns.name = level_df.columns.name
    fitted_df = pd.DataFrame(fitted_results)
    fitted_df.columns.name = level_df.columns.name
    return fc_df, fitted_df
