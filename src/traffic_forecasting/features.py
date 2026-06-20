from __future__ import annotations

import pandas as pd


def make_supervised_features(
    data: pd.DataFrame,
    target: str = "total_mbps",
    lags: tuple[int, ...] = (1, 2, 3, 6, 12, 24),
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert a time series into supervised-learning features."""
    df = data.copy()
    if "timestamp" not in df.columns:
        raise ValueError("Input data must contain a timestamp column")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    for lag in lags:
        df[f"lag_{lag}"] = df[target].shift(lag)
    df["rolling_mean_6"] = df[target].shift(1).rolling(6).mean()
    df["rolling_mean_24"] = df[target].shift(1).rolling(24).mean()
    df = df.dropna().reset_index(drop=True)
    feature_columns = ["hour", "dayofweek", *[f"lag_{lag}" for lag in lags], "rolling_mean_6", "rolling_mean_24"]
    return df[feature_columns], df[target]
