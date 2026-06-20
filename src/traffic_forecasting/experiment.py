from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .data import generate_traffic_series
from .features import make_supervised_features
from .model import evaluate_forecaster, train_forecaster


def run_experiment(output_dir: str | Path = "results", periods: int = 24 * 60, seed: int = 42) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = generate_traffic_series(periods=periods, seed=seed)
    x, y = make_supervised_features(data)
    split = int(len(x) * 0.80)
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = train_forecaster(x_train, y_train, seed=seed)
    metrics = evaluate_forecaster(model, x_test, y_test)
    pred = model.predict(x_test)

    data.to_csv(output_path / "synthetic_5g_traffic.csv", index=False)
    pd.DataFrame([metrics.__dict__]).to_csv(output_path / "forecast_metrics.csv", index=False)
    predictions = pd.DataFrame({"actual_mbps": y_test.values, "predicted_mbps": pred, "naive_lag1_mbps": x_test["lag_1"].values})
    predictions.to_csv(output_path / "forecast_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_size = min(160, len(predictions))
    ax.plot(predictions["actual_mbps"].values[:plot_size], label="Actual")
    ax.plot(predictions["predicted_mbps"].values[:plot_size], label="Predicted")
    ax.set_title("5G traffic forecasting: actual vs predicted")
    ax.set_xlabel("Test time step")
    ax.set_ylabel("Total traffic, Mbps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "forecast_actual_vs_predicted.png", dpi=300)
    plt.close(fig)

    importance = pd.DataFrame({"feature": x.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(output_path / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(importance["feature"], importance["importance"])
    ax.set_title("Forecasting feature importance")
    ax.set_ylabel("Importance")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path / "feature_importance.png", dpi=300)
    plt.close(fig)

    return pd.DataFrame([metrics.__dict__])
