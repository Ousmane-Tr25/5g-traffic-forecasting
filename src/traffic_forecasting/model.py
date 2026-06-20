from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class ForecastMetrics:
    mae: float
    rmse: float
    r2: float
    baseline_mae: float


def train_forecaster(x_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=180,
        max_depth=10,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_forecaster(model: RandomForestRegressor, x_test: pd.DataFrame, y_test: pd.Series) -> ForecastMetrics:
    pred = model.predict(x_test)
    baseline = x_test["lag_1"].values
    return ForecastMetrics(
        mae=float(mean_absolute_error(y_test, pred)),
        rmse=float(np.sqrt(mean_squared_error(y_test, pred))),
        r2=float(r2_score(y_test, pred)),
        baseline_mae=float(mean_absolute_error(y_test, baseline)),
    )
