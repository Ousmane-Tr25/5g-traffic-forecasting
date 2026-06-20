from traffic_forecasting.data import generate_traffic_series
from traffic_forecasting.features import make_supervised_features
from traffic_forecasting.model import evaluate_forecaster, train_forecaster


def test_feature_generation():
    data = generate_traffic_series(periods=100, seed=1)
    x, y = make_supervised_features(data)
    assert len(x) == len(y)
    assert "lag_24" in x.columns


def test_model_training_small_dataset():
    data = generate_traffic_series(periods=150, seed=1)
    x, y = make_supervised_features(data)
    model = train_forecaster(x.iloc[:80], y.iloc[:80], seed=1)
    metrics = evaluate_forecaster(model, x.iloc[80:], y.iloc[80:])
    assert metrics.mae >= 0
