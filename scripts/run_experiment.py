from traffic_forecasting.experiment import run_experiment

if __name__ == "__main__":
    metrics = run_experiment("results")
    print(metrics.to_string(index=False))
