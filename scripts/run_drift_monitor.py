from src.monitoring.retrain_trigger import should_retrain


if __name__ == "__main__":
    print(should_retrain(drift_detected=False, min_new_samples=100, new_samples_count=0))
