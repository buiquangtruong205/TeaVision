from src.monitoring.retrain_trigger import should_retrain


def test_should_retrain():
    assert should_retrain(True, min_new_samples=10, new_samples_count=10)
