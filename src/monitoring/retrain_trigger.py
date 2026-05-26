def should_retrain(drift_detected, min_new_samples, new_samples_count):
    return bool(drift_detected and new_samples_count >= min_new_samples)
