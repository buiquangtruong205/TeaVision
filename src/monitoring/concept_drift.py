import numpy as np


def confidence_drift(mean_confidences, min_confidence=0.7):
    return float(np.mean(mean_confidences)) < min_confidence
