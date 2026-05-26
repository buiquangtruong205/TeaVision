import numpy as np


def apply_threshold(probabilities, threshold=0.5):
    probabilities = np.asarray(probabilities)
    return (probabilities >= threshold).astype(int)
