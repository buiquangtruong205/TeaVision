from scipy.stats import ks_2samp


def ks_drift_score(reference_values, current_values):
    statistic, pvalue = ks_2samp(reference_values, current_values)
    return {"statistic": float(statistic), "pvalue": float(pvalue)}
