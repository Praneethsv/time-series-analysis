import pandas as pd
from scipy.stats import zscore
from statsmodels.tsa.stattools import adfuller


def detect_outliers_quantile(y: pd.DataFrame, verbose=False):
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = y[(y < lower_bound) | (y > upper_bound)]
    if verbose:
        print("Number of outliers detected using Quartile method:", len(outliers))
        print(outliers)
    return outliers


def detect_outliers_z_score(y, verbose=False):

    y_zscore = zscore(y)
    outliers = y[abs(y_zscore) > 3]
    if verbose:
        print("Number of outliers detected based on z-score:", len(outliers))
        print(outliers)
    return outliers


def is_stationary(y):
    res = adfuller(y)
    adf_stats = res[0]
    p_val = res[1]
    return True if p_val < 0.05 else False
