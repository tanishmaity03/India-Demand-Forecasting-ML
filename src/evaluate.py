import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SPLITS = {
    "train": ("2019-01-01", "2021-12-31"),
    "val": ("2022-01-01", "2022-12-31"),
    "test": ("2023-01-01", "2024-04-30"),
}

def split_mask(target_time, name):
    start, end = SPLITS[name]
    return ((target_time >= pd.Timestamp(start)) & (target_time < pd.Timestamp(end) + pd.Timedelta(days=1))).values


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "MAE (MW)": mean_absolute_error(y_true, y_pred),
        "RMSE (MW)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE (%)": float(np.mean(np.abs(y_true - y_pred) / y_true) * 100),
        "R2": r2_score(y_true, y_pred),
    }