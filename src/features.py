from typing import NamedTuple

import holidays
import numpy as np
import pandas as pd

DEMAND_LAGS = [0, 1, 2, 3, 6, 12, 24, 48, 168]  # hours before the origin
WEATHER_COLS = ["temp_c", "rh_pct", "dewpoint_c", "wind_ms", "solar_wm2", "precip_mm"]
COVID_START, COVID_END = "2020-03-22", "2020-06-07"  # approximate lockdown period
FEATURE_SETS = ("demand", "calendar", "full")  # nested: demand < calendar < full


class Dataset(NamedTuple):
    X: pd.DataFrame            # features, indexed by origin time
    y_rel: pd.Series           # target divided by level
    y_true: pd.Series          # target in MW
    level: pd.Series           # recent mean demand at origin (MW)
    baselines: pd.DataFrame    # persistence / previous_day / previous_week forecasts in MW
    target_time: pd.Series     # origin time + horizon


def build_features(df, horizon, feature_set="full"):
    """Build (features, target, baselines) for forecasting `horizon` hours ahead."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
    d = df.sort_values("datetime").set_index("datetime")
    y = d["national"]
    level = y.rolling(168).mean()

    X = pd.DataFrame(index=d.index)
    # --- demand history (known at origin) ---
    for k in DEMAND_LAGS:
        X[f"lag_{k}"] = y.shift(k) / level
    if 24 - horizon not in DEMAND_LAGS:
        X["same_hour_yesterday"] = y.shift(24 - horizon) / level
    if 168 - horizon not in DEMAND_LAGS:
        X["same_hour_last_week"] = y.shift(168 - horizon) / level
    last24 = y.rolling(24)
    X["roll_mean_24"] = last24.mean() / level
    X["roll_std_24"] = last24.std() / level
    X["roll_max_24"] = last24.max() / level
    X["roll_min_24"] = last24.min() / level
    X["weekly_growth"] = level / level.shift(168)

    if feature_set in ("calendar", "full"):
        T = d.index + pd.Timedelta(hours=horizon)
        for name, values, period in (("hour", T.hour, 24), ("dow", T.dayofweek, 7), ("doy", T.dayofyear, 366)):
            X[f"{name}_sin"] = np.sin(2 * np.pi * np.asarray(values) / period)
            X[f"{name}_cos"] = np.cos(2 * np.pi * np.asarray(values) / period)
        X["is_weekend"] = (np.asarray(T.dayofweek) >= 5).astype(int)
        hol = set(holidays.country_holidays("IN", years=range(T.year.min(), T.year.max() + 1)).keys())
        X["is_holiday"] = pd.Series(T.date).isin(hol).astype(int).values
        X["covid_lockdown"] = ((T >= COVID_START) & (T <= COVID_END)).astype(int)

    if feature_set == "full":
        for c in WEATHER_COLS:
            X[f"{c}_target"] = d[c].shift(-horizon)
        X["cdd_target"] = (d["temp_c"].shift(-horizon) - 25).clip(lower=0)  # cooling degrees above 25 C
        X["temp_roll24"] = d["temp_c"].rolling(24).mean()
        X["temp_delta"] = X["temp_c_target"] - d["temp_c"]

    y_true = y.shift(-horizon)
    baselines = pd.DataFrame({
        "persistence": y,
        "previous_day": y.shift(24 - horizon),
        "previous_week": y.shift(168 - horizon),
    })
    target_time = pd.Series(d.index + pd.Timedelta(hours=horizon), index=d.index)

    ok = X.notna().all(axis=1) & y_true.notna() & level.notna() & baselines.notna().all(axis=1)
    return Dataset(X[ok], (y_true / level)[ok], y_true[ok], level[ok], baselines[ok], target_time[ok])