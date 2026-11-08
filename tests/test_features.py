import unittest
import numpy as np 
import pandas as pd 
from src.features import build_features,WEATHER_COLS

def synthetic_frame(days=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=24 * days, freq="h")
    df = pd.DataFrame({"datetime": idx})
    df["national"] = 150000 + 20000 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 2000, len(idx))
    for c in WEATHER_COLS:
        df[c] = rng.uniform(0, 30, len(idx))
    return df


class FeatureTests(unittest.TestCase):
    def test_no_future_demand_in_features(self):
        """Changing demand after time t0 must not change any feature row with origin <= t0."""
        df = synthetic_frame()
        t0 = pd.Timestamp("2022-02-01 12:00")
        altered = df.copy()
        altered.loc[altered["datetime"] > t0, "national"] *= 3
        for h in (1, 24):
            a = build_features(df, h).X
            b = build_features(altered, h).X
            rows = a.index[a.index <= t0]
            pd.testing.assert_frame_equal(a.loc[rows], b.loc[rows])

    def test_target_alignment(self):
        df = synthetic_frame()
        for h in (1, 24):
            ds = build_features(df, h)
            demand = df.set_index("datetime")["national"]
            origin = ds.y_true.index[100]
            self.assertAlmostEqual(ds.y_true.loc[origin], demand.loc[origin + pd.Timedelta(hours=h)])

    def test_baselines_are_known_at_origin(self):
        df = synthetic_frame()
        demand = df.set_index("datetime")["national"]
        ds = build_features(df, 24)
        origin = ds.baselines.index[50]
        self.assertAlmostEqual(ds.baselines.loc[origin, "persistence"], demand.loc[origin])
        self.assertAlmostEqual(ds.baselines.loc[origin, "previous_day"], demand.loc[origin])  # t+24-24 = t

    def test_no_missing_values(self):
        ds = build_features(synthetic_frame(), 1)
        self.assertFalse(ds.X.isna().any().any())
        self.assertFalse(ds.y_rel.isna().any())


if __name__ == "__main__":
    unittest.main()