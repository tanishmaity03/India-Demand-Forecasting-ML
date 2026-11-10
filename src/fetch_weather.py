
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"

# One representative city per grid region (region keys match the demand dataset columns).
CITIES = {
    "northern": ("Delhi", 28.61, 77.21),
    "western": ("Mumbai", 19.08, 72.88),
    "eastern": ("Kolkata", 22.57, 88.36),
    "southern": ("Chennai", 13.08, 80.27),
    "north_eastern": ("Guwahati", 26.14, 91.74),
}

VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "precipitation",
]


def fetch_city(region, city, lat, lon, start, end, retries=5):
    """Download hourly weather for one location; retry with back-off on rate limits."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(VARIABLES),
        "wind_speed_unit": "ms",
        "timezone": "Asia/Kolkata",  # IST, to align with Indian grid demand timestamps
    }
    for attempt in range(retries):
        resp = requests.get(URL, params=params, timeout=120)
        if resp.status_code == 200:
            df = pd.DataFrame(resp.json()["hourly"]).rename(columns={"time": "datetime"})
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.insert(1, "region", region)
            df.insert(2, "city", city)
            return df
        wait = 20 * (attempt + 1)
        print(f"  {city}: HTTP {resp.status_code}, retrying in {wait}s ...")
        time.sleep(wait)
    raise RuntimeError(f"Could not fetch weather for {city} (last status {resp.status_code}): {resp.text[:200]}")


def main(start="2019-01-01", end="2024-04-30", out="data/raw/india_weather_hourly.csv"):
    frames = []
    for region, (city, lat, lon) in CITIES.items():
        print(f"Fetching {city} ({region}) ...")
        frames.append(fetch_city(region, city, lat, lon, start, end))
        time.sleep(5)  # be polite to the free API
    weather = pd.concat(frames, ignore_index=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(out, index=False)
    print(f"Saved {len(weather):,} rows to {out}")
    print(weather.groupby("city").size())


if __name__ == "__main__":
    main()