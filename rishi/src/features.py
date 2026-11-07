"""Causal hourly features shared by training and prediction."""
import numpy as np
import pandas as pd

TARGET = "PM2.5 (µg/m³)"


def load_data(path):
    data = pd.read_csv(path, encoding="utf-8-sig")
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="raise")
    data = data.sort_values("Timestamp").set_index("Timestamp")
    if data.index.has_duplicates:
        raise ValueError("Duplicate timestamps must be resolved explicitly.")
    data = data.apply(pd.to_numeric, errors="raise")
    if np.isinf(data.to_numpy()).any() or (data < 0).any().any():
        raise ValueError("Infinite or negative concentrations require review.")
    # Explicitly insert absent hours so shift(-1) always means one hour ahead.
    return data.asfreq("h")


def make_features(data):
    x = data.copy()
    for lag in (1, 2, 3, 24, 168):
        x[f"pm25_lag_{lag}h"] = data[TARGET].shift(lag)
    for window in (3, 24, 168):
        past = data[TARGET].rolling(window, min_periods=1)
        x[f"pm25_mean_{window}h"] = past.mean()
        x[f"pm25_std_{window}h"] = past.std()
    # The forecast hour is known at issue time.
    forecast_time = x.index + pd.Timedelta(hours=1)
    for name, values, period in (
        ("hour", forecast_time.hour, 24),
        ("weekday", forecast_time.day_of_week, 7),
        ("year_day", forecast_time.day_of_year, 365.25),
    ):
        x[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        x[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
    return x


def chronological_masks(data):
    """70/15/15 by timeline; purge labels reaching the next partition."""
    a, b = data.index[int(len(data) * .70)], data.index[int(len(data) * .85)]
    issue = data.index
    label_time = issue + pd.Timedelta(hours=1)
    observed = data[TARGET].shift(-1).notna().to_numpy()
    return {
        "train": (issue < a) & (label_time < a) & observed,
        "validation": (issue >= a) & (issue < b) & (label_time < b) & observed,
        "test": (issue >= b) & observed,
    }
