"""Forecast the hour following the most recent row in a history CSV."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from src.features import load_data, make_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "models/model_4_xgboost.joblib")
    args = parser.parse_args()
    d = load_data(args.data)
    p = joblib.load(args.model)
    value = float(p.predict(make_features(d).iloc[[-1]])[0])
    print(f"Target time: {d.index[-1] + pd.Timedelta(hours=1)}")
    print(f"Forecast PM2.5: {value:.2f} µg/m³")


if __name__ == "__main__":
    main()
