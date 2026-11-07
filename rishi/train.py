"""Train Model 3 and Model 4; select settings without using test outcomes."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.features import TARGET, chronological_masks, load_data, make_features

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data/Delhi_AnandVihar_2025_Aug2026_11pollutants_combined_unimputed.csv"


def metrics(y, pred):
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": float(r2_score(y, pred)), "n": len(y)}


def pipeline(kind, params):
    model = (RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=4, **params)
             if kind == "model_3_random_forest" else
             XGBRegressor(objective="reg:squarederror", tree_method="hist",
                          random_state=42, n_jobs=4, **params))
    return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                     ("model", model)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    for folder in ("reports", "models"):
        (ROOT / folder).mkdir(exist_ok=True)
    d = load_data(args.data)
    x, y = make_features(d), d[TARGET].shift(-1)
    masks = chronological_masks(d)
    tr, va, te = (masks[k] for k in ("train", "validation", "test"))
    fit = tr | va
    grids = {
        "model_3_random_forest": [
            {"max_depth": 12, "min_samples_leaf": 2, "max_features": .8},
            {"max_depth": 20, "min_samples_leaf": 4, "max_features": 1.0},
            {"max_depth": None, "min_samples_leaf": 2, "max_features": 1.0}],
        "model_4_xgboost": [
            {"n_estimators": 250, "max_depth": 3, "learning_rate": .04,
             "subsample": .8, "colsample_bytree": .8, "reg_lambda": 5},
            {"n_estimators": 400, "max_depth": 4, "learning_rate": .03,
             "subsample": .8, "colsample_bytree": .9, "reg_lambda": 10},
            {"n_estimators": 250, "max_depth": 6, "learning_rate": .03,
             "subsample": .9, "colsample_bytree": .9, "reg_lambda": 10}],
    }
    results, tuning, saved = [], [], {}
    predictions = pd.DataFrame({"issue_time": d.index[te],
                                "target_time": d.index[te] + pd.Timedelta(hours=1),
                                "actual_pm25": y[te].values})
    # Carry the latest observed value forward, never backwards from the future.
    baseline = d[TARGET].ffill().fillna(d.loc[fit, TARGET].median())[te]
    predictions["persistence"] = baseline.values
    results.append({"model": "persistence", **metrics(y[te], baseline)})
    for kind, candidates in grids.items():
        scores = []
        for params in candidates:
            p = pipeline(kind, params)
            p.fit(x[tr], y[tr])
            score = metrics(y[va], p.predict(x[va]))
            tuning.append({"model": kind, "parameters": params, **score})
            scores.append(score["RMSE"])
            print(kind, params, "validation RMSE", score["RMSE"], flush=True)
        best = candidates[int(np.argmin(scores))]
        p = pipeline(kind, best)
        p.fit(x[fit], y[fit])
        pred = p.predict(x[te])
        joblib.dump(p, ROOT / f"models/{kind}.joblib", compress=3)
        restored = joblib.load(ROOT / f"models/{kind}.joblib")
        np.testing.assert_allclose(restored.predict(x[te].iloc[:20]), pred[:20])
        if kind == "model_4_xgboost":
            p.named_steps["model"].save_model(ROOT / "models/model_4_xgboost.json")
        saved[kind] = best
        predictions[kind] = pred
        results.append({"model": kind, **metrics(y[te], pred)})
        names = p.named_steps["imputer"].get_feature_names_out()
        importance = pd.Series(p.named_steps["model"].feature_importances_, index=names)
        importance.sort_values(ascending=False).to_csv(ROOT / f"reports/{kind}_importance.csv", header=["importance"])
        fig, ax = plt.subplots(figsize=(9, 6))
        importance.nlargest(15).sort_values().plot.barh(ax=ax, color="#237A83")
        ax.set(title=kind.replace("_", " ").title(), xlabel="Model importance (not causal influence)")
        fig.tight_layout()
        fig.savefig(ROOT / f"reports/{kind}_importance.png", dpi=150)
        plt.close(fig)
    scores = pd.DataFrame(results)
    scores.to_csv(ROOT / "reports/test_metrics.csv", index=False)
    predictions.to_csv(ROOT / "reports/test_predictions.csv", index=False)
    pd.DataFrame([{**r, "parameters": json.dumps(r["parameters"])} for r in tuning]).to_csv(ROOT / "reports/validation_search.csv", index=False)
    metadata = {
        "target": TARGET, "horizon_hours": 1, "seed": 42,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "input_rows": len(d), "start": str(d.index.min()), "end": str(d.index.max()),
        "missing_by_column": d.isna().sum().to_dict(), "feature_columns": list(x.columns),
        "splits": {k: {"n": int(m.sum()), "issue_start": str(d.index[m].min()),
                          "issue_end": str(d.index[m].max()),
                          "target_end": str((d.index[m] + pd.Timedelta(hours=1)).max())}
                   for k, m in masks.items()},
        "selected_parameters": saved,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__,
                     "xgboost": xgboost.__version__},
    }
    (ROOT / "reports/run_metadata.json").write_text(json.dumps(metadata, indent=2))
    lines = ["# Held-out evaluation", "", "Target: PM2.5 one hour ahead. Errors are in µg/m³.", "",
             "| Model | MAE | RMSE | R² | Test rows |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for r in results:
        lines.append(f"| {r['model']} | {r['MAE']:.3f} | {r['RMSE']:.3f} | {r['R2']:.4f} | {r['n']} |")
    lines += ["", "## Time partitions", "", "Final models were refitted on training plus validation rows. No test labels were used for fitting or parameter selection.", ""]
    for k, split in metadata["splits"].items():
        lines.append(f"- {k.title()}: {split['n']:,} observed targets; issue times {split['issue_start']} to {split['issue_end']}; final target {split['target_end']}.")
    lines += ["", "## Comparison with persistence", ""]
    for r in results[1:]:
        improvement = 100 * (results[0]["RMSE"] - r["RMSE"]) / results[0]["RMSE"]
        lines.append(f"- {r['model']}: RMSE reduction versus persistence = {improvement:.2f}% (negative means worse).")
    lines += ["", "## Selected settings", "", "```json", json.dumps(saved, indent=2), "```", "",
              "Settings were selected by the lowest validation RMSE from three candidates per model; see validation_search.csv.", "",
              "This is a rolling one-hour-ahead simulation on one station and one seasonal holdout. It is not a year-round or multi-step forecast validation. Missing targets are omitted; actual historical readings available at each issue time are used. See the README for preprocessing, availability assumptions, and limitations.", ""]
    (ROOT / "reports/RESULTS.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plotted = predictions.set_index("target_time")
    plotted[["actual_pm25", "model_3_random_forest", "model_4_xgboost"]].iloc[:168].plot(ax=axes[0], linewidth=1)
    axes[0].set(title="First test week: hourly PM2.5", ylabel="PM2.5 (µg/m³)", xlabel="Target time (source timezone unspecified)")
    scores.set_index("model")[["MAE", "RMSE"]].plot.bar(ax=axes[1], rot=0)
    axes[1].set(title="Full held-out test set: lower error is better", ylabel="µg/m³", xlabel="")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/evaluation.png", dpi=150)
    plt.close(fig)
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
