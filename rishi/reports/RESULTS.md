# Held-out evaluation

Target: PM2.5 one hour ahead. Errors are in µg/m³.

| Model | MAE | RMSE | R² | Test rows |
| --- | ---: | ---: | ---: | ---: |
| persistence | 15.364 | 22.325 | 0.7016 | 2050 |
| model_3_random_forest | 14.945 | 21.808 | 0.7152 | 2050 |
| model_4_xgboost | 15.315 | 21.740 | 0.7170 | 2050 |

## Time partitions

Final models were refitted on training plus validation rows. No test labels were used for fitting or parameter selection.

- Train: 9,572 observed targets; issue times 2025-01-01 00:00:00 to 2026-03-02 12:00:00; final target 2026-03-02 13:00:00.
- Validation: 2,026 observed targets; issue times 2026-03-02 14:00:00 to 2026-06-01 16:00:00; final target 2026-06-01 17:00:00.
- Test: 2,050 observed targets; issue times 2026-06-01 19:00:00 to 2026-08-31 22:00:00; final target 2026-08-31 23:00:00.

## Comparison with persistence

- model_3_random_forest: RMSE reduction versus persistence = 2.32% (negative means worse).
- model_4_xgboost: RMSE reduction versus persistence = 2.62% (negative means worse).

## Selected settings

```json
{
  "model_3_random_forest": {
    "max_depth": 12,
    "min_samples_leaf": 2,
    "max_features": 0.8
  },
  "model_4_xgboost": {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "reg_lambda": 10
  }
}
```

Settings were selected by the lowest validation RMSE from three candidates per model; see validation_search.csv.

This is a rolling one-hour-ahead simulation on one station and one seasonal holdout. It is not a year-round or multi-step forecast validation. Missing targets are omitted; actual historical readings available at each issue time are used. See the README for preprocessing, availability assumptions, and limitations.
