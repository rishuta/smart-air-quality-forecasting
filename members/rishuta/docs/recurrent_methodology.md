# LSTM and GRU forecasting experiment

Model 5 is LSTM; Model 6 is GRU. Both forecast PM2.5, PM10, NO2, NH3, SO2, CO and ozone. All fitting uses 2025; testing uses January–August 2026.

## Comparisons

| Target mean | Finer input route | Direct route |
|---|---|---|
| Daily | Previous 48 hours → next 24 hourly values → average | Previous 14 daily means → next daily mean |
| Weekly | Previous 28 daily means → next 7 daily means → average | Previous 8 weekly means → next weekly mean |
| Monthly | Previous 8 seven-day means → next 4–5 block means → duration-weighted average | Previous 3 calendar-month means → next monthly mean |

Each forecast is issued at the beginning of the target period and predicts that whole period at once. Weekly test periods run Monday to Monday. Monthly blocks start on the first of the month; the last block ends at the month boundary and may contain fewer than seven days. The model predicts that partial block's mean separately. February's unused fifth block is excluded from loss and aggregation. Seven-day input blocks end at the issue time, so they do not always match calendar weeks.

Hourly-to-daily training origins occur every six hours to reduce redundant overlapping windows; testing occurs at midnight. Daily-input training origins occur daily. Weekly-input training origins occur on Mondays; for weekly-to-monthly training, each target extends one calendar month from that Monday. Monthly testing occurs on calendar month starts. This origin difference is a limitation of the monthly comparison.

## Training

Both networks use one recurrent layer with 32 hidden units, a linear output head, Adam at learning rate 0.001, batch size 64, masked standardized mean squared error, gradient clipping at 1 and seed 42. Epoch selection uses targets ending by 1 October 2025 for fitting and origins in October–December for validation. Windows crossing the validation boundary are excluded. Training stops after six epochs without improvement, up to 60 epochs. A fresh model then trains on all eligible 2025 samples for the selected epoch count.

Only measurement columns shared by both years are used. Rainfall, MP-Xylene and VWS are excluded. Cyclical hour and day-of-year features are added. Scaling and fallback medians use training samples only. Missing inputs fill forward within each input window, with training medians for initial gaps. No future observations fill gaps. Negative forecasts are clipped at zero.

An observed aggregate requires at least 75% of its expected hourly measurements for that pollutant. Missing training targets are masked; missing actual period means are excluded from scoring separately by pollutant. Observed means use available measurements, while predicted block aggregation uses block duration; uneven missingness can therefore affect the comparison.

The model never fits 2026 data. Later 2026 forecasts use earlier observed 2026 measurements available before their issue time. This is rolling evaluation, not a single forecast of the whole year issued in December 2025. Incomplete final weeks are excluded.

## Outputs

- `results/error_comparison.csv`: per-pollutant MAE and RMSE, with scored test counts.
- `results/predictions_2026.csv`: forecast boundaries, actuals and predictions.
- `results/training_summary.csv`: sample counts and selected epochs.
- `results/*_history.csv`: epoch-selection losses.
- `results/*.pt`: trained weights, architecture settings, feature order and preprocessing. Only load trusted checkpoints.
- `results/observed_*_means.csv` and `observed_*_counts.csv`: unfiltered source averages and counts for inspection. Boundary periods may be partial. The predictions table contains the authoritative coverage-filtered actuals used for evaluation.

Errors retain source concentration units: CO uses mg/m³; the other targets use µg/m³ as indicated in the source headers. Compare approaches within the same pollutant and target scale; do not average raw errors across pollutants.

## Reproduce

Install `numpy pandas torch` in Python, then run:

```powershell
python members/rishuta/models/recurrent/train.py --data-dir "data/raw" --output-dir results
```

## Interpretation limits

One station, one training year and one seeded run do not establish general superiority. Direct monthly models have just nine supervised training windows and eight test periods; weekly and monthly results are exploratory. Several selected epoch counts reach the maximum of 60, so further tuning may change results. Earlier cleaning of the supplied CSVs has not been audited for leakage or interpolation.

The experiment covers concentration forecasting and errors. The separate AQI calculation/classification stage from the presentation still requires pollutant-specific averaging periods and breakpoints.
