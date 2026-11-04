# Forecasting results

Training: 2025. Testing: January–August 2026. Lower MAE and RMSE are better.

## PM2.5 comparison

| Target | Model | Input scale | Scored periods | MAE | RMSE |
|---|---|---|---:|---:|---:|
| daily | LSTM | hourly | 228 | 30.93 | 44.62 |
| daily | GRU | hourly | 228 | 33.19 | 45.85 |
| daily | LSTM | daily | 228 | 28.24 | 43.02 |
| daily | GRU | daily | 228 | 33.25 | 45.60 |
| weekly | LSTM | daily | 33 | 19.60 | 30.81 |
| weekly | GRU | daily | 33 | 27.51 | 39.29 |
| weekly | LSTM | weekly | 33 | 38.81 | 51.00 |
| weekly | GRU | weekly | 33 | 35.99 | 44.69 |
| monthly | LSTM | weekly | 8 | 41.70 | 53.76 |
| monthly | GRU | weekly | 8 | 45.21 | 56.95 |
| monthly | LSTM | monthly | 8 | 41.41 | 49.98 |
| monthly | GRU | monthly | 8 | 48.12 | 60.41 |

PM2.5 errors are in µg/m³. Missing actuals are excluded under the 75% coverage rule.

## Lowest RMSE in this run, by pollutant

| Target | Pollutant | Model | Input scale | MAE | RMSE |
|---|---|---|---|---:|---:|
| daily | PM2.5 | LSTM | daily | 28.24 | 43.02 |
| daily | PM10 | GRU | daily | 67.11 | 92.61 |
| daily | NO2 | GRU | hourly | 10.94 | 14.84 |
| daily | NH3 | LSTM | hourly | 8.63 | 11.50 |
| daily | SO2 | GRU | hourly | 7.68 | 9.33 |
| daily | CO | GRU | hourly | 0.59 | 0.72 |
| daily | Ozone | GRU | hourly | 4.07 | 5.40 |
| weekly | PM2.5 | LSTM | daily | 19.60 | 30.81 |
| weekly | PM10 | LSTM | weekly | 64.41 | 74.25 |
| weekly | NO2 | LSTM | daily | 15.15 | 19.71 |
| weekly | NH3 | GRU | daily | 9.40 | 11.74 |
| weekly | SO2 | LSTM | daily | 6.68 | 8.20 |
| weekly | CO | GRU | daily | 0.69 | 0.88 |
| weekly | Ozone | GRU | weekly | 4.01 | 4.75 |
| monthly | PM2.5 | LSTM | monthly | 41.41 | 49.98 |
| monthly | PM10 | GRU | monthly | 41.87 | 56.71 |
| monthly | NO2 | LSTM | weekly | 17.43 | 19.79 |
| monthly | NH3 | LSTM | monthly | 4.89 | 6.37 |
| monthly | SO2 | LSTM | weekly | 6.87 | 8.07 |
| monthly | CO | LSTM | weekly | 0.52 | 0.66 |
| monthly | Ozone | GRU | weekly | 4.55 | 5.09 |

These are descriptive comparisons from one seeded run, not proof of statistical superiority. Errors use each pollutant’s source units; CO is in mg/m³ and other targets in µg/m³.

## What the results support

For PM2.5, the daily-input LSTM has the lowest daily and weekly RMSE. The direct monthly LSTM has the lowest monthly RMSE, but only eight monthly test points and nine training windows make that finding highly uncertain. No method wins every pollutant and scale.

Both approaches use identical actual period means and matching test periods within each scale. All 84 reported MAE/RMSE pairs were recalculated from the saved predictions and matched.

See README.md for the architecture, train/validation separation, missing-data handling, month alignment and limitations. AQI classification has not been implemented in this experiment.