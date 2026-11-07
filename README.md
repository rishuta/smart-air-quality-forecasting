# Multi-Pollutant Time-Series Forecasting and Air Quality Index Classification for Smart Air Quality Monitoring

This project studies pollutant concentration forecasts for Anand Vihar, Delhi, and their use in smart air-quality monitoring. The current evaluation compares daily, weekly and monthly concentration forecasts.

## Team and models

| Member | Models |
|---|---|
| Hari | Persistence and SARIMA |
| Rishi | Random Forest and XGBoost |
| Rishuta | LSTM and GRU |
| Akash | Patch TCN and Patch TST |

## Dataset and evaluation

The dataset contains **14,592 hourly observations and 11 pollutants**, covering January 2025 to August 2026. Models train and validate on **2025** and test on **2026**.

Daily, weekly and monthly predictions are evaluated using **MAE**, **RMSE** and the signed error **predicted minus actual**. Each observed period average requires at least 75% hourly coverage. Monthly findings are exploratory because only one training year is available.

## Results

- [LSTM report](members/rishuta/lstm/LSTM_Report.pdf)
- [GRU report](members/rishuta/gru/GRU_Report.pdf)
- [LSTM and GRU comparison](members/rishuta/comparison/Comparison_Report.pdf)
- [Rishi: Random Forest and XGBoost](rishi/README.md) — previously trained one-hour-ahead PM2.5 models, with a separate 70/15/15 chronological evaluation. These scores are not directly comparable with the shared daily/weekly/monthly protocol above.

The LSTM and GRU reports contain concentration forecasts, error tables and comparison graphs for all 11 pollutants. Rishi's separate hourly PM2.5 submission is also available. The remaining team models have not yet been submitted to this repository. AQI classification and prototype deployment are future stages.
