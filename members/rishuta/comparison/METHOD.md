# How these models work

- Input: the supplied hourly CSV with 11 pollutants and missing readings.
- Training: 2025. Testing: January-August 2026.
- Models: one LSTM or GRU layer with 32 hidden units and a linear output.
- Daily: predict 24 hourly values and average them, or predict the daily average directly.
- Weekly: predict 7 daily averages and average them, or predict the weekly average directly.
- Monthly: predict 4-5 weekly blocks and average by duration, or predict the monthly average directly. The final block stops at the month boundary.
- Lookback: 48 hours for hourly inputs; 14 days for direct daily; 28 days for daily-to-weekly; 8 weeks for weekly inputs; 3 months for monthly inputs.
- Validation: fit targets ending by October 1 and validate on origins from October through December 2025. Use this to choose epochs, then retrain on all eligible 2025 samples.
- Settings: Adam, learning rate 0.001, batch size 64, up to 60 epochs, early stopping after 6 epochs, seed 42.
- Missing inputs: carry forward earlier values within each window; use training medians for initial gaps. Scale using training values only.
- Actual averages need at least 75% hourly coverage. Missing targets are masked during training and excluded from scoring.
- Each forecast uses only observations before its start. Later 2026 forecasts can use earlier observed 2026 data, without retraining.
- Training origins: every 6 hours for hourly inputs, daily for daily inputs, Monday for weekly inputs and month starts for monthly inputs. Test origins match the target period start.
- Weekly-to-monthly training forecasts one calendar month from each Monday; testing starts on the first of the month. This origin difference is a limitation.
- Errors: MAE and RMSE in original units. CO uses mg/m3, NOx uses ppb, others use source micrograms/m3.

Only 12 monthly training observations exist, leaving 9 direct monthly windows and 8 test months. Monthly results are exploratory. Source cleaning has not been audited. Missingness can make observed and duration-weighted block averages differ. One seeded run does not prove superiority. AQI classification is not included yet.
