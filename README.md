# Multi-Pollutant Time-Series Forecasting and Air Quality Index Classification for Smart Air Quality Monitoring

We are currently training individual base models.

| Member | Model 1 | Model 2 | Work folder |
|---|---|---|---|
| Hari | Persistence | SARIMA | [Hari](members/hari) |
| Rishi | Random Forest | XGBoost | [Rishi](members/rishi) |
| Rishuta | LSTM | GRU | [Rishuta](members/rishuta) |
| Akash | Patch TCN | Patch TST | [Akash](members/aakash) |

Each member has **three folders**: two for models and one called `comparison` for results and graphs. Keep your work in your own folders.

## Data

Place the supplied combined 11-pollutant CSV in [data](data). The source file stays local; obtain it from the team. Train on **2025** and test on **2026**. Do not train or choose settings using test data.

## Run LSTM and GRU

Use Python 3.11. Create and activate a virtual environment, then run from this repository:

```bash
python -m pip install -r members/rishuta/requirements.txt
python members/rishuta/lstm/train.py
python members/rishuta/gru/train.py
python members/rishuta/make_reports.py
```

Each model uses the same dataset and predicts pollutant averages for days, weeks and months. MAE and RMSE show the errors; lower is better. Monthly results are exploratory because there is only one training year.

## Add your work

Add code and results inside your own member folder. Make a branch, commit your changes and open a pull request for a teammate to review.

AQI classification and deployment are pending.

## Reports

Each model folder contains its report. The `comparison` folder holds one combined report. Other members' folders are ready for their own work.
