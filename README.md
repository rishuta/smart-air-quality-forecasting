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

Use your own GitHub account. Make a branch, add files inside your folder, commit, push and open a pull request. Ask a teammate to review it. See [CONTRIBUTING.md](CONTRIBUTING.md).

AQI classification and deployment are pending. Existing work was imported as a current snapshot; future commits should record real progress.

## Reports

Open Rishuta's LSTM and GRU PDFs in their model folders. Her comparison folder contains only one PDF report with all comparison tables and graphs. Other members' model and comparison folders are empty for their own contributions. `.gitkeep` only preserves empty folders in Git.
