# Multi-Pollutant Time-Series Forecasting and Air Quality Index Classification for Smart Air Quality Monitoring

Shared final-year project repository. Current stage: individual base-model training. Each member keeps their code, dependencies, documentation, checkpoints and results in a separate workspace.

| Member | Assigned base models | Workspace | Repository status |
|---|---|---|---|
| Hari | Model 1: Persistence; Model 2: SARIMA | [members/hari](members/hari) | Awaiting files |
| Rishi | Model 3: Random Forest Regressor; Model 4: XGBoost Regressor | [members/rishi](members/rishi) | Awaiting files |
| Rishuta | Model 5: LSTM; Model 6: GRU | [members/rishuta](members/rishuta) | Code, trained models and evaluation included |
| Aakash | Assignment pending | [members/aakash](members/aakash) | Awaiting files |

## Keep contributions separate

Work only in your own member folder. Use your own GitHub account and branch, then open a pull request. Do not change another person's files without discussing it. Add your dependency list and run instructions to your own workspace. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [model template](templates/README.md).

Source dataset instructions and checksums are shared in [data/raw](data/raw); CSV data is excluded from Git. The agreed split is 2025 for training/validation and 2026 for testing. Each member must document the targets, horizon, missing-data handling and evaluation independently. No team-wide model ranking or integration is claimed at this stage.

## Run Rishuta's models

Use Python 3.11. Create and activate a virtual environment, then run from the repository root:

```bash
python -m pip install -r members/rishuta/requirements.txt
python members/rishuta/models/lstm/train.py
python members/rishuta/models/gru/train.py
```

Place the two datasets in `data/raw/` first, or supply `--data-dir PATH`. Existing [results](members/rishuta/docs/recurrent_results.md) and [methodology](members/rishuta/docs/recurrent_methodology.md) describe only LSTM/GRU. Monthly findings are exploratory. AQI classification, model integration and deployment remain pending.

[Project diary](docs/project_diary.md) | [Pending work](docs/roadmap.md)

The initial import records existing work as a current snapshot. Future contributions must use genuine commits and actual dates.
