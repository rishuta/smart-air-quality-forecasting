# Multi-Pollutant Time-Series Forecasting and Air Quality Index Classification for Smart Air Quality Monitoring

Shared final-year project repository for team model development and evaluation.

## Model contributions

| Model | Contributor | Status | Folder |
|---|---|---|---|
| Model 5: LSTM | Rishuta Sheel (@rishuta) | Trained and evaluated | [models/lstm](models/lstm) |
| Model 6: GRU | Rishuta Sheel (@rishuta) | Trained and evaluated | [models/gru](models/gru) |
| Other team models | Team members | Awaiting contributions | Copy [models/_template](models/_template) |

LSTM and GRU forecast seven pollutants using 2025 for training and January-August 2026 for testing. Daily, weekly and monthly comparisons are included. AQI classification and deployment remain pending. Weekly and monthly results are exploratory due to limited training data.

## Setup and run

Use Python 3.11. From the repository root, create and activate an environment:

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.
Place the two source CSVs in `data/raw/` (see its README), then run:

```bash
python models/lstm/train.py
python models/gru/train.py
```

Each command saves its own results. Options include `--data-dir PATH` and `--epochs NUMBER`. The shared implementation is `models/recurrent/train.py`. Completed checkpoints, predictions and metrics are included.

## Team workflow

Read [CONTRIBUTING.md](CONTRIBUTING.md). Each teammate uses their own GitHub account, branch and pull request. Copy the model template into a uniquely named model folder; add code, dependencies, documentation and results.

- [Experiment methodology](docs/recurrent_methodology.md)
- [Results summary](docs/recurrent_results.md)
- [Combined recurrent results](results/recurrent)
- [Project diary](docs/project_diary.md)
- [Pending work](docs/roadmap.md)

Initialization imports existing work as a current snapshot; it does not represent earlier development history. Record future work through genuine commits and diary entries.
