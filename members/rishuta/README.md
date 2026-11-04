# Rishuta

- `lstm/LSTM_Report.pdf`: LSTM results and graphs.
- `gru/GRU_Report.pdf`: GRU results and graphs.
- `comparison/Comparison_Report.pdf`: both models compared.

Each model's `results/predictions_2026.csv` includes actual, predicted and error for every forecast. Error = predicted - actual. Missing actuals have blank errors.

Shared training: `train_models.py`. Rebuild reports: `make_reports.py`.
