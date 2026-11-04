# LSTM vs GRU

- `train_models.py`: shared, commented training code for both models.
- `compare.py`: make comparison tables and graphs.
- `results/`: saved comparison tables.
- `graphs/`: MAE, RMSE and actual-vs-predicted plots.
- `RESULTS.md`: results with all graphs.
- `METHOD.md`: short explanation of training and testing.

From the repository root:

```bash
python members/rishuta/comparison/compare.py
```

Run each model first if you want to regenerate its results. The graphs use the new combined 11-pollutant dataset.
