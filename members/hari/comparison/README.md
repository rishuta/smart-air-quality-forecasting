# Comparison

Put results from your two models in `results/`, and graphs in `graphs/`. No results have been submitted yet.

Save `error_comparison.csv` with columns: model,input_scale,forecast_scale,pollutant,n_test,MAE,RMSE.
Save `predictions_2026.csv` with columns: model,input_scale,forecast_scale,period_start,period_end_exclusive,pollutant,actual,predicted. Use original concentration units.

Then run `python compare.py` from this folder. Install pandas, numpy and matplotlib first.
