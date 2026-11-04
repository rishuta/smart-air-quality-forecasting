# Team contribution guide

1. Accept your invitation and clone the repository using your own GitHub account.
2. Create a branch such as `models/yourname-modelname`.
3. Copy `models/_template` to `models/your_model_name`.
4. Add code/notebooks, dependencies, a model card, results and reproduction commands. Use configurable paths. Do not overwrite another person's model folder.
5. Train and validate on 2025. Keep 2026 for final testing. Explain preprocessing, missing data, lookback and forecast horizon. Compare approaches only on matching targets and test periods.
6. Commit meaningful work with clear messages using your own Git identity. Never invent or backdate contributions.
7. Push your branch, open a pull request, request teammate review and address comments before merging.
8. Update the project diary with evidence links and actual dates.

Report per-pollutant MAE and RMSE in original units, scored counts and actual/predicted values with forecast boundaries. Do not average raw errors across different pollutants. Keep datasets, secrets and environment folders out of Git. Small checkpoints may be committed; agree on artifact storage for large files.

## Upload through the browser

After accepting the invitation, open your model folder, choose **Add file > Upload files**, upload your files to a new branch and open a pull request. Each member must use their own account for attribution.
