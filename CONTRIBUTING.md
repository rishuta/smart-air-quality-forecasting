# How to contribute

1. Accept the repository invitation.
2. Make a branch using your own GitHub account.
3. Add code and results only inside your member folder.
4. Give each model a short README: how to run it, what it predicts and its errors.
5. Put your comparison tables and graphs in your `comparison/` folder.
6. Commit and push your changes. Open a pull request for a teammate to review.
7. Add an actual-date entry to `docs/project_diary.md`.

Use 2025 for training/validation and 2026 for testing. Report MAE/RMSE separately for each pollutant. Compare models using the same actual periods. Never fill test targets and present them as real observations.

Do not overwrite another member's files or upload passwords. Do not invent past commits. Dependencies belong in your own requirements.txt.
