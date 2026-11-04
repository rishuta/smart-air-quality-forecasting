# Rishuta - base models

- Model 5: LSTM: `models/lstm/`
- Model 6: GRU: `models/gru/`
- Shared implementation for these two models only: `models/recurrent/`
- Combined results for these two models only: `results/recurrent/`
- Methodology and results discussion: `docs/`

Existing checkpoints and evaluated predictions are included. These files are Rishuta's contributions; they are separate from other team members' work.

From the repository root:

```bash
python -m pip install -r members/rishuta/requirements.txt
python members/rishuta/models/lstm/train.py
python members/rishuta/models/gru/train.py
```

The source datasets belong in the root `data/raw/` directory, or supply `--data-dir PATH`.
