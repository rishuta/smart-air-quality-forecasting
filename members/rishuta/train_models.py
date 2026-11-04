"""LSTM/GRU temporal-resolution comparison. Requires numpy, pandas, torch."""
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
TARGETS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'SO2', 'CO', 'Ozone', 'Benzene', 'Toluene']
CONFIGS = [('daily', 'hourly', 1, 48, 24), ('daily', 'daily', 24, 14, 1), ('weekly', 'daily', 24, 28, 7), ('weekly', 'weekly', 168, 8, 1), ('monthly', 'weekly', 168, 8, 5), ('monthly', 'monthly', 0, 3, 1)]
AVERAGE_CACHE = {}

def load(path):
    """Read the CSV and give each pollutant a short column name."""
    d = pd.read_csv(path)
    d.columns = [c.split(' (')[0] for c in d.columns]
    d['Timestamp'] = pd.to_datetime(d.Timestamp)
    return d.set_index('Timestamp').sort_index().apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)

class Network(nn.Module):

    def __init__(self, kind, nin, nout):
        super().__init__()
        self.rnn = getattr(nn, kind)(nin, 32, batch_first=True)
        self.head = nn.Linear(32, nout)

    def forward(self, x):
        z, _ = self.rnn(x)
        return self.head(z[:, -1])

def end_of(origin, period):
    """Find the end of a day, week or calendar month."""
    return origin + {'daily': pd.Timedelta(days=1), 'weekly': pd.Timedelta(days=7)}.get(period, pd.DateOffset(months=1))

def average(raw, a, b):
    """Calculate a mean only when at least 75% of hourly readings exist."""
    key = tuple(raw.columns)
    if key not in AVERAGE_CACHE:
        values = raw.to_numpy()
        good = np.isfinite(values)
        AVERAGE_CACHE[key] = (raw.index[0], len(raw), np.vstack([np.zeros(raw.shape[1]), np.cumsum(np.where(good, values, 0), axis=0)]), np.vstack([np.zeros(raw.shape[1]), np.cumsum(good, axis=0)]))
    start, length, sums, counts = AVERAGE_CACHE[key]
    lo = int((a - start) / pd.Timedelta(hours=1))
    hi = int((b - start) / pd.Timedelta(hours=1))
    expected = int((b - a) / pd.Timedelta(hours=1))
    if lo < 0 or hi > length:
        return np.full(raw.shape[1], np.nan)
    count = counts[hi] - counts[lo]
    return np.where(count >= 0.75 * expected, (sums[hi] - sums[lo]) / np.maximum(count, 1), np.nan)

def examples(raw, cols, period, base, hours, lookback, horizon):
    """Build past-input and future-target windows without crossing year boundaries."""
    first = raw.index.min()
    stop = raw.index.max() + pd.Timedelta(hours=1)
    if base == 'monthly':
        origins = pd.date_range(first.normalize(), stop, freq='MS')
    elif base == 'weekly':
        origins = pd.date_range(first.normalize(), stop, freq='W-MON')
    elif base == 'hourly':
        origins = pd.date_range(first, stop, freq='6h')
    else:
        origins = pd.date_range(first.normalize(), stop, freq='D')
    freq = {'daily': 'D', 'weekly': 'W-MON', 'monthly': 'MS'}[period]
    tests = pd.date_range('2026-01-01', stop, freq=freq)
    origins = origins[origins < pd.Timestamp('2026-01-01')].union(tests)
    xs = []
    ys = []
    masks = []
    weights = []
    dates = []
    ends = []
    actuals = []
    inputs = raw[cols]
    targets = raw[TARGETS]
    for o in origins:
        end = end_of(o, period)
        if end > stop or (o.year == 2025 and end > pd.Timestamp('2026-01-01')):
            continue
        starts = [o - pd.DateOffset(months=lookback - i) for i in range(lookback)] if base == 'monthly' else [o - pd.Timedelta(hours=hours * (lookback - i)) for i in range(lookback)]
        if starts[0] < first:
            continue
        bounds = starts + [o]
        x = np.stack([average(inputs, a, b) for a, b in zip(bounds[:-1], bounds[1:])])
        calendar = np.array([[np.sin(2 * np.pi * b.day_of_year / 365.25), np.cos(2 * np.pi * b.day_of_year / 365.25), np.sin(2 * np.pi * b.hour / 24), np.cos(2 * np.pi * b.hour / 24)] for b in starts])
        x = np.concatenate([x, calendar], axis=1)
        yy = np.full((horizon, len(TARGETS)), np.nan)
        ww = np.zeros(horizon)
        for j in range(horizon):
            a = o if horizon == 1 else o + pd.Timedelta(hours=hours * j)
            b = end if horizon == 1 else min(end, a + pd.Timedelta(hours=hours))
            if a >= end:
                continue
            yy[j] = average(targets, a, b)
            ww[j] = (b - a) / pd.Timedelta(hours=1)
        xs.append(x)
        ys.append(yy)
        weights.append(ww)
        dates.append(o)
        ends.append(end)
        actuals.append(average(targets, o, end))
    return (np.array(xs), np.array(ys), np.array(weights), pd.DatetimeIndex(dates), pd.DatetimeIndex(ends), np.array(actuals))

def train(kind, x, y, mask, epochs, validation=None):
    """Fit a recurrent model; use validation to select the epoch count."""
    torch.manual_seed(42)
    model = Network(kind, x.shape[-1], y.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(42))
    best = np.inf
    chosen = 1
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        count = 0
        for bx, by, bm in loader:
            opt.zero_grad()
            loss = ((model(bx) - by) ** 2 * bm).sum() / bm.sum().clamp(min=1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            total += loss.item() * len(bx)
            count += len(bx)
        row = {'epoch': epoch, 'training_scaled_mse': total / count}
        if validation is not None:
            vx, vy, vm = validation
            p = predict(model, vx)
            score = float(((p - vy) ** 2 * vm).sum() / max(vm.sum(), 1))
            row['validation_scaled_mse'] = score
            if score < best - 1e-05:
                best = score
                chosen = epoch
                stale = 0
            else:
                stale += 1
        history.append(row)
        if validation is not None and stale >= 6:
            break
    return (model, chosen, history)

def predict(model, x):
    """Run the model in batches without changing its weights."""
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.tensor(x[i:i + 256], dtype=torch.float32)).numpy() for i in range(0, len(x), 256)])

def preprocess(x, y, idx):
    """Fill past inputs and scale values using training samples only."""
    flat = x[idx].reshape(-1, x.shape[-1])
    med = np.nanmedian(flat, axis=0)
    med = np.nan_to_num(med)
    filled = x.copy()
    for t in range(x.shape[1]):
        fallback = med if t == 0 else filled[:, t - 1, :]
        filled[:, t, :] = np.where(np.isfinite(filled[:, t, :]), filled[:, t, :], fallback)
    mean = filled[idx].mean(axis=(0, 1))
    std = filled[idx].std(axis=(0, 1))
    std[std < 1e-08] = 1
    ym = np.nanmean(y[idx], axis=(0, 1))
    ys = np.nanstd(y[idx], axis=(0, 1))
    ys[ys < 1e-08] = 1
    mask = np.isfinite(y).reshape(len(y), -1)
    target = np.nan_to_num((y - ym) / ys).reshape(len(y), -1)
    return ((filled - mean) / std, target, mask, dict(median=med.tolist(), mean=mean.tolist(), std=std.tolist(), target_mean=ym.tolist(), target_std=ys.tolist()))

def main():
    """Train on 2025, test on 2026 and save predictions, errors and weights."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-file', default=str(Path(__file__).resolve().parents[2] / 'data' / 'Delhi_AnandVihar_2025_Aug2026_11pollutants_combined_unimputed.csv'))
    ap.add_argument('--output-dir', default=str(Path(__file__).parent / 'results'))
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--model', choices=['LSTM', 'GRU', 'both'], default='both')
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    AVERAGE_CACHE.clear()
    torch.set_num_threads(4)
    np.random.seed(42)
    random.seed(42)
    dataset = load(args.data_file)
    a = dataset[dataset.index.year == 2025]
    b = dataset[dataset.index.year == 2026]
    assert (a.index.year == 2025).all() and (b.index.year == 2026).all()
    raw = pd.concat([a, b])
    assert not raw.index.has_duplicates
    assert raw.index.equals(pd.date_range(raw.index.min(), raw.index.max(), freq='h'))
    cols = [c for c in a if c in b]
    rows = []
    logs = []
    predrows = []
    for period, base, hours, lookback, horizon in CONFIGS:
        x, y, w, dates, ends, actual = examples(raw, cols, period, base, hours, lookback, horizon)
        trainidx = np.where((ends <= pd.Timestamp('2026-01-01')) & np.isfinite(y).any(axis=(1, 2)))[0]
        validx = np.where((ends <= pd.Timestamp('2026-01-01')) & (dates >= pd.Timestamp('2025-10-01')) & np.isfinite(y).any(axis=(1, 2)))[0]
        earlyidx = np.where((ends <= pd.Timestamp('2025-10-01')) & np.isfinite(y).any(axis=(1, 2)))[0]
        testidx = np.where(dates >= pd.Timestamp('2026-01-01'))[0]
        if min(len(earlyidx), len(validx), len(testidx)) == 0:
            raise ValueError(f'Empty partition: {period}/{base}')
        assert ends[earlyidx].max() <= dates[validx].min()
        for kind in ['LSTM', 'GRU'] if args.model == 'both' else [args.model]:
            xx, yy, mask, prep = preprocess(x, y, earlyidx)
            _, epochs, history = train(kind, xx[earlyidx], yy[earlyidx], mask[earlyidx], args.epochs, (xx[validx], yy[validx], mask[validx]))
            xx, yy, mask, prep = preprocess(x, y, trainidx)
            model, _, _ = train(kind, xx[trainidx], yy[trainidx], mask[trainidx], epochs)
            forecast = predict(model, xx[testidx]).reshape(-1, horizon, len(TARGETS))
            forecast = np.maximum(0, forecast * np.array(prep['target_std']) + np.array(prep['target_mean']))
            agg = (forecast * w[testidx, :, None]).sum(axis=1) / w[testidx].sum(axis=1)[:, None]
            label = f'{kind}_{base}_to_{period}'
            torch.save({'state_dict': model.state_dict(), 'kind': kind, 'hidden_size': 32, 'input_columns': cols + ['day_sin', 'day_cos', 'hour_sin', 'hour_cos'], 'targets': TARGETS, 'lookback': lookback, 'horizon_blocks': horizon, 'base': base, 'period': period, 'preprocessing': prep}, out / f'{label}.pt')
            pd.DataFrame(history).to_csv(out / f'{label}_history.csv', index=False)
            for j, t in enumerate(TARGETS):
                observed = actual[testidx, j]
                ok = np.isfinite(observed)
                err = agg[ok, j] - observed[ok]
                rows.append(dict(model=kind, input_scale=base, forecast_scale=period, pollutant=t, n_test=int(ok.sum()), MAE=float(np.mean(abs(err))), RMSE=float(np.sqrt(np.mean(err ** 2)))))
                for k, i in enumerate(testidx):
                    predrows.append(dict(model=kind, input_scale=base, forecast_scale=period, period_start=str(dates[i]), period_end_exclusive=str(ends[i]), pollutant=t, actual=actual[i, j], predicted=agg[k, j]))
            log = dict(model=kind, input_scale=base, forecast_scale=period, train_samples=len(trainidx), validation_samples=len(validx), early_training_samples=len(earlyidx), test_periods=len(testidx), selected_epochs=epochs)
            logs.append(log)
            print(json.dumps(log), flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / 'error_comparison.csv', index=False)
    pd.DataFrame(predrows).to_csv(out / 'predictions_2026.csv', index=False)
    pd.DataFrame(logs).to_csv(out / 'training_summary.csv', index=False)
    for freq, name in [('D', 'daily'), ('W-MON', 'weekly'), ('MS', 'monthly')]:
        grouped = raw[cols].resample(freq, label='left', closed='left')
        grouped.mean().to_csv(out / f'observed_{name}_means.csv')
        grouped.count().to_csv(out / f'observed_{name}_counts.csv')
    (out / 'metadata.json').write_text(json.dumps({'seed': 42, 'coverage_threshold': 0.75, 'train_year': 2025, 'test_year': 2026, 'shared_columns': cols, 'targets': TARGETS, 'train_rows': len(a), 'test_rows': len(b), 'epochs_max': args.epochs}, indent=2))
    print('Finished. Results:', out, flush=True)
if __name__ == '__main__':
    main()
