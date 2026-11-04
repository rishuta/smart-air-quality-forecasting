"""Make error charts and actual-versus-predicted charts from saved results."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
GRAPHS = HERE / 'graphs'


def read_results(filename):
    """Use each model's outputs when present, otherwise use combined results."""
    files = sorted(HERE.parent.glob('*/results/' + filename))
    files = [file for file in files if file.parent.parent != HERE]
    if files:
        return pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    path = RESULTS / filename
    if not path.exists():
        raise SystemExit('No results yet. Add model outputs first. See README.md.')
    return pd.read_csv(path)


def unit(pollutant):
    if pollutant == 'CO':
        return 'mg/m³'
    if pollutant == 'NOx':
        return 'ppb'
    return 'µg/m³'


def make_axes(pollutants):
    rows = math.ceil(len(pollutants) / 3)
    figure, axes = plt.subplots(rows, 3, figsize=(16, rows * 3.4), squeeze=False)
    axes = axes.ravel()
    for axis in axes[len(pollutants):]:
        axis.set_visible(False)
    return figure, axes


def error_graphs(metrics):
    for scale, data in metrics.groupby('forecast_scale', sort=False):
        pollutants = data.pollutant.unique()
        for error in ['MAE', 'RMSE']:
            figure, axes = make_axes(pollutants)
            for axis, pollutant in zip(axes, pollutants):
                values = data[data.pollutant == pollutant]
                labels = values.model + '\n' + values.input_scale
                axis.bar(labels, values[error], color=['#2864ac', '#dc7c30', '#41966a', '#9264aa'][:len(values)])
                axis.set_title(pollutant)
                axis.set_ylabel(error + ' (' + unit(pollutant) + ')')
                axis.tick_params(axis='x', labelsize=8)
                axis.grid(axis='y', alpha=.2)
                axis.set_axisbelow(True)
            note = 'Lower is better. Monthly results are exploratory.' if scale == 'monthly' else 'Lower is better.'
            figure.suptitle(f'{scale.capitalize()} forecasts: {error} comparison\n{note}', fontsize=15)
            figure.tight_layout(rect=(0, 0, 1, .94))
            figure.savefig(GRAPHS / f'{scale}_{error.lower()}.png', dpi=150)
            plt.close(figure)


def forecast_graphs(predictions):
    for scale, data in predictions.groupby('forecast_scale', sort=False):
        pollutants = data.pollutant.unique()
        figure, axes = make_axes(pollutants)
        for axis, pollutant in zip(axes, pollutants):
            values = data[data.pollutant == pollutant]
            actual = values.drop_duplicates('period_start').sort_values('period_start')
            axis.plot(pd.to_datetime(actual.period_start), actual.actual, color='black', label='Actual', linewidth=1.6)
            for (model, source), group in values.groupby(['model', 'input_scale'], sort=False):
                group = group.sort_values('period_start')
                axis.plot(pd.to_datetime(group.period_start), group.predicted, label=f'{model} ({source})', alpha=.8, linewidth=1)
            axis.set_title(pollutant)
            axis.set_ylabel(unit(pollutant))
            axis.tick_params(axis='x', rotation=30, labelsize=8)
            axis.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', ncol=min(5, len(labels)), fontsize=9)
        figure.suptitle(f'{scale.capitalize()} forecasts: actual vs predicted (2026)', fontsize=15)
        figure.tight_layout(rect=(0, .05, 1, .96))
        figure.savefig(GRAPHS / f'{scale}_actual_vs_predicted.png', dpi=150)
        plt.close(figure)


def main():
    RESULTS.mkdir(exist_ok=True)
    GRAPHS.mkdir(exist_ok=True)
    metrics = read_results('error_comparison.csv')
    predictions = read_results('predictions_2026.csv')
    if metrics.duplicated(['model', 'input_scale', 'forecast_scale', 'pollutant']).any():
        raise ValueError('Duplicate model metrics: check the input result files.')
    # Every model must be compared against the same actuals for each scale.
    for _, group in predictions.groupby(['forecast_scale', 'period_start', 'pollutant']):
        values = group.actual.to_numpy()
        if not np.allclose(values, values[0], equal_nan=True):
            raise ValueError('Models have different actual targets. Compare matching periods first.')
    metrics.to_csv(RESULTS / 'error_comparison.csv', index=False)
    predictions.to_csv(RESULTS / 'predictions_2026.csv', index=False)
    error_graphs(metrics)
    forecast_graphs(predictions)
    lines = ['# Results', '', 'Train: 2025. Test: January-August 2026. Lower errors are better.', '',
             '| Target | Pollutant | Lowest RMSE in this run | Input | MAE | RMSE |',
             '|---|---|---|---|---:|---:|']
    for (scale, pollutant), group in metrics.groupby(['forecast_scale', 'pollutant'], sort=False):
        best = group.loc[group.RMSE.idxmin()]
        lines.append(f'| {scale} | {pollutant} | {best.model} | {best.input_scale} | {best.MAE:.2f} | {best.RMSE:.2f} |')
    lines += ['', 'These are results from one seeded run. Monthly results are exploratory. Errors use original units; do not average errors across pollutants.', '', '## Graphs', '']
    for file in sorted(GRAPHS.glob('*.png')):
        lines += [f'### {file.stem.replace("_", " ").capitalize()}', '', f'![{file.stem}](graphs/{file.name})', '']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved tables, graphs and RESULTS.md in', HERE)


if __name__ == '__main__':
    main()
