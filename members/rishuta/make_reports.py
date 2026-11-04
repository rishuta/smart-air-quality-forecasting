import tempfile, shutil, math
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def unit(pollutant):
    if pollutant == 'CO':
        return 'mg/mÂ³'
    if pollutant == 'NOx':
        return 'ppb'
    return 'Âµg/mÂ³'

def make_axes(pollutants):
    rows = math.ceil(len(pollutants) / 3)
    figure, axes = plt.subplots(rows, 3, figsize=(16, rows * 3.4), squeeze=False)
    axes = axes.ravel()
    for axis in axes[len(pollutants):]:
        axis.set_visible(False)
    return (figure, axes)

def error_graphs(metrics):
    for scale, data in metrics.groupby('forecast_scale', sort=False):
        pollutants = data.pollutant.unique()
        for error in ['RMSE']:
            figure, axes = make_axes(pollutants)
            for axis, pollutant in zip(axes, pollutants):
                values = data[data.pollutant == pollutant]
                labels = values.model + '\n' + values.input_scale if data.model.nunique() > 1 else values.input_scale
                axis.bar(labels, values[error], color=['#2864ac', '#dc7c30', '#41966a', '#9264aa'][:len(values)])
                axis.set_title(pollutant)
                axis.set_ylabel(error + ' (' + unit(pollutant) + ')')
                axis.tick_params(axis='x', labelsize=8)
                axis.grid(axis='y', alpha=0.2)
                axis.set_axisbelow(True)
            note = 'Lower is better. Monthly results are exploratory.' if scale == 'monthly' else 'Lower is better.'
            figure.suptitle(f'{scale.capitalize()} forecasts: {error} comparison\n{note}', fontsize=15)
            figure.tight_layout(rect=(0, 0, 1, 0.94))
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
                axis.plot(pd.to_datetime(group.period_start), group.predicted, label=f'{model} ({source})' if data.model.nunique() > 1 else source.capitalize(), alpha=0.8, linewidth=1)
            axis.set_title(pollutant)
            axis.set_ylabel(unit(pollutant))
            axis.tick_params(axis='x', rotation=30, labelsize=8)
            axis.grid(alpha=0.2)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', ncol=min(5, len(labels)), fontsize=9)
        figure.suptitle(f'{scale.capitalize()} forecasts: actual vs predicted (2026)', fontsize=15)
        figure.tight_layout(rect=(0, 0.05, 1, 0.96))
        figure.savefig(GRAPHS / f'{scale}_actual_vs_predicted.png', dpi=150)
        plt.close(figure)
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = landscape(A4)

def heading(doc, title, subtitle=''):
    doc.setFillColor(colors.black)
    doc.setFont('Helvetica-Bold', 21)
    doc.drawString(35, HEIGHT - 45, title)
    doc.setFont('Helvetica', 10)
    doc.drawString(35, HEIGHT - 65, subtitle)

def finish(doc):
    doc.setFont('Helvetica', 9)
    doc.setFillColor(colors.grey)
    doc.drawRightString(WIDTH - 35, 20, str(doc.getPageNumber()))
    doc.showPage()

def overview(doc, name):
    heading(doc, name + ' results report', 'Multi-pollutant forecasting | 4 November 2026')
    text = doc.beginText(35, HEIGHT - 105)
    text.setFont('Helvetica', 12)
    text.setLeading(22)
    lines = ['Dataset: supplied combined hourly CSV with 11 pollutants and 14,592 rows.', 'Training and validation use 2025. Testing uses January to August 2026.', 'Daily: hourly forecasts averaged to a day versus a direct daily forecast.', 'Weekly: daily forecasts averaged to a week versus a direct weekly forecast.', 'Monthly: weekly blocks averaged to a month versus a direct monthly forecast.', 'Error = predicted - actual. Each prediction row includes this exact signed error.', 'Find every prediction and its error in each model folder: results/predictions_2026.csv.', 'MAE is the average absolute error. RMSE gives larger errors more weight.', 'Lower MAE and RMSE are better. Compare values within the same pollutant.', 'Units: CO is mg/m3; NOx is ppb; other pollutants use source micrograms/m3.', 'Each actual average needs 75% hourly coverage. Missing actuals are not scored.', 'Forecasts use measurements available before the period starts.', 'Later 2026 forecasts can use earlier observed 2026 data, without retraining.', 'Monthly findings are exploratory: only nine direct training windows and eight test months.', 'One seeded run does not establish superiority. AQI classification is still pending.', 'Network: one recurrent layer with 32 hidden units; Adam at 0.001; seed 42.', 'Validate on October-December 2025; select epochs, then refit on eligible 2025 data.', 'Missing past inputs fill forward or use training medians; scaling uses training data only.']
    for line in lines:
        text.textLine(line)
    doc.drawText(text)
    finish(doc)

def table_page(doc, name, scale, rows):
    heading(doc, scale.capitalize() + ' errors', 'MAE and RMSE in original units. Lower is better. N is the number of scored periods.')
    methods = list(dict.fromkeys(((row['model'], row['input_scale']) for row in rows)))
    multiple_models = len({model for model, source in methods}) > 1
    data = [['Pollutant', 'N']]
    for model, source in methods:
        label = model + ' ' + source if multiple_models else source.capitalize()
        data[0].extend([label + '\nMAE', label + '\nRMSE'])
    for pollutant in dict.fromkeys((row['pollutant'] for row in rows)):
        values = [row for row in rows if row['pollutant'] == pollutant]
        counts = {str(row['n_test']) for row in values}
        if len(counts) != 1:
            raise ValueError('Comparison must use matching actual periods.')
        record = [pollutant, values[0]['n_test']]
        for model, source in methods:
            value = next((row for row in values if row['model'] == model and row['input_scale'] == source))
            record.extend([f"{float(value['MAE']):.2f}", f"{float(value['RMSE']):.2f}"])
        data.append(record)
    widths = [100, 45] + [(WIDTH - 215) / (2 * len(methods))] * (2 * len(methods))
    table = Table(data, colWidths=widths, rowHeights=[35] + [26] * (len(data) - 1))
    table.setStyle(TableStyle([('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'), ('FONTSIZE', (0, 0), (-1, -1), 9), ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8edf2')), ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#d9d9d9')), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('ALIGN', (1, 0), (-1, -1), 'CENTER')]))
    width, height = table.wrap(WIDTH - 70, HEIGHT - 110)
    table.drawOn(doc, 35, HEIGHT - 92 - height)
    finish(doc)

def graph_page(doc, title, path):
    image = ImageReader(str(path))
    w, h = image.getSize()
    factor = min((WIDTH - 65) / w, (HEIGHT - 100) / h)
    iw, ih = (w * factor, h * factor)
    doc.drawImage(image, (WIDTH - iw) / 2, 45 + (HEIGHT - 100 - ih) / 2, width=iw, height=ih)
    finish(doc)

def build(name, path, rows, assets):
    doc = canvas.Canvas(str(path), pagesize=(WIDTH, HEIGHT))
    doc.setTitle(name + ' results report')
    doc.setAuthor('Rishuta Sheel')
    overview(doc, name)
    for scale in ['daily', 'weekly', 'monthly']:
        selected = [r for r in rows if r['forecast_scale'] == scale]
        selected.sort(key=lambda r: (r['pollutant'], r['model'], r['input_scale']))
        table_page(doc, name, scale, selected)
        graph_page(doc, scale.capitalize() + ' RMSE comparison', assets / f'{scale}_rmse.png')
        graph_page(doc, scale.capitalize() + ' actual and predicted values', assets / f'{scale}_actual_vs_predicted.png')
    doc.save()

def main():
    global GRAPHS
    ASSETS = Path(tempfile.mkdtemp(prefix='model_report_graphs_'))
    all_metrics = []
    all_predictions = []
    for model in ['LSTM', 'GRU']:
        metrics = pd.read_csv(ROOT / model.lower() / 'results/error_comparison.csv')
        metrics['model'] = model
        predictions = pd.read_csv(ROOT / model.lower() / 'results/predictions_2026.csv')
        predictions['model'] = model
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        GRAPHS = ASSETS / model
        GRAPHS.mkdir()
        error_graphs(metrics)
        forecast_graphs(predictions)
        build(model, ROOT / model.lower() / (model + '_Report.pdf'), metrics.to_dict('records'), GRAPHS)
    GRAPHS = ASSETS / 'comparison'
    GRAPHS.mkdir()
    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    error_graphs(metrics)
    forecast_graphs(predictions)
    build('LSTM and GRU comparison', ROOT / 'comparison/Comparison_Report.pdf', metrics.to_dict('records'), GRAPHS)
    shutil.rmtree(ASSETS)
    print('Updated the LSTM, GRU and comparison reports.')
if __name__ == '__main__':
    main()
