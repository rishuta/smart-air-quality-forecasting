"""Train this model and save its results in this folder."""
import runpy
import sys
from pathlib import Path

if __name__ == '__main__':
    folder = Path(__file__).resolve().parent
    sys.argv[1:1] = ['--model', 'LSTM', '--output-dir', str(folder / 'results')]
    runpy.run_path(str(folder.parent / 'train_models.py'), run_name='__main__')
