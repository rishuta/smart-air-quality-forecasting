import runpy, sys
from pathlib import Path

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    sys.argv[1:1] = ["--model", "LSTM", "--output-dir", str(here / "results")]
    runpy.run_path(str(here.parent / "recurrent" / "train.py"), run_name="__main__")
