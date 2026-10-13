"""Every paper table, figure and number macro: `uv run --no-project --with matplotlib --with numpy --with scipy --with tqdm LlMBO/scripts/all.py` from the repo root."""

import budget
import gap
import headline
import obsmodel
import scaling
import scatter
import shift
import tables
from tqdm import tqdm
from common import load_data

if __name__ == "__main__":
    data = load_data()
    scripts = (tables, headline, scaling, scatter, gap, shift, budget)
    for script in (bar := tqdm(scripts, desc="figures", unit="script")):
        bar.set_postfix_str(script.__name__)
        script.main(*data)
    obsmodel.main()
