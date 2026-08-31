"""Bind the current data into a standalone, shareable HTML snapshot: `uv run dashboard/build.py` -> dashboard/dashboard.html."""

import argparse
import json
from pathlib import Path

from serve import TEMPLATE, collect

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines-dir", type=Path, default="outputs/baselines")
    parser.add_argument("--runs-dir", type=Path, default="outputs")
    parser.add_argument("--out", type=Path, default="dashboard/dashboard.html")
    args = parser.parse_args()

    payload = json.dumps(collect(args.baselines_dir, args.runs_dir)).replace(
        "</", "<\\/"
    )
    args.out.write_text(TEMPLATE.read_text().replace("/*__DATA__*/null", payload))
    print(f"wrote {args.out}")
