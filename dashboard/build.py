"""Bind the current data into standalone, shareable HTML snapshots: `uv run dashboard/build.py` -> dashboard/dashboard.html plus a curve-thinned dashboard-slim.html."""

import argparse
import json
from pathlib import Path

from serve import TEMPLATE, collect

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines-dir", type=Path, default="outputs/baselines")
    parser.add_argument("--runs-dir", type=Path, default="outputs")
    parser.add_argument("--out", type=Path, default="dashboard/dashboard.html")
    parser.add_argument("--slim-every", type=int, default=25, help="curve step stride of the slim build; 0 to skip it")
    args = parser.parse_args()

    template = TEMPLATE.read_text()
    builds = [(args.out, 1)]
    if args.slim_every:
        builds.append((args.out.with_name(f"{args.out.stem}-slim{args.out.suffix}"), args.slim_every))
    for out, every in builds:
        payload = json.dumps(collect(args.baselines_dir, args.runs_dir, every, wait_gp=True)).replace("</", "<\\/")
        out.write_text(template.replace("/*__DATA__*/null", payload))
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
