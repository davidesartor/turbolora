"""Bind the current data into standalone, shareable HTML snapshots: `uv run dashboard/build.py` -> dashboard/dashboard.html plus a dashboard-slim.html without misses and full θ vectors."""

import argparse
import os
from pathlib import Path

from serve import collect, page

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines-dir", type=Path, default="outputs/runs", help="holds <family>/<model>/base/eval@K")
    parser.add_argument("--runs-dir", type=Path, default="outputs")
    parser.add_argument("--out", type=Path, default="dashboard/dashboard.html")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="scan processes")
    parser.add_argument("--slim-every", type=int, default=25, help="curve step stride of the slim build; 0 to skip it")
    args = parser.parse_args()

    write = lambda out, data: (out.write_bytes(page(data)), print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)"))

    full, slim = collect(args.baselines_dir, args.runs_dir, workers=args.workers, every=args.slim_every)
    write(args.out, full)
    if args.slim_every:
        write(args.out.with_name(f"{args.out.stem}-slim{args.out.suffix}"), slim)
