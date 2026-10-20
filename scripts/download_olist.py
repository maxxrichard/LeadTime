#!/usr/bin/env python
"""Download the public Olist dataset from Kaggle into data/raw (requires `pip install kaggle` + API token).

    python scripts/download_olist.py --out data/raw

Afterwards run `leadtime prepare && leadtime features && leadtime backtest` on the real ~100k orders.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET = "olistbr/brazilian-ecommerce"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if shutil.which("kaggle") is None:
        print(
            "kaggle CLI not found: pip install kaggle and place your token in ~/.kaggle/kaggle.json",
            file=sys.stderr,
        )
        return 1
    subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-p", str(out)], check=True)
    for z in out.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
        z.unlink()
    print(f"Olist CSVs extracted to {out}:")
    for p in sorted(out.glob("*.csv")):
        print(" ", p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
