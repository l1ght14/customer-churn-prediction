"""One command to rebuild every artefact in the project.

    python -m src.run_all

Order matters: evaluate.py needs metrics.json to know which model won, and train.py
needs the cleaned data that data_loader produces.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from . import config, data_loader, eda, evaluate, train

RAW_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)


def fetch_raw_data() -> Path:
    """Download the dataset if it is not already present.

    The raw CSV is gitignored because it is a 1 MB third-party file, which means a fresh
    clone has nothing to read. This is the documented bootstrap step; without it
    `run_all` cannot execute on a clean checkout.
    """
    if config.RAW_CSV.exists():
        return config.RAW_CSV
    import urllib.request

    config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading dataset to {config.RAW_CSV}")
    urllib.request.urlretrieve(RAW_URL, config.RAW_CSV)
    return config.RAW_CSV


def clean_outputs() -> None:
    """Delete every generated artefact so a rebuild cannot leave stale files behind.

    Without this, a figure that is renamed or dropped from eda.py survives forever and
    silently becomes part of the portfolio. It already happened once: a stale PNG pushed
    the documented figure count out of date.

    Generated JSON and the ranked CSV are matched by extension rather than by an explicit
    list. An allowlist silently rots - a newly written report file is not on it, so it
    survives the rebuild that should have removed it. RETENTION_PLAYBOOK.md is
    hand-authored and .md, so this does not touch it.
    """
    for directory in (config.FIG_DIR, config.MODEL_DIR):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    reports = config.ROOT / "reports"
    for stale in reports.glob("*.json"):
        stale.unlink()
    for stale in config.SCORED_CSV.parent.glob("*.csv"):
        stale.unlink()
    print("cleared generated outputs")


def main() -> int:
    fetch_raw_data()
    clean_outputs()

    steps = [
        ("1/4  clean data", lambda: print(data_loader.data_quality_report(data_loader.load_raw(), data_loader.load_clean()))),
        ("2/4  exploratory analysis", eda.run),
        ("3/4  train and compare models", train.run),
        ("4/4  evaluate and export ranked list", evaluate.run),
    ]

    for label, fn in steps:
        print("\n" + "#" * 100)
        print(f"# {label}")
        print("#" * 100)
        started = time.time()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - surface the step, keep the traceback
            print(f"\nFAILED at {label}: {exc}", file=sys.stderr)
            raise
        print(f"\n-- {label} done in {time.time() - started:.1f}s")

    print("\n" + "=" * 100)
    print("PIPELINE COMPLETE")
    print(f"  figures  : {config.FIG_DIR}")
    print(f"  metrics  : {config.METRICS_JSON}")
    print(f"  ranked   : {config.SCORED_CSV}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
