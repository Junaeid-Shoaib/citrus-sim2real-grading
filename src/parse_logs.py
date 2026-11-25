"""Turn each Stage 3 training log in logs/ into results/<run_id>_history.csv, one row per epoch.

logs/ is gitignored, so the per-epoch numbers quoted in the README need a tracked copy.
"""

import re
from pathlib import Path

import pandas as pd

from compare_runs import RESULTS, STAGE3_RUN_PATTERN

LOGS = Path("logs")
# Matches the per-epoch line train.py prints; the watch-only part is absent for real-only runs
EPOCH_LINE = re.compile(
    r"Phase (?P<phase>\w) epoch +(?P<epoch>\d+): "
    r"train loss (?P<train_loss>[\d.]+) acc (?P<train_acc>[\d.]+) \| "
    r"val loss (?P<val_loss>[\d.]+) acc (?P<val_acc>[\d.]+)"
    r"(?: \| (?P<watch_set>real val|synthetic val) loss (?P<watch_loss>[\d.]+) acc (?P<watch_acc>[\d.]+))?"
)


def parse_log(path):
    rows = [match.groupdict() for match in map(EPOCH_LINE.search, path.read_text().splitlines()) if match]
    history = pd.DataFrame(rows)
    # Synthetic runs validate on their own synthetic val; real-only and mixed runs validate on real val
    history["val_set"] = "synthetic val" if path.stem.startswith("synth_") else "real val"
    return history[["epoch", "phase", "train_loss", "train_acc", "val_set", "val_loss", "val_acc",
                    "watch_set", "watch_loss", "watch_acc"]]


def main():
    run_ids = pd.read_csv(RESULTS / "experiments.csv")["run_id"]
    run_ids = run_ids[run_ids.str.fullmatch(STAGE3_RUN_PATTERN)]
    for run_id in run_ids:
        log_path = LOGS / f"{run_id}.log"
        if not log_path.exists():
            print(f"MISSING {log_path}")
            continue
        history = parse_log(log_path)
        out_path = RESULTS / f"{run_id}_history.csv"
        history.to_csv(out_path, index=False)
        print(f"{out_path}: {len(history)} epochs")


if __name__ == "__main__":
    main()
