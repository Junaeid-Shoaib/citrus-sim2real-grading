"""Write nested 5%, 10% and 25% subsets of real train for the Stage 3.3 runs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Percent of real train -> number of 5% folds. Taking folds 0..k-1 makes each subset contain the smaller ones
SUBSETS = {5: 1, 10: 2, 25: 5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/real/splits/train.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    # Same method as the Stage 1 split: duplicate groups stay whole, source classes stay balanced
    cv = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=args.seed)
    fold = np.empty(len(train), dtype=int)
    for k, (_, idx) in enumerate(cv.split(train, train["source_class"], train["group_id"])):
        fold[idx] = k

    out_dir = Path(args.train_csv).parent
    for pct, n_folds in SUBSETS.items():
        subset = train[fold < n_folds]
        out_path = out_dir / f"train_{pct}.csv"
        subset.to_csv(out_path, index=False)
        print(f"{out_path}: {len(subset)} images")
        print(subset["source_class"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
