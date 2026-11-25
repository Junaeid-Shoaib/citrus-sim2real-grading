"""Split finished synthetic images 85/15 into train and val, keeping near-duplicates together.

Runs after src/camera_process.py, because the perceptual-hash check needs the final images.
Stratified on grade and defect type. There is no synthetic test split: the test set is always real.

    uv run python src/split_synthetic.py --synthetic-dir data/synthetic/wide_v2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from prepare_real import duplicate_groups

N_FOLDS = 20  # folds of 5%: 3 val folds give the 85/15 split
VAL_FOLDS = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-dir", default="data/synthetic/wide_v2")
    parser.add_argument("--hash-threshold", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.synthetic_dir)
    labels = pd.read_csv(out_dir / "labels.csv")
    missing = (~labels.path.map(lambda p: Path(p).exists())).sum()
    if missing:
        raise SystemExit(f"{missing} images missing; render and camera-process the whole batch first")

    # Stratum: grade for fresh and green, defect type for defective, so the three defect types are balanced
    stratum = pd.Series(np.where(labels.grade == "defective", labels.defect_type, labels.grade), name="stratum")
    if labels.randomisation.iloc[0] == "narrow":
        # Every narrow image has the same framing and background, so phash chains most of them into
        # one group (2,328 of 3,000 in the first batch) although no two images match. Grouping would
        # put nearly all images in one split, so narrow is split without groups; narrow val is
        # therefore only for watching curves, never for choosing a checkpoint (decisions.md).
        labels["group_id"] = np.arange(len(labels))
    else:
        labels["group_id"] = duplicate_groups(labels.path.tolist(), args.hash_threshold)
    sizes = labels.group_id.value_counts()
    print(f"Duplicate groups: {(sizes > 1).sum()}, images in them: {sizes[sizes > 1].sum()}, largest {sizes.max()}")

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=args.seed)
    fold = np.empty(len(labels), dtype=int)
    for k, (_, idx) in enumerate(cv.split(labels, stratum, labels.group_id)):
        fold[idx] = k
    labels["split"] = np.where(fold < VAL_FOLDS, "val", "train")

    (out_dir / "splits").mkdir(exist_ok=True)
    for name in ("train", "val"):
        split = labels[labels.split == name]
        split[["path", "grade", "defect_type", "group_id"]].to_csv(out_dir / "splits" / f"{name}.csv", index=False)
    print(pd.crosstab(labels.split, stratum, margins=True).to_string())


if __name__ == "__main__":
    main()
