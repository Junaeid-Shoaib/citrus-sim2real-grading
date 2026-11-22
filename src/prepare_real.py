"""Map the real orange dataset to 3 grades, group near-duplicates, and write train/val/test split CSVs."""

import argparse
import random
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

GRADE_OF_FOLDER = {
    "fresh_orange": "fresh",
    "green_orange": "green",
    "blackspotted_orange": "defective",
    "cankered_orange": "defective",
    "rotten_orange": "defective",
}
DEFECTIVE_PER_FOLDER = 1000  # 3 defective folders x 1000 = 3000, same as fresh and green


def list_images(raw_dir, seed):
    rng = random.Random(seed)
    rows = []
    for folder, grade in GRADE_OF_FOLDER.items():
        paths = sorted((raw_dir / folder).glob("*.jpg"))
        if grade == "defective":
            paths = sorted(rng.sample(paths, DEFECTIVE_PER_FOLDER))
        rows += [{"path": str(p), "grade": grade, "source_class": folder} for p in paths]
    return pd.DataFrame(rows)


def hash_bits(image):
    # 256-bit phash: the 64-bit default matched different oranges, which all look alike at low resolution
    return imagehash.phash(image, hash_size=16).hash.flatten().astype(np.float32)


def hash_variants(path):
    # The authors' augmentation includes flips and rotations, which change the phash, so hash all 8 orientations
    image = Image.open(path).convert("RGB")
    return np.stack([hash_bits(m.rotate(angle)) for m in (image, ImageOps.mirror(image)) for angle in (0, 90, 180, 270)])


def duplicate_groups(paths, threshold):
    variants = np.stack([hash_variants(p) for p in paths])  # (n, 8, 256)
    originals = variants[:, 0]
    min_dist = None
    for k in range(variants.shape[1]):
        # Hamming distance between 0/1 vectors as two matrix products
        dist = originals @ (1 - variants[:, k]).T + (1 - originals) @ variants[:, k].T
        min_dist = dist if min_dist is None else np.minimum(min_dist, dist)
    # Connected components so that if A~B and B~C, all three land in the same split
    _, group_ids = connected_components(csr_matrix(min_dist <= threshold), directed=False)
    return group_ids


def assign_splits(df, seed):
    # 20 folds of 5%: 3 test, 3 val, 14 train gives 70/15/15 while keeping each duplicate group whole.
    # Stratify on source_class so each defect type is balanced too; grade balance follows from it.
    cv = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(cv.split(df, df["source_class"], df["group_id"])):
        fold[idx] = k
    return np.where(fold < 3, "test", np.where(fold < 6, "val", "train"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/real/raw")
    parser.add_argument("--out-dir", default="data/real/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hash-threshold", type=int, default=24)
    args = parser.parse_args()

    df = list_images(Path(args.raw_dir), args.seed)
    print("Images per grade:")
    print(df["grade"].value_counts().to_string())

    print(f"Hashing {len(df)} images...")
    df["group_id"] = duplicate_groups(df["path"], args.hash_threshold)
    sizes = df["group_id"].value_counts()
    dup_groups = sizes[sizes > 1]
    grades_per_group = df.groupby("group_id")["grade"].nunique()
    print(f"Duplicate groups: {len(dup_groups)}, images in them: {dup_groups.sum()}, "
          f"largest group: {sizes.max()}, cross-grade groups: {(grades_per_group > 1).sum()}")

    df["split"] = assign_splits(df, args.seed)
    print("Split x grade counts:")
    print(pd.crosstab(df["split"], df["grade"], margins=True).to_string())
    print("Split x source_class counts:")
    print(pd.crosstab(df["split"], df["source_class"], margins=True).to_string())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        df[df["split"] == split].drop(columns="split").to_csv(out_dir / f"{split}.csv", index=False)
    print(f"Wrote splits to {out_dir}")


if __name__ == "__main__":
    main()
