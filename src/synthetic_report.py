"""Compare rendered synthetic oranges with the real train split, and check for duplicates.

Writes three figures: a 16-image sample grid, a real-vs-synthetic grid per grade, and
hue/saturation histograms per grade. Uses the real TRAIN split only, never val or test.

    uv run python src/synthetic_report.py --synthetic-dir data/synthetic/wide
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from dataset import GRADES
from prepare_real import duplicate_groups

# Rough fruit mask: real backgrounds are white or grey, so saturated pixels are mostly fruit
FRUIT_SATURATION = 0.25
# Synthetic backgrounds can be beige or a random colour, which the mask cannot remove, so the
# histograms only use renders on low-saturation backgrounds (light surface or dark belt)
NEUTRAL_BACKGROUNDS = ["light", "dark_belt"]


def fruit_hsv(path):
    hsv = np.asarray(Image.open(path).convert("RGB").resize((112, 112)).convert("HSV")).reshape(-1, 3) / 255
    return hsv[hsv[:, 1] > FRUIT_SATURATION]


def coverage(real_pixels, synth_pixels):
    """% of real images whose median hue/sat/value lies inside the synthetic 5th-95th percentile.

    Answers "does the synthetic set span the real images?" per channel, which a match of
    medians alone does not.
    """
    real_medians = np.array([np.median(px, axis=0) for px in real_pixels])
    synth_medians = np.array([np.median(px, axis=0) for px in synth_pixels])
    low, high = np.percentile(synth_medians, [5, 95], axis=0)
    inside = (real_medians >= low) & (real_medians <= high)
    return 100 * inside.mean(axis=0), low, high


def image_grid(rows, out_path, title_size=8, dpi=110):
    # rows: list of (row label, [(path, title), ...]); every row has the same length
    n_cols = max(len(items) for _, items in rows)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(1.6 * n_cols, 1.75 * len(rows)), squeeze=False)
    for r, (label, items) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(items):
                path, title = items[c]
                ax.imshow(Image.open(path).convert("RGB"))
                ax.set_title(title, fontsize=title_size)
        axes[r][0].text(-0.1, 0.5, label, transform=axes[r][0].transAxes, ha="right", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def sample_grid(synth, out_path, seed):
    picked = synth.sample(16, random_state=seed)
    items = [(row.path, row.grade if row.defect_type == "none" else f"{row.grade}: {row.defect_type}")
             for row in picked.itertuples()]
    image_grid([("", items[:8]), ("", items[8:])], out_path)


def real_vs_synthetic_grid(real, synth, out_path, seed):
    rows = []
    for grade in GRADES:
        real_rows = real[real.grade == grade].sample(8, random_state=seed)
        synth_rows = synth[synth.grade == grade].sample(8, random_state=seed)
        rows.append((f"real\n{grade}", [(p, "") for p in real_rows.path]))
        rows.append((f"synthetic\n{grade}", [(p, "") for p in synth_rows.path]))
    # 48 photos in one figure: a lower dpi keeps the tracked PNG small
    image_grid(rows, out_path, dpi=55)


def hue_sat_histograms(real, synth, out_path, n_images, seed):
    fig, axes = plt.subplots(2, len(GRADES), figsize=(4 * len(GRADES), 6))
    bins = {"hue": np.linspace(0, 0.35, 71), "saturation": np.linspace(FRUIT_SATURATION, 1, 61)}
    summary = []
    for c, grade in enumerate(GRADES):
        real_paths = real[real.grade == grade].sample(min(n_images, (real.grade == grade).sum()), random_state=seed).path
        neutral = synth[(synth.grade == grade) & synth.bg_type.isin(NEUTRAL_BACKGROUNDS)]
        synth_paths = neutral.sample(min(n_images, len(neutral)), random_state=seed).path
        real_images = [fruit_hsv(p) for p in real_paths]
        synth_images = [fruit_hsv(p) for p in synth_paths]
        real_px, synth_px = np.concatenate(real_images), np.concatenate(synth_images)
        covered, low, high = coverage(real_images, synth_images)
        for r, (name, channel) in enumerate((("hue", 0), ("saturation", 1))):
            ax = axes[r][c]
            ax.hist(real_px[:, channel], bins=bins[name], density=True, alpha=0.5, label=f"real train ({len(real_paths)})")
            ax.hist(synth_px[:, channel], bins=bins[name], density=True, alpha=0.5, label=f"synthetic ({len(synth_paths)})")
            ax.set_title(f"{grade}: {name}")
            ax.legend(fontsize=7)
        summary.append({"grade": grade, "real images": len(real_paths), "synthetic images": len(synth_paths),
                        "real hue": np.median(real_px[:, 0]), "synthetic hue": np.median(synth_px[:, 0]),
                        "real sat": np.median(real_px[:, 1]), "synthetic sat": np.median(synth_px[:, 1]),
                        "real value": np.median(real_px[:, 2]), "synthetic value": np.median(synth_px[:, 2]),
                        **{f"{name} coverage %": covered[i] for i, name in enumerate(("hue", "sat", "value"))},
                        **{f"synthetic {name} 5-95%": f"{low[i]:.3f}-{high[i]:.3f}"
                           for i, name in enumerate(("hue", "sat", "value"))}})
    fig.suptitle("Fruit pixels (saturation > 0.25); synthetic only on light or dark-belt backgrounds")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-dir", default="data/synthetic/wide")
    parser.add_argument("--real-train", default="data/real/splits/train.csv")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--tag", help="suffix for the figure names; default: the randomisation in labels.csv")
    parser.add_argument("--n-images", type=int, default=500, help="images per grade and source for the histograms")
    parser.add_argument("--phash", action="store_true", help="run the Stage 1 near-duplicate check (D8)")
    parser.add_argument("--hash-threshold", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    synth = pd.read_csv(Path(args.synthetic_dir) / "labels.csv")
    synth = synth[synth.path.map(lambda p: Path(p).exists())]
    real = pd.read_csv(args.real_train)
    tag = args.tag or synth.randomisation.iloc[0]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Synthetic images found: {len(synth)}")
    print(synth.groupby(["grade", "bg_type"]).size().unstack(fill_value=0).to_string())

    sample_grid(synth, out_dir / "sample_synthetic.png", args.seed)
    real_vs_synthetic_grid(real, synth, out_dir / f"real_vs_synthetic_{tag}.png", args.seed)
    summary = hue_sat_histograms(real, synth, out_dir / f"hue_sat_{tag}.png", args.n_images, args.seed)
    medians = [c for c in summary.columns if "coverage" not in c and "5-95" not in c]
    print("\nMedian of fruit pixels:")
    print(summary[medians].round(3).to_string(index=False))
    print("\nCoverage: % of real train images whose median lies in the synthetic 5th-95th percentile:")
    print(summary[["grade"] + [c for c in summary.columns if "coverage" in c or "5-95" in c]].round(1).to_string(index=False))

    if args.phash:
        groups = duplicate_groups(synth.path.tolist(), args.hash_threshold)
        sizes = pd.Series(groups).value_counts()
        print(f"\nphash check (threshold {args.hash_threshold}): {(sizes > 1).sum()} duplicate groups, "
              f"{sizes[sizes > 1].sum()} images in them, largest {sizes.max()}")
    print(f"\nFigures written to {out_dir}")


if __name__ == "__main__":
    main()
