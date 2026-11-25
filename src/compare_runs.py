"""Stage 3 summary across runs: one table, the real-fraction plot, the section 9 difference rule,
and a Grad-CAM comparison of four models on the same real val images.

Reads experiments.csv and the saved test predictions CSVs, never the test images. The Grad-CAM
comparison uses real val images only, so the test set is still read only by evaluate.py.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

from dataset import GRADES, IMAGENET_MEAN, IMAGENET_STD, eval_transform
from evaluate import grad_cam
from model import build_model, get_device

RESULTS = Path("results")
UPPER_BOUND_RUN = "real_lrb1e-4"
STAGE3_RUN_PATTERN = r"(synth_(wide|narrow)|real_\d+|mixed_wide_\d+)_s\d+"
CEILING = 0.97  # above this, differences come down to a handful of images, so report error counts
# Columns of the Grad-CAM comparison, fixed in the spec (S7) before any 3.3 result
GRADCAM_RUNS = ["real_lrb1e-4", "synth_narrow_s42", "synth_wide_s42", "mixed_wide_5_s42"]


def arm(run_id):
    """Which setting a run belongs to: synth_wide, synth_narrow, real_5, mixed_wide_5, ..."""
    if run_id == UPPER_BOUND_RUN:
        return "real_full"
    return run_id.rsplit("_s", 1)[0]


def summary_table(runs):
    rows = []
    upper = runs.loc[runs["run_id"] == UPPER_BOUND_RUN, "test_acc"].item()
    for _, run in runs.iterrows():
        preds = pd.read_csv(RESULTS / f"{run['run_id']}_test_predictions.csv")
        correct = preds["true"] == preds["pred"]
        row = {"run_id": run["run_id"], "arm": arm(run["run_id"]), "real_fraction": run["real_fraction"],
               "seed": run["seed"], "test_acc": run["test_acc"], "errors": int((~correct).sum()),
               "gap": round(upper - run["test_acc"], 4)}
        row.update({f"recall_{g}": round(v, 4) for g, v in correct.groupby(preds["true"]).mean().items()})
        row.update({f"acc_{c}": round(v, 4) for c, v in correct.groupby(preds["source_class"]).mean().items()})
        rows.append(row)
    return pd.DataFrame(rows)


def seed_range(table, arm_name):
    acc = table.loc[table["arm"] == arm_name, "test_acc"]
    return acc.max() - acc.min()


def compare(table, arm_a, arm_b, noise):
    """Section 9 rule: a real difference beats 2x the noise range and 2 points."""
    a, b = (table[table["arm"] == name] for name in (arm_a, arm_b))
    diff = a["test_acc"].mean() - b["test_acc"].mean()
    verdict = "real difference" if abs(diff) > max(2 * noise, 0.02) else "no clear difference"
    print(f"{arm_a} mean {a['test_acc'].mean():.4f} vs {arm_b} mean {b['test_acc'].mean():.4f}: "
          f"diff {diff * 100:+.1f} pts, bar {max(2 * noise, 0.02) * 100:.1f} pts -> {verdict}")
    if min(a["test_acc"].mean(), b["test_acc"].mean()) > CEILING:
        print(f"  near ceiling, errors out of 1,350: {arm_a} {a['errors'].tolist()}, {arm_b} {b['errors'].tolist()}")


def plot_real_fraction(table, out_path):
    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4))
    for source, label, shift in (("mixed_wide", "synthetic wide + real", -0.3), ("real", "real only", 0.3)):
        stats = [table.loc[table["arm"] == f"{source}_{pct}", "test_acc"].agg(["mean", "min", "max"])
                 for pct in (5, 10, 25)]
        means = [s["mean"] for s in stats]
        # Error bars show the seed range (only the 5% points have 3 seeds)
        err = [[s["mean"] - s["min"] for s in stats], [s["max"] - s["mean"] for s in stats]]
        ax.errorbar([5, 10, 25], means, yerr=err, marker="o", capsize=4, label=label)
        # Zoomed copy; a small sideways shift stops the two 5% bars hiding each other
        ax_zoom.errorbar([5 + shift, 10 + shift, 25 + shift], means, yerr=err, marker="o", capsize=4, label=label)
    synth = table.loc[table["arm"] == "synth_wide", "test_acc"]
    ax.errorbar([0], [synth.mean()], yerr=[[synth.mean() - synth.min()], [synth.max() - synth.mean()]],
                marker="s", color="grey", capsize=4, linestyle="none", label="synthetic wide only (3 seeds)")
    upper = table.loc[table["arm"] == "real_full", "test_acc"].item()
    ax.axhline(upper, color="black", linestyle="--", label="all real train (upper bound)")
    ax.set_xlabel("real train used (%)")
    ax.set_ylabel("real test accuracy")
    ax.set_xticks([0, 5, 10, 25])
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax_zoom.set_xlabel("real train used (%)")
    ax_zoom.set_ylabel("real test accuracy")
    ax_zoom.set_xticks([5, 10, 25])
    ax_zoom.set_ylim(0.96, 1.0)
    ax_zoom.set_title("Zoom: real only vs synthetic wide + real", fontsize=10)
    ax_zoom.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def pick_gradcam_images(val_csv, seed):
    """S7 rule: one real val image per source class, plus one more fresh and one more green (7 in all)."""
    val = pd.read_csv(val_csv)
    extra = {"fresh_orange": 2, "green_orange": 2}
    return pd.concat([val[val["source_class"] == c].sample(extra.get(c, 1), random_state=seed)
                      for c in sorted(val["source_class"].unique())])


def save_gradcam_compare(images, device, out_path):
    transform = eval_transform()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    fig, axes = plt.subplots(len(images), len(GRADCAM_RUNS), figsize=(2.4 * len(GRADCAM_RUNS), 2.4 * len(images)))
    for col, run_id in enumerate(GRADCAM_RUNS):
        model = build_model()
        model.load_state_dict(torch.load(f"checkpoints/{run_id}.pt", map_location="cpu")["model_state"])
        model.to(device).eval()
        for row, (_, item) in enumerate(images.iterrows()):
            image = transform(Image.open(item["path"]).convert("RGB")).to(device)
            with torch.no_grad():
                pred = model(image.unsqueeze(0)).argmax(1).item()
            # Heatmap for the model's own prediction, as in evaluate.py
            cam = grad_cam(model, image, pred)
            ax = axes[row, col]
            ax.imshow((image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0))
            ax.imshow(cam, cmap="jet", alpha=0.4)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{item['source_class'].replace('_orange', '')} / {GRADES[pred]}", fontsize=13)
            if row == 0:
                ax.set_title(f"{run_id}\n{ax.get_title()}", fontsize=13)
    fig.suptitle("Grad-CAM on real val (true source class / predicted grade)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))  # leaves room for the suptitle
    fig.savefig(out_path, dpi=48)  # 28 heatmaps compress badly, so a lower dpi keeps the tracked PNG small
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="draws the Grad-CAM comparison images")
    args = parser.parse_args()

    runs = pd.read_csv(RESULTS / "experiments.csv")
    # Only the Stage 3 arms plus the upper bound, so a stray Stage 1 trial cannot add a row or crash the script
    runs = runs[runs["run_id"].str.fullmatch(STAGE3_RUN_PATTERN) | (runs["run_id"] == UPPER_BOUND_RUN)]
    table = summary_table(runs)
    table.to_csv(RESULTS / "stage3_summary.csv", index=False)
    print(table[["run_id", "test_acc", "errors", "gap"]].to_string(index=False))

    print("\nStep 3.2, narrow vs wide (noise = larger seed range):")
    compare(table, "synth_narrow", "synth_wide", max(seed_range(table, "synth_narrow"), seed_range(table, "synth_wide")))

    # Amended section 9 rule: the larger 5% seed range is the noise estimate at every fraction
    noise = max(seed_range(table, "mixed_wide_5"), seed_range(table, "real_5"))
    print(f"\nStep 3.3, mixed vs real-only (noise = larger 5% seed range, {noise * 100:.1f} pts):")
    for pct in (5, 10, 25):
        compare(table, f"mixed_wide_{pct}", f"real_{pct}", noise)

    plot_real_fraction(table, RESULTS / "real_fraction.png")
    print(f"\nSaved {RESULTS / 'stage3_summary.csv'} and {RESULTS / 'real_fraction.png'}")

    images = pick_gradcam_images("data/real/splits/val.csv", seed=args.seed)
    print(f"\nGrad-CAM comparison images (real val, S7 rule, seed {args.seed}):")
    print(images[["path", "source_class"]].to_string(index=False))
    save_gradcam_compare(images, get_device(), RESULTS / "gradcam_compare.png")
    print(f"Saved {RESULTS / 'gradcam_compare.png'}")


if __name__ == "__main__":
    main()
