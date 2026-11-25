"""Evaluate a finished run once on the test set: metrics, confusion matrix, Grad-CAM, and one row in experiments.csv."""

import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader

from dataset import GRADES, IMAGENET_MEAN, IMAGENET_STD, OrangeDataset, eval_transform
from model import build_model, get_device

EXPERIMENTS_CSV = Path("results/experiments.csv")


def predict(model, loader, device):
    model.eval()
    labels, preds = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            preds += model(images.to(device)).argmax(1).cpu().tolist()
            labels += batch_labels.tolist()
    return labels, preds


def grad_cam(model, image, target_class):
    """Heatmap of where layer4 evidence for target_class sits in the image."""
    saved = {}

    def keep_activations(module, inputs, output):
        saved["activations"] = output
        output.register_hook(lambda grad: saved.__setitem__("gradients", grad))

    hook = model.layer4.register_forward_hook(keep_activations)
    model.eval()
    model.zero_grad()
    logits = model(image.unsqueeze(0))
    logits[0, target_class].backward()
    hook.remove()

    # Each of the 512 channels is weighted by how much raising it would raise the class score
    weights = saved["gradients"].mean(dim=(2, 3), keepdim=True)
    # ReLU keeps only regions that push towards the class, not against it
    cam = F.relu((weights * saved["activations"]).sum(dim=1, keepdim=True))
    # layer4 is 7x7, so upsample to the input size to overlay it
    cam = F.interpolate(cam, size=image.shape[1:], mode="bilinear", align_corners=False)[0, 0]
    return (cam / (cam.max() + 1e-8)).detach().cpu()


def save_gradcam_grid(model, predictions, device, seed, out_path, max_errors=4):
    # 2 random images per grade, plus misclassified ones because errors show what the model relies on
    per_grade = [predictions[predictions["true"] == g] for g in GRADES]
    samples = pd.concat([rows.sample(min(2, len(rows)), random_state=seed) for rows in per_grade]
                        + [predictions[predictions["true"] != predictions["pred"]].head(max_errors)])
    samples = samples[~samples.index.duplicated()]  # an error may also have been drawn at random
    transform = eval_transform()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    n_cols = (len(samples) + 1) // 2
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, row) in zip(axes.T.flat, samples.iterrows()):
        image = transform(Image.open(row["path"]).convert("RGB"))
        # Explain the logged prediction, not a fresh single-image one, so titles match the predictions CSV
        cam = grad_cam(model, image.to(device), GRADES.index(row["pred"]))
        ax.imshow((image * std + mean).clamp(0, 1).permute(1, 2, 0))
        ax.imshow(cam, cmap="jet", alpha=0.4)
        error = " (error)" if row["true"] != row["pred"] else ""
        ax.set_title(f"{row['true']} / {row['pred']}{error}", fontsize=10)
    fig.suptitle("Grad-CAM (true / predicted)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=64)  # keeps the tracked PNG small
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv", default="data/real/splits/test.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    run = checkpoint["args"]
    run_id = run["run_id"]

    # The test set is scored once per experiment; refuse a second look so it cannot be tuned on
    logged = pd.read_csv(EXPERIMENTS_CSV)
    if run_id in logged["run_id"].astype(str).values:
        raise SystemExit(f"{run_id} is already in {EXPERIMENTS_CSV}; the test set is evaluated once per run.")

    device = get_device()
    print(f"Device: {device}")
    model = build_model()
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    predictions = predict_test(model, args.test_csv, device)
    test_acc = (predictions["true"] == predictions["pred"]).mean()
    print(f"Test accuracy: {test_acc:.4f} on {len(predictions)} images")

    # Logged straight after scoring, before any plotting, so a later crash still counts as evaluated
    # and the guard above blocks a second look at the test set
    row = {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "train_source": run["train_source"],
        # real_fraction is stored as a percent (5, 10, 25); absent means all of real train
        "real_fraction": 0.0 if run["train_source"] == "synthetic" else (run.get("real_fraction") or 100) / 100,
        "model": "resnet18",
        "lr": f"{run['lr_a']:g}/{run['lr_b']:g}",
        "epochs": f"{run['epochs_a']}+{run['epochs_b']}",
        "seed": run["seed"],
        # Stage 1 checkpoints saved it under the older key name
        "val_acc": round(checkpoint.get("val_acc", checkpoint.get("best_val_acc")), 4),
        "test_acc": round(test_acc, 4),
        "notes": args.notes,
    }
    pd.DataFrame([row])[logged.columns].to_csv(EXPERIMENTS_CSV, mode="a", header=False, index=False)
    print(f"Logged {run_id} to {EXPERIMENTS_CSV}")

    write_test_outputs(model, predictions, run_id, device, args.seed)


def predict_test(model, test_csv, device):
    """Per-image predictions on the test set as a table of path, source_class, true and pred grades."""
    test_df = pd.read_csv(test_csv)
    loader = DataLoader(OrangeDataset(test_csv, eval_transform()), batch_size=64)
    labels, preds = predict(model, loader, device)
    return pd.DataFrame({"path": test_df["path"], "source_class": test_df["source_class"],
                         "true": [GRADES[i] for i in labels], "pred": [GRADES[i] for i in preds]})


def write_test_outputs(model, predictions, run_id, device, seed):
    """Per-class report, predictions CSV, per-source_class accuracy, confusion matrix and Grad-CAM."""
    print(classification_report(predictions["true"], predictions["pred"], labels=GRADES, digits=4))

    # Per-image predictions so errors can be inspected later without re-running the test set
    predictions_path = Path("results") / f"{run_id}_test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    print(f"Saved {predictions_path}")
    correct = predictions["true"] == predictions["pred"]
    print("Accuracy per source_class:")
    print(correct.groupby(predictions["source_class"]).mean().round(4).to_string())

    confusion_path = Path("results") / f"{run_id}_confusion.png"
    disp = ConfusionMatrixDisplay.from_predictions(predictions["true"], predictions["pred"], labels=GRADES,
                                                   cmap="Blues", colorbar=False)
    disp.ax_.set_title(f"{run_id} test confusion matrix")
    disp.figure_.tight_layout()
    disp.figure_.savefig(confusion_path, dpi=80)
    plt.close(disp.figure_)
    print(f"Saved {confusion_path}")

    gradcam_path = Path("results") / f"{run_id}_gradcam.png"
    save_gradcam_grid(model, predictions, device, seed, gradcam_path)
    print(f"Saved {gradcam_path}")


if __name__ == "__main__":
    main()
