"""Two-phase ResNet18 training: head only, then full fine-tune.

Runs on all of real train keep the best checkpoint by val accuracy. Stage 3 runs (synthetic, mixed,
and real-only subsets) keep the last epoch, so no real labels pick the checkpoint (spec_sim2real section 5).
"""

import argparse
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import OrangeDataset, eval_transform, train_transform
from model import build_model, get_device, set_backbone_trainable

REAL_VAL_CSV = "data/real/splits/val.csv"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loaders(train_source, synthetic_dir, real_fraction, batch_size, seed):
    if train_source == "synthetic":
        train_csv, val_csv = f"{synthetic_dir}/splits/train.csv", f"{synthetic_dir}/splits/val.csv"
    elif real_fraction is None:
        train_csv, val_csv = "data/real/splits/train.csv", REAL_VAL_CSV
    else:
        # Real-only and mixed runs at the same fraction read the same subset file
        train_csv, val_csv = f"data/real/splits/train_{real_fraction}.csv", REAL_VAL_CSV
    # Seeded generator so the shuffle order is the same on every run
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(OrangeDataset(train_csv, train_transform()), batch_size=batch_size,
                              shuffle=True, generator=generator)
    val_loader = DataLoader(OrangeDataset(val_csv, eval_transform()), batch_size=batch_size)
    return train_loader, val_loader


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over the loader. Trains if an optimizer is given, otherwise only evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
    return total_loss / count, correct / count


def plot_history(history, phase_a_epochs, watch_name, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in ((ax_loss, "loss", "Loss"), (ax_acc, "acc", "Accuracy")):
        ax.plot(epochs, history[f"train_{key}"], marker="o", label="train")
        ax.plot(epochs, history[f"val_{key}"], marker="o", label="val")
        if history[f"watch_{key}"]:
            ax.plot(epochs, history[f"watch_{key}"], marker="o", label=f"{watch_name} (watch only)")
        # Dashed line marks where the backbone is unfrozen
        ax.axvline(phase_a_epochs + 0.5, color="grey", linestyle="--", label="unfreeze")
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.legend()
    # Log scale so near-zero synthetic loss and large real val loss are both readable on one panel
    ax_loss.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", choices=["real", "synthetic", "mixed"], default="real")
    parser.add_argument("--synthetic-dir", help="e.g. data/synthetic/wide_v2; needed for synthetic and mixed")
    parser.add_argument("--real-fraction", type=int, choices=[5, 10, 25],
                        help="percent of real train (subset from make_real_subsets.py); omit for all of it")
    parser.add_argument("--init-checkpoint", help="synthetic checkpoint a mixed run starts from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs-a", type=int, default=5)
    parser.add_argument("--epochs-b", type=int, default=5)
    parser.add_argument("--lr-a", type=float, default=1e-3)
    parser.add_argument("--lr-b", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--run-id", default="real_baseline")
    args = parser.parse_args()
    if args.train_source in ("synthetic", "mixed") and not args.synthetic_dir:
        parser.error(f"--synthetic-dir is needed for --train-source {args.train_source}")
    if args.train_source == "mixed" and not (args.real_fraction and args.init_checkpoint):
        parser.error("--train-source mixed needs --real-fraction and --init-checkpoint")
    if args.train_source == "synthetic" and args.real_fraction:
        parser.error("--real-fraction does not apply to --train-source synthetic")

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader = make_loaders(args.train_source, args.synthetic_dir, args.real_fraction,
                                            args.batch_size, args.seed)
    # A second val set, scored every epoch for watching only; it never picks the checkpoint.
    # Synthetic runs watch real val (how real accuracy moves); mixed runs watch synthetic val (forgetting)
    watch_loader, watch_name = None, None
    if args.train_source == "synthetic":
        watch_loader, watch_name = DataLoader(OrangeDataset(REAL_VAL_CSV, eval_transform()),
                                              batch_size=args.batch_size), "real val"
    elif args.train_source == "mixed":
        watch_loader, watch_name = DataLoader(OrangeDataset(f"{args.synthetic_dir}/splits/val.csv", eval_transform()),
                                              batch_size=args.batch_size), "synthetic val"

    model = build_model()
    if args.init_checkpoint:
        # Keep the synthetic-trained head too: it already maps to the same three grades
        model.load_state_dict(torch.load(args.init_checkpoint, map_location="cpu")["model_state"])
        print(f"Starting from {args.init_checkpoint}")
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    # Phase A trains only the new head so the random weights cannot wreck the pretrained features.
    # Phase B fine-tunes everything at a lower learning rate so the pretrained filters move gently.
    phases = [("A", False, args.lr_a, args.epochs_a), ("B", True, args.lr_b, args.epochs_b)]
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "watch_loss": [], "watch_acc": []}
    best_val_acc, best_epoch = 0.0, 0
    total_epochs = args.epochs_a + args.epochs_b
    checkpoint_path = Path("checkpoints") / f"{args.run_id}.pt"
    checkpoint_path.parent.mkdir(exist_ok=True)

    epoch = 0
    for phase, backbone_trainable, lr, n_epochs in phases:
        set_backbone_trainable(model, backbone_trainable)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        for _ in range(n_epochs):
            epoch += 1
            start = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
            for key, value in zip(("train_loss", "val_loss", "train_acc", "val_acc"),
                                  (train_loss, val_loss, train_acc, val_acc)):
                history[key].append(value)
            line = (f"Phase {phase} epoch {epoch:2d}: train loss {train_loss:.4f} acc {train_acc:.4f} | "
                    f"val loss {val_loss:.4f} acc {val_acc:.4f}")
            if watch_loader is not None:
                watch_loss, watch_acc = run_epoch(model, watch_loader, loss_fn, device)
                history["watch_loss"].append(watch_loss)
                history["watch_acc"].append(watch_acc)
                line += f" | {watch_name} loss {watch_loss:.4f} acc {watch_acc:.4f}"
            print(f"{line} | {time.time() - start:.0f}s")

            if args.train_source == "real" and args.real_fraction is None:
                save_now = val_acc > best_val_acc
            else:
                # Fixed epochs, keep the last (Stage 3): real-only subsets follow the same rule as mixed,
                # so both arms of the 3.3 comparison pick their checkpoint the same way
                save_now = epoch == total_epochs
            if save_now:
                best_val_acc, best_epoch = val_acc, epoch
                # experiments.csv logs real val for every run, so all runs share one val set
                logged_val_acc = watch_acc if args.train_source == "synthetic" else val_acc
                torch.save({"model_state": model.state_dict(), "args": vars(args),
                            "val_acc": logged_val_acc, "saved_epoch": epoch}, checkpoint_path)

    plot_path = Path("results") / f"{args.run_id}_curves.png"
    plot_history(history, args.epochs_a, watch_name, plot_path)
    print(f"Saved epoch {best_epoch} (val acc {best_val_acc:.4f}) to {checkpoint_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
