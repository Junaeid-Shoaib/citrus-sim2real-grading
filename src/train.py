"""Two-phase ResNet18 training: head only, then full fine-tune. Keeps the best checkpoint by val accuracy."""

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loaders(train_source, batch_size, seed):
    if train_source == "real":
        train_csv, val_csv = "data/real/splits/train.csv", "data/real/splits/val.csv"
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


def plot_history(history, phase_a_epochs, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in ((ax_loss, "loss", "Loss"), (ax_acc, "acc", "Accuracy")):
        ax.plot(epochs, history[f"train_{key}"], marker="o", label="train")
        ax.plot(epochs, history[f"val_{key}"], marker="o", label="val")
        # Dashed line marks where the backbone is unfrozen
        ax.axvline(phase_a_epochs + 0.5, color="grey", linestyle="--", label="unfreeze")
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", choices=["real"], default="real")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs-a", type=int, default=5)
    parser.add_argument("--epochs-b", type=int, default=5)
    parser.add_argument("--lr-a", type=float, default=1e-3)
    parser.add_argument("--lr-b", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--run-id", default="real_baseline")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader = make_loaders(args.train_source, args.batch_size, args.seed)
    model = build_model().to(device)
    loss_fn = nn.CrossEntropyLoss()

    # Phase A trains only the new head so the random weights cannot wreck the pretrained features.
    # Phase B fine-tunes everything at a lower learning rate so the pretrained filters move gently.
    phases = [("A", False, args.lr_a, args.epochs_a), ("B", True, args.lr_b, args.epochs_b)]
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_epoch = 0.0, 0
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
            for key, value in zip(history, (train_loss, val_loss, train_acc, val_acc)):
                history[key].append(value)
            print(f"Phase {phase} epoch {epoch:2d}: train loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f} | {time.time() - start:.0f}s")
            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch
                torch.save({"model_state": model.state_dict(), "args": vars(args),
                            "best_val_acc": best_val_acc, "best_epoch": best_epoch}, checkpoint_path)

    plot_path = Path("results") / f"{args.run_id}_curves.png"
    plot_history(history, args.epochs_a, plot_path)
    print(f"Best val acc {best_val_acc:.4f} at epoch {best_epoch}, saved {checkpoint_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
