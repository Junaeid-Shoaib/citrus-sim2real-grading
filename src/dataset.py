"""PyTorch Dataset for the split CSVs, plus train and eval transforms."""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Fixed order so label index 0/1/2 means the same thing in every script
GRADES = ["defective", "fresh", "green"]
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class OrangeDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), GRADES.index(row["grade"])


def train_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Fruit is seen from above on a belt, so there is no true "up": both flips are valid
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # No hue jitter: hue is what separates green from fresh, so shifting it could contradict the label
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def save_sample_grid(dataset, out_path, seed, n=16):
    torch.manual_seed(seed)  # fixes both which images are picked and the random augmentations
    indices = torch.randperm(len(dataset))[:n]
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, idx in zip(axes.flat, indices):
        image, label = dataset[int(idx)]
        # Undo normalisation so colours look natural on screen
        image = (image * std + mean).clamp(0, 1)
        ax.imshow(image.permute(1, 2, 0))
        ax.set_title(GRADES[label], fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=64)  # keeps the tracked PNG around 300 KB
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/real/splits/train.csv")
    parser.add_argument("--out", default="results/sample_real.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = OrangeDataset(args.csv, train_transform())
    print(f"{len(dataset)} images in {args.csv}")
    image, label = dataset[0]
    print(f"First item: tensor {tuple(image.shape)}, label {label} ({GRADES[label]})")
    save_sample_grid(dataset, args.out, args.seed)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
