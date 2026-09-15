"""
Stage 7: multi-seed comparison.

A single paired bootstrap only captures uncertainty from which test
images we happened to draw. It says nothing about uncertainty from
which random training run we happened to get, and we just proved
that source of variance is real and large enough to flip our
conclusion. The correct fix: train each model multiple times with
different seeds, and look at the SPREAD of results, not one run.

With only 3 seeds, a formal significance test has very little power,
so we report the raw per-seed results and the paired differences
directly, and let the spread speak for itself, rather than compute
a p-value that would overstate precision we don't have.
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import f1_score

CLASSES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 2024]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class MRIDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return transform(Image.open(row["filepath"])), LABEL_TO_IDX[row["label"]]

def make_weighted_sampler(csv_path):
    df = pd.read_csv(csv_path)
    class_counts = df["label"].value_counts()
    weights = df["label"].apply(lambda c: 1.0 / class_counts[c]).values
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model.to(DEVICE)

def evaluate_f1(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds.extend(outputs.argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    return f1_score(targets, preds, average="macro")

def train_one_run(train_csv, seed, epochs=15, batch_size=32, lr=1e-4):
    torch.manual_seed(seed)
    train_ds = MRIDataset(train_csv)
    val_ds = MRIDataset("splits/val.csv")
    sampler = make_weighted_sampler(train_csv)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        val_f1 = evaluate_f1(model, val_loader)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model

if __name__ == "__main__":
    test_loader = DataLoader(MRIDataset("splits/test.csv"), batch_size=32)

    results = []
    for seed in SEEDS:
        print(f"\n=== Seed {seed}: baseline ===")
        base_model = train_one_run("splits/train.csv", seed)
        base_f1 = evaluate_f1(base_model, test_loader)
        print(f"Seed {seed} baseline test macro-F1: {base_f1:.4f}")

        print(f"=== Seed {seed}: augmented ===")
        aug_model = train_one_run("splits/train_augmented.csv", seed)
        aug_f1 = evaluate_f1(aug_model, test_loader)
        print(f"Seed {seed} augmented test macro-F1: {aug_f1:.4f}")

        results.append((seed, base_f1, aug_f1, aug_f1 - base_f1))

    print("\n" + "=" * 60)
    print("SUMMARY: seed | baseline | augmented | difference")
    print("=" * 60)
    diffs = []
    for seed, base_f1, aug_f1, diff in results:
        print(f"{seed:6d} | {base_f1:.4f}   | {aug_f1:.4f}    | {diff:+.4f}")
        diffs.append(diff)

    import numpy as np
    diffs = np.array(diffs)
    print(f"\nMean difference across {len(SEEDS)} seeds: {diffs.mean():+.4f}")
    print(f"Std across seeds: {diffs.std():.4f}")
    print(f"Positive (augmentation helped) in {(diffs > 0).sum()}/{len(SEEDS)} seeds")
    print("\nNote: with only 3 seeds this is a descriptive summary of the "
          "spread, not a formal significance test.")
