"""
Stage 6c: train on augmented data, save separately as best_model_augmented.pt
so we keep the original baseline model (best_model.pt) intact for comparison.
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import f1_score

torch.manual_seed(42)

CLASSES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        img = transform(Image.open(row["filepath"]))
        return img, LABEL_TO_IDX[row["label"]]

def make_weighted_sampler(csv_path):
    df = pd.read_csv(csv_path)
    class_counts = df["label"].value_counts()
    weights = df["label"].apply(lambda c: 1.0 / class_counts[c]).values
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model.to(DEVICE)

def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds.extend(outputs.argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    return f1_score(targets, preds, average="macro")

def train(epochs=15, batch_size=32, lr=1e-4):
    train_ds = MRIDataset("splits/train_augmented.csv")
    val_ds = MRIDataset("splits/val.csv")

    sampler = make_weighted_sampler("splits/train_augmented.csv")
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_f1 = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs} | train loss: {total_loss/len(train_loader):.4f} | val macro-F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_model_augmented.pt")

    print(f"\nBest val macro-F1: {best_f1:.4f}")

if __name__ == "__main__":
    train()
