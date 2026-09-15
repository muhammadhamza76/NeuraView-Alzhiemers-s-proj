"""
Stage 4: evaluation with bootstrap CIs (self-contained version).

Run this after training with stage3_train_baseline.py in the same
notebook session, so best_model.pt already exists in the working
directory. This version duplicates the small pieces it needs instead
of importing stage3 as a module, since Kaggle notebook cells aren't
separate importable files by default.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import f1_score, classification_report

CLASSES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_BOOTSTRAP = 2000

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

def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model.to(DEVICE)

def get_predictions(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds.extend(outputs.argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    return np.array(preds), np.array(targets)

def bootstrap_macro_f1(preds, targets, n=N_BOOTSTRAP):
    n_samples = len(preds)
    rng = np.random.default_rng(42)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        scores.append(f1_score(targets[idx], preds[idx], average="macro"))
    return np.percentile(scores, [2.5, 50, 97.5])

if __name__ == "__main__":
    test_ds = MRIDataset("splits/test.csv")
    test_loader = DataLoader(test_ds, batch_size=32)

    model = build_model()
    model.load_state_dict(torch.load("best_model.pt"))

    preds, targets = get_predictions(model, test_loader)

    point_f1 = f1_score(targets, preds, average="macro")
    ci_low, ci_mid, ci_high = bootstrap_macro_f1(preds, targets)

    print(f"Test macro-F1: {point_f1:.3f}")
    print(f"Bootstrap 95% CI: [{ci_low:.3f}, {ci_high:.3f}] (median {ci_mid:.3f})")
    print("\nFull classification report:")
    print(classification_report(targets, preds, target_names=CLASSES))
    print("\nNote: ModerateDemented has only 4 test images. Its per-class "
          "precision/recall should be read as a rough signal, not a precise "
          "estimate, given the sample size.")
