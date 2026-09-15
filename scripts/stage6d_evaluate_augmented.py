"""
Stage 6d: evaluate augmented model on the same real test.csv (untouched).
Compare this directly against the baseline's 0.909 test macro-F1 and
[0.877, 0.936] CI from Stage 4.
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
    model.load_state_dict(torch.load("best_model_augmented.pt"))

    preds, targets = get_predictions(model, test_loader)
    point_f1 = f1_score(targets, preds, average="macro")
    ci_low, ci_mid, ci_high = bootstrap_macro_f1(preds, targets)

    print(f"[AUGMENTED MODEL] Test macro-F1: {point_f1:.3f}")
    print(f"Bootstrap 95% CI: [{ci_low:.3f}, {ci_high:.3f}] (median {ci_mid:.3f})")
    print("\nFull classification report:")
    print(classification_report(targets, preds, target_names=CLASSES))
    print("\nCompare against baseline: test macro-F1 0.909, CI [0.877, 0.936]")
