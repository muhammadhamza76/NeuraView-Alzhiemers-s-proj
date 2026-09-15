"""
Stage 6e: paired bootstrap comparison.

Comparing two separate confidence intervals for overlap is a rough
heuristic, not a real test. The correct approach: for each bootstrap
resample of the test set, compute BOTH models' macro-F1 on the exact
same resampled indices, then look at the distribution of
(augmented - baseline). If that distribution's 95% CI includes 0,
the improvement is not statistically significant.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import f1_score

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

def get_predictions(model_path, loader):
    model = build_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds.extend(outputs.argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    return np.array(preds), np.array(targets)

if __name__ == "__main__":
    test_loader = DataLoader(MRIDataset("splits/test.csv"), batch_size=32)

    preds_base, targets = get_predictions("best_model.pt", test_loader)
    preds_aug, _ = get_predictions("best_model_augmented.pt", test_loader)

    point_base = f1_score(targets, preds_base, average="macro")
    point_aug = f1_score(targets, preds_aug, average="macro")
    print(f"CURRENT baseline model test macro-F1 (live, this session): {point_base:.4f}")
    print(f"CURRENT augmented model test macro-F1 (live, this session): {point_aug:.4f}")
    print(f"Raw point difference: {point_aug - point_base:.4f}")
    print("(This should be close to the bootstrap mean difference below. "
          "If not, something is still inconsistent.)\n")

    n = len(targets)
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        f1_base = f1_score(targets[idx], preds_base[idx], average="macro")
        f1_aug = f1_score(targets[idx], preds_aug[idx], average="macro")
        diffs.append(f1_aug - f1_base)
    diffs = np.array(diffs)

    ci_low, ci_mid, ci_high = np.percentile(diffs, [2.5, 50, 97.5])
    pct_positive = (diffs > 0).mean() * 100

    print(f"Mean difference (augmented - baseline): {diffs.mean():.4f}")
    print(f"95% CI of difference: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Augmented model wins in {pct_positive:.1f}% of bootstrap resamples")
    if ci_low <= 0 <= ci_high:
        print("\nCI includes 0: the improvement is NOT statistically significant.")
    else:
        print("\nCI excludes 0: the improvement IS statistically significant.")
