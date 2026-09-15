"""
Stage 2c: leakage-safe split.

We already confirmed near-duplicate images cross our train/val/test
boundary. The fix: cluster images that are near-identical (small
perceptual-hash distance) into groups, then split by GROUP, not by
individual file. This guarantees every near-duplicate of a source
image ends up on the same side of the split.

Hamming distance threshold of 5 (out of 64 bits) is a reasonably
strict "these are basically the same image" cutoff. If you find
this groups together images that are clearly different scans on
manual inspection, tighten it (lower the number).
"""

import numpy as np
import pandas as pd
import imagehash
from PIL import Image
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split

DATA_DIR = Path(
    "/kaggle/input/datasets/preetpalsingh25/"
    "alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
HAMMING_THRESHOLD = 5
OUTPUT_DIR = Path("splits")
OUTPUT_DIR.mkdir(exist_ok=True)

def collect_files():
    records = []
    for class_dir in sorted(DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in class_dir.iterdir():
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                records.append((str(f), class_dir.name))
    return records

def hash_to_bits(h):
    return np.array(h.hash.flatten(), dtype=np.uint8)

if __name__ == "__main__":
    records = collect_files()
    filepaths = [r[0] for r in records]
    labels = [r[1] for r in records]
    n = len(filepaths)
    print(f"Hashing {n} images...")

    bits = np.zeros((n, 64), dtype=np.uint8)
    for i, fp in enumerate(filepaths):
        h = imagehash.phash(Image.open(fp))
        bits[i] = hash_to_bits(h)

    print("Computing pairwise Hamming distances...")
    # distance[i,j] = number of differing bits
    dist = bits @ (1 - bits).T + (1 - bits) @ bits.T
    adjacency = (dist <= HAMMING_THRESHOLD).astype(np.uint8)
    np.fill_diagonal(adjacency, 0)

    n_components, group_ids = connected_components(csr_matrix(adjacency), directed=False)
    print(f"Found {n_components} groups from {n} images "
          f"({n - n_components} images merged into existing groups)")

    df = pd.DataFrame({"filepath": filepaths, "label": labels, "group": group_ids})

    # majority label per group, used to stratify the group-level split
    group_labels = df.groupby("group")["label"].agg(lambda x: x.value_counts().idxmax())
    groups = group_labels.index.values
    group_label_values = group_labels.values

    train_val_groups, test_groups = train_test_split(
        groups, test_size=0.15, stratify=group_label_values, random_state=42
    )
    train_val_label_values = group_labels.loc[train_val_groups].values
    train_groups, val_groups = train_test_split(
        train_val_groups, test_size=0.1765, stratify=train_val_label_values, random_state=42
    )

    def save_split(name, group_subset):
        subset = df[df["group"].isin(group_subset)][["filepath", "label"]]
        subset.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(subset)} images, {subset['label'].value_counts().to_dict()}")

    save_split("train", set(train_groups))
    save_split("val", set(val_groups))
    save_split("test", set(test_groups))
