import pandas as pd
import matplotlib.pyplot as plt

CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

fig, ax = plt.subplots(figsize=(9, 5))
width = 0.25
x = range(len(CLASSES))

for i, split in enumerate(["train", "val", "test"]):
    df = pd.read_csv(f"splits/{split}.csv")
    counts = df["label"].value_counts().reindex(CLASSES).fillna(0)
    ax.bar([p + i * width for p in x], counts, width, label=split)

ax.set_xticks([p + width for p in x])
ax.set_xticklabels(CLASSES, rotation=15)
ax.set_ylabel("Number of images")
ax.set_title("Class distribution across splits (leakage-safe, group-based)")
ax.legend()
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
print("Saved class_distribution.png")
