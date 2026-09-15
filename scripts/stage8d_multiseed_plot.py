import matplotlib.pyplot as plt
import numpy as np

# from your Stage 7 output, no retraining needed
seeds = [42, 123, 2024]
baseline = [0.9085, 0.9184, 0.8899]
augmented = [0.8621, 0.8885, 0.8674]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(seeds))
width = 0.35

ax.bar(x - width/2, baseline, width, label="Baseline", color="#4C72B0")
ax.bar(x + width/2, augmented, width, label="+ GAN augmentation", color="#DD8452")

for i in range(len(seeds)):
    ax.text(x[i] - width/2, baseline[i] + 0.005, f"{baseline[i]:.3f}", ha="center", fontsize=9)
    ax.text(x[i] + width/2, augmented[i] + 0.005, f"{augmented[i]:.3f}", ha="center", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([f"Seed {s}" for s in seeds])
ax.set_ylabel("Test macro-F1")
ax.set_ylim(0.80, 0.95)
ax.set_title("Baseline consistently outperforms augmented model across all 3 seeds")
ax.legend()
plt.tight_layout()
plt.savefig("multiseed_comparison.png", dpi=150)
print("Saved multiseed_comparison.png")
