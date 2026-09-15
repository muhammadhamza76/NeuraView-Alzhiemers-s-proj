"""
Stage 6b: augmented manifest.

Critical rule: synthetic images go ONLY into the training set. Val
and test stay exactly as they were, real images only. Adding
synthetic images to evaluation data would make the comparison
meaningless, since we'd be testing the model partly on images it
was never meant to see as "ground truth."
"""

import os
import pandas as pd

SYNTH_DIR = "synthetic_moderate"
synth_paths = [os.path.join(SYNTH_DIR, f) for f in os.listdir(SYNTH_DIR) if f.endswith(".png")]

train_df = pd.read_csv("splits/train.csv")
synth_df = pd.DataFrame({
    "filepath": synth_paths,
    "label": "ModerateDemented",
})

augmented = pd.concat([train_df, synth_df], ignore_index=True)
augmented.to_csv("splits/train_augmented.csv", index=False)

print(f"Original train size: {len(train_df)}")
print(f"Added synthetic ModerateDemented: {len(synth_df)}")
print(f"Augmented train size: {len(augmented)}")
print(augmented["label"].value_counts())
