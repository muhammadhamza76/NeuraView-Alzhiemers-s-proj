# NeuraView: Alzheimer's MRI Classification with Leakage-Safe Evaluation and Generative Augmentation

## Overview
A 4-class Alzheimer's severity classifier (NonDemented, VeryMildDemented, MildDemented, ModerateDemented) trained on MRI slices, built with an emphasis on catching and correcting data leakage before trusting any reported metric, and testing whether generative augmentation helps a severely underrepresented class.

## Dataset
Kaggle "Alzheimer's Dataset (4 class of Images)" (preetpalsingh25), 128x128 grayscale MRI slices.

**Two problems found and fixed before modeling:**
1. The dataset's provided train/test folders were exact duplicates of each other (6400/6400 images, 100% hash overlap). The real usable pool is 6400 unique images, not 12,800.
2. Within that pool, only 1213 truly distinct source images exist; the rest are augmented copies (avg. ~5.3 copies per source image), confirmed via perceptual hashing. A naive random split leaked these near-duplicates across train/val/test, inflating an early baseline to a false 97.1% macro-F1.

**Final approach:** images were grouped by perceptual-hash similarity (Hamming distance <= 5), and the 70/15/15 train/val/test split was performed at the group level, guaranteeing no near-duplicate of any image crosses a split boundary.

Class distribution (train): NonDemented 2826, VeryMildDemented 1978, MildDemented 772, ModerateDemented 56 (50:1 imbalance ratio, largest to smallest).

## Method
- Backbone: ResNet18, ImageNet-pretrained, fine-tuned end to end
- Grayscale images converted to 3-channel for compatibility
- WeightedRandomSampler to counter class imbalance during training
- Model selection by validation macro-F1, not accuracy, since raw accuracy is misleading under this imbalance
- All test-set metrics reported with bootstrap 95% confidence intervals (2000 resamples), given the small size of the rarest class

## Results

Baseline and augmented models were each trained with 3 different random seeds to separate a genuine treatment effect from ordinary training-run variance, since an initial single-run comparison gave a misleading result (see note below).

| Model | Mean test macro-F1 (3 seeds) | Std | Per-seed results |
|---|---|---|---|
| Baseline (final model) | 0.906 | 0.012 | 0.9085, 0.9184, 0.8899 |
| + Generative augmentation | 0.873 | 0.011 | 0.8621, 0.8885, 0.8674 |

Augmentation reduced test macro-F1 in **3 out of 3 seeds** (mean difference -0.033, std of the per-seed differences 0.010). **The baseline model, without augmentation, is the final reported model.**

Representative single-run baseline classification report (seed 42, test macro-F1 0.903, bootstrap 95% CI [0.868, 0.931]):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| MildDemented | 0.96 | 0.72 | 0.82 | 60 |
| ModerateDemented | 1.00 | 1.00 | 1.00 | 4 |
| NonDemented | 0.91 | 0.97 | 0.94 | 186 |
| VeryMildDemented | 0.85 | 0.86 | 0.85 | 133 |

**Important methodological note:** an initial single-run comparison suggested augmentation gave a statistically significant improvement (paired bootstrap, +0.033, 95% CI [0.003, 0.065]). A subsequent 3-seed comparison reversed this entirely, augmentation lost in every seed. This discrepancy arose because a paired bootstrap test only quantifies uncertainty from test-set sampling, not from training-run randomness, and the two sources of variance were comparable in size here. This is retained in the writeup as a deliberate methodological finding: single-run comparisons, even with proper significance testing, are insufficient for small-data deep learning experiments, and results should be verified across multiple training seeds before being trusted.

## Interpretability (Grad-CAM)
Grad-CAM on the last convolutional block shows all four classes attending primarily to the periventricular region (around the brain's ventricles), consistent with ventricular enlargement as an established real biomarker of atrophy. No attention on background or scan borders was observed, ruling out the most common shortcut-learning failure mode for this type of task.

## Generative augmentation experiment
A small DCGAN was trained on the 56 real ModerateDemented training images to test whether synthetic augmentation can help an extremely underrepresented class. 200 synthetic images were generated and added only to the training set (val/test kept real-only).

**Finding:** across a 3-seed comparison, augmentation consistently reduced overall test macro-F1 (mean -0.033, std 0.010, worse in 3/3 seeds). We attribute this to the visible texture noise in the DCGAN's outputs, a known limitation of training a GAN on only 56 source images. Because the weighted sampler drew real and synthetic images with equal probability, the model spent meaningful training time learning from lower-fidelity images, and since ResNet18's early and mid-level features are shared across all four classes, this noise likely degraded the shared representation rather than affecting only the augmented class. Notably, an earlier single-run comparison had suggested a significant improvement, a result that did not replicate across additional random seeds (see Results and Limitations). This negative result is a genuine, useful finding: at this level of data scarcity, synthetic image quality was the binding constraint, not the quantity of minority-class examples.

## Limitations
- ModerateDemented's test set (n=4) is too small to treat its precision/recall as a precise estimate; results for this class should be read qualitatively.
- Baseline test macro-F1 varied by roughly 3 points across 3 training seeds (0.890 to 0.918), and even with a fixed seed value, GPU convolution nondeterminism (cuDNN) prevents exact reproducibility run to run. This variance is comparable in size to the effect being tested, which is why single-run comparisons were insufficient and a multi-seed comparison was necessary.
- Grad-CAM from a deep layer produces coarse, low-resolution heatmaps; findings describe general regions of attention, not pixel-level precision.
- 2D MRI slices only; no volumetric (3D) information is used.
- The generative augmentation experiment used one DCGAN architecture, one synthetic sample count (200), and one class; results should not be generalized to other augmentation setups without further testing.

## Reproducing this work
1. `stage2c_leakage_safe_split.py` - build the leakage-safe split
2. `stage3_train_baseline.py` - train baseline
3. `stage4_evaluate_bootstrap_standalone.py` - evaluate baseline with bootstrap CI
4. `stage5_gradcam.py` - generate interpretability figure
5. `stage6a_train_gan.py` through `stage6e_paired_significance_test.py` - generative augmentation experiment and significance test
