"""
Stage 5: Grad-CAM (self-contained, no extra library).

How it works, briefly: we hook layer4 (the last conv block) to
capture its activations during the forward pass, and its gradients
during the backward pass. For a chosen class, "how much did each
channel in layer4 matter for this prediction" is the gradient
averaged over space. Weighting each channel's activation map by
that importance, summing, and applying ReLU (we only care about
features that increased the score, not decreased it) gives the
Grad-CAM heatmap. We then resize it up to image size and overlay it.

Run this after Stage 3/4, in the same session, so best_model.pt exists.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

CLASSES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model.to(DEVICE)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, output.argmax(1).item()

def overlay_heatmap(original_img, cam):
    heatmap = cm.jet(cam)[:, :, :3]
    original = np.array(original_img.resize((224, 224)).convert("RGB")) / 255.0
    return 0.5 * original + 0.5 * heatmap

if __name__ == "__main__":
    model = build_model()
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()

    gradcam = GradCAM(model, model.layer4)

    df = pd.read_csv("splits/test.csv")
    # take one correctly-labeled example per class for a clean comparison grid
    sample_paths = df.groupby("label").first()["filepath"].to_dict()

    fig, axes = plt.subplots(2, len(CLASSES), figsize=(4 * len(CLASSES), 8))

    for i, (label, path) in enumerate(sample_paths.items()):
        img = Image.open(path)
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)
        cam, pred_idx = gradcam.generate(input_tensor, LABEL_TO_IDX[label])
        overlay = overlay_heatmap(img, cam)

        axes[0, i].imshow(img.resize((224, 224)), cmap="gray")
        axes[0, i].set_title(f"True: {label}")
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"Pred: {CLASSES[pred_idx]}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_grid.png", dpi=150)
    print("Saved gradcam_grid.png")
    plt.show()
