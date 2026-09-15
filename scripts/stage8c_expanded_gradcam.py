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
N_PER_CLASS = 4

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
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

def overlay(img, cam_map):
    heatmap = cm.jet(cam_map)[:, :, :3]
    base = np.array(img.resize((224, 224)).convert("RGB")) / 255.0
    return 0.5 * base + 0.5 * heatmap

if __name__ == "__main__":
    model = build_model()
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    gradcam = GradCAM(model, model.layer4)

    df = pd.read_csv("splits/test.csv")
    fig, axes = plt.subplots(len(CLASSES), N_PER_CLASS, figsize=(3 * N_PER_CLASS, 3 * len(CLASSES)))

    for row, label in enumerate(CLASSES):
        paths = df[df["label"] == label]["filepath"].head(N_PER_CLASS).tolist()
        for col, path in enumerate(paths):
            img = Image.open(path)
            input_tensor = transform(img).unsqueeze(0).to(DEVICE)
            cam_map = gradcam.generate(input_tensor, LABEL_TO_IDX[label])
            axes[row, col].imshow(overlay(img, cam_map))
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(label, fontsize=10)
        axes[row, 0].text(-30, 112, label, rotation=90, va="center", fontsize=11)

    plt.suptitle("Grad-CAM across multiple examples per class")
    plt.tight_layout()
    plt.savefig("gradcam_grid_expanded.png", dpi=150)
    print("Saved gradcam_grid_expanded.png")
