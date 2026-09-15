"""
Stage 6a: DCGAN for ModerateDemented augmentation.

Honest expectation: 44 real images is very little for GAN training.
Standard GANs are typically trained on thousands of images. We use a
small architecture and heavy standard augmentation (flips, rotations)
applied to the real images each epoch, which effectively shows the
discriminator more varied views of the same 44 images. This does not
fix the fundamental data scarcity, it just gives the GAN a slightly
better chance. If generated samples look like noise or collapse to
one repeated pattern, that IS the result, report it as such.
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils as vutils
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 64
LATENT_DIM = 100
EPOCHS = 300
BATCH_SIZE = 8
N_SYNTHETIC = 200

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

class ModerateDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.paths = df[df["label"] == "ModerateDemented"]["filepath"].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return train_transform(Image.open(self.paths[idx]))

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 256, 4, 1, 0), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 1, 4, 1, 0), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1)

if __name__ == "__main__":
    dataset = ModerateDataset("splits/train.csv")
    print(f"Training GAN on {len(dataset)} real ModerateDemented images")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    G, D = Generator().to(DEVICE), Discriminator().to(DEVICE)
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(EPOCHS):
        for real in loader:
            real = real.to(DEVICE)
            bs = real.size(0)
            real_labels = torch.full((bs,), 0.9, device=DEVICE)  # label smoothing
            fake_labels = torch.zeros(bs, device=DEVICE)

            # train discriminator
            opt_d.zero_grad()
            d_real = D(real)
            loss_d_real = criterion(d_real, real_labels)
            z = torch.randn(bs, LATENT_DIM, 1, 1, device=DEVICE)
            fake = G(z)
            d_fake = D(fake.detach())
            loss_d_fake = criterion(d_fake, fake_labels)
            (loss_d_real + loss_d_fake).backward()
            opt_d.step()

            # train generator
            opt_g.zero_grad()
            d_fake_for_g = D(fake)
            loss_g = criterion(d_fake_for_g, torch.full((bs,), 0.9, device=DEVICE))
            loss_g.backward()
            opt_g.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | D loss: {(loss_d_real+loss_d_fake).item():.3f} | G loss: {loss_g.item():.3f}")

    # save a visual grid to inspect before trusting these for anything
    z = torch.randn(16, LATENT_DIM, 1, 1, device=DEVICE)
    with torch.no_grad():
        samples = G(z).cpu()
    vutils.save_image(samples, "gan_samples_preview.png", normalize=True, nrow=4)
    print("Saved gan_samples_preview.png - LOOK AT THIS before proceeding")

    # generate the full synthetic set for later use, only if preview looks reasonable
    import os
    os.makedirs("synthetic_moderate", exist_ok=True)
    with torch.no_grad():
        for i in range(N_SYNTHETIC):
            z = torch.randn(1, LATENT_DIM, 1, 1, device=DEVICE)
            img = G(z).cpu()
            vutils.save_image(img, f"synthetic_moderate/synth_{i}.png", normalize=True)
    print(f"Saved {N_SYNTHETIC} synthetic images to synthetic_moderate/")
