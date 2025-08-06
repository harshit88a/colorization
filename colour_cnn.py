import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage import color
import os
import matplotlib.pyplot as plt

# -------------- Utilities --------------
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def lab_to_rgb(L, ab):
    L = to_numpy(L)
    ab = to_numpy(ab)
    if L.ndim == 3 and L.shape[0] == 1:
        L = L[0]
    if ab.shape[0] == 2:
        ab = ab.transpose(1, 2, 0)
    L = np.array(L, dtype=np.float32)
    ab = np.array(ab, dtype=np.float32)
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float32)
    lab[:, :, 0] = L * 100
    lab[:, :, 1:] = ab * 128
    rgb = color.lab2rgb(lab)
    rgb = np.clip(rgb, 0, 1)
    return rgb

# ------------ Model Definition -------------
class UNetColorization(nn.Module):
    def __init__(self):
        super().__init__()
        def CBR(in_ch, out_ch): return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU())
        self.enc1 = CBR(1, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2, 2)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = CBR(128, 64)
        self.final = nn.Conv2d(64, 2, kernel_size=1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d1 = self.up1(e4)
        d1 = self.dec1(torch.cat([d1, e3], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, e1], dim=1))
        return torch.tanh(self.final(d3))

# ------------ Colorization Process -------------
def colorize_images(input_dir, output_dir, model_weights, device='cpu', save=True, show=False):
    # Prepare model
    model = UNetColorization()
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fname in os.listdir(input_dir):
        if not (fname.lower().endswith(".png") or fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg")):
            continue
        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert('L')   # B&W input

        img_resized = transform(img)
        L = img_resized.unsqueeze(0).to(device)   # shape [1,1,H,W]
        with torch.no_grad():
            pred_ab = model(L)                   # shape [1,2,H,W]

        L_np = L.cpu().squeeze().numpy()
        pred_ab_np = pred_ab.cpu().squeeze(0)
        colorized = lab_to_rgb(L_np, pred_ab_np)

        # Save results (side-by-side)
        fig, axs = plt.subplots(1,2, figsize=(8,4))
        axs[0].imshow(L_np, cmap='gray')
        axs[0].set_title('Input B&W')
        axs[0].axis('off')
        axs[1].imshow(colorized)
        axs[1].set_title('Colorized')
        axs[1].axis('off')
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(output_dir, fname+"_colorized.png"))
        if show:
            plt.show()
        plt.close()


# Directory with your B&W test images
input_dir = "bw_img"
output_dir = "output_cnn"
model_weights = "best_colorization_model.pth"
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print("Using device:", device)

colorize_images(input_dir, output_dir, model_weights, device, save=True, show=True)

