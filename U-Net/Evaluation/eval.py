import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.upconv4 = self.upconv_block(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        self.decoder1 = self.conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        up4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat((up4, enc4), dim=1))
        up3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat((up3, enc3), dim=1))
        up2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((up2, enc2), dim=1))
        up1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((up1, enc1), dim=1))
        return torch.sigmoid(self.final_conv(dec1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load('../Model/U-Net_1_model.pth', weights_only = False))

img_path = input("Evaluate on file: ")

image = cv2.imread(img_path, cv2.IMREAD_COLOR)
h, w, _ = image.shape
image = np.transpose(image, (2, 0, 1))

output = torch.zeros((2, h, w), dtype=torch.float32).to(device)
overlap = 0.5
step = int(256 * (1 - overlap))

batch_size = 18
patches = []
coords = []

random_patches = []
random_predictions = []

def process_batch(patches, coords, model, output):
    patches = torch.stack(patches)
    with torch.no_grad():
        patch_evals = model(patches)

    for i, (y, x) in enumerate(coords):
        output[0, y:y + 256, x:x + 256] += patch_evals[i, 0, :, :]
        output[1, y:y + 256, x:x + 256] += 1

def show_progress_bar(progress, total, message):
    percent = 100 * (progress / float(total))
    bar = '█' * int(percent/10) + '-' * (10 - int(percent/10))
    message = '\r' + message + f'[{bar}] {percent:.2f}%'
    sys.stdout.write(message)
    sys.stdout.flush()

with torch.no_grad():
    for y in range(0, h, step):
        for x in range(0, w, step):
            show_progress_bar(x + (y+1)*w, h*w, "Progress: ")
            if y + 256 > h:
                y = h - 256
            if x + 256 > w:
                x = w - 256

            patch = torch.tensor(image[:, y:y + 256, x:x + 256] / 255.0, dtype=torch.float32).to(device)
            patches.append(patch)
            coords.append((y, x))

            if len(patches) == batch_size:
                process_batch(patches, coords, model, output)
                patches = []
                coords = []

    if patches:
        process_batch(patches, coords, model, output)

output = output[0] / output[1]

output_image = output.cpu().numpy()

name, ext = os.path.splitext(img_path)
cv2.imwrite(name + '_eval' + ext, (output_image * 255).astype(np.uint8))
