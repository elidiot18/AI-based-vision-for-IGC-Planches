import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pickle
import sys
import logging
import time
import cv2

####### logging

logging.basicConfig(
    filename='./Model/U-Net_1.log',  # File where logs will be saved
    level=logging.INFO,  # Log level (e.g., INFO, DEBUG, ERROR)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

def log_training_progress(epoch, loss, loss_eval):
    logging.info(f'(Partial) epoch with ratio {ratio}: {epoch},\nLoss of training set: {loss:.4f},\nLoss of evaluation set: {loss_eval:.4f}')

####### display progress

def show_progress_bar(progress, total, message):
    percent = 100 * (progress / float(total))
    bar = '█' * int(percent/10) + '-' * (10 - int(percent/10))
    sys.stdout.write(message + f'\r[{bar}] {percent:.2f}%')  # Dynamic update on the same line
    sys.stdout.flush()

####### U-Net

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Contracting Path (Encoder)
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Expanding Path (Decoder)
        self.upconv4 = self.upconv_block(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        self.decoder1 = self.conv_block(128, 64)

        # Final Convolution
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
        # Contracting Path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Expanding Path
        up4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat((up4, enc4), dim=1))
        up3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat((up3, enc3), dim=1))
        up2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((up2, enc2), dim=1))
        up1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((up1, enc1), dim=1))

        return torch.sigmoid(self.final_conv(dec1))

####### hyperparameters and loss

# Experiments with Dice loss
def dice_loss(preds, targets, smooth=1.0):
    preds = preds.contiguous()
    targets = targets.contiguous()

    intersection = (preds * targets).sum(dim=2).sum(dim=2)

    loss = (2. * intersection + smooth) / (preds.sum(dim=2).sum(dim=2) + targets.sum(dim=2).sum(dim=2) + smooth)

    return 1 - loss.mean()

# Hyperparameters
n = int(input("Training with batch n° : ").strip())
m = int(input("Testing with batch n° : ").strip())
learning_rate = 1e-4
batch_size = 8
ratio = float(input("Ratio for partial epochs : "))
epochs = int(input("Number of partial epochs to execute : "))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####### loading the model

model = UNet(in_channels=3, out_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

model.load_state_dict(torch.load('./Model/U-Net_1_model.pth'))
optimizer.load_state_dict(torch.load('./Model/U-Net_1_optimizer.pth'))

# SampleDataSet has .sample(ratio) and a __getitem__ that allows randomely selected smaller samples
# for partial epochs
class SampleDataset(Dataset):
    def __init__(self, images, masks):
        self.images, self.masks = (images, masks)
        self.indices = np.arange(len(self.images))  # All indices
        self.subset_indices = self.indices  # Default to all indices

    def sample(self, percentage):
        num_samples = int(len(self.indices) * percentage)
        self.subset_indices = np.random.choice(self.indices, num_samples, replace=False)

    def __len__(self):
        return len(self.subset_indices)

    def __getitem__(self, idx):
        index = self.subset_indices[idx]
        image = self.images[index]
        mask = self.masks[index]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

def load_training_data(data_dir):
    images = []
    masks = []
    # Find all test images and truth images for this angle
    test_filenames = sorted([f for f in os.listdir(data_dir) if f.endswith('_test.png')])
    truth_filenames = sorted([f for f in os.listdir(data_dir) if f.endswith('_truth.png')])
    # Loop through the files for this angle and load images and corresponding masks
    for test_file, truth_file in zip(test_filenames, truth_filenames):
        # Load the test image (color image)
        test_img = cv2.imread(os.path.join(data_dir, test_file), cv2.IMREAD_COLOR)

        # Load the corresponding truth mask (grayscale image)
        truth_img = cv2.imread(os.path.join(data_dir, truth_file), cv2.IMREAD_GRAYSCALE)/255

        # Convert images and masks to PyTorch tensors
        # Convert to (channels, height, width) format
        test_img = np.transpose(test_img, (2, 0, 1))  # Change from (height, width, channels) to (channels, height, width)
        truth_img = np.expand_dims(truth_img, axis=0)  # Change from (height, width) to (1, height, width)


        # Append images and masks to the lists
        images.append(test_img)
        masks.append(truth_img)

    return np.array(images), np.array(masks)

data_dir = f'./Training/batch_{n}/'
data_dir_eval = f'./Training/batch_{m}/'
data_pkl = data_dir + 'data.pkl'
data_pkl_eval = data_dir_eval + 'data.pkl'

if os.path.exists(data_pkl) and input(f"{data_pkl} already exists. Do you want to overwrite it? (y/n): ").strip().lower() == 'n':
    with open(data_pkl, 'rb') as file:
        images, masks = pickle.load(file)
else:
    with open(data_pkl, 'wb') as file:
        images, masks = load_training_data(data_dir)
        pickle.dump((images, masks), file)

if os.path.exists(data_pkl_eval) and input(f"{data_pkl_eval} already exists. Do you want to overwrite it? (y/n): ").strip().lower() == 'n':
    with open(data_pkl_eval, 'rb') as file:
        images_eval, masks_eval = pickle.load(file)
else:
    with open(data_pkl_eval, 'wb') as file:
        images_eval, masks_eval = load_training_data(data_dir_eval)
        pickle.dump((images_eval, masks_eval), file)


dataset = SampleDataset(images, masks)
dataset_eval = SampleDataset(images_eval, masks_eval)
dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=True)

####### training loop

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    loss_eval = 0
    dataset.sample(ratio)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"(Partial) epoch [{epoch + 1}/{epochs}], training on {len(dataset)} samples...")
    for batch_idx, (images, masks) in enumerate(dataloader):
        show_progress_bar(batch_idx, len(dataloader), "Progression: ")
        images = torch.tensor(images, dtype=torch.float32).to(device)
        masks = torch.tensor(masks, dtype=torch.float32).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        bce_loss = criterion(outputs, masks)
        dice_loss_value = dice_loss(outputs, masks)

        # experiment, doesn't seem to give good results
        loss = 0.7 * bce_loss + 0.3 * dice_loss_value
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    model.eval()
    dataset_eval.sample(0.1)
    print(f"Evaluating on a portion of batch {m}...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader_eval):
            show_progress_bar(batch_idx + 2, len(dataloader_eval), "Progression: ")
            images = torch.tensor(images, dtype=torch.float32).to(device)
            masks = torch.tensor(masks, dtype=torch.float32).to(device)

            outputs_eval = model(images)
            bce_loss = criterion(outputs_eval, masks)
            dice_loss_value = dice_loss(outputs_eval, masks)

            loss = 0.5 * bce_loss + 0.5 * dice_loss_value
            loss_eval += loss.item()


    print("Done.")
    print(f"Loss for the training set: {epoch_loss / len(dataloader)}")
    print(f"Loss for the evaluation set: {loss_eval / len(dataloader_eval)}")
    log_training_progress(epoch, epoch_loss / len(dataloader), loss_eval / len(dataloader_eval))
    # Save the model state dictionary
    torch.save(model.state_dict(), f'./Model/U-Net_1_model.pth')

    # Save the optimizer state dictionary (if you want to resume training)
    torch.save(optimizer.state_dict(), f'./Model/U-Net_1_optimizer.pth')