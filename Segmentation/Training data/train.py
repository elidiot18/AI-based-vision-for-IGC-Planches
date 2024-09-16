import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pickle
import sys
import argparse
import logging
import time
import cv2

####### logging

logging.basicConfig(
    filename='./Model/U-Net_1.log',  # File where logs will be saved
    level=logging.INFO,  # Log level (e.g., INFO, DEBUG, ERROR)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

def log_training_progress(epoch, loss, loss_eval, ratio):
    logging.info(f'(Partial) epoch with ratio {ratio}: {epoch},\nLoss of training set: {loss:.4f},\nLoss of evaluation set: {loss_eval:.4f}')

####### display progress

def show_progress_bar(progress, total, message):
    percent = 100 * (progress / float(total))
    bar = '█' * int(percent/10) + '-' * (10 - int(percent/10))
    sys.stdout.write('\r' + message + f'[{bar}] {percent:.2f}%')  # Dynamic update on the same line
    sys.stdout.flush()

####### U-Net

import UNet3

####### warning

# to avoid the warning from torch.load
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


####### hyperparameters and loss

# experiments with Dice loss
def dice_loss(preds, targets, smooth=1.0):
    preds = preds.contiguous()
    targets = targets.contiguous()

    intersection = (preds * targets).sum(dim=2).sum(dim=2)

    loss = (2. * intersection + smooth) / (preds.sum(dim=2).sum(dim=2) + targets.sum(dim=2).sum(dim=2) + smooth)

    return 1 - loss.mean()

# experiments with log-modified TV loss
def log_TV_loss(output, alpha=1.0):
    diff_i = torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :])
    diff_j = torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1])

    penalty_i = torch.log(1 + alpha * diff_i)
    penalty_j = torch.log(1 + alpha * diff_j)

    return torch.mean(penalty_i) + torch.mean(penalty_j)

# Hyperparameters
learning_rate = 1e-4

####### gpu

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
        return torch.tensor(image, dtype=torch.float32).to(device), torch.tensor(mask, dtype=torch.float32).to(device)

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

    return images, masks

####### training loop

def train(train_dataset, test_dataset, batch_size, ratio, epochs):
    print(f"Samples per (partial) epoch: {ratio * len(train_dataset)} samples.")
    for epoch in range(epochs):
        train_dataset.sample(ratio)
        test_dataset.sample(0.1) # default, might change by hand

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        train_loss = torch.zeros(1, device=device)
        test_loss = torch.zeros(1, device=device)

        print(f"(Partial) epoch [{epoch + 1}/{epochs}]:")
        for batch_idx, (images, masks) in enumerate(train_dataloader):
            show_progress_bar(batch_idx, len(train_dataloader), "Progression: ")

            optimizer.zero_grad()
            #print("computing model")
            outputs = model(images)
        #    print("criterion")
            bce_loss = criterion(outputs, masks)
            #dice_loss_value = dice_loss(outputs, masks)
            # experiment, doesn't seem to give good results
            loss = bce_loss + 0.2*log_TV_loss(outputs)#0.7 * bce_loss + 0.3 * dice_loss_value
        #    print("backward")
            loss.backward()
        #    print("step")
            optimizer.step()

            train_loss += loss
        model.eval()
        print(f"Testing performance;")
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(test_dataloader):
                show_progress_bar(batch_idx + 2, len(test_dataloader), "Progression: ")

                outputs_eval = model(images)
                bce_loss = criterion(outputs_eval, masks)
                #dice_loss_value = dice_loss(outputs_eval, masks)
                loss = bce_loss + 0.2*log_TV_loss(outputs) #0.7 * bce_loss + 0.3 * dice_loss_value
                test_loss += loss

        train_loss, test_loss = train_loss.item(), test_loss.item()

        print("Done.")
        print(f"Loss for the training set: {train_loss / len(train_dataloader)}")
        print(f"Loss for the testing set: {test_loss / len(test_dataloader)}")
        log_training_progress(epoch, train_loss / len(train_dataloader), test_loss / len(test_dataloader), ratio)

        # Save the model
        torch.save(model.state_dict(), f'./Model/U-Net_1_model.pth')
        torch.save(optimizer.state_dict(), f'./Model/U-Net_1_optimizer.pth')

####### main

def main(batches, test=None, batch_size=None, ratio=None, epochs=None):
    print(f"Training on batches: {batches}")
    if test is None:
        test = 1
    print(f"Testing on batch: {test}")
    if batch_size is None:
        batch_size = 10
    print(f"Batch Size: {batch_size}")
    if ratio is None:
        ratio = 1
    print(f"Ratio for partial epochs: {ratio}")
    if epochs is None:
        epochs = 1
    print(f"Number of (partial) epochs: {epochs}")

    train_dirs = [f'./Training/batch_{n}/' for n in batches]
    test_dir = f'./Training/batch_{test}/'

    train_images = []
    train_masks = []

    for dir in train_dirs:
        if not os.path.exists(dir):
            raise ValueError("There is no such directory: " + dir)
        if os.path.exists(dir + 'data.pkl'):
            with open(dir + 'data.pkl', 'rb') as file:
                print("Loading training pickle data...")
                images, masks = pickle.load(file)
                print("done.")
                train_images += images
                train_masks += masks
        else:
            with open(dir + 'data.pkl', 'wb') as file:
                print("Creating pickle data in " + dir + "...")
                images, masks = load_training_data(dir)
                pickle.dump((images, masks), file)
                print("done.")
                train_images += images
                train_masks += masks

    if not os.path.exists(test_dir):
        raise ValueError("There is no such directory: " + test_dir)
    if os.path.exists(test_dir + 'data.pkl'):
        with open(test_dir + 'data.pkl', 'rb') as file:
            print("Loading testing pickle data...")
            test_images, test_masks = pickle.load(file)
            print("done.")
    else:
        with open(test_dir + 'data.pkl', 'wb') as file:
            print("Creating pickle data in " + test_dir + "...")
            test_images, test_masks = load_training_data(test_dir)
            pickle.dump((test_images, test_masks), file)
            print("done.")

    train_dataset = SampleDataset(train_images, train_masks)
    test_dataset = SampleDataset(test_images, test_masks)

    train(train_dataset, test_dataset, batch_size, ratio, epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for IGC Masse Calcaire vision with the given parameters.")

    # Required positional argument that accepts at least one value
    parser.add_argument('batches', nargs='+', type=int, help="One or more batches of training data to train on")

    # Optional arguments
    parser.add_argument('-t', '--test', type=int, help="Test on the given batch (default=1)")
    parser.add_argument('-b', '--batch_size', type=int, help="Batch size (default=10)")
    parser.add_argument('-r', '--ratio', type=float, help="Ratio for partial epochs (default=1)")
    parser.add_argument('-e', '--epochs', type=int, help="Number of (partial) epochs (default=1)")


    args = parser.parse_args()
    main(args.batches, args.test, args.batch_size, args.ratio, args.epochs)
