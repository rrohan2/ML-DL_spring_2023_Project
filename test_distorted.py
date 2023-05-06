import torch
import torchvision.models as models
import numpy as np 
import pandas as pd 
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torch.optim as optims
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description='Test a ResNet model on a dataset')

# Add arguments for the test dataset path and model checkpoint directory
parser.add_argument('--test_path', type=str, help='Path to the test dataset')
parser.add_argument('--checkpoint_dir', type=str, help='Path to the model checkpoint directory')

# Parse the command-line arguments
args = parser.parse_args()

# Get the test dataset path and model checkpoint directory from the parsed arguments
test_path = args.test_path
checkpoint_dir = args.checkpoint_dir


def load(test_path):
    test_dataset = tv.datasets.ImageFolder(root=test_path, transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )

    return test_loader

test_path = '/content/drive/MyDrive/imagenet-mini/test_undistorted'

# Step 1: Prepare testing data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the input data
])

test_loader = load(test_path)

# Step 2: Load the model checkpoint
model = models.resnet50(pretrained=False)  # Use a similar model architecture
checkpoint_dir = '/content/drive/MyDrive/imagenet-mini/res50model_checkpoint.pth'
checkpoint = torch.load(checkpoint_dir)
model.load_state_dict(checkpoint)


# Step 3: Evaluate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = 100 * correct / total
print('Test Accuracy: {:.2f}%'.format(accuracy))
