import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description='Train a ResNet model on image datasets')

# Add arguments for train and validation dataset paths
parser.add_argument('--train_path', type=str, help='Path to the train dataset')
parser.add_argument('--val_path', type=str, help='Path to the validation dataset')

# Parse the command-line arguments
args = parser.parse_args()

# Get the train and validation dataset paths from the parsed arguments
train_path = args.train_path
val_path = args.val_path

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define the source and target datasets
source_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
target_dataset = torchvision.datasets.ImageFolder(root=val_path, transform=transform)

source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=48, shuffle=True)
target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=48, shuffle=True)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)

# Modify the last fully-connected layer to match the number of classes in ImageNet dataset
num_classes = 1000  # Number of classes in ImageNet dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Set model to device
model = model.to(device)

# Define loss functions
classification_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# # Define domain labels (0 for source, 1 for target)
source_labels = torch.zeros(48, dtype=torch.float32, device=device)
target_labels = torch.ones(48, dtype=torch.float32, device=device)

train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    # Train on source domain
    model.train()
    train_epoch_loss = 0.0
    train_correct = 0
    for inputs, labels in source_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        classification_loss = classification_criterion(outputs, labels)
        domain_outputs = model(inputs, domain_adaptation=True)
        domain_loss = domain_criterion(domain_outputs, source_labels)
        loss = classification_loss + domain_loss
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
    train_epoch_loss = train_epoch_loss / len(source_loader.dataset)
    train_epoch_acc = train_correct / len(source_loader.dataset)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_acc)

    # Evaluate on target domain
    model.eval()
    val_epoch_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in target_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = classification_criterion(outputs, labels)
            val_epoch_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()

    val_epoch_loss = val_epoch_loss / len(target_loader.dataset)
    val_epoch_acc = val_correct / len(target_loader.dataset)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_acc)
    print("Epoch %d, Source Domain Loss: %.3f, Source Domain Accuracy: %.3f, Target Domain Loss: %.3f, Target Domain Accuracy: %.3f" %
        (epoch + 1, train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc))

# Save the model checkpoint
torch.save(model.state_dict(), '/content/drive/MyDrive/imagenet-mini/res50model_checkpoint.pth')

# Plotting loss and accuracy
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
epochs = range(1, num_epochs + 1)

ax1.plot(epochs, train_loss, label='Train Loss')
ax1.plot(epochs, val_loss, label='Val Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()

ax2.plot(epochs, train_accuracy, label='Train Accuracy')
ax2.plot(epochs, val_accuracy, label='Val Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()

