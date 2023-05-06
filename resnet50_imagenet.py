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
torch.cuda.empty_cache()


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

train_dataset = torchvision.datasets.ImageFolder(root='/content/drive/MyDrive/imagenet-mini/train', transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=48, shuffle=True)

val_dataset = torchvision.datasets.ImageFolder(root='/content/drive/MyDrive/imagenet-mini/val', transform=transform)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=48, shuffle=True)


# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)

# Modify the last fully-connected layer to match the number of classes in ImageNet dataset
num_classes = 1000 # Number of classes in ImageNet dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Set model to device
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []
# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    # Train on training set
    model.train()
    train_epoch_loss = 0.0
    train_correct = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
    train_epoch_loss = train_epoch_loss / len(train_loader.dataset)
    train_epoch_acc = train_correct / len(train_loader.dataset)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_acc)
    # Evaluate on validation set
    model.eval()
    val_epoch_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_epoch_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
    val_epoch_loss = val_epoch_loss / len(val_loader.dataset)
    val_epoch_acc = val_correct / len(val_loader.dataset)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_acc)
    print("Epoch %d, Training Loss: %.3f, Training Accuracy: %.3f, Validation Loss: %.3f, Validation Accuracy: %.3f" %
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