import numpy as np 
import pandas as pd 
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optims
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

def resize_image(src, size=(128, 128), bgc="white"):
    src.thumbnail(size, Image.ANTIALIAS)
    
    new_image = Image.new("RGB", size, bgc)
    
    new_image.paste(src, (int((size[0]-src.size[0]) / 2)), int((size[1] - src.size[1]) / 2))
    
    return new_image

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load(train_path, val_path):
    train_dataset = tv.datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = tv.datasets.ImageFolder(root=val_path, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=50,
        num_workers=0,
        shuffle=False
    )

    return train_loader, val_loader

train_path = 'path/imagenetmini-1000/imagenet-mini/train'
val_path = 'path/imagenetmini-1000/imagenet-mini/val'

train_loader, val_loader = load(train_path, val_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = tv.models.resnet50(weights=tv.models.ResNet50_Weights.DEFAULT)

model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []
num_epochs = 500

for epoch in range(num_epochs):
    train_epoch_loss = 0.0
    train_correct = 0
    train_total = 0
    
    val_epoch_loss = 0.0
    val_correct = 0
    val_total = 0
    
    # Training phase
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        train_epoch_loss += loss.item()
        
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    
    train_epoch_accuracy = 100 * train_correct / train_total
    train_loss.append(train_epoch_loss / len(train_loader))
    train_accuracy.append(train_epoch_accuracy)
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_epoch_loss += loss.item()
            
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_epoch_accuracy = 100 * val_correct / val_total
    val_loss.append(val_epoch_loss / len(val_loader))
    val_accuracy.append(val_epoch_accuracy)
    
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"Train Loss: {train_loss[epoch]:.4f} - Train Accuracy: {train_epoch_accuracy:.2f}%")
    print(f"Val Loss: {val_loss[epoch]:.4f} - Val Accuracy: {val_epoch_accuracy:.2f}%")

# Save the model checkpoint
torch.save(model.state_dict(), 'model_checkpoint.pth')
    
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
