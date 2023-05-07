import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model

# get test dataset path....
class Classifier(nn.Module):
    def __init__(self, nin, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(nin, num_classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

# Define the adversarial domain adaptation training function
def train_model(model, source_loader, target_loader, optimizer, criterion, alpha):
    model.train()

    for batch_idx, (source_data, source_target) in enumerate(source_loader):
        try:
            target_data, target_target = next(target_iter)
        except:
            target_iter = iter(target_loader)
            target_data, target_target = next(target_iter)
            
        optimizer.zero_grad()

        # Forward pass on the source data
        source_pred = model(source_data)
        source_loss = criterion(source_pred, source_target)

        # Forward pass on the target data
        target_pred = model(target_data)
        target_loss = criterion(target_pred, target_target)

        # Compute the domain confusion loss
        source_features = model.features(source_data)
        target_features = model.features(target_data)
        domain_loss = alpha * criterion(source_features.mean(0) - target_features.mean(0), torch.zeros_like(source_features.mean(0)))

        # Backward pass and optimization
        total_loss = source_loss + target_loss + domain_loss
        total_loss.backward()
        optimizer.step()

        # Print progress
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(source_data), len(source_loader.dataset),
                100. * batch_idx / len(source_loader), total_loss.item()))

# Define the main function
def main():
    # Set up the data loaders
    transform = transforms.Compose([transforms.Resize(32),
                                    transforms.CenterCrop(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    source_dataset = ImageFolder('source_data_path', transform=transform)
    target_dataset = ImageFolder('target_data_path', transform=transform)
    source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True, num_workers=4)
    target_loader = DataLoader(target_dataset, batch_size=64, shuffle=True, num_workers=4)

    # Set up the model, optimizer, and criterion
    # Load pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True)

    # Modify the last fully-connected layer to match the number of classes in ImageNet dataset
    num_classes = 1000 # Number of classes in ImageNet dataset    
    model = Classifier(model.fc.in_features, num_classes)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Set the alpha parameter for the domain confusion loss
    alpha = 0.2 # playing around with this

    # Train the model using domain adaptation
    for epoch in range(15): #may change
        train_model(model, source_loader, target_loader, optimizer, criterion, alpha)

    # Evaluate
    test_loss, test_acc = model.evaluate(x_hold, y_hold)
    print('Test accuracy:', test_acc)

