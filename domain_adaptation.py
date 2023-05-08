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
    def __init__(self, num_classes, final_layer):
        super(Classifier, self).__init__()
        self.features = models.resnet50(pretrained=True)
        self.classifier = nn.Linear(self.features.fc.in_features, num_classes)
        self.classifier.load_state_dict(final_layer)
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Define the adversarial domain adaptation training function
def train_model(epoch, discriminator, target_model, source_model, target_classifier, source_loader, target_loader, 
                disc_optimizer, target_optimizer, disc_criterion, target_criterion):
    for batch_idx, (source_data, source_target) in enumerate(source_loader):
        try:
            target_data, target_target = next(target_iter)
        except:
            target_iter = iter(target_loader)
            target_data, target_target = next(target_iter)
        
        # discriminator training
        target_model.eval()
        discriminator.train()

        target_data, source_data = target_data.to(device), source_data.to(device)
        
        # Forward pass on the source data
        source_pred = source_model(source_data)
        # Forward pass on the target data
        target_pred = target_model(target_data)

        # Setup discriminator task
        source_features = source_pred.view(source_pred.shape[0], -1)
        target_features = target_pred.view(target_pred.shape[0], -1)
        features = torch.cat([source_features, target_features])
        true_labels = torch.cat([torch.ones(source_features.shape[0]),
                                 torch.zeros(target_features.shape[0])])
        features, true_labels = features.to(device), true_labels.to(device)
        loss = disc_criterion(discriminator(features).squeeze(), true_labels)

        # Backward pass and optimization
        disc_optimizer.zero_grad()
        loss.backward()
        disc_optimizer.step()

        ## above trains the discriminator to tell if a given feature tensor 
        ## is from the source or target model


        ### Now train the target classifier to be better at fooling the discriminator
        target_model.train()
        discriminator.eval()

        target_pred = target_model(target_data)
        features = target_pred.view(target_pred.shape[0], -1)
        true_labels = torch.cat([torch.ones(features.shape[0])])
        features, true_labels = features.to(device), true_labels.to(device)
        loss = disc_criterion(discriminator(features).squeeze(), true_labels)

        # Backward pass and optimization
        target_optimizer.zero_grad()
        loss.backward()
        target_optimizer.step()

        ## above trains the target model to produce features that 
        ## more closely resemble that of the source model

        target_classifier.eval()
        if batch_idx % 10 == 0:
            outputs = target_classifier(target_data)
            _, predicted = torch.max(outputs.data, 1)
            total += target_target.size(0)
            correct += (predicted == target_target).sum().item()
            accuracy = 100 * correct / total
            print('Test Accuracy: {:.2f}%'.format(accuracy))
        
        # Print progress
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(source_data), len(source_loader.dataset),
                100. * batch_idx / len(source_loader), loss.item()))


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
    num_classes = 1000 # Number of classes in ImageNet dataset
    full_undistored_model = models.resnet50(pretrained=True)
    full_undistored_model.fc = nn.Linear(full_undistored_model.fc.in_features, num_classes)
    checkpoint_dir = '/content/drive/MyDrive/imagenet-mini/res50model_checkpoint.pth'
    checkpoint = torch.load(checkpoint_dir)
    full_undistored_model.load_state_dict(checkpoint)

    ## same model as above, but now just the feature producer, not the final layer
    source_model = models.resnet50(pretrained=True)
    checkpoint = torch.load(checkpoint_dir)
    source_model.load_state_dict(checkpoint)
    source_model = source_model.to(device)

    # setup new model based on resnet
    # model is trained up until final layer (to produce features on distorted images
    # similar to their undistorted counterparts)
    # and final layer is undisturbed to classify, as if features were undistorted
    target_model = Classifier(num_classes, full_undistored_model.fc.state_dict())
    target_model_features = target_model.features.to(device)

    discriminator = nn.Sequential(
            nn.Linear(target_model.features.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    discriminator = discriminator.to(device)

    target_optimizer = optim.Adam(target_model.parameters(), lr=0.001)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    # Set the alpha parameter for the domain confusion loss
    alpha = 0.2 # playing around with this

    # Train the model using domain adaptation
    for epoch in range(15): #may change
        train_model(epoch, discriminator, target_model_features, source_model, target_model, source_loader, target_loader, 
                disc_optimizer, target_optimizer, bce, criterion)

    # Evaluate
    test_loss, test_acc = target_model.evaluate(x_hold, y_hold)
    print('Test accuracy:', test_acc)

