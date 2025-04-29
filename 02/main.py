import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import random_split

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Data Preparation
# Choose dataset: CIFAR-10
# Note: You can change to MNIST by commenting/uncommenting the appropriate sections

# Define transformations for data augmentation
# For CIFAR-10
cifar_train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        # Normalize using ImageNet mean and std
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)

cifar_test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)

# For MNIST (uncomment if using MNIST instead)
"""
mnist_train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
"""

# Load the datasets
# CIFAR-10
train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=cifar_train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=cifar_test_transform
)

# Split training data into training and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# For MNIST (uncomment if using MNIST instead)
"""
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=mnist_train_transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=mnist_test_transform
)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
"""

# Create data loaders for batch processing
batch_sizes = {"small": 16, "medium": 32, "large": 64}
batch_size = batch_sizes["medium"]  # Choose batch size

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

# Get class names for CIFAR-10
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# For MNIST, classes would be digits 0-9
# classes = tuple(map(str, range(10)))


# Visualize some training images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.show()


# Get random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Show images
print("Sample training images:")
imshow(torchvision.utils.make_grid(images[:8]))
print(" ".join(f"{classes[labels[j]]:5s}" for j in range(8)))


# 2. Model Architecture
# Custom CNN Model for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # For CIFAR-10 (RGB images)
        # For MNIST: self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        # 128 * 4 * 4 = 2048 for CIFAR-10
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        # For MNIST: self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 classes for both CIFAR-10 and MNIST

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(-1, 128 * 4 * 4)  # For CIFAR-10
        # For MNIST: x = x.view(-1, 128 * 3 * 3)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Transfer Learning with ResNet18 (alternative approach)
class TransferLearningModel(nn.Module):
    def __init__(self, num_classes=10, freeze=False):
        super(TransferLearningModel, self).__init__()
        # Load pre-trained ResNet18 model
        self.model = torchvision.models.resnet18(weights="ResNet18_Weights.DEFAULT")

        # Freeze parameters
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        # self.model.fc = nn.Sequential(
        #     nn.Linear(num_features, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(512, num_classes),
        # )

    def forward(self, x):
        return self.model(x)


# Define the training function
def train_model(model, criterion, optimizer, dataloaders, num_epochs=25):
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    train_loss_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                dataloader = dataloaders["train"]
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = dataloaders["val"]

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # Backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc.item())
                val_loss_history.append(epoch_loss)
            if phase == "train":
                train_acc_history.append(epoch_acc.item())
                train_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return (
        model,
        train_loss_history,
        train_acc_history,
        val_loss_history,
        val_acc_history,
    )


# Function to evaluate model on test data
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Collect predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    return cm, 100 * correct / total


# Visualize training results
def plot_training_results(train_loss, val_loss, train_acc, val_acc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(train_loss, label="Training Loss")
    ax1.plot(val_loss, label="Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()

    ax2.plot(train_acc, label="Training Accuracy")
    ax2.plot(val_acc, label="Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()

    plt.tight_layout()
    plt.show()


# Plot confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


# Visualize misclassified images
def visualize_misclassified(model, test_loader, classes, num_images=10):
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Find misclassified images
            for i in range(len(preds)):
                if preds[i] != labels[i]:
                    misclassified_images.append(images[i].cpu())
                    misclassified_labels.append(labels[i].item())
                    misclassified_preds.append(preds[i].item())

                    if len(misclassified_images) >= num_images:
                        break

            if len(misclassified_images) >= num_images:
                break

    # Show misclassified images
    plt.figure(figsize=(12, 4))
    for i in range(min(num_images, len(misclassified_images))):
        plt.subplot(1, num_images, i + 1)
        img = misclassified_images[i] / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(
            f"True: {classes[misclassified_labels[i]]}\nPred: {classes[misclassified_preds[i]]}"
        )
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# 3. Main execution
if __name__ == "__main__":
    # Define model types and optimizers to compare
    model_types = ["SimpleCNN", "TransferLearning"]
    optimizer_types = ["adam", "sgd", "rmsprop"]
    dropout_rate = 0.25  # Dropout rate for regularization
    lr = 0.001  # Learning rate
    weight_decay = 1e-4  # L2 regularization
    num_epochs = 15  # Reduced epochs for faster comparison

    # Create dataloaders dictionary
    dataloaders = {"train": train_loader, "val": val_loader}

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Store results for comparison
    results = {}

    # Loop through each model type
    for model_type in model_types:
        results[model_type] = {}

        print(f"\n{'=' * 50}")
        print(f"Training with {model_type} architecture")
        print(f"{'=' * 50}\n")

        # Loop through each optimizer type
        for opt_type in optimizer_types:
            print(f"\n{'-' * 50}")
            print(f"Training {model_type} with {opt_type} optimizer")
            print(f"{'-' * 50}\n")

            # Initialize model
            if model_type == "SimpleCNN":
                model = SimpleCNN(dropout_rate=dropout_rate)
            else:
                model = TransferLearningModel()

            model = model.to(device)

            # Initialize optimizer
            if opt_type == "adam":
                optimizer = optim.Adam(
                    model.parameters(), lr=lr, weight_decay=weight_decay
                )
            elif opt_type == "sgd":
                optimizer = optim.SGD(
                    model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
                )
            else:  # RMSprop
                optimizer = optim.RMSprop(
                    model.parameters(), lr=lr, weight_decay=weight_decay
                )

            # Train model
            trained_model, train_loss, train_acc, val_loss, val_acc = train_model(
                model, criterion, optimizer, dataloaders, num_epochs=num_epochs
            )

            # Evaluate model on test set
            cm, test_accuracy = evaluate_model(trained_model, test_loader)

            # Store results
            results[model_type][opt_type] = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_acc": test_accuracy,
                "confusion_matrix": cm,
                "model": trained_model,
            }

            # Save the model
            torch.save(
                trained_model.state_dict(),
                f"{model_type}_{opt_type}_dropout{dropout_rate}.pth",
            )
            print(f"Model saved as {model_type}_{opt_type}_dropout{dropout_rate}.pth")

    # Visualize and compare results
    # Compare training and validation accuracy
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    for model_type in model_types:
        for opt_type in optimizer_types:
            plt.plot(
                results[model_type][opt_type]["train_acc"],
                label=f"{model_type}-{opt_type}",
            )
    plt.title("Training Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(2, 2, 2)
    for model_type in model_types:
        for opt_type in optimizer_types:
            plt.plot(
                results[model_type][opt_type]["val_acc"],
                label=f"{model_type}-{opt_type}",
            )
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(2, 2, 3)
    for model_type in model_types:
        for opt_type in optimizer_types:
            plt.plot(
                results[model_type][opt_type]["train_loss"],
                label=f"{model_type}-{opt_type}",
            )
    plt.title("Training Loss Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 4)
    for model_type in model_types:
        for opt_type in optimizer_types:
            plt.plot(
                results[model_type][opt_type]["val_loss"],
                label=f"{model_type}-{opt_type}",
            )
    plt.title("Validation Loss Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("optimizer_model_comparison.png")
    plt.show()

    # Compare test accuracies
    plt.figure(figsize=(12, 6))
    bar_width = 0.25
    x = np.arange(len(optimizer_types))

    for i, model_type in enumerate(model_types):
        test_accs = [results[model_type][opt]["test_acc"] for opt in optimizer_types]
        plt.bar(x + i * bar_width, test_accs, width=bar_width, label=model_type)

    plt.xlabel("Optimizer")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy Comparison")
    plt.xticks(x + bar_width / 2, optimizer_types)
    plt.legend()
    plt.savefig("test_accuracy_comparison.png")
    plt.show()

    # Find best model configuration
    best_acc = 0
    best_config = None

    for model_type in model_types:
        for opt_type in optimizer_types:
            acc = results[model_type][opt_type]["test_acc"]
            if acc > best_acc:
                best_acc = acc
                best_config = (model_type, opt_type)

    print(
        f"\nBest model configuration: {best_config[0]} with {best_config[1]} optimizer"
    )
    print(f"Best test accuracy: {best_acc:.2f}%")

    # Visualize misclassified images from the best model
    best_model = results[best_config[0]][best_config[1]]["model"]
    print("\nVisualization of misclassified images using the best model:")
    visualize_misclassified(best_model, test_loader, classes)

    # Plot confusion matrix for the best model
    print("\nConfusion matrix for the best model:")
    best_cm = results[best_config[0]][best_config[1]]["confusion_matrix"]
    plot_confusion_matrix(best_cm, classes)
