import torch
from early_stopping import EarlyStopping
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int,
    device: str = "cpu",
    early_stopping: EarlyStopping | None = None,
) -> dict:
    model_stats = {
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
    }

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_accuracy = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            x, y = batch
            x = x.view(-1, 28 * 28)
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            train_loss += loss.item()

            y_hat = output.argmax(dim=1)
            train_accuracy += (y_hat == y).float().mean().item()

            loss.backward()
            optimizer.step()
        model_stats["train_loss"].append(train_loss / len(train_loader))
        model_stats["train_accuracy"].append(train_accuracy / len(train_loader))
        model.eval()
        with torch.no_grad():
            test_loss = 0
            test_accuracy = 0
            for batch in test_loader:
                x, y = batch
                x = x.view(-1, 28 * 28)
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = nn.CrossEntropyLoss()(output, y)
                test_loss += loss.item()
                test_accuracy += (output.argmax(dim=1) == y).float().mean().item()
            test_loss = test_loss / len(test_loader)
            test_accuracy = test_accuracy / len(test_loader)

            model_stats["test_loss"].append(test_loss)
            model_stats["test_accuracy"].append(test_accuracy)

            print(
                f"Epoch {epoch + 1}: Train Loss: {model_stats['train_loss'][-1]:.4f}, Train Accuracy: {model_stats['train_accuracy'][-1]:.4f}, Test Loss: {model_stats['test_loss'][-1]:.4f}, Test Accuracy: {model_stats['test_accuracy'][-1]:.4f}"
            )

            if early_stopping is not None and early_stopping.should_stop(test_loss):
                break
    print("Training complete!")
    print(
        f"Final Train Loss: {model_stats['train_loss'][-1]:.4f}, Final Train Accuracy: {model_stats['train_accuracy'][-1]:.4f}, Final Test Loss: {model_stats['test_loss'][-1]:.4f}, Final Test Accuracy: {model_stats['test_accuracy'][-1]:.4f}"
    )

    return model_stats
