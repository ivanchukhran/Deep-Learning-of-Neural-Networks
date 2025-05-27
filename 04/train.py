import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

# Import our custom modules
from dataset import TwitterSentimentDataset
from rnn import TinyGRU, TinyLSTM, TinyRNN
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


class SentimentClassifier(nn.Module):
    """
    Sentiment classifier using RNN-based architectures.

    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Hidden dimension in the RNN/LSTM/GRU
        output_dim: Number of output classes
        model_type: Type of RNN model to use ('rnn', 'lstm', or 'gru')
        num_layers: Number of recurrent layers
        dropout: Dropout probability
        pad_idx: Index of the padding token
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        model_type: str = "lstm",
        num_layers: int = 2,
        dropout: float = 0.5,
        pad_idx: int = 0,
    ):
        super(SentimentClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Initialize the appropriate RNN model
        if model_type.lower() == "rnn":
            self.rnn = TinyRNN(
                input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers
            )
        elif model_type.lower() == "lstm":
            self.rnn = TinyLSTM(
                input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers
            )
        elif model_type.lower() == "gru":
            self.rnn = TinyGRU(
                input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers
            )
        else:
            raise ValueError("model_type must be 'rnn', 'lstm', or 'gru'")

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.model_type = model_type.lower()
        self.num_layers = num_layers

    def forward(self, text):
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]

        output, hidden = self.rnn(embedded)
        # since lstm is the only model that packs two tensors as the second return value
        # and we only need the first one in this case
        if self.model_type == "lstm":
            hidden, _ = hidden
        hidden = hidden[-1] if self.num_layers > 1 else hidden

        # Apply dropout and pass through linear layer
        dropped = self.dropout(hidden)
        return self.fc(dropped)


def train(args):
    """
    Main training function for the Twitter sentiment classifier.

    Args:
        args: Command line arguments containing training configuration
    """
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    train_dataset = TwitterSentimentDataset(
        root=args.data_dir,
        split="train",
        max_length=args.max_seq_length,
        vocab_size=args.vocab_size,
    )

    # Split into train and validation
    train_size = int(args.train_ratio * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Load test dataset
    test_dataset = TwitterSentimentDataset(
        root=args.data_dir, split="test", max_length=args.max_seq_length
    )

    print(
        f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    vocab_size = args.vocab_size
    pad_idx = 0

    # Define sentiment classes
    sentiment_classes = ["negative", "neutral", "positive"]

    # Create model
    print(f"Creating {args.model_type.upper()} model with {args.num_layers} layers...")
    model = SentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=len(sentiment_classes),
        model_type=args.model_type,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=pad_idx,
    )

    # Move model to device
    model = model.to(device)

    # Define loss function and optimizer
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Define learning rate scheduler
    scheduler = None
    if args.use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

    # Initialize training variables
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    # Initialize history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
    }

    # Save model configuration
    config = {
        "model_type": args.model_type,
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
        "vocab_size": vocab_size,
        "max_seq_length": args.max_seq_length,
        "dropout": args.dropout,
        "classes": sentiment_classes,
    }

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f)

    print("Starting training...")
    # Training loop
    for epoch in range(args.epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for batch in train_bar:
            optimizer.zero_grad()
            if isinstance(batch, list) and len(batch) == 2:
                texts, labels = batch
            else:
                texts, labels = batch

            texts, labels = texts.to(device), labels.to(device)

            outputs = model(texts)
            loss = criterion(outputs, labels)

            loss.backward()

            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

            # Update metrics
            train_loss += loss.item() * texts.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

            # Update progress bar
            train_bar.set_postfix(
                {"loss": loss.item(), "acc": train_correct / train_total}
            )

        # Calculate training metrics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]")
            for batch in val_bar:
                # Get batch
                if isinstance(batch, list) and len(batch) == 2:
                    texts, labels = batch
                else:
                    # Handle different dataset return types if needed
                    texts, labels = batch

                texts, labels = texts.to(device), labels.to(device)

                # Forward pass
                outputs = model(texts)
                loss = criterion(outputs, labels)

                # Update metrics
                val_loss += loss.item() * texts.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                # Update progress bar
                val_bar.set_postfix(
                    {"loss": loss.item(), "acc": val_correct / val_total}
                )

        # Calculate validation metrics
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])

        # Calculate epoch time
        epoch_time = time.time() - start_time

        # Print epoch summary
        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Time: {epoch_time:.2f}s - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save model
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, "best_model.pt")
            )
            print(f"Saved best model at epoch {epoch + 1}")
        else:
            patience_counter += 1

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "history": history,
            }
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt"),
            )

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Save training history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f)

    # Plot training history
    plot_training_history(
        history, os.path.join(args.output_dir, "training_history.png")
    )

    # Load best model for evaluation
    print(f"Loading best model from epoch {best_epoch + 1}")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt")))

    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc, y_pred, y_true = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Print classification report
    print("\nClassification Report:")
    report = classification_report(
        y_true, y_pred, target_names=sentiment_classes, digits=4
    )
    print(report)

    # Save classification report
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(str(report))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(
        cm, sentiment_classes, os.path.join(args.output_dir, "confusion_matrix.png")
    )

    print("Training completed!")
    return model, history


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on a dataset.

    Args:
        model: The model to evaluate
        data_loader: DataLoader for the dataset
        criterion: Loss function
        device: Device to use for evaluation

    Returns:
        loss: Average loss on the dataset
        accuracy: Accuracy on the dataset
        y_pred: Predicted labels
        y_true: True labels
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Get batch
            if isinstance(batch, list) and len(batch) == 2:
                texts, labels = batch
            else:
                # Handle different dataset return types if needed
                texts, labels = batch

            texts, labels = texts.to(device), labels.to(device)

            # Forward pass
            outputs = model(texts)
            loss = criterion(outputs, labels)

            # Update metrics
            total_loss += loss.item() * texts.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Store predictions and true labels
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, y_pred, y_true


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss and accuracy.

    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))

    # Plot training and validation loss
    plt.subplot(2, 1, 1)
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot training and validation accuracy
    plt.subplot(2, 1, 2)
    plt.plot(history["train_acc"], label="Training Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title("Accuracy vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(cm, classes, save_path=None):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        classes: Class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train Twitter Sentiment Classifier")

    # Data parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the dataset files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the model and results",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=50, help="Maximum sequence length"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=20000, help="Maximum vocabulary size"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data to use (rest is validation)",
    )

    # Model parameters
    parser.add_argument(
        "--model_type",
        type=str,
        default="lstm",
        choices=["rnn", "lstm", "gru"],
        help="Type of RNN model to use",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of recurrent layers"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=300, help="Dimension of word embeddings"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Dimension of hidden states"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Patience for early stopping (0 to disable)",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=0,
        help="Gradient clipping value (0 to disable)",
    )
    parser.add_argument(
        "--save_every", type=int, default=5, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--use_lr_scheduler", action="store_true", help="Use learning rate scheduler"
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no_cuda", action="store_true", help="Disable CUDA even if available"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    args.output_dir = os.path.join(
        args.output_dir, f"{args.model_type}_{args.num_layers}_layers"
    )

    # Train the model
    model, history = train(args)

    return model, history


if __name__ == "__main__":
    main()
